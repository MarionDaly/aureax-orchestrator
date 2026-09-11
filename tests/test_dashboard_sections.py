"""Behavioural guards for the dashboard's polled sections.

``tests/test_dashboard_static.py`` covers the whole of ``static/`` at the level
of "does it parse" and "does it call a route that exists". This module goes one
layer in on the sections that poll — lifecycle, configs, overview and the
per-node detail view — where the failures are not syntax errors but
stale-but-plausible UI: dots that stay green after contact is lost, an editor
buffer discarded on a tab switch, a grid rebuilt under a selection.

The sections import from ``/utils.js`` and ``/main.js``, which are server
absolute paths node cannot resolve. Rather than add a bundler or a DOM library —
the project ships no package.json and serves these files directly — each test
rewrites the module into a temp file: the ``/utils.js`` specifier repointed at
the real file so the helpers under it are the shipped ones, ``/main.js``
stripped in favour of a global, a small hand-rolled DOM prepended, and the
internals under test appended to an export list. What runs is therefore the real
section source over the real utils, unmodified below its import block.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from tests.conftest import PROJECT_ROOT

STATIC_DIR = PROJECT_ROOT / "orchestrator" / "dashboard" / "static"
SECTIONS = STATIC_DIR / "sections"
STYLE = STATIC_DIR / "style.css"

LIFECYCLE = SECTIONS / "lifecycle.js"
CONFIGS = SECTIONS / "configs.js"
OVERVIEW = SECTIONS / "overview.js"
CHAT = SECTIONS / "chat.js"
NODE = SECTIONS / "node.js"
# Not a section, but it is run the same way: the helpers the sections share live
# here, and asserting their output means running the shipped file.
UTILS = STATIC_DIR / "utils.js"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

# Enough of a browser to run the sections' logic. Deliberately minimal and
# hand-written: every element the code asks for is registered by id, and
# anything unasked-for answers null so a render bails the way it does in the
# real app when its container is gone.
DOM_STUB = """
const ELS = new Map();

function makeEl(id = "") {
    const node = {
        id, className: "", textContent: "", innerHTML: "", hidden: false,
        value: "", disabled: false, readOnly: false, dataset: {}, children: [],
        scrollTop: 0, clientHeight: 100, scrollHeight: 100,
        classList: { add() {}, toggle() {} },
        get options() { return this.children; },
        get childElementCount() { return this.children.length; },
        get firstElementChild() { return this.children[0]; },
        appendChild(c) {
            if (c && c.__fragment) { this.children.push(...c.children); return c; }
            this.children.push(c);
            return c;
        },
        removeChild(c) { this.children.splice(this.children.indexOf(c), 1); },
        remove() {},
        querySelector() { return null; },
        querySelectorAll() { return []; },
        addEventListener() {},
    };
    if (id) ELS.set(id, node);
    return node;
}

globalThis.__el = makeEl;
globalThis.document = {
    getElementById: (id) => ELS.get(id) || null,
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => makeEl(),
    createDocumentFragment: () => ({
        __fragment: true, children: [],
        appendChild(c) { this.children.push(c); return c; },
    }),
};
globalThis.window = { addEventListener() {}, confirm: () => true, alert() {}, prompt: () => "" };
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
globalThis.CSS = { escape: (s) => s };

// main.js's export. The section only asks whether its mount is still the
// current one, and every test here drives one mount at a time.
globalThis.isCurrent = () => true;
// The network. Every test that cares replaces this before calling in; the
// shim below routes the section's imported fetchJSON through it, so a section
// can be exercised without a server while still using the real utils.js for
// everything that has logic worth diverging from.
globalThis.fetchJSON = async () => ({});
"""

# The section's own import block. `/utils.js` is a server-absolute path node
# cannot resolve, so it is repointed at the real file rather than deleted:
# re-implementing escapeHtml and friends here would leave every section test
# asserting against a private copy that no longer matches what ships.
_UTILS_IMPORT_RE = re.compile(r'^(import .*? from )"/utils\.js";$', re.MULTILINE)
# Everything else in the block — `/main.js` — is stubbed as a global above.
# Matches only the server-absolute specifiers still left, so the rewritten
# utils.js line survives.
_IMPORT_RE = re.compile(r'^import .*? from "/[^"]*";$', re.MULTILINE)

_ALREADY_EXPORTED_RE = re.compile(r"^export (?:async )?function (\w+)", re.MULTILINE)


def _utils_shim(tmp_path) -> str:
    """The real utils.js, with fetchJSON alone routed through globalThis.

    An explicit local export shadows the same name coming from `export *`, so
    every other helper is the shipped one and only the network is swappable.
    """
    shim = tmp_path / "utils_shim.mjs"
    shim.write_text(
        f'export * from "{(STATIC_DIR / "utils.js").as_uri()}";\n'
        "export const fetchJSON = (...args) => globalThis.fetchJSON(...args);\n"
    )
    return shim.as_uri()


def run_section(tmp_path, section, exports: str, script: str) -> list:
    """Run `script` against `section`'s real source and return its JSON output."""
    src = _UTILS_IMPORT_RE.sub(rf'\1"{_utils_shim(tmp_path)}";', section.read_text())
    src = _IMPORT_RE.sub("", src)
    public = set(_ALREADY_EXPORTED_RE.findall(src))
    extra = [name.strip() for name in exports.split(",") if name.strip() not in public]
    mod = tmp_path / f"{section.stem}.mjs"
    tail = f"export {{ {', '.join(extra)} }};\n" if extra else ""
    mod.write_text(f"{DOM_STUB}\n{src}\n{tail}")
    proc = subprocess.run(
        [
            "node", "--input-type=module", "-e",
            f'import * as M from "{mod.as_uri()}";\n{script}',
        ],
        capture_output=True, text=True, check=False,
        # A section under test can leave a timer or an unresolved promise behind,
        # and node then keeps the event loop alive with the suite blocked on it.
        timeout=60,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return json.loads(proc.stdout)


# --- utils.js ---------------------------------------------------------------


@needs_node
def test_metric_unit_suffixes_are_anchored(tmp_path) -> None:
    """`includes("_c")` matched kv_cache_usage_ratio, so KV-cache occupancy —
    a 0..1 ratio the collector emits — rendered as "0.4°C".

    In utils.js because the Overview grid and the node detail view both format
    the same metric names, and two copies would render one screen's temperature
    as the other's ratio.
    """
    out = run_section(
        tmp_path, UTILS, "formatMetricValue",
        """
        console.log(JSON.stringify([
            M.formatMetricValue("kv_cache_usage_ratio", 0.4),
            M.formatMetricValue("temperature_c", 45.23),
            M.formatMetricValue("cpu_percent", 12.5),
            M.formatMetricValue("prompt_eval_ms", 1234.6),
            M.formatMetricValue("memory_used_gb", 3.456),
            M.formatMetricValue("uptime_seconds", 1209600),
            M.formatMetricValue("uptime_seconds", 7200),
            M.formatMetricValue("uptime_seconds", 42.4),
        ]));
        """,
    )
    assert out == ["0.40", "45.2°C", "12.5%", "1235 ms", "3.46 GB",
                   "14.0 d", "2.0 h", "42 s"]


def test_the_restart_badge_escapes_its_count() -> None:
    """api.py proxies restartCount straight out of the kubectl JSON with no int
    coercion; it was the one server value interpolated into HTML unescaped.

    In utils.js because both the Overview grid and the node detail view render
    the badge, and a second copy is a second place for the escape to be lost."""
    src = UTILS.read_text()
    assert "${escapeHtml(c.restarts)} restarts" in src
    assert "${c.restarts} restarts" not in src


# --- overview.js ------------------------------------------------------------


@needs_node
def test_grids_are_not_rewritten_when_the_payload_is_unchanged(tmp_path) -> None:
    """An unconditional innerHTML write six times a minute destroys whatever the
    operator has selected — typically a .pod-error crash message mid-copy.

    The memo is keyed on the container element (utils.js::setHtmlIfChanged), so
    a re-mount — which rebuilds the grids via ``root.innerHTML`` — starts from
    an empty memo and cannot leave the new, empty container unwritten.
    """
    out = run_section(
        tmp_path, OVERVIEW, "setHtmlIfChanged",
        """
        const el = __el("cluster-info");
        M.setHtmlIfChanged(el, "<p>same</p>");
        el.innerHTML = "SELECTION";
        M.setHtmlIfChanged(el, "<p>same</p>");
        const skipped = el.innerHTML === "SELECTION";
        // What mount() does: a fresh container, which the memo has never seen.
        const rebuilt = __el("cluster-info-2");
        M.setHtmlIfChanged(rebuilt, "<p>same</p>");
        console.log(JSON.stringify([skipped, rebuilt.innerHTML]));
        """,
    )
    assert out == [True, "<p>same</p>"]
    # The grids are rebuilt on every mount, which is what makes the per-element
    # memo self-clearing. If mount ever stopped doing that, the memo would need
    # an explicit reset again.
    assert "root.innerHTML = `" in OVERVIEW.read_text()


@needs_node
def test_overview_poll_does_not_overlap(tmp_path) -> None:
    """Each tick opens three fetches; without a guard a slow backend stacked
    them against the browser's six-connection-per-origin budget."""
    out = run_section(
        tmp_path, OVERVIEW, "mount, unmount, refresh",
        """
        let calls = 0;
        globalThis.fetchJSON = async () => {
            calls++;
            await new Promise(r => setTimeout(r, 20));
            return { _error: "x" };
        };
        M.mount(__el("root"));
        M.refresh();
        M.refresh();
        await new Promise(r => setTimeout(r, 60));
        M.unmount();
        console.log(JSON.stringify(calls));
        """,
    )
    assert out == 3, "expected one refreshOnce (three fetches), not three"


@needs_node
def test_last_update_is_not_stamped_after_unmount(tmp_path) -> None:
    """#last-update lives in the page header, so an in-flight refresh re-stamped
    it after the tab switch and the timestamp then sat frozen on every other
    tab."""
    out = run_section(
        tmp_path, OVERVIEW, "mount, unmount, refreshOnce",
        """
        const last = __el("last-update");
        globalThis.fetchJSON = async () => ({ _error: "x" });
        await M.refreshOnce();                 // never mounted
        const beforeMount = last.textContent;
        M.mount(__el("root"));
        await new Promise(r => setTimeout(r, 10));
        const stamped = last.textContent.startsWith("Updated ");
        M.unmount();
        last.textContent = "MARK";
        await M.refreshOnce();                 // in-flight poll resuming late
        console.log(JSON.stringify([beforeMount, stamped, last.textContent]));
        """,
    )
    assert out == ["", True, "MARK"]


def test_overview_names_the_schema_fields_its_label_literals_copy() -> None:
    """`aureax.io/role` / `edge` are configurable (kubernetes.node_label,
    kubernetes.edge_label_value) — changing either drops the badge with no
    error, so the coupling has to be greppable from here."""
    src = OVERVIEW.read_text()
    comment = src[: src.index('node.labels?.["aureax.io/role"]')]
    for token in ("node_label", "edge_label_value", "schema.py", "shard_label"):
        assert token in comment.rsplit("const nodesHtml", 1)[-1], f"missing {token}"


# --- lifecycle.js -----------------------------------------------------------


@needs_node
def test_source_filter_options_come_back_after_a_remount(tmp_path) -> None:
    """A module-level Set of seen sources outlived unmount(), so every node:<id>
    option was suppressed on the rebuilt <select> and could never be selected
    again."""
    out = run_section(
        tmp_path, LIFECYCLE, "registerSource, unmount",
        """
        const first = __el("log-source");
        M.registerSource("node:edge-01");
        M.registerSource("node:edge-01");
        const added = first.options.map(o => o.value);
        M.unmount();
        const rebuilt = __el("log-source");       // mount() rewrites the element
        M.registerSource("node:edge-01");
        console.log(JSON.stringify([added, rebuilt.options.map(o => o.value)]));
        """,
    )
    assert out == [["node:edge-01"], ["node:edge-01"]]


@needs_node
def test_lost_contact_greys_the_dots_and_re_enables_the_controls(tmp_path) -> None:
    """Swallowing the poll error left every dot on its last `online` class, so
    the operator read green for nodes whose reachability was unknown — and an
    error landing while STARTING left every control disabled."""
    out = run_section(
        tmp_path, LIFECYCLE, "refreshState",
        """
        const dots = [__el(), __el()];
        for (const d of dots) d.className = "status-dot online";
        const buttons = [
            { dataset: { action: "start", node: "" }, disabled: true },
            { dataset: { action: "stop", node: "n1" }, disabled: true },
        ];
        const alert = __el("lifecycle-alert");
        alert.hidden = true;
        document.querySelectorAll = (q) => (q.includes("data-node-dot") ? dots : buttons);
        globalThis.fetchJSON = async () => ({ _error: "timeout", _message: "no response after 10s" });
        await M.refreshState();
        console.log(JSON.stringify([
            dots.map(d => d.className),
            buttons.map(b => b.disabled),
            alert.hidden,
            alert.textContent,
        ]));
        """,
    )
    assert out == [
        ["status-dot unknown", "status-dot unknown"],
        [False, False],
        False,
        "Lifecycle state unavailable: no response after 10s",
    ]


@needs_node
def test_lifecycle_poll_does_not_overlap(tmp_path) -> None:
    """1.5s ticks against a backend that answers slowly rather than not at all."""
    out = run_section(
        tmp_path, LIFECYCLE, "refreshState",
        """
        let calls = 0;
        document.querySelectorAll = () => [];
        globalThis.fetchJSON = async () => {
            calls++;
            await new Promise(r => setTimeout(r, 20));
            return { _error: "x" };
        };
        await Promise.all([M.refreshState(), M.refreshState(), M.refreshState()]);
        console.log(JSON.stringify(calls));
        """,
    )
    assert out == 1


@needs_node
def test_log_lines_are_batched_into_one_dom_write(tmp_path) -> None:
    """The server replays its whole buffer as one event per line on every mount;
    appending each separately read the pane's geometry twice per line."""
    out = run_section(
        tmp_path, LIFECYCLE, "appendLine",
        """
        const pane = __el("log-pane");
        let reads = 0;
        Object.defineProperty(pane, "scrollHeight", { get() { reads++; return 100; } });
        __el("log-level").value = "debug";
        __el("log-source").value = "";
        for (let i = 0; i < 5; i++) {
            M.appendLine({ level: "info", source: "lifecycle", text: `line ${i}` });
        }
        const beforeFrame = pane.children.length;
        await new Promise(r => setTimeout(r, 10));
        console.log(JSON.stringify([beforeFrame, pane.children.length, reads]));
        """,
    )
    assert out == [0, 5, 2], "expected one geometry read pair for the whole batch"


@needs_node
def test_409_reports_the_reason_the_backend_gave(tmp_path) -> None:
    """lifecycle_api answers 409 for a BusyError AND for an Access-tab config
    apply, which is the case that sets `detail`. The old text named the wrong
    one, sending the operator to the wrong tab."""
    out = run_section(
        tmp_path, LIFECYCLE, "onClick",
        """
        const alerts = [];
        globalThis.window = { confirm: () => true, alert: (m) => alerts.push(m) };
        globalThis.fetchJSON = async () => ({ _error: 409, detail: "a config apply is running" });
        await M.onClick({ target: { closest: () => ({ dataset: { action: "stop", node: "shard-head" } }) } });
        console.log(JSON.stringify(alerts));
        """,
    )
    assert out == ["Cannot stop: a config apply is running. Wait for it to finish."]


@needs_node
def test_unknown_state_gets_the_cancel_warning(tmp_path) -> None:
    """fetchJSON never throws, so a failed state probe simply has no `state` —
    and try_stop cancels whenever the manager is STARTING. Consenting to "state
    is preserved" and getting a SIGTERMed setup-all.sh is the wrong way round."""
    out = run_section(
        tmp_path, LIFECYCLE, "onClick",
        """
        const prompts = [];
        globalThis.window = { confirm: (m) => { prompts.push(m); return false; }, alert() {} };
        globalThis.fetchJSON = async () => ({ _error: "network", _message: "down" });
        await M.onClick({ target: { closest: () => ({ dataset: { action: "stop", node: "" } }) } });
        console.log(JSON.stringify(prompts));
        """,
    )
    assert len(out) == 1
    assert "setup-all.sh" in out[0]
    assert "State is preserved" not in out[0]


@needs_node
def test_node_list_failure_is_shown_and_still_retried(tmp_path) -> None:
    """/api/lifecycle/nodes answers 503 when the network config is unreadable.
    An empty panel is indistinguishable from an empty fleet — and the self-heal
    must not read the error placeholder as "already populated"."""
    out = run_section(
        tmp_path, LIFECYCLE, "populateNodes, refreshState",
        """
        let nodeCalls = 0;
        globalThis.fetchJSON = async (url) => {
            if (url === "/api/lifecycle/nodes") {
                nodeCalls++;
                return { _error: 503, _message: "network config unreadable: YAMLError" };
            }
            return { state: "idle", nodes: [] };
        };
        const panel = __el("per-node-controls");
        panel.querySelectorAll = () => [];        // no .node-controls rows rendered
        document.querySelectorAll = () => [];
        await M.populateNodes();
        const shown = panel.innerHTML;
        await M.refreshState();
        console.log(JSON.stringify([shown, nodeCalls]));
        """,
    )
    shown, node_calls = out
    assert "Node list unavailable" in shown
    assert "network config unreadable: YAMLError" in shown
    assert node_calls == 2, "self-heal must retry through the error placeholder"


@needs_node
def test_the_node_panel_self_heal_backs_off(tmp_path) -> None:
    """The error placeholder never satisfies the retry condition, so an endpoint
    that stays down would be re-fetched on every 1.5s tick — and each call
    re-reads and re-parses the whole network config."""
    out = run_section(
        tmp_path, LIFECYCLE, "refreshState",
        """
        let nodeCalls = 0;
        globalThis.fetchJSON = async (url) => {
            if (url === "/api/lifecycle/nodes") {
                nodeCalls++;
                return { _error: 503, _message: "network config unreadable" };
            }
            return { state: "idle", nodes: [] };
        };
        const panel = __el("per-node-controls");
        panel.querySelectorAll = () => [];        // no .node-controls rows rendered
        document.querySelectorAll = () => [];
        for (let i = 0; i < 10; i++) await M.refreshState();
        console.log(JSON.stringify(nodeCalls));
        """,
    )
    assert out >= 2, "the panel must still be retried"
    assert out < 10, "a persistently failing /nodes must not be re-fetched every tick"


def test_lifecycle_mount_sequences_its_two_fetches() -> None:
    """/state is served from a 3s cache while /nodes re-parses the network config
    per call, so /state won the race and the self-heal fired a second /nodes."""
    src = LIFECYCLE.read_text()
    assert "Promise.all([populateNodes()" not in src
    assert src.index("await populateNodes();") < src.index("await refreshState();")


def test_lifecycle_log_stream_clears_its_retry() -> None:
    """A non-event-stream response closes the connection for good, and every
    handle the stream opened is released on unmount."""
    src = LIFECYCLE.read_text()
    assert "EventSource.CLOSED" in src
    unmount = src[src.index("export function unmount"):src.index("const nodeMeta")]
    for handle in ("retryId", "flushHandle", "pending", "stateInFlight"):
        assert handle in unmount, f"unmount() must release {handle}"


def test_the_state_poll_ticks_faster_than_the_snapshot_it_reads() -> None:
    """`nodes` in /api/lifecycle/state comes from a snapshot cached for
    ``_NODE_STATUS_TTL``. Polling at or above that period means the buttons —
    which read the state-machine half of the same response, and that half is
    uncached — react a whole cache generation late."""
    from orchestrator.lifecycle import _NODE_STATUS_TTL

    poll_ms = int(re.search(r"const STATE_POLL_MS = (\d+);", LIFECYCLE.read_text()).group(1))

    assert poll_ms / 1000 < _NODE_STATUS_TTL


def test_the_chat_and_api_buttons_are_withheld_using_the_schemas_role_literal() -> None:
    """A shard worker runs rpc-server, serves no MCP endpoint and holds no key,
    so a Chat or API link on one lands elsewhere without saying so. The literal
    the section tests is SHARD_WORKER_ROLE, and a rename there has to be
    greppable from here or the check quietly matches nothing."""
    from orchestrator.config.schema import SHARD_WORKER_ROLE

    assert f'n.role !== "{SHARD_WORKER_ROLE}"' in LIFECYCLE.read_text()


def test_debug_sorts_below_every_selectable_log_level() -> None:
    """`debug` arrives on entries shipped up from edge nodes — the run-up a
    crashing child ships. It has to sort BELOW info or Info+ cannot hide it,
    and below every other option or the lowest selectable level would hide it
    with nothing able to show it again."""
    src = LIFECYCLE.read_text()
    order = {
        k: int(v)
        for k, v in re.findall(
            r"(\w+): (-?\d+)",
            re.search(r"const LEVEL_ORDER = \{([^}]*)\}", src).group(1),
        )
    }
    levels = re.findall(
        r'<option value="(\w+)"',
        re.search(r'id="log-level".*?</select>', src, re.S).group(0),
    )

    assert order["debug"] < min(v for k, v in order.items() if k != "debug")
    assert levels[0] == "debug", "the lowest selectable level must be the lowest-sorting one"


@needs_node
def test_a_row_below_the_selected_level_is_filtered_out(tmp_path) -> None:
    """The filter is a >= comparison on LEVEL_ORDER, so a level missing from
    that table scores 0 and renders as if it were info."""
    out = run_section(
        tmp_path, LIFECYCLE, "passesFilter",
        """
        const row = (level) => ({ dataset: { level, source: "lifecycle" } });
        console.log(JSON.stringify([
            M.passesFilter(row("debug"), "info", ""),
            M.passesFilter(row("debug"), "debug", ""),
            M.passesFilter(row("info"), "info", ""),
            M.passesFilter(row("info"), "warn", ""),
            M.passesFilter(row("error"), "warn", ""),
            M.passesFilter(row("info"), "info", "node:edge-01"),
        ]));
        """,
    )
    assert out == [False, True, True, False, True, False]


@needs_node
def test_a_per_node_stop_names_every_member_of_the_group(tmp_path) -> None:
    """A group serves ONE model across all its members, so stopping any one of
    them stops all of them and a prompt naming one node understates it. The
    member list is the wire's `group_members`, kept under a different key in the
    section's own map."""
    out = run_section(
        tmp_path, LIFECYCLE, "populateNodes, onClick",
        """
        const prompts = [];
        globalThis.window = { confirm: (m) => { prompts.push(m); return false; }, alert() {} };
        globalThis.fetchJSON = async (url) => {
            if (url === "/api/lifecycle/nodes") return [
                { node_id: "shard-w1", role: "shard_worker", group: "qwen-a",
                  group_members: ["shard-head", "shard-w1", "shard-w2"] },
                { node_id: "edge-01", role: "edge", group: null, group_members: [] },
            ];
            return { state: "idle", nodes: [] };
        };
        __el("per-node-controls");
        await M.populateNodes();
        const stop = (node) =>
            M.onClick({ target: { closest: () => ({ dataset: { action: "stop", node } }) } });
        await stop("shard-w1");
        await stop("edge-01");
        console.log(JSON.stringify(prompts));
        """,
    )
    grouped, standalone = out
    assert 'group "qwen-a"' in grouped
    assert "shard-head, shard-w2" in grouped
    assert "shard-w1" in grouped
    assert standalone == "Stop node edge-01? The pod will be terminated."


def test_the_state_poll_handle_is_only_assigned_by_a_live_mount() -> None:
    """The two fetches mount() awaits mean unmount() may already have run and
    cleared the handle, and a later assignment then leaves an interval nothing
    can stop."""
    src = LIFECYCLE.read_text()
    bail = src.index("if (!isCurrent(token)) return;")
    assert src.index("pollId = setInterval(refreshState, STATE_POLL_MS);") > bail


# --- configs.js -------------------------------------------------------------


@needs_node
def test_unsaved_draft_survives_a_tab_switch(tmp_path) -> None:
    """main.js unmounts on every hashchange, so clicking another tab mid-edit
    wiped the buffer with no prompt — while this same section confirms before
    strictly smaller losses."""
    out = run_section(
        tmp_path, CONFIGS, "state, unmount",
        """
        M.state.selected = { name: "a.yaml", dirty: true, content: "edited" };
        M.unmount();
        const kept = M.state.selected && M.state.selected.content;
        M.state.selected.dirty = false;
        M.unmount();
        console.log(JSON.stringify([kept, M.state.selected]));
        """,
    )
    assert out == ["edited", None]


@needs_node
def test_status_line_has_one_precedence_rule(tmp_path) -> None:
    """renderEditor and renderEditorMeta disagreed: a full render() while dirty
    reinstated the stale "Saved 14:02" the incremental path suppresses."""
    out = run_section(
        tmp_path, CONFIGS, "state, statusText",
        """
        M.state.errors = [];
        M.state.statusMessage = "Saved 14:02:11";
        const dirty = M.statusText({ dirty: true });
        const clean = M.statusText({ dirty: false });
        M.state.errors = [{ loc: "model", msg: "bad" }];
        M.state.statusMessage = "Validation failed (1)";
        const invalid = M.statusText({ dirty: true });
        console.log(JSON.stringify([dirty, clean, invalid]));
        """,
    )
    assert out == ["Unsaved changes", "Saved 14:02:11", "Validation failed (1)"]
    src = CONFIGS.read_text()
    assert src.count("statusText(sel)") >= 3, "renderEditor and renderEditorMeta must share it"


@needs_node
def test_a_failed_save_is_reported_over_the_dirty_flag(tmp_path) -> None:
    """A 400 detail, a 403 view-only, a 404 or a fetchJSON timeout carries no
    `errors` array, so the dirty flag swallowed "Save failed: ..." — the only
    evidence the operator gets that the write did not happen."""
    out = run_section(
        tmp_path, CONFIGS, "state, statusText, save",
        """
        globalThis.fetchJSON = async () => ({
            _error: 400, _message: "password lines were added or removed",
        });
        M.state.selected = {
            name: "aureax-network.yaml", kind: "network", editable: true,
            original: "a", content: "b", dirty: true,
        };
        await M.save();
        console.log(JSON.stringify([
            M.statusText(M.state.selected), M.state.selected.dirty,
        ]));
        """,
    )
    assert out == ["Save failed: password lines were added or removed", True]


@needs_node
def test_a_redacted_config_says_what_the_marker_means(tmp_path) -> None:
    """GET /api/config/{name} replaces secrets with a visible marker.

    Without a note the operator sees the marker in the buffer with nothing
    saying whether saving writes it out over the real password — and the
    answer differs by kind, which is why the text is chosen from `editable`.
    """
    out = run_section(
        tmp_path, CONFIGS, "state, selectFile, redactionNote",
        """
        globalThis.fetchJSON = async () => ({
            name: "aureax-network.yaml", kind: "network", editable: true,
            content: "password: <redacted>", redacted: true,
        });
        await M.selectFile("aureax-network.yaml");
        console.log(JSON.stringify([
            M.state.selected.redacted,
            M.redactionNote({ editable: true }).includes("restored on save"),
            M.redactionNote({ editable: false }).includes("view-only"),
        ]));
        """,
    )
    assert out == [True, True, True]


@needs_node
def test_a_stale_file_read_cannot_win(tmp_path) -> None:
    """Two quick sidebar clicks run two concurrent loads; whichever landed last
    used to win, so a slower first read left the operator editing file A with
    the highlight moved back to A as well."""
    out = run_section(
        tmp_path, CONFIGS, "state, selectFile",
        """
        globalThis.fetchJSON = async (url) => {
            const name = decodeURIComponent(url.split("/").pop());
            await new Promise(r => setTimeout(r, name === "slow.yaml" ? 40 : 5));
            return { name, kind: "edge-node", editable: true, content: name };
        };
        await Promise.all([M.selectFile("slow.yaml"), M.selectFile("fast.yaml")]);
        console.log(JSON.stringify(M.state.selected.name));
        """,
    )
    assert out == "fast.yaml"


@needs_node
def test_tab_leaves_view_only_buffers_alone_and_save_survives_caps_lock(tmp_path) -> None:
    """A readonly <textarea> still fires keydown and .value assignment bypasses
    the guard, so Tab marked a generated config dirty. KeyboardEvent.key carries
    the produced character, so Cmd+S with Caps Lock on was "S" and fell through
    to the browser's Save-Page dialog."""
    out = run_section(
        tmp_path, CONFIGS, "state, onEditorKeyDown",
        """
        M.state.selected = null;
        const ta = (readOnly) => ({
            id: "editor-textarea", readOnly, value: "abc",
            selectionStart: 1, selectionEnd: 1, dispatchEvent() {},
        });
        let prevented = 0;
        const view = ta(true);
        M.onEditorKeyDown({ key: "Tab", target: view, preventDefault() { prevented++; } });
        const edit = ta(false);
        M.onEditorKeyDown({ key: "Tab", target: edit, preventDefault() { prevented++; } });
        let savePrevented = 0;
        M.onEditorKeyDown({ key: "S", metaKey: true, target: view,
                            preventDefault() { savePrevented++; } });
        console.log(JSON.stringify([view.value, prevented, edit.value, savePrevented]));
        """,
    )
    assert out == ["abc", 1, "a  bc", 1]


@needs_node
def test_save_keeps_keystrokes_typed_during_the_put(tmp_path) -> None:
    """sel.content was re-read AFTER the await, so anything typed while the PUT
    was in flight was marked saved and dropped on the next tab switch. The same
    run proves every successful save raises the Regenerate banner."""
    out = run_section(
        tmp_path, CONFIGS, "state, save",
        """
        M.state.selected = {
            name: "aureax-shard-head-35b.yaml", kind: "edge-node", editable: true,
            original: "one", content: "one two", dirty: true,
        };
        globalThis.fetchJSON = async (url, opts) => {
            if (opts && opts.method === "PUT") {
                M.state.selected.content = "one two three";   // typed mid-request
                return { ok: true };
            }
            return [];
        };
        await M.save();
        const sel = M.state.selected;
        console.log(JSON.stringify([sel.original, sel.content, sel.dirty, M.state.templateStale]));
        """,
    )
    assert out == ["one two", "one two three", True, True]


@needs_node
def test_a_whitespace_prompt_keeps_the_remembered_ip(tmp_path) -> None:
    """" " is truthy, so the guard passed, ip.trim() stored "" over the
    remembered value and the empty POST was rejected — a failure alert AND a
    lost prefill."""
    out = run_section(
        tmp_path, CONFIGS, "onBannerClick",
        """
        const store = { "aureax.orchestrator_ip": "10.77.0.5" };
        globalThis.localStorage = {
            getItem: (k) => (k in store ? store[k] : null),
            setItem: (k, v) => { store[k] = v; },
        };
        let posted = 0;
        globalThis.fetchJSON = async () => { posted++; return {}; };
        globalThis.window = { addEventListener() {}, prompt: () => "   ", alert() {} };
        await M.onBannerClick({
            target: { closest: () => ({ dataset: { action: "regenerate" } }) },
        });
        console.log(JSON.stringify([store["aureax.orchestrator_ip"], posted]));
        """,
    )
    assert out == ["10.77.0.5", 0]


@needs_node
def test_in_flight_status_does_not_rebuild_the_textarea(tmp_path) -> None:
    """render() rewrites the editor including the <textarea>, so announcing
    "Saving…"/"Loading…" through it discarded focus, caret, scroll and the
    native undo stack of the element being edited."""
    out = run_section(
        tmp_path, CONFIGS, "state, setStatus",
        """
        const editor = __el("config-editor");
        editor.innerHTML = "UNTOUCHED";
        const status = { textContent: "Clean" };
        let staleErrorsCleared = false;
        editor.querySelector = (q) => (q === ".editor-status" ? status : null);
        editor.querySelectorAll = (q) =>
            (q === ".pod-error" ? [{ remove() { staleErrorsCleared = true; } }] : []);
        M.setStatus("Saving…");
        console.log(JSON.stringify([
            status.textContent, editor.innerHTML, M.state.statusMessage, staleErrorsCleared,
        ]));
        """,
    )
    assert out == ["Saving…", "UNTOUCHED", "Saving…", True]
    src = CONFIGS.read_text()
    for phrase in ('setStatus("Saving…")', 'setStatus("Loading…")'):
        assert phrase in src, f"{phrase} must replace the pre-request render()"


def test_configs_uses_the_shared_fetch_error_helpers() -> None:
    """selectFile hand-rolled both halves of the fetchJSON contract while the
    imported isError sat unused, so one file carried two conventions."""
    src = CONFIGS.read_text()
    assert "if (isError(data))" in src
    assert "errorText(data)" in src
    assert "data._error" not in src
    assert "data._message" not in src


@needs_node
def test_a_failed_config_list_is_reported_rather_than_read_as_an_empty_tree(tmp_path) -> None:
    """fetchJSON never throws, so a transport error or the 10s timeout arrives
    as a non-array. Without the isError() check it collapsed to [] and the
    sidebar printed "No configs found." — a claim about the repository, made
    when nothing was read at all."""
    out = run_section(
        tmp_path, CONFIGS, "loadFileList, renderSidebar, state",
        """
        globalThis.fetchJSON = async () => ({ _error: "timeout", _message: "no response after 10s" });
        const el = __el("config-sidebar");
        await M.loadFileList();
        M.renderSidebar();
        console.log(JSON.stringify([el.innerHTML, M.state.files.length]));
        """,
    )
    shown, files = out
    assert "No configs found." not in shown
    assert "no response after 10s" in shown
    assert files == 0


@needs_node
def test_chat_survives_a_browser_that_blocks_site_data(tmp_path) -> None:
    """A browser configured to block site data throws on window.localStorage
    itself. On the read that aborted mount() before the selects were built or a
    session opened — main.js only console.errors it, so the tab stayed blank —
    and on the write it aborted the target switch before closeStream(), leaking
    the previous target's EventSource."""
    out = run_section(
        tmp_path, CHAT, "mount, onTargetChange, state",
        """
        globalThis.localStorage = {
            get getItem() { throw new Error("site data blocked"); },
            get setItem() { throw new Error("site data blocked"); },
        };
        globalThis.fetchJSON = async (url) =>
            (url === "/api/chat/targets"
                ? [{ id: "node-a", label: "a" }, { id: "node-b", label: "b" }]
                : []);
        // mount() writes root.innerHTML and then looks its controls up by id;
        // the stub DOM does not parse HTML, so they are registered up front.
        for (const id of [
            "chat-form", "chat-prompt", "chat-send", "chat-target", "chat-session",
            "chat-reset", "chat-delete", "chat-messages", "chat-status",
        ]) __el(id);
        globalThis.window.location = { hash: "" };
        globalThis.history = { replaceState() {} };
        const root = __el("root");
        await M.mount(root, "", 1);
        const mounted = M.state.targetId;

        // The stream the switch must close before it opens the next one.
        let closed = false;
        M.state.stream = { close() { closed = true; } };
        await M.onTargetChange({ target: { value: "node-b" } });
        console.log(JSON.stringify([mounted, M.state.targetId, closed, M.state.stream]));
        """,
    )
    mounted, switched, closed, stream = out
    assert mounted == "node-a", "mount() aborted on the localStorage read"
    assert switched == "node-b", "the target switch aborted on the localStorage write"
    assert closed and stream is None, (
        "closeStream() was skipped, leaking the previous target's EventSource"
    )


def test_the_apply_dialog_names_the_wider_push_set() -> None:
    """access_api._apply_push_set adds every node whose config embeds this
    node's MCP key — a planner's ${mcp_endpoints} list — and the server pushes
    and restarts those too. A dialog saying "(node only)" has the operator
    consent to something narrower than what runs."""
    src = (SECTIONS / "access.js").read_text()
    assert "only)`" not in src, "the confirm dialog still claims a single-node scope"
    assert "MCP key" in src


def test_every_saveable_config_raises_the_regenerate_banner() -> None:
    """Every SSH target's config_file is a template resolved per target, and
    config_api 403s the only non-source kind — so the old two-kind allowlist let
    a save of any file the fleet actually uses leave the generated configs stale
    with no banner."""
    src = CONFIGS.read_text()
    assert "TEMPLATE_KINDS" not in src
    assert "state.templateStale = true;" in src


# --- node.js ----------------------------------------------------------------


# The four containers mount() writes into. The stub DOM does not parse HTML, so
# they are registered up front exactly as the chat test does.
_NODE_CONTAINERS = """
for (const id of ["node-identity", "node-k8s", "node-metrics", "node-logs", "node-updated"]) __el(id);
"""

# One payload in the shape GET /api/nodes/{node_id} answers with; `host` is the
# field the assertions look for on screen.
_NODE_PAYLOAD = """
const payload = (host) => ({
    node_id: "edge-01",
    identity: { display_name: "Edge 01", host, tags: ["arm64"], role: "edge",
                mcp_enabled: true, group: null, model: {} },
    k8s: { status: "ok", matched_by: null, node: null, pods: [] },
    metrics: { status: "ok", last_seen: null, latest: {}, history: {} },
    logs: { status: "ok", entries: [] },
});
"""


@needs_node
def test_a_sparkline_only_plots_what_it_can_measure(tmp_path) -> None:
    """One point is a dot, not a trend; a flat series has no range to scale
    against and divided by zero it painted `NaN` into the attribute; and a
    non-numeric sample — the metrics column has REAL affinity but the ingest
    route does not type-check — must not be plotted as a zero."""
    out = run_section(
        tmp_path, NODE, "sparklinePoints",
        """
        console.log(JSON.stringify([
            M.sparklinePoints([["t0", 0], ["t1", 5], ["t2", 10]]),
            M.sparklinePoints([["t0", 4], ["t1", 4]]),
            M.sparklinePoints([["t0", 1], ["t1", "boom"], ["t2", 3]]),
            M.sparklinePoints([["t0", 7]]),
            M.sparklinePoints(null),
        ]));
        """,
    )
    rising, flat, junk, single, missing = out
    # Inset one stroke width inside the 0..24 viewBox at each end: with
    # `vector-effect: non-scaling-stroke` a point at 0 or 24 has half its stroke
    # clipped, which flattens the peak of every chart that reaches its own max.
    assert rising == "0.00,22.00 50.00,12.00 100.00,2.00", "y is inverted: 22 is the floor"
    assert flat == "0.00,12.00 100.00,12.00", "a flat series is drawn down the middle"
    assert junk == "0.00,22.00 100.00,2.00", "the unreadable sample is dropped, not zeroed"
    assert single == ""
    assert missing == ""


@needs_node
def test_a_log_row_escapes_its_text_and_classes_only_known_levels(tmp_path) -> None:
    """Both fields reach the store through the unauthenticated /api/v1/logs
    ingest. `text` is a crash line rendered as markup if unescaped, and `level`
    lands in a class attribute — where an unlisted value would be interpolated
    straight into it."""
    out = run_section(
        tmp_path, NODE, "renderLogs",
        """
        const pane = __el("node-logs");
        M.renderLogs({ status: "ok", entries: [
            { level: "warn", text: "<img src=x onerror=alert(1)>" },
            { level: 'error" onmouseover="steal(', text: "spoofed level" },
        ]});
        console.log(JSON.stringify(pane.innerHTML));
        """,
    )
    assert "log-line log-warn" in out
    assert "<img" not in out and "&lt;img" in out
    assert "onmouseover" not in out
    assert out.count("log-line log-info") == 1, "an unlisted level must fall back to info"


@needs_node
def test_the_detail_view_stops_painting_after_unmount(tmp_path) -> None:
    """main.js empties the root and re-mounts on every hashchange, so a poll
    still in flight would paint the previous node's answer into the new view's
    containers — and setHtmlIfChanged would then memoise it as what they hold."""
    out = run_section(
        tmp_path, NODE, "mount, unmount, refresh",
        f"""
        {_NODE_PAYLOAD}
        let calls = 0;
        globalThis.fetchJSON = async () => {{
            calls++;
            await new Promise(r => setTimeout(r, 20));
            // A different host each time, or setHtmlIfChanged would skip the
            // second write on its own and prove nothing.
            return payload(calls === 1 ? "10.0.0.5" : "10.0.0.9");
        }};
        {_NODE_CONTAINERS}
        M.mount(__el("root"), "edge-01", 1);
        await new Promise(r => setTimeout(r, 40));
        const identity = document.getElementById("node-identity");
        const updated = document.getElementById("node-updated");
        const painted = identity.innerHTML.includes("10.0.0.5");
        const stamped = updated.textContent.startsWith("Updated ");

        M.refresh();                 // a tick that lands after the tab switch
        M.unmount();
        updated.textContent = "MARK";
        await new Promise(r => setTimeout(r, 40));
        console.log(JSON.stringify([
            painted, stamped, identity.innerHTML.includes("10.0.0.9"), updated.textContent,
        ]));
        """,
    )
    assert out == [True, True, False, "MARK"]


@needs_node
def test_one_node_s_answer_never_lands_in_another_node_s_view(tmp_path) -> None:
    """#node/<a> → #node/<b> re-mounts this same section rather than switching
    to another one, so the mount token alone does not separate them: the poll
    read the module-level nodeId AFTER its await, by which time the re-mount had
    already reassigned it, and a's slow answer painted into b's cards — where
    setHtmlIfChanged memoised it as what they hold."""
    out = run_section(
        tmp_path, NODE, "mount, unmount, refresh",
        f"""
        {_NODE_PAYLOAD}
        {_NODE_CONTAINERS}
        globalThis.fetchJSON = async (url) => {{
            const first = url.includes("edge-a");
            // Still in flight when the operator opens the next node.
            if (first) await new Promise(r => setTimeout(r, 40));
            return payload(first ? "10.0.0.5" : "10.0.0.9");
        }};
        M.mount(__el("root"), "edge-a", 1);
        M.unmount();                    // what main.js does on a hashchange
        M.mount(__el("root"), "edge-b", 2);
        await new Promise(r => setTimeout(r, 80));
        const identity = document.getElementById("node-identity").innerHTML;
        M.unmount();
        console.log(JSON.stringify([
            identity.includes("10.0.0.9"), identity.includes("10.0.0.5"),
        ]));
        """,
    )
    assert out == [True, False]


@needs_node
def test_the_detail_poll_does_not_overlap(tmp_path) -> None:
    """History is re-sent in full on every tick, so a backend answering slowly
    rather than not at all is exactly the case that stacks requests."""
    out = run_section(
        tmp_path, NODE, "mount, unmount, refresh",
        """
        let calls = 0;
        globalThis.fetchJSON = async () => {
            calls++;
            await new Promise(r => setTimeout(r, 20));
            return { _error: "x" };
        };
        M.mount(__el("root"), "edge-01", 1);
        M.refresh();
        M.refresh();
        await new Promise(r => setTimeout(r, 60));
        M.unmount();
        console.log(JSON.stringify(calls));
        """,
    )
    assert out == 1, "expected one in-flight request, not three"


@needs_node
def test_an_absent_section_says_so_rather_than_rendering_blank(tmp_path) -> None:
    """Each of the four blocks can be absent on its own — no config entry, no
    K3s node matching the host, no store open — and a card that renders empty is
    indistinguishable from one whose data is genuinely zero."""
    out = run_section(
        tmp_path, NODE, "mount, unmount, renderIdentity, renderK8s, renderMetrics",
        f"""
        {_NODE_CONTAINERS}
        M.mount(__el("root"), "edge-01", 1);
        M.unmount();
        M.renderIdentity(null, null);
        const identity = document.getElementById("node-identity").innerHTML;
        M.renderK8s({{ status: "ok", node: null, pods: [] }}, {{ host: "10.0.0.5" }}, null);
        const k8s = document.getElementById("node-k8s").innerHTML;
        M.renderMetrics({{ status: "unavailable" }});
        const metrics = document.getElementById("node-metrics").innerHTML;
        console.log(JSON.stringify([identity, k8s, metrics]));
        """,
    )
    identity, k8s, metrics = out
    assert "no entry for this node" in identity
    assert "No K3s node matched host 10.0.0.5" in k8s
    assert "no telemetry store open" in metrics


# --- style.css --------------------------------------------------------------


def test_aria_disabled_placeholders_are_styled_as_disabled() -> None:
    """:disabled matches form elements only, so lifecycle's <span class="btn">
    placeholders kept the pointer cursor and the full hover highlight — and in
    btn-ghost they lit up the same danger red as the live Stop beside them."""
    css = STYLE.read_text()
    assert '.btn[aria-disabled="true"]' in css
    assert '.btn:hover:not(:disabled):not([aria-disabled="true"])' in css
    assert '.btn-ghost:hover:not(:disabled):not([aria-disabled="true"])' in css
    lifecycle = LIFECYCLE.read_text()
    for placeholder in re.findall(r'<span class="[^"]*btn[^"]*"[^>]*aria-disabled', lifecycle):
        assert "btn-ghost" not in placeholder, "placeholders must not be painted danger-red"


def test_access_sourced_log_badges_have_their_own_colour() -> None:
    """access_api emits key-rotation and apply lines with source="access"; with
    no rule the badge fell back to .log-badge, byte-identical to
    .log-src-stream, so those lines read as stream chatter."""
    css = STYLE.read_text()
    rules = dict(re.findall(r"\.log-src-(\w+)\s*\{([^}]*)\}", css))
    assert "access" in rules
    assert rules["access"].strip() != rules["stream"].strip()


def test_dead_section_subheader_rule_is_gone() -> None:
    """Its one declaration was always overridden by `.log-header h2 { margin: 0 }`
    at its one use site; the spacing comes from `.log-header`."""
    assert ".section-subheader" not in STYLE.read_text()
    for js in sorted((STATIC_DIR).rglob("*.js")):
        assert "section-subheader" not in js.read_text(), js.name


def test_a_filtered_log_row_outranks_every_display_rule() -> None:
    """`.log-hidden` and `.log-line` are both single-class selectors, so they tie
    on specificity and only `!important` settles it. Drop it and a row the
    filter marks hidden goes on rendering — every log filter silently stops
    working."""
    css = STYLE.read_text()
    hidden = re.search(r"\.log-hidden\s*\{([^}]*)\}", css).group(1)

    assert re.search(r"display:\s*none\s*!important", hidden)
    assert re.search(r"\.log-line\s*\{[^}]*display:\s*flex", css), "the rule it must beat"


@pytest.mark.parametrize(
    ("selector", "css_class"),
    [
        (".pod-error", "pod-error"),
        (".select-control", "select-control"),
        (".spacer", "spacer"),
        (".endpoint-row", "endpoint-row"),
        (".endpoint-label", "endpoint-label"),
    ],
)
def test_the_shared_skins_are_declared_once_above_the_sections(selector, css_class) -> None:
    """These are declared outside any section block precisely because more
    than one section paints with them: restyling one repaints every call site,
    and a copy inside a section would drift from the others unnoticed. Declared
    ABOVE the section rules so a section can still override."""
    css = STYLE.read_text()
    callers = [js.name for js in sorted(SECTIONS.glob("*.js")) if css_class in js.read_text()]
    first_section_banner = min(m.start() for m in re.finditer(r"/\* --- \w+ section --- \*/", css))

    assert len(callers) > 1, f"{selector} has one call site ({callers}); it is not shared"
    assert css.count(f"\n{selector} {{") == 1, f"{selector} is declared more than once"
    assert css.index(f"\n{selector} {{") < first_section_banner
