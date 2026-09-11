"""Guards for the hand-written dashboard frontend.

`orchestrator/dashboard/static/` is git-tracked, is excluded from vulture
(pyproject.toml), and has no JS linter — so nothing checked it at all. These
tests are deliberately dependency-free: the project ships no package.json and
serves these files directly, and adding an npm toolchain is a bigger decision
than this gap warrants.

They cover the failure modes that are cheap to detect and expensive to hit: a
file that does not parse, JS calling an endpoint the backend does not serve,
and the handful of pure helpers whose output is asserted by running them —
escapeHtml's attribute-safe escaping, timeAgo's clock skew, the chat tool-label
split, and the render/poll guards in the chat and access sections.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from tests.conftest import PROJECT_ROOT

STATIC_DIR = PROJECT_ROOT / "orchestrator" / "dashboard" / "static"


def _js_files() -> list:
    return sorted(STATIC_DIR.rglob("*.js"))


def test_static_dir_is_present() -> None:
    """Guard the guards: an empty glob would make everything below vacuous."""
    files = _js_files()
    assert len(files) >= 7, f"expected the dashboard's ES modules, found {files}"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.parametrize("js", _js_files(), ids=lambda p: p.name)
def test_javascript_parses(js) -> None:
    """Every module parses as ESM — these are served without a build step.

    Fed through stdin with ``--input-type=module`` rather than as
    ``node --check <file>``: for a ``.js`` file carrying ``import``, current
    Node exits 0 without parsing it (verified on Node 25 with a deliberately
    broken file), so the by-path form passed every module unconditionally.
    """
    proc = subprocess.run(
        ["node", "--input-type=module", "--check"],
        input=js.read_text(), capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"{js.name} does not parse:\n{proc.stderr}"


# Matches the API paths the frontend calls, in plain strings and in template
# literals. A `${...}` segment becomes a wildcard so it can be compared against
# FastAPI's "{param}" placeholders.
_URL_RE = re.compile(r"""["'`](/api/[^"'`\s?]*)""")


def _normalise(url: str) -> str:
    """Collapse template-literal interpolations to a single placeholder."""
    url = re.sub(r"\$\{[^}]*\}", "{}", url)
    return url.rstrip("/") or "/"


def test_every_frontend_api_call_hits_a_real_route(tmp_aureax_home, network_config_path) -> None:
    """A fetch to an endpoint that does not exist fails only at runtime.

    The frontend is served as static files, so a renamed or removed route shows
    up as a 404 in the browser console and a silently empty panel — never as a
    build or import error.
    """
    from orchestrator.dashboard.app import create_app

    app = create_app(network_config_path=str(network_config_path))
    registered = [_normalise(re.sub(r"\{[^}]*\}", "{}", p)) for p in app.openapi()["paths"]]

    called: dict[str, str] = {}
    for js in _js_files():
        for raw in _URL_RE.findall(js.read_text()):
            called.setdefault(_normalise(raw), js.name)

    def matches(call: str, route: str) -> bool:
        """Segment-wise, with `{}` on either side acting as a wildcard.

        A call may interpolate a path param (`/api/config/${name}`) or a fixed
        choice (`/api/lifecycle/${action}` for start|stop), so a `{}` on the
        JS side has to match a literal route segment as well as a param.
        """
        a, b = call.split("/"), route.split("/")
        return len(a) == len(b) and all(
            x == y or x == "{}" or y == "{}" for x, y in zip(a, b, strict=True)
        )

    missing = {
        url: src
        for url, src in called.items()
        if not any(matches(url, r) for r in registered)
    }
    assert not missing, (
        "frontend calls endpoints the backend does not serve:\n"
        + "\n".join(f"  {u}  (in {s})" for u, s in sorted(missing.items()))
        + "\n\nregistered:\n" + "\n".join(f"  {p}" for p in sorted(registered))
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_escape_html_escapes_quotes() -> None:
    """escapeHtml is used in attribute position, so quotes must be escaped.

    node_ids and config filenames are operator-supplied and land in
    `data-node-dot="..."`, `href="#chat/node-..."` and `<option value="...">`.
    The previous implementation round-tripped through textContent/innerHTML,
    which escapes &, < and > only — a value containing a double quote closed
    the attribute early.

    Asserts the OUTPUT, by running the real function. Grepping the source for
    each character instead only proved it was mentioned: a `.replace(/"/g,
    "&quot")` missing its semicolon, or the `&` pass moved last so every entity
    is double-escaped into `&amp;lt;`, both still contain `/"/g` and `/&/g` and
    stayed green — while a correct single-pass rewrite through a lookup table
    failed. `&` must run FIRST, which is the ordering this pins.
    """
    script = (
        f'import {{ escapeHtml }} from "{(STATIC_DIR / "utils.js").as_uri()}";\n'
        'process.stdout.write(escapeHtml(`<a href="x">&\'`));\n'
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "&lt;a href=&quot;x&quot;&gt;&amp;&#39;"


def _section(name: str) -> str:
    return (STATIC_DIR / "sections" / name).read_text()


def _function_body(src: str, signature: str) -> str:
    """The text of one top-level function, from its signature to the next one.

    Cheap and good enough for these files: every function here is declared at
    column 0, so `\\n}` closes it.
    """
    start = src.index(signature)
    return src[start : src.index("\n}", start)]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_chat_tool_labels_split_the_node_prefix_once() -> None:
    """MCPPool namespaces a tool as ``<node_id>__<tool name>`` and splits it
    with ``split("__", 1)``. JS's second argument to split() is a result-length
    cap that DISCARDS the remainder, so the label for a tool whose own name
    contains "__" was silently truncated.
    """
    import json

    src = _section("chat.js")
    match = re.search(
        r"( *const i = name\.indexOf\(\"__\"\);\n *const \[nodeId, toolName\] = .*)\n", src
    )
    assert match, "chat.js no longer splits the namespaced tool name in one place"

    script = (
        "const out = [];\n"
        'for (const name of ["edge-01__read__sensor", "edge-01__read_sensor", "local_tool"]) {\n'
        f"{match.group(1)}\n"
        "  out.push([nodeId, toolName]);\n"
        "}\n"
        "console.log(JSON.stringify(out));\n"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [
        ["edge-01", "read__sensor"],
        ["edge-01", "read_sensor"],
        ["", "local_tool"],
    ]


def test_access_cards_skip_an_identical_re_render() -> None:
    """The 20s live poll and the 2s apply poll both call render(), and writing
    innerHTML replaces every input, <pre> and button in the list — dropping a
    text selection, the focus in a copy field and the 1.5s "Copied ✓" label.

    The comparison is against the whole rendered string on purpose: a
    hand-picked signature goes stale the moment cardHtml reads one more field.
    """
    body = _function_body(_section("access.js"), "function renderCards(")
    # The memo itself lives in utils.js — one copy, shared with overview.js.
    assert "setHtmlIfChanged(el, html);" in body
    assert "el.innerHTML" not in body, "renderCards must not bypass the memo"
    # Set on every branch, or an error -> cards transition compares against a
    # signature the screen never showed and is skipped.
    assert body.count("html = ") == 3, "each renderCards branch must feed the cache"


def test_the_shared_helpers_have_exactly_one_implementation() -> None:
    """utils.js is the home for a helper more than one section needs. A second
    copy in a section drifts — and a reader then cannot tell whether the
    difference between the two means anything."""
    utils = (STATIC_DIR / "utils.js").read_text()
    for name in ("setHtmlIfChanged", "readLocal", "writeLocal",
                 "containerStateClass", "restartBadge"):
        assert f"export function {name}(" in utils, f"{name} left utils.js"

    for section in ("overview.js", "access.js", "chat.js", "configs.js", "node.js"):
        src = _section(section)
        for call in ("localStorage.getItem(", "localStorage.setItem("):
            assert call not in src, (
                f"{section} calls {call} directly instead of readLocal/writeLocal — "
                "a browser configured to block site data throws on the property itself"
            )
        assert "function setHtml(" not in src, f"{section} re-implements the innerHTML memo"


def test_apply_polling_survives_one_failed_status_poll() -> None:
    """A transient failure of /api/access/apply/state used to stop the poller
    mid-apply: applyRunning cleared, the "Apply running" note vanished, every
    Apply button re-enabled, and nothing was left to poll for the completion
    that clears the per-node notes."""
    body = _function_body(_section("access.js"), "async function pollApply(")
    error_branch = body[body.index("if (isError(snap))") : body.index("state.applyPollError = null;")]
    assert "stopApplyPolling()" not in error_branch
    assert "state.applyPollError = errorText(snap);" in error_branch


def test_apply_outcomes_are_adopted_only_for_a_tracked_run() -> None:
    """ApplyRunner keeps its `nodes` dict after a run ends, for the life of the
    dashboard process, so the mount-time probe adopted the outcomes of an apply
    that may have finished hours ago and printed them as current."""
    body = _function_body(_section("access.js"), "async function pollApply(")
    assert "if (snap.running || state.applyRunning) {" in body


# --- the Access tab's two keyed services -------------------------------------


def test_access_renders_one_auth_block_per_key_kind() -> None:
    """A node has two keyed services and the card has to show both — a single
    block leaves a staged-but-unapplied MCP key invisible on the one screen
    that exists to surface it."""
    src = _section("access.js")
    body = _function_body(src, "function authHtml(")
    assert 'authKindHtml(entry, "llama"' in body
    assert 'authKindHtml(entry, "mcp"' in body
    assert "function authKindHtml(entry, kind, label)" in src


def test_access_key_actions_carry_the_kind_on_the_button() -> None:
    """The kind is read off the clicked button, never inferred from where in
    the DOM the click landed: an inference that goes wrong reveals or rotates
    the OTHER service's credential."""
    body = _function_body(_section("access.js"), "async function onCardClick(")
    assert 'const kind = btn.dataset.kind || "llama";' in body
    for action in ("revealKey(entry, kind)", "hideKey(entry, kind)",
                   "copyKey(btn, entry, kind)", "copyPreviousKey(btn, entry, kind)",
                   "rotateKey(entry, kind)"):
        assert action in body, f"{action} does not carry the kind"


def test_access_reveals_and_hides_one_kind_at_a_time() -> None:
    """state.revealed is per node AND per kind: revealing one service's key
    must not put the other service's key on screen under the wrong label."""
    src = _section("access.js")
    assert "function revealedKey(nodeId, kind)" in src
    reveal = _function_body(src, "async function revealKey(")
    assert "[kind]: data.api_key" in reveal
    hide = _function_body(src, "function hideKey(")
    assert "delete held[kind];" in hide


def test_access_names_the_two_key_routes_in_exactly_one_place() -> None:
    """/key and /mcp-key differ by one path segment, and a hand-built URL in
    each of the five call sites is five chances to fetch the wrong one."""
    src = _section("access.js")
    assert 'path: "key"' in src and 'path: "mcp-key"' in src
    assert "function keyUrl(nodeId, kind, previous = false)" in src
    # Every key fetch goes through it.
    assert src.count("/key`") == 0, "a key URL is still built by hand"


def test_the_mcp_toggle_confirm_says_what_actually_opens() -> None:
    """Reusing the llama wording ("the endpoint becomes open") understates it:
    the open MCP port reaches run_inference, which forwards to llama-server
    WITH this node's own API key attached."""
    body = _function_body(_section("access.js"), "async function onCardChange(")
    assert "run_inference" in body
    assert ":30900" in body
    assert 'body: JSON.stringify({ enabled: wanted, kind })' in body


def test_the_rotate_request_names_the_kind() -> None:
    """The route defaults to llama, so a rotate posted with no body from the
    MCP sub-block would silently re-key the wrong service and report success."""
    body = _function_body(_section("access.js"), "async function rotateKey(")
    assert "JSON.stringify({ kind })" in body


def test_the_card_badge_shows_the_worse_of_the_two_services() -> None:
    """A card reading "Applied" while its :30900 drifts is the false green the
    whole tab exists to prevent."""
    src = _section("access.js")
    assert "const BADGE_RANK" in src
    body = _function_body(src, "function badgeFor(")
    assert "Math.min" in body
    assert "kindBadge(entry, kind)" in body


def test_the_snippet_block_carries_the_mcp_variable() -> None:
    """AUREAX_MCP_API_KEY is what the harness and scripts/test-mcp-tools.py
    read, and it comes from the backend so the two cannot drift."""
    src = _section("access.js")
    assert 'envVar: "AUREAX_MCP_API_KEY"' in src
    body = _function_body(src, "function snippetLines(")
    assert "entry.mcp_snippet" in body


def test_chat_loaders_report_a_failed_fetch() -> None:
    """fetchJSON never throws: a failed load arrives as a non-array and used to
    become [] silently — two empty dropdowns, a null target, no reason."""
    src = _section("chat.js")
    for signature in ("async function loadTargets(", "async function loadSessions("):
        body = _function_body(src, signature)
        assert "isError(data)" in body, f"{signature} ignores a failed fetch"
    # A target_id of null lists every session of every target, not none.
    assert "if (!state.targetId) {" in _function_body(src, "async function loadSessions(")


def test_chat_send_arms_its_reentry_guard_before_the_first_await() -> None:
    """onSubmit's own `if (state.pending) return;` guard is inert unless the
    flag is set before it awaits — createNewSession() is a full round trip."""
    body = _function_body(_section("chat.js"), "async function onSubmit(")
    assert body.index("state.pending = true;") < body.index("await "), (
        "onSubmit awaits before arming its guard"
    )


def test_chat_destructive_actions_check_the_response() -> None:
    """Reset 404s on an unknown session and emits no 'reset' frame; a discarded
    DELETE result made a failed delete look like a successful one."""
    src = _section("chat.js")
    for signature in ("async function onReset(", "async function onDelete("):
        body = _function_body(src, signature)
        assert "isError(resp)" in body, f"{signature} discards the response"


def test_chat_escapes_selectors_with_the_platform_helper() -> None:
    """chat.js hand-rolled `"` and `\\` escaping for its [data-msg-id=...]
    selectors while access.js and lifecycle.js used CSS.escape, which covers
    more cases."""
    src = _section("chat.js")
    assert "cssId" not in src
    assert "CSS.escape" in src


# --- the per-node detail view ------------------------------------------------


def test_the_overview_card_links_to_the_detail_view() -> None:
    """The telemetry card is the only way into #node, and it is the whole card
    rather than a corner link — a metric grid with one clickable word in it
    hides the affordance."""
    src = _section("overview.js")
    assert '<a class="card node-card" href="#node/${escapeHtml(node.node_id)}">' in src
    assert '<div class="card">' not in _function_body(src, "function renderNodes("), (
        "the telemetry card is still an inert div"
    )


def test_the_headline_metric_shortlist_has_one_definition() -> None:
    """The Overview card leads with it and the detail view sorts by it. A copy
    per section drifts, and the two screens then disagree about which readings
    matter — with nothing on either saying so."""
    assert "export const HEADLINE_METRICS" in (STATIC_DIR / "utils.js").read_text()
    for section in ("overview.js", "node.js"):
        src = _section(section)
        assert "HEADLINE_METRICS" in src, f"{section} does not use the shortlist"
        assert "HEADLINE_METRICS = [" not in src, f"{section} re-declares the shortlist"


def test_the_router_lights_the_overview_tab_for_a_detail_view() -> None:
    """#node has no tab of its own — it is opened from an Overview card — so
    without the alias the whole nav goes dark and nothing shows the way back."""
    main = (STATIC_DIR / "main.js").read_text()
    assert 'import * as node from "/sections/node.js";' in main
    assert '"#node":' in main
    assert 'const lit = tab === "#node" ? "#overview" : tab;' in main
    assert 't.getAttribute("href") === lit' in main


def test_the_sparkline_takes_its_paint_from_the_stylesheet() -> None:
    """node.js re-emits every polyline on each 10s poll. A stroke or fill
    presentation attribute there would put the dashboard's accent colour in a
    JS file, where no stylesheet search finds it and a theme change misses it."""
    css = (STATIC_DIR / "style.css").read_text()
    assert re.search(r"\.sparkline polyline\s*\{[^}]*stroke:", css), "the polyline has no paint"
    src = _section("node.js")
    assert "stroke=" not in src and "fill=" not in src
    # A bare "#" proves nothing here: every hash route starts with one.
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", src), "node.js carries a colour literal"


def test_no_inline_styles_anywhere_in_static() -> None:
    """The dashboard's CSP is `default-src 'self'` with no style-src override
    and no `unsafe-inline`, so a style="" attribute is dropped by the browser
    with nothing on the page saying the rule did not apply."""
    for path in sorted(STATIC_DIR.rglob("*")):
        if path.suffix in {".js", ".html"}:
            assert 'style="' not in path.read_text(), f"{path.name} carries an inline style"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_is_fresh_is_two_missed_samples_and_never_throws() -> None:
    """The collector reports every 60s, so one late sample is jitter and two is
    a node that stopped talking. A missing or unparseable stamp has to read as
    stale rather than throw inside a render, and a node whose clock runs ahead
    of the Mac's is still reporting."""
    import json

    script = (
        f'import {{ isFresh, FRESH_WINDOW_MS }} from "{(STATIC_DIR / "utils.js").as_uri()}";\n'
        "const at = (ms) => new Date(Date.now() - ms).toISOString();\n"
        "console.log(JSON.stringify([FRESH_WINDOW_MS, isFresh(at(1000)),\n"
        "    isFresh(at(FRESH_WINDOW_MS + 5000)), isFresh(at(-60000)),\n"
        '    isFresh("not-a-date"), isFresh(null)]));\n'
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [120000, True, False, True, False, False]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_time_ago_never_renders_a_negative_or_nan_age() -> None:
    """Nodes stamp their own metrics and are re-provisioned regularly, so one
    whose clock runs ahead of the Mac's rendered "-37s ago", and an
    unparseable stamp rendered "NaNh ago"."""
    import json

    script = (
        f'import {{ timeAgo }} from "{(STATIC_DIR / "utils.js").as_uri()}";\n'
        "const future = new Date(Date.now() + 60000).toISOString();\n"
        'console.log(JSON.stringify([timeAgo(future), timeAgo("not-a-date"), timeAgo(null)]));\n'
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == ["0s ago", "unknown", "never"]
