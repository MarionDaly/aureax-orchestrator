// Lifecycle section — Start/Stop controls + live log stream.

import { errorText, escapeHtml, fetchJSON, isError, stickyScroll } from "/utils.js";
import { isCurrent } from "/main.js";

// Each tick GETs /api/lifecycle/state, whose `nodes` come from a snapshot cached
// for _NODE_STATUS_TTL = 3.0s (orchestrator/lifecycle.py). Polling faster than
// that only refreshes the state-machine half of the response, which is the half
// the buttons read — hence a cadence below the TTL rather than at it.
const STATE_POLL_MS = 1500;

// Delay before reopening a stream the browser will not retry itself.
const LOG_RETRY_MS = 5000;

let es = null;
let pollId = null;
let retryId = null;
let highestLineIndex = -1;
let stateInFlight = false;
let mountToken;

export async function mount(root, _arg, token) {
    mountToken = token;
    root.innerHTML = `
        <section>
            <h2>Lifecycle</h2>
            <div class="card lifecycle-panel">
                <div class="lifecycle-controls" id="lifecycle-controls">
                    <div class="control-group">
                        <span class="control-label">Whole stack</span>
                        <button class="btn btn-primary" data-action="start" data-node="">Start</button>
                        <button class="btn btn-danger" data-action="stop" data-node="">Stop</button>
                    </div>
                    <div class="control-group" id="per-node-controls"></div>
                </div>
                <div class="lifecycle-alert" id="lifecycle-alert" hidden></div>
            </div>
            <div class="log-header">
                <h2>Log</h2>
                <div class="log-controls">
                    <label>Min level:
                        <select id="log-level" class="select-control">
                            <option value="debug">All</option>
                            <option value="info">Info+</option>
                            <option value="success">Success+</option>
                            <option value="warn">Warn+</option>
                            <option value="error">Error</option>
                        </select>
                    </label>
                    <label>Source:
                        <select id="log-source" class="select-control">
                            <option value="">All</option>
                            <option value="lifecycle">Lifecycle</option>
                            <option value="chat">Chat</option>
                            <option value="telemetry">Telemetry</option>
                            <option value="pods">Pods</option>
                            <option value="nodes">Nodes</option>
                            <option value="drift">Drift</option>
                            <option value="config">Config</option>
                            <option value="access">Access</option>
                            <option value="stream">Streams</option>
                        </select>
                    </label>
                    <button class="btn" id="log-clear" title="Hide entries currently on screen (does not delete from DB)">Clear</button>
                </div>
            </div>
            <div class="log-pane" id="log-pane"></div>
        </section>
    `;

    document.getElementById("lifecycle-controls").addEventListener("click", onClick);
    document.getElementById("log-level").addEventListener("change", applyLogFilter);
    document.getElementById("log-source").addEventListener("change", applyLogFilter);
    document.getElementById("log-clear").addEventListener("click", () => {
        document.getElementById("log-pane").innerHTML = "";
    });
    // Opened first, and before any await: it needs no data, and the log pane is
    // the only view of a Start already in flight. /api/lifecycle/state TCP-probes
    // every configured node, so waiting on it leaves the pane blank for as long
    // as the unreachable ones take to time out.
    openLogStream();
    // Sequential, not concurrent: /api/lifecycle/state is served from the 3s
    // cache above while /api/lifecycle/nodes re-reads and re-parses the network
    // config on every call, so run concurrently /state lands first with the
    // per-node panel still empty — and refreshState()'s self-heal then fires a
    // SECOND /nodes fetch on every mount.
    await populateNodes();
    await refreshState();
    // The awaits above mean the user may have switched tabs by now. unmount()
    // has already run in that case, so assigning pollId here would overwrite the
    // handle it cleared and leave an interval nothing can ever stop.
    if (!isCurrent(token)) return;
    pollId = setInterval(refreshState, STATE_POLL_MS);
}

export function unmount() {
    if (es) { es.close(); es = null; }
    if (pollId) { clearInterval(pollId); pollId = null; }
    if (retryId) { clearTimeout(retryId); retryId = null; }
    if (flushHandle !== null) { cancelAnimationFrame(flushHandle); flushHandle = null; }
    pending = [];
    highestLineIndex = -1;
    healFailures = 0;
    healSkipTicks = 0;
    // Released here too: a poll still in flight when the tab was switched would
    // otherwise leave the guard set, making the next mount's first refresh a
    // silent no-op.
    stateInFlight = false;
}

// node_id -> {role, group, members}, from /api/lifecycle/nodes (`members` is the
// wire's `group_members`). Both fields are load-bearing in the UI: a per-node
// Stop on a group member has to name the whole group it actually stops, and only
// a node that serves MCP may be offered a Chat button.
const nodeMeta = new Map();

async function populateNodes() {
    const nodes = await fetchJSON("/api/lifecycle/nodes");
    const el = document.getElementById("per-node-controls");
    if (!el) return;
    if (isError(nodes)) {
        // Said out loud. The endpoint answers 503 when the network config is
        // unreadable (lifecycle_api.list_configured_nodes), and an empty panel
        // is indistinguishable from a fleet with no nodes configured.
        el.innerHTML = `<div class="empty-state">Node list unavailable: ${escapeHtml(errorText(nodes))}</div>`;
        return;
    }
    if (!Array.isArray(nodes)) return;

    nodeMeta.clear();
    for (const n of nodes) {
        nodeMeta.set(n.node_id, {
            role: n.role || "",
            group: n.group || null,
            members: Array.isArray(n.group_members) ? n.group_members : [],
        });
    }

    el.innerHTML = nodes.map(n => {
        // A shard worker runs rpc-server and no MCP server, so it is not a chat
        // target: chat_service excludes it, and a link to one lands on a
        // different node without saying so. The literal is SHARD_WORKER_ROLE
        // (orchestrator/config/schema.py), whose header comment inventories the
        // places that must agree; this is the dashboard's copy of it.
        const chattable = n.role !== "shard_worker";
        const chat = chattable
            ? `<a class="btn" href="#chat/node-${escapeHtml(n.node_id)}">Chat</a>`
            : `<span class="btn" aria-disabled="true" title="Shard workers run rpc-server and serve no MCP endpoint">Chat</span>`;
        // Deep link into the Access tab: endpoint URLs + the API-key controls
        // for this node. Same reasoning as Chat — a worker serves no HTTP
        // endpoint of its own and holds no key.
        const api = chattable
            ? `<a class="btn" href="#access/${escapeHtml(n.node_id)}">API</a>`
            : `<span class="btn" aria-disabled="true" title="Shard workers run rpc-server, which has no HTTP endpoint or API key">API</span>`;
        // Unconditional, unlike Chat and API: the detail view is assembled from
        // the fleet config and the K8s objects, so it has something to show for
        // a worker, and for a node that has never reported a metric.
        const detail = `<a class="btn" href="#node/${escapeHtml(n.node_id)}">Detail</a>`;
        const groupTag = n.group
            ? ` <span class="control-hint" title="Start/Stop acts on the whole group">(${escapeHtml(n.group)})</span>`
            : "";
        return `
        <div class="node-controls">
            <span class="control-label">
                <span class="status-dot unknown" data-node-dot="${escapeHtml(n.node_id)}"></span>
                ${escapeHtml(n.display_name)}${groupTag}
            </span>
            <button class="btn" data-action="start" data-node="${escapeHtml(n.node_id)}">Start</button>
            <button class="btn btn-ghost" data-action="stop" data-node="${escapeHtml(n.node_id)}">Stop</button>
            ${chat}
            ${api}
            ${detail}
        </div>`;
    }).join("");
}

async function onClick(e) {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;
    const node = btn.dataset.node || null;

    if (action === "stop" && !node) {
        // While a Start is in flight, Stop = cancel that Start (kill the
        // setup-all.sh subprocess + abort any wait loops). Use a different
        // prompt so the user knows what they're actually doing.
        const state = await fetchJSON("/api/lifecycle/state");
        // An unknown state takes the stricter prompt. fetchJSON never throws, so
        // a failed probe simply has no `state` field — and try_stop cancels
        // whenever the manager is STARTING, so consenting to "state is
        // preserved" and getting a SIGTERMed setup-all.sh is the wrong way round.
        if (isError(state) || state.state === "starting") {
            if (!window.confirm(
                "Stop Aureax, cancelling any Start now in flight?\n\n" +
                "If a Start is running this SIGTERMs the setup-all.sh " +
                "subprocess (and any ssh/scp/docker children) and aborts its " +
                "wait loops, leaving the cluster in whatever partial state it " +
                "had reached. Otherwise it terminates all edge pods and stops " +
                "the k3d cluster."
            )) return;
        } else {
            if (!window.confirm(
                "Stop Aureax?\n\n" +
                "This terminates all edge pods and stops the k3d cluster.\n" +
                "The dashboard will remain running. State is preserved — " +
                "use Start to bring everything back."
            )) return;
        }
    }
    if (action === "stop" && node) {
        // Name the whole group when the node belongs to one. A group serves a
        // single model across all its members, so stopping any one of them
        // stops all of them — a prompt naming one node understates it.
        const meta = nodeMeta.get(node);
        const others = (meta?.members || []).filter(m => m !== node);
        const msg = meta?.group && others.length
            ? `Stop node ${node}?\n\n` +
              `${node} belongs to group "${meta.group}", which serves ONE model across ` +
              `all its members — so this stops the whole group:\n\n` +
              `  ${[node, ...others].join(", ")}\n\n` +
              `Everything in the group will be terminated.`
            : `Stop node ${node}? The pod will be terminated.`;
        if (!window.confirm(msg)) return;
    }

    const resp = await fetchJSON(`/api/lifecycle/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node }),
    });
    if (resp._error === 409) {
        // 409 has two causes, and they send the operator to different tabs: a
        // BusyError from the lifecycle manager, or an Access-tab config apply
        // holding the lock (lifecycle_api._apply_in_flight, which is the one
        // that sets `detail`).
        window.alert(
            `Cannot ${action}: ${resp.detail || "another lifecycle operation is already running"}. ` +
            "Wait for it to finish."
        );
    } else if (isError(resp)) {
        // Surfaced, not only logged: the button appears to do nothing
        // otherwise, and "did my Stop work?" is the worst question to leave
        // an operator holding.
        window.alert(`${action} failed: ${errorText(resp)}`);
    } else {
        await refreshState();
    }
}

async function refreshState() {
    // One poll at a time. fetchJSON bounds each request at 10s, but a backend
    // answering slowly rather than not at all would still let ticks overlap.
    if (stateInFlight) return;
    stateInFlight = true;
    try {
        const state = await fetchJSON("/api/lifecycle/state");
        if (isError(state)) {
            // Freshness is not asserted past the last successful poll. A dot
            // left holding its `online` class reads green for a node whose
            // reachability has become unknown.
            document.querySelectorAll("[data-node-dot]").forEach(dot => {
                dot.className = "status-dot unknown";
            });
            setStateAlert(`Lifecycle state unavailable: ${errorText(state)}`);
            // Falls to the else branch and re-enables everything, so an error
            // landing while STARTING cannot leave the controls dead until the
            // backend answers again.
            updateButtons({});
            return;
        }
        setStateAlert("");

        // Self-heal. The per-node rows are rendered once during mount, so
        // anything that made that attempt fail — the endpoint briefly erroring,
        // or a re-mount landing between the fetch and the write — would leave
        // the panel empty for the life of the page with no way back short of a
        // reload.
        const perNode = document.getElementById("per-node-controls");
        // Counts rows, not children: populateNodes writes an .empty-state div on
        // failure, which a childElementCount test reads as "already populated".
        if (perNode && perNode.querySelectorAll(".node-controls").length === 0) {
            await healNodePanel(perNode);
        }

        updateButtons(state);
        updateNodeRows(state.nodes || []);
    } finally {
        stateInFlight = false;
    }
}

// Backoff for refreshState's self-heal. A failed populateNodes leaves the panel
// empty, so the condition that triggered the retry is still true on the next
// tick — and /api/lifecycle/nodes re-parses the whole network config per call.
// Each consecutive failure skips one more tick than the last, up to this cap, so
// a config that stays unreadable costs one request a minute rather than forty.
const HEAL_MAX_SKIP_TICKS = 40;
let healFailures = 0;
let healSkipTicks = 0;

async function healNodePanel(perNode) {
    if (healSkipTicks > 0) {
        healSkipTicks--;
        return;
    }
    await populateNodes();
    if (perNode.querySelectorAll(".node-controls").length === 0) {
        healFailures++;
        healSkipTicks = Math.min(healFailures, HEAL_MAX_SKIP_TICKS);
    } else {
        healFailures = 0;
    }
}

function setStateAlert(text) {
    const el = document.getElementById("lifecycle-alert");
    if (!el) return;
    el.textContent = text;
    el.hidden = !text;
}

function updateNodeRows(nodes) {
    for (const n of nodes) {
        const dot = document.querySelector(`[data-node-dot="${CSS.escape(n.node_id)}"]`);
        if (dot) dot.className = `status-dot ${n.running ? "online" : "offline"}`;
    }
}

function updateButtons(state) {
    // Per-state button enablement:
    //   IDLE     — everything enabled
    //   STARTING — Stop (whole-stack) stays enabled so user can cancel; the
    //              rest disabled to prevent double-clicks while busy.
    //   STOPPING — everything disabled (stop is fast, no need to interrupt).
    //   ERROR    — everything enabled (try to recover).
    const s = state.state;
    document.querySelectorAll("#lifecycle-controls button[data-action]").forEach(b => {
        const action = b.dataset.action;
        const node = b.dataset.node;  // empty string = whole-stack
        const isWholeStackStop = action === "stop" && !node;
        if (s === "starting") {
            b.disabled = !isWholeStackStop;
        } else if (s === "stopping") {
            b.disabled = true;
        } else {
            b.disabled = false;
        }
    });
}

function openLogStream() {
    if (es) es.close();
    if (retryId) { clearTimeout(retryId); retryId = null; }
    const since = highestLineIndex + 1;
    es = new EventSource(`/api/lifecycle/logs?since=${since}`);
    es.addEventListener("log", (evt) => {
        try {
            const line = JSON.parse(evt.data);
            // Skip anything already rendered. EventSource reconnects on its own
            // and re-requests the SAME url — `?since=` is fixed at open time —
            // so the server replays its whole buffer (up to 2000 lines) again.
            // The server's own dedupe is per-connection, and no `id:` field is
            // emitted, so Last-Event-ID resume is not available: this is the
            // only thing standing between a reconnect and a duplicated log.
            if (typeof line.line_index === "number" && line.line_index <= highestLineIndex) return;
            appendLine(line);
            if (line.line_index > highestLineIndex) highestLineIndex = line.line_index;
        } catch (e) {
            console.error("bad log line", e, evt.data);
        }
    });
    es.onerror = () => {
        // Only TRANSPORT failures self-retry. A response that is not 200
        // text/event-stream fails the connection outright: readyState goes
        // CLOSED and the browser never reconnects, so the reopen has to be ours
        // or the pane stays empty and silent for the life of the page.
        if (!es || es.readyState !== EventSource.CLOSED) return;
        es = null;
        appendLine({
            level: "error",
            source: "stream",
            text: `Log stream closed — retrying in ${LOG_RETRY_MS / 1000}s`,
        });
        retryId = setTimeout(() => {
            retryId = null;
            // unmount() clears this timer, but a re-mount of this same section
            // resets the handle, so check the mount token too rather than
            // reopening a second stream beside the new mount's own.
            if (mountToken !== undefined && !isCurrent(mountToken)) return;
            openLogStream();
        }, LOG_RETRY_MS);
    };
}

// `debug` arrives on entries shipped up from edge nodes. Without an entry here
// it scores 0 and renders as if it were info, which the "Info+" filter then
// cannot hide.
//
// It sorts BELOW info, which is why "All" selects "debug" rather than "info":
// with the lowest selectable option at info, the setting labelled All would hide
// every debug entry and nothing could show them — and those entries are the
// run-up a crashing child ships (ChildLogCapture.flush_crash_context).
const LEVEL_ORDER = { debug: -1, info: 0, success: 1, warn: 2, error: 3, hint: 0 };

// Cap on rendered rows. The server replays up to 2000 on connect and then
// streams indefinitely, and the manual Clear button is the only other thing that
// removes a row — so without this a dashboard left open on a long deploy grows
// without bound.
const MAX_LOG_ROWS = 2000;

// Arriving lines are batched into one DOM write per frame. The server replays
// its whole buffer on connect as one `log` event per line, and appending each
// separately reads the pane's geometry twice per line — two forced reflows
// each, up to 2000 times on every visit to this tab.
let pending = [];
let flushHandle = null;

function appendLine(line) {
    pending.push(line);
    // requestAnimationFrame does not fire in a background tab while the stream
    // keeps arriving, so cap the buffer at what the pane would keep anyway.
    if (pending.length > MAX_LOG_ROWS) pending.splice(0, pending.length - MAX_LOG_ROWS);
    if (flushHandle === null) flushHandle = requestAnimationFrame(flushLines);
}

function flushLines() {
    flushHandle = null;
    const lines = pending;
    pending = [];
    const pane = document.getElementById("log-pane");
    if (!pane || !lines.length) return;
    // Geometry and both filter values read once for the whole batch, not once
    // per row.
    const restoreScroll = stickyScroll(pane);
    const minLevel = document.getElementById("log-level")?.value || "info";
    const source = document.getElementById("log-source")?.value || "";
    const frag = document.createDocumentFragment();
    for (const line of lines) frag.appendChild(buildRow(line, minLevel, source));
    pane.appendChild(frag);
    while (pane.childElementCount > MAX_LOG_ROWS) pane.removeChild(pane.firstElementChild);
    restoreScroll();
}

function buildRow(line, minLevel, source) {
    const src = line.source || "unknown";
    registerSource(src);

    const row = document.createElement("div");
    row.className = `log-line log-${line.level}`;
    row.dataset.level = line.level;
    row.dataset.source = src;

    const badge = document.createElement("span");
    // Sanitised: shipped entries carry "node:<id>", and a raw colon in a class
    // name needs escaping at every use site for no benefit.
    badge.className = `log-badge log-src-${src.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
    badge.textContent = src;

    const text = document.createElement("span");
    text.textContent = line.text;

    row.appendChild(badge);
    row.appendChild(text);
    if (!passesFilter(row, minLevel, source)) row.classList.add("log-hidden");
    return row;
}

// Ceiling on sources ADDED here; the static options do not count against it.
// `source` is "node:" + a node_id taken straight from the unauthenticated
// /api/v1/logs ingest, so the option list is caller-growable exactly as the
// pane is — MAX_LOG_ROWS bounds one, this bounds the other. No catch-all entry
// past the cap: its value would match no row's dataset.source, so selecting it
// would hide every line.
const MAX_LOG_SOURCES = 50;

// Sources the Source filter already offers. The static <select> lists the
// orchestrator's own subsystems only, so entries shipped up from edge nodes —
// tagged "node:<id>", the whole point of log shipping — are added here or are
// selectable by nothing but "All".
//
// The Set is tied to the element it was built from, and mount() rebuilds that
// <select> with the static options only: a Set outliving the rebuild would
// suppress every already-seen node source and never re-add it.
let sourceSelect = null;
let knownSources = new Set();
let dynamicSources = 0;

function registerSource(src) {
    const sel = document.getElementById("log-source");
    if (!src || !sel) return;
    if (sel !== sourceSelect) {
        sourceSelect = sel;
        knownSources = new Set([...sel.options].map(o => o.value));
        dynamicSources = 0;
    }
    if (knownSources.has(src)) return;
    if (dynamicSources >= MAX_LOG_SOURCES) return;
    knownSources.add(src);
    dynamicSources++;
    const opt = document.createElement("option");
    opt.value = src;
    opt.textContent = src.startsWith("node:") ? `Node: ${src.slice(5)}` : src;
    sel.appendChild(opt);
}

function passesFilter(row, minLevel, source) {
    const rowLevel = row.dataset.level || "info";
    const rowSource = row.dataset.source || "unknown";
    if ((LEVEL_ORDER[rowLevel] ?? 0) < (LEVEL_ORDER[minLevel] ?? 0)) return false;
    if (source && rowSource !== source) return false;
    return true;
}

function applyLogFilter() {
    const pane = document.getElementById("log-pane");
    if (!pane) return;
    const minLevel = document.getElementById("log-level")?.value || "info";
    const source = document.getElementById("log-source")?.value || "";
    pane.querySelectorAll(".log-line").forEach(row => {
        row.classList.toggle("log-hidden", !passesFilter(row, minLevel, source));
    });
}
