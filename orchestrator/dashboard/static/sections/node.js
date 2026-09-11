// Node detail — everything the orchestrator already knows about ONE node:
// identity from the fleet config, its K3s node and pods, an hour of metric
// history, and the log lines it shipped. Polls every 10s.
//
// Hash: #node/<node_id>
//
// Reached from an Overview telemetry card and from a Lifecycle row. One fetch
// per tick: GET /api/nodes/<node_id> assembles all four blocks server-side, so
// a node that has never reported a metric still renders its identity — which is
// the case this view exists for.

import { containerStateClass, errorText, escapeHtml, fetchJSON, formatMetricName, formatMetricValue, HEADLINE_METRICS, isError, isFresh, restartBadge, setHtmlIfChanged, timeAgo } from "/utils.js";
import { isCurrent } from "/main.js";

const REFRESH_INTERVAL = 10000;

// The levels orchestrator/logs/receiver.py accepts. `level` reaches the store
// through the unauthenticated ingest route, so it is caller-supplied and lands
// in a class attribute here — anything else is rendered as info rather than
// interpolated.
const LOG_LEVELS = ["debug", "info", "warn", "error", "success", "hint"];

// What an absent field renders as. The route reports null for anything the
// node's config does not carry — including a value still spelled ${...} in an
// ungenerated template — and a blank row reads as a rendering bug.
const NO_VALUE = "—";

let intervalId = null;
let refreshing = false;
let mountToken;
let nodeId = null;

export function mount(root, arg, token) {
    mountToken = token;
    nodeId = arg || null;
    root.innerHTML = `
        <section>
            <div class="node-head">
                <a class="btn" href="#overview">← Overview</a>
                <h2>${escapeHtml(nodeId || "Node")}</h2>
                <span class="update-time" id="node-updated"></span>
            </div>
            <div class="card-grid">
                <div id="node-identity"></div>
                <div id="node-k8s"></div>
            </div>
        </section>

        <section>
            <h2>Metrics</h2>
            <div id="node-metrics"></div>
        </section>

        <section>
            <h2>Logs</h2>
            <div id="node-logs"></div>
        </section>
    `;
    if (!nodeId) {
        // "#node" with no id — a hand-edited hash, or a link built without one.
        // Say so instead of polling /api/nodes/ every 10s forever.
        paint("node-identity", card("Node", '<div class="empty-state">No node selected. Open one from the Overview.</div>'));
        return;
    }
    refresh();
    intervalId = setInterval(refresh, REFRESH_INTERVAL);
}

export function unmount() {
    if (intervalId) clearInterval(intervalId);
    intervalId = null;
    // Released so the next mount's first refresh is not skipped by a poll that
    // was still in flight when the tab was switched.
    refreshing = false;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function paint(id, html) {
    const el = document.getElementById(id);
    if (el) setHtmlIfChanged(el, html);
}

function card(title, body, header = "") {
    return `
        <div class="card">
            <div class="card-header">${header}<h3>${escapeHtml(title)}</h3></div>
            ${body}
        </div>
    `;
}

function row(label, value) {
    // "" as well as null: the K8s summary reports a field it could not read as
    // an empty string, and a row with a label and nothing beside it reads as a
    // rendering fault rather than as an absent value.
    const absent = value === null || value === undefined || value === "";
    return `
        <div class="endpoint-row">
            <span class="endpoint-label">${escapeHtml(label)}</span>
            <span>${escapeHtml(absent ? NO_VALUE : value)}</span>
        </div>
    `;
}

// The parts of a compound value that the node actually reported. A template
// that was never generated still carries its `${...}` placeholders, which the
// route reports as null — joining them in blindly would print "null · null".
function joinParts(parts) {
    const kept = parts.filter(p => p !== null && p !== undefined && p !== "");
    return kept.length ? kept.join(" · ") : NO_VALUE;
}

// A number with its unit, or nothing at all — so joinParts drops it rather
// than rendering "null ctx".
function suffixed(value, unit) {
    return value === null || value === undefined || value === "" ? null : `${value} ${unit}`;
}

function renderIdentity(identity, metrics) {
    if (!identity) {
        paint("node-identity", card("Identity", '<div class="empty-state">The fleet config carries no entry for this node.</div>'));
        return;
    }
    const lastSeen = metrics ? metrics.last_seen : null;
    const dot = `<span class="status-dot ${isFresh(lastSeen) ? "online" : "offline"}"></span>`;
    const tags = Array.isArray(identity.tags) ? identity.tags : [];
    const model = identity.model || {};
    const group = identity.group;

    // A shard worker runs rpc-server: it serves no MCP endpoint and holds no
    // API key, so both links would land on a different node without saying so.
    // The literal is SHARD_WORKER_ROLE (orchestrator/config/schema.py), whose
    // header comment inventories the places that must agree.
    const addressable = identity.role !== "shard_worker";
    const links = addressable
        ? `<a class="btn" href="#access/${escapeHtml(nodeId)}">API</a>
           <a class="btn" href="#chat/node-${escapeHtml(nodeId)}">Chat</a>`
        : `<span class="btn" aria-disabled="true" title="Shard workers run rpc-server, which has no HTTP endpoint or API key">API</span>
           <span class="btn" aria-disabled="true" title="Shard workers run rpc-server and serve no MCP endpoint">Chat</span>`;

    paint("node-identity", card(identity.display_name || nodeId, `
        <div class="pod-meta">
            <span>${escapeHtml(nodeId)}</span>
            <span>Last seen ${escapeHtml(timeAgo(lastSeen))}</span>
        </div>
        ${row("Host", identity.host)}
        ${row("Tags", tags.join(", "))}
        ${row("Role", identity.role)}
        ${row("Model", joinParts([
            model.name,
            model.quantization,
            suffixed(model.context_size, "ctx"),
            suffixed(model.gpu_layers, "gpu layers"),
        ]))}
        ${row("File", model.filename)}
        ${row("Group", group && joinParts([group.name, `head ${group.head}`, `${(group.workers || []).length} workers`]))}
        ${row("MCP", identity.mcp_enabled ? "enabled" : "disabled")}
        <div class="endpoint-row">${links}</div>
    `, dot));
}

function renderK8s(k8s, identity, latest) {
    if (!k8s) {
        paint("node-k8s", card("Kubernetes", '<div class="empty-state">No cluster information in this response.</div>'));
        return;
    }
    if (k8s.status === "error") {
        paint("node-k8s", card("Kubernetes", `<div class="empty-state">${escapeHtml(k8s.message || "cluster unavailable")}</div>`));
        return;
    }
    const node = k8s.node;
    const pods = Array.isArray(k8s.pods) ? k8s.pods : [];
    if (!node && !pods.length) {
        // The match is by address (api.py), so this is what an unlabelled,
        // renamed or not-yet-joined node looks like — not an empty cluster.
        const host = identity ? identity.host : null;
        paint("node-k8s", card("Kubernetes", `<div class="empty-state">No K3s node matched host ${escapeHtml(host || nodeId)}.</div>`));
        return;
    }

    // `matched_by` names which of api.py's three address rules identified this
    // node. Shown because the identification is a match, not a stored link: a
    // node found by a pod's hostIP is a weaker claim than one found by its own
    // InternalIP, and the operator is the only one who can tell whether the
    // box on screen is the one they meant.
    let nodeHtml = "";
    if (node) {
        const ready = node.status === "Ready";
        const pressure = Array.isArray(node.pressure) ? node.pressure : [];
        nodeHtml = `
            <div class="pod-meta">
                <span class="status-dot ${ready ? "online" : "offline"}"></span>
                <span>${escapeHtml(node.name)}</span>
                <span class="badge ${ready ? "ready" : "not-ready"}">${escapeHtml(node.status)}</span>
                <span>${escapeHtml(node.kubelet_version || NO_VALUE)}</span>
                ${k8s.matched_by ? `<span class="update-time">matched by ${escapeHtml(k8s.matched_by)}</span>` : ""}
            </div>
            ${row("OS", joinParts([node.os_image, node.kernel_version, node.architecture]))}
            ${row("Runtime", node.container_runtime)}
            ${row("Capacity", joinParts([capacityText(node, latest), node.memory_capacity]))}
            ${pressure.length ? `<div class="endpoint-row"><span class="endpoint-label">Pressure</span>${pressure.map(p => `<span class="badge crash">${escapeHtml(p)}</span>`).join("")}</div>` : ""}
        `;
    }

    // A pod that is Pending or still creating its container carries no
    // startTime, and timeAgo answers an absent stamp with "never" — "started
    // never" reads as a pod that failed rather than one that has not begun.
    const podsHtml = pods.map(pod => `
        <div class="containers-list">
            <div class="pod-meta">
                <span>${escapeHtml(pod.name)}</span>
                <span class="badge ${pod.phase === "Running" ? "ready" : "not-ready"}">${escapeHtml(pod.phase)}</span>
                <span>${escapeHtml(pod.start_time ? `started ${timeAgo(pod.start_time)}` : "not started yet")}</span>
            </div>
            ${(pod.containers || []).map(containerHtml).join("")}
            ${row("Host", pod.host_ip)}
        </div>
    `).join("");

    paint("node-k8s", card("Kubernetes", `${nodeHtml}${podsHtml || '<div class="empty-state">No pod scheduled on this node.</div>'}`));
}

function containerHtml(c) {
    const state = c.state || "unknown";
    const stateClass = containerStateClass(c);
    // Only for a container that is running again: `last_finished_at` times the
    // PREVIOUS run's exit, and on one that is exited right now `last_error` is
    // that same exit — "OOMKilled (exit 137) — ended 2m ago" would then read as
    // a restart that has not happened.
    const ended = c.last_finished_at && !state.startsWith("exited")
        ? ` — ended ${escapeHtml(timeAgo(c.last_finished_at))}`
        : "";
    const errorHtml = c.last_error
        ? `<div class="pod-error">${escapeHtml(c.last_error)}${ended}</div>`
        : "";
    return `
        <div class="container-row">
            <span class="status-dot ${stateClass}"></span>
            <span class="container-name">${escapeHtml(c.name)}</span>
            <span class="badge ${stateClass === "online" ? "ready" : "not-ready"}">${escapeHtml(state)}</span>
            ${restartBadge(c)}
            <span>${escapeHtml(c.started_at ? `started ${timeAgo(c.started_at)}` : "")}</span>
        </div>
        ${errorHtml}
    `;
}

// "load 1m 3.2 / 4 cpus" — a load average means nothing without the core count
// it is measured against, and the node's own capacity is the only place that
// number exists.
function capacityText(node, latest) {
    const cpus = node.cpu_capacity;
    if (!cpus) return null;
    const load = latest ? latest.load_1m : undefined;
    const loadText = Number.isFinite(Number(load)) ? `load 1m ${Number(load).toFixed(2)} / ` : "";
    return `${loadText}${cpus} cpus`;
}

function renderMetrics(metrics) {
    if (!metrics || metrics.status !== "ok") {
        paint("node-metrics", '<div class="card"><div class="empty-state">The orchestrator has no telemetry store open.</div></div>');
        return;
    }
    const latest = metrics.latest || {};
    const history = metrics.history || {};
    const names = Object.keys(latest);
    if (!names.length) {
        paint("node-metrics", '<div class="card"><div class="empty-state">This node has not reported any metrics yet.</div></div>');
        return;
    }
    // The headline set first, in its own order, so the two screens agree on
    // what matters; the rest alphabetically, so a new collector metric lands
    // somewhere predictable instead of wherever SQLite returned it.
    const ordered = [
        ...HEADLINE_METRICS.filter(name => name in latest),
        ...names.filter(name => !HEADLINE_METRICS.includes(name)).sort(),
    ];
    const rows = ordered.map(name => {
        const points = sparklinePoints(history[name]);
        return `
            <div class="metric-row">
                <div class="metric-row-head">
                    <span class="metric-name">${escapeHtml(formatMetricName(name))}</span>
                    <span class="metric-value">${escapeHtml(formatMetricValue(name, latest[name]))}</span>
                </div>
                ${points ? `<svg class="sparkline" viewBox="0 0 100 24" preserveAspectRatio="none"><polyline points="${points}"/></svg>` : ""}
            </div>
        `;
    }).join("");
    paint("node-metrics", `<div class="card">${rows}</div>`);
}

// `points` for one sparkline, from a history series of [timestamp, value] pairs
// in oldest-first order. "" when there is nothing to draw — one point is a dot,
// not a trend, and an empty attribute would render as a stray mark at 0,0.
//
// The x axis is the sample INDEX, not the timestamp: samples are one a minute
// by construction, and spacing them by time would only encode the collector's
// jitter. A non-numeric sample (the metrics column has REAL affinity but the
// ingest route does not type-check) is dropped rather than plotted as 0.
function sparklinePoints(series) {
    if (!Array.isArray(series)) return "";
    const values = series
        .map(point => Number(Array.isArray(point) ? point[1] : point))
        .filter(Number.isFinite);
    if (values.length < 2) return "";
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min;
    const stepX = 100 / (values.length - 1);
    // Inset by one stroke width at each end rather than spanning the full 0..24
    // viewBox: `vector-effect: non-scaling-stroke` (style.css) draws the line at
    // a constant device width, so a series touching either edge had half its
    // stroke clipped and the peak of every chart read flatter than the data.
    // A flat series has no range to scale against and is drawn down the middle
    // rather than divided by zero.
    const y = value => (span === 0 ? 12 : 22 - ((value - min) / span) * 20);
    return values.map((value, i) => `${(i * stepX).toFixed(2)},${y(value).toFixed(2)}`).join(" ");
}

function renderLogs(logs) {
    if (!logs || logs.status !== "ok") {
        paint("node-logs", '<div class="empty-state">The orchestrator has no log store open.</div>');
        return;
    }
    const entries = Array.isArray(logs.entries) ? logs.entries : [];
    if (!entries.length) {
        paint("node-logs", '<div class="empty-state">No log lines shipped by this node yet.</div>');
        return;
    }
    // Newest first, which is why nothing here scrolls: the line that matters is
    // already at the top, so there is no autoscroll to fight with a selection.
    // Each carries its age because the store holds 30 days: without one, a crash
    // line from last week's wipe reads as the crash happening now.
    // One row per line with no whitespace between the tags: .log-pane is
    // white-space: pre-wrap so a shipped line keeps its own breaks, which also
    // means any indentation in this template would render as blank lines
    // between rows.
    const rows = entries.map(entry => {
        const level = LOG_LEVELS.includes(entry.level) ? entry.level : "info";
        return `<div class="log-line log-${level}">`
            + `<span class="log-badge">${escapeHtml(level)}</span>`
            + `<span class="log-time">${escapeHtml(timeAgo(entry.ts))}</span>`
            + `<span>${escapeHtml(entry.text)}</span></div>`;
    }).join("");
    paint("node-logs", `<div class="log-pane">${rows}</div>`);
}

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

async function refresh() {
    // One poll at a time, as on Overview: a backend answering slowly rather
    // than not at all would otherwise stack requests behind the browser's
    // six-connection-per-origin budget.
    if (refreshing) return;
    refreshing = true;
    try {
        await refreshOnce();
    } catch (e) {
        // Unawaited at both call sites, so anything escaping here becomes an
        // unhandled rejection and silently stops the render.
        console.error("node refresh failed", e);
    } finally {
        refreshing = false;
    }
}

async function refreshOnce() {
    // Both captured BEFORE the fetch: a #node/<a> → #node/<b> switch re-mounts
    // this module and reassigns both module-level values, so reading them after
    // the await compares the new mount against itself — and a's answer then
    // paints into b's cards, where setHtmlIfChanged memoises it as what they
    // hold. `nodeId` is checked as well as the token because main.js re-mounts
    // the same section for that transition.
    const id = nodeId;
    const token = mountToken;
    const data = await fetchJSON(`/api/nodes/${encodeURIComponent(id)}`);
    // unmount() nulls intervalId, which is what "not mounted at all" looks like.
    if (!isCurrent(token) || intervalId === null || id !== nodeId) return;
    if (isError(data)) {
        // 404 for an unknown or disabled node, 503 for an unreadable config, or
        // a transport failure. Said once: the other three cards have nothing to
        // add that this does not already say.
        paint("node-identity", card("Node", `<div class="empty-state">${escapeHtml(errorText(data))}</div>`));
        paint("node-k8s", "");
        paint("node-metrics", "");
        paint("node-logs", "");
        return;
    }
    renderIdentity(data.identity, data.metrics);
    renderK8s(data.k8s, data.identity, data.metrics ? data.metrics.latest : null);
    renderMetrics(data.metrics);
    renderLogs(data.logs);

    const updated = document.getElementById("node-updated");
    if (updated) updated.textContent = `Updated ${new Date().toLocaleTimeString()}`;
}
