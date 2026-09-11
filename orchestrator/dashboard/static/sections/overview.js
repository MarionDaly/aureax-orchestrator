// Overview section — cluster, pod, and telemetry state. Polls every 10s.

import { containerStateClass, errorText, escapeHtml, fetchJSON, formatMetricName, formatMetricValue, HEADLINE_METRICS, isError, isFresh, restartBadge, setHtmlIfChanged, timeAgo } from "/utils.js";
import { isCurrent } from "/main.js";

const REFRESH_INTERVAL = 10000;

let intervalId = null;
let refreshing = false;
let mountToken;

export function mount(root, _arg, token) {
    mountToken = token;
    root.innerHTML = `
        <section>
            <h2>Cluster</h2>
            <div id="cluster-info" class="card-grid"></div>
        </section>

        <section>
            <h2>Edge Pods</h2>
            <div id="pods-grid" class="card-grid"></div>
        </section>

        <section>
            <h2>Telemetry</h2>
            <div id="nodes-grid" class="card-grid"></div>
        </section>
    `;
    refresh();
    intervalId = setInterval(refresh, REFRESH_INTERVAL);
}

export function unmount() {
    if (intervalId) clearInterval(intervalId);
    intervalId = null;
    // Released so the next mount's first refresh is not skipped by a poll that
    // was still in flight when the tab was switched.
    refreshing = false;
    // #last-update lives in the page header, outside this section's root, so
    // it survives the tab switch and would sit there frozen on another tab.
    const last = document.getElementById("last-update");
    if (last) last.textContent = "";
}

function renderCluster(data) {
    const el = document.getElementById("cluster-info");
    if (!el) return;
    // Both failure shapes, and a defensive check on `nodes`. fetchJSON returns
    // {_error, _message} on a transport failure — no `.status`, no `.nodes` —
    // and `.map` on a missing array throws inside an unawaited refresh(). That
    // becomes an unhandled rejection which skips renderPods and renderNodes
    // too, freezing the whole tab on stale data with nothing to explain it.
    if (isError(data)) {
        setHtmlIfChanged(el, `<div class="card"><div class="empty-state">Cluster unavailable: ${escapeHtml(errorText(data))}</div></div>`);
        return;
    }
    if (data.status === "error") {
        setHtmlIfChanged(el, `<div class="card"><div class="empty-state">Cluster unavailable: ${escapeHtml(data.message || "unknown")}</div></div>`);
        return;
    }
    if (!Array.isArray(data.nodes)) {
        setHtmlIfChanged(el, '<div class="card"><div class="empty-state">Cluster returned no node list.</div></div>');
        return;
    }
    const nodesHtml = data.nodes.map(node => {
        const isReady = node.status === "Ready";
        // CONTRACT: both halves are configuration, not constants —
        // `kubernetes.node_label` and `kubernetes.edge_label_value` in
        // orchestrator/config/schema.py, mirrored in configs/aureax-network.yaml,
        // k8s/daemonset-edge.yaml, scripts/setup-all.sh and scripts/test-all.sh.
        // Nothing reconciles them, so changing a value there drops this badge
        // with no error. Shard workers carry the orthogonal `aureax.io/shard`
        // label (kubernetes.shard_label) and match nothing here.
        const isEdge = node.labels?.["aureax.io/role"] === "edge";
        return `
            <div class="cluster-node">
                <span class="status-dot ${isReady ? "online" : "offline"}"></span>
                <span>${escapeHtml(node.name)}</span>
                <span class="badge ${isReady ? "ready" : "not-ready"}">${escapeHtml(node.status)}</span>
                ${isEdge ? '<span class="badge edge">edge</span>' : ""}
            </div>
        `;
    }).join("");
    setHtmlIfChanged(el, `<div class="card"><div class="card-header"><h3>K3s Nodes (${data.nodes.length})</h3></div>${nodesHtml}</div>`);
}

function renderPods(data) {
    const el = document.getElementById("pods-grid");
    if (!el) return;
    if (isError(data)) {
        setHtmlIfChanged(el, `<div class="empty-state">Could not fetch pod status: ${escapeHtml(errorText(data))}</div>`);
        return;
    }
    if (data.status === "error") {
        setHtmlIfChanged(el, `<div class="empty-state">Could not fetch pod status: ${escapeHtml(data.message || "unknown")}</div>`);
        return;
    }
    if (!data.pods || data.pods.length === 0) {
        setHtmlIfChanged(el, '<div class="empty-state">No edge pods deployed yet.</div>');
        return;
    }
    setHtmlIfChanged(el, data.pods.map(pod => {
        const statusClass = pod.ready ? "online" : (pod.phase === "Running" ? "degraded" : "offline");
        const containersHtml = pod.containers.map(c => {
            const stateClass = containerStateClass(c);
            const errorHtml = c.last_error
                ? `<div class="pod-error">${escapeHtml(c.last_error)}</div>`
                : "";
            return `
                <div class="container-row">
                    <span class="status-dot ${stateClass}"></span>
                    <span class="container-name">${escapeHtml(c.name)}</span>
                    <span class="badge ${stateClass === "online" ? "ready" : "not-ready"}">${escapeHtml(c.state)}</span>
                    ${restartBadge(c)}
                </div>
                ${errorHtml}
            `;
        }).join("");
        return `
            <div class="card">
                <div class="card-header">
                    <span class="status-dot ${statusClass}"></span>
                    <h3>${escapeHtml(pod.name)}</h3>
                </div>
                <div class="pod-meta">
                    <span>Node: ${escapeHtml(pod.node || "unscheduled")}</span>
                    <span class="badge ${pod.phase === "Running" ? "ready" : "not-ready"}">${escapeHtml(pod.phase)}</span>
                </div>
                <div class="containers-list">
                    ${containersHtml}
                </div>
            </div>
        `;
    }).join(""));
}

function renderNodes(nodes) {
    const el = document.getElementById("nodes-grid");
    if (!el) return;
    // Errors first, like renderCluster and renderPods. fetchJSON returns
    // {_error, _message} on a failure — not an array — and collapsing that to
    // [] would report an unreachable dashboard or an unreadable metrics DB as
    // "the fleet is not reporting", which is a different problem entirely.
    if (isError(nodes)) {
        setHtmlIfChanged(el, `<div class="empty-state">Could not fetch telemetry: ${escapeHtml(errorText(nodes))}</div>`);
        return;
    }
    if (!Array.isArray(nodes) || nodes.length === 0) {
        setHtmlIfChanged(el, '<div class="empty-state">No edge nodes reporting metrics yet.</div>');
        return;
    }
    setHtmlIfChanged(el, nodes.map(node => {
        const lastSeen = timeAgo(node.last_seen);
        const statusClass = isFresh(node.last_seen) ? "online" : "offline";
        const metrics = node.metrics || {};
        // A shortlist, not everything the node ships: the card is one of nine in
        // a grid, and listing every metric made each new collector reading push
        // the next card further down the page. The count is what says the rest
        // exist, and the card links to where they are shown.
        const headline = HEADLINE_METRICS.filter(name => name in metrics);
        const metricsHtml = headline.map(name => `
            <li>
                <span class="metric-name">${escapeHtml(formatMetricName(name))}</span>
                <span class="metric-value">${escapeHtml(formatMetricValue(name, metrics[name]))}</span>
            </li>
        `).join("");
        const others = Object.keys(metrics).length - headline.length;
        const moreHtml = others > 0 ? `<li class="more-metrics">+${others} more</li>` : "";
        return `
            <a class="card node-card" href="#node/${escapeHtml(node.node_id)}">
                <div class="card-header">
                    <span class="status-dot ${statusClass}"></span>
                    <h3>${escapeHtml(node.node_id)}</h3>
                    <span class="update-time">${lastSeen}</span>
                </div>
                <ul class="metrics-list">${metricsHtml || '<li class="empty-state">No metrics</li>'}${moreHtml}</ul>
            </a>
        `;
    }).join(""));
}

async function refresh() {
    // One poll at a time. Each tick opens three fetches, so a backend answering
    // slowly rather than not at all would stack them against the browser's
    // six-connection-per-origin budget.
    if (refreshing) return;
    refreshing = true;
    try {
        await refreshOnce();
    } catch (e) {
        // Unawaited at both call sites, so anything escaping here becomes an
        // unhandled rejection and silently stops the render.
        console.error("overview refresh failed", e);
    } finally {
        refreshing = false;
    }
}

async function refreshOnce() {
    const [cluster, pods, nodes] = await Promise.all([
        fetchJSON("/api/cluster"),
        fetchJSON("/api/pods"),
        fetchJSON("/api/nodes"),
    ]);
    // A superseded mount must not paint. main.js empties the root and re-mounts,
    // so a response belonging to the previous mount would otherwise land in the
    // new mount's containers, be memoised by setHtmlIfChanged as what they
    // hold, and be stamped fresh below.
    if (!isCurrent(mountToken)) return;
    renderCluster(cluster);
    renderPods(pods);
    renderNodes(nodes);

    // #last-update is in the page header, not this section's root, so it
    // survives a tab switch: a refresh resolving after one must not re-stamp it,
    // or the timestamp sits frozen on Lifecycle/Configs/Chat. unmount() nulls
    // intervalId, which is what "this section is not mounted" looks like here.
    if (intervalId === null) return;
    const last = document.getElementById("last-update");
    if (last) last.textContent = `Updated ${new Date().toLocaleTimeString()}`;
}
