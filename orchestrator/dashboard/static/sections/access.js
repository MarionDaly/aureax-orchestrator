// Access section — per-node endpoints (OpenAI-compatible API, web UI, MCP) and
// the API-key auth workflow: stage in the template, rotate, then Apply.
//
// Hash: #access            — every node
//       #access/<node_id>  — same list, scrolled to and highlighting that card
//
// Three separate truths come back from the backend and are NOT recomputed here:
// the staged verdict (auth.state), the live probe, and the final `badge` the
// live route computes from both. Render the badge; do not second-guess it.
//
// A node has TWO keyed services, so everything below is per KIND: "llama"
// reads entry.auth and the /key routes, "mcp" reads entry.auth.mcp and the
// /mcp-key routes. The kind travels on the button (data-kind) rather than
// being inferred from which block a click landed in — an inference that goes
// wrong hands the operator the wrong service's credential.

import { copyText, errorText, escapeHtml, fetchJSON, isError, setHtmlIfChanged } from "/utils.js";
import { isCurrent } from "/main.js";

// Each tick re-probes EVERY llama-serving node over the network
// (access_service.probe_all: a TCP connect plus one or two HTTP GETs per node,
// ~1s wall-clock). This is the only browser-driven timer that issues HTTP
// requests to llama-server itself. It is not the only one that touches the
// fleet: the Lifecycle tab's 1.5s state poll TCP-probes every node's llama
// NodePort, though behind a ~3s server-side cache. Weigh both before
// shortening either.
const LIVE_POLL_MS = 20000;
const APPLY_POLL_MS = 2000;
// Only armed after a lifecycle-busy 409, and only until that run ends. The
// route is served from LifecycleManager's in-memory state machine plus a
// node snapshot cached for 3s, so this touches no node of its own.
const LIFECYCLE_POLL_MS = 2000;
const COPY_FEEDBACK_MS = 1500;

// Both listeners this page is served on are potentially trustworthy origins —
// https on the LAN address, http://localhost on loopback — so the async
// Clipboard API is the path that normally runs and "copied" is the normal
// label. "select" is copyText's last resort, reached only when writeText is
// refused (the document is not focused, or the permission was denied) and
// execCommand cannot rescue it. Name both modifiers — the page is served to
// every platform on the LAN.
const COPY_LABEL = {
    copied: "Copied ✓",
    select: "Selected — press Ctrl/Cmd+C",
    failed: "Copy failed",
};

// Badge text + class. `in_sync` is the staged-only verdict shown before the
// first live probe lands: config matches what was pushed, but nothing has
// confirmed the running pod yet — so it is deliberately not the green that
// `applied` gets. classify() in access_service.py holds the same line.
const BADGE = {
    applied:       { label: "Applied",       cls: "badge-applied" },
    in_sync:       { label: "In sync",       cls: "badge-in-sync" },
    pending:       { label: "Pending apply", cls: "badge-pending" },
    never_applied: { label: "Never applied", cls: "badge-pending" },
    drift:         { label: "Drift",         cls: "badge-drift" },
    unknown:       { label: "Unknown",       cls: "badge-unknown" },
    unsupported:   { label: "n/a",           cls: "badge-unsupported" },
};

// The two keyed services, and everything that differs between them. The URL
// segment is the only place the route names appear: /key vs /mcp-key, each
// with its own /previous.
const KINDS = {
    llama: { label: "llama-server (:30800)", path: "key",     envVar: "OPENAI_API_KEY" },
    mcp:   { label: "MCP (:30900/mcp)",      path: "mcp-key", envVar: "AUREAX_MCP_API_KEY" },
};

let state = emptyState();
let liveTimer = null;
let applyTimer = null;
let lifecycleTimer = null;
let mountToken = null;

function emptyState() {
    return {
        entries: [],        // /api/access/nodes, in the order returned
        loadError: null,
        live: {},           // node_id -> live blob (incl. the final badge)
        liveCheckedAt: null,
        liveError: null,
        revealed: {},       // node_id -> {llama, mcp} plaintext keys, memory-only
        notes: {},          // node_id -> transient message from the last action
        applyRunning: false,
        applyOutcomes: {},  // node_id -> outcome string from the apply run
        applyError: null,
        applyPollError: null, // last failed /apply/state poll, shown only while
                              // a run is being tracked
        busyNodes: [],      // node_ids a 409 marked "Busy"; their notes clear
                            // when the run that blocked them finishes
        focusNode: null,     // deep-link target, consumed once
        highlightNode: null, // card currently flashing; kept in state so a
                             // re-render does not drop the highlight
    };
}

export async function mount(root, arg, token) {
    state = emptyState();
    state.focusNode = arg || null;
    mountToken = token;

    root.innerHTML = `
        <section>
            <h2>Access</h2>
            <div class="access-bar" id="access-bar"></div>
            <div class="access-cards" id="access-cards">
                <div class="empty-state">Loading endpoints…</div>
            </div>
        </section>
    `;

    const cards = document.getElementById("access-cards");
    cards.addEventListener("click", onCardClick);
    cards.addEventListener("change", onCardChange);
    document.getElementById("access-bar").addEventListener("click", onBarClick);

    await loadNodes();
    if (!isCurrent(token)) return;
    render();

    // An apply may already be running — started from another browser tab, or
    // before a reload. Probe once so the buttons disable themselves rather than
    // offering an action whose only answer is a 409.
    await pollApply();
    if (!isCurrent(token)) return;

    await loadLive();
    if (!isCurrent(token)) return;
    render();

    // Scroll last: render() rewrites the whole card list, so a render after
    // this point destroys the target node and cancels the smooth scroll
    // mid-flight, leaving the page at the top.
    consumeFocus();

    liveTimer = setInterval(refreshLive, LIVE_POLL_MS);
}

export function unmount() {
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    if (applyTimer) { clearInterval(applyTimer); applyTimer = null; }
    stopLifecycleWait();
    mountToken = null;
    state = emptyState();
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

async function loadNodes() {
    const data = await fetchJSON("/api/access/nodes");
    if (isError(data)) {
        state.loadError = errorText(data);
        state.entries = [];
        return;
    }
    state.loadError = null;
    state.entries = Array.isArray(data) ? data : [];
}

async function loadLive() {
    const data = await fetchJSON("/api/access/live");
    if (isError(data)) {
        state.liveError = errorText(data);
        return;
    }
    state.liveError = null;
    state.live = data.nodes || {};
    state.liveCheckedAt = data.checked_at || null;
}

async function refreshLive() {
    await loadLive();
    if (!isCurrent(mountToken)) return;
    render();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render() {
    renderBar();
    renderCards();
}

function entryById(nodeId) {
    return state.entries.find(e => e.node_id === nodeId) || null;
}

// How bad each verdict is, worst first — used to collapse the node's two
// keyed services into the ONE badge in the card header. A card that reads
// "Applied" while its :30900 is drifting is the false green this tab exists to
// prevent, so the header shows the worse of the two and the sub-blocks say
// which is which.
const BADGE_RANK = ["drift", "unknown", "never_applied", "pending", "in_sync", "applied"];

// The effective badge: the live route's verdict once it has answered,
// otherwise the staged state that came with the node list.
function badgeFor(entry) {
    const badges = Object.keys(KINDS)
        .filter(kind => authFor(entry, kind).supported)
        .map(kind => kindBadge(entry, kind));
    if (!badges.length) return (entry.auth || {}).state || "unknown";
    const worst = Math.min(...badges.map(b => {
        const i = BADGE_RANK.indexOf(b);
        return i === -1 ? BADGE_RANK.indexOf("unknown") : i;
    }));
    return BADGE_RANK[worst];
}

function messageFor(entry) {
    const live = state.live[entry.node_id];
    if (live && live.message) return live.message;
    return (entry.auth || {}).message || "";
}

// The same two rules for the MCP sub-block, against the live route's nested
// `mcp` blob. Kept separate from badgeFor/messageFor rather than parameterised:
// the card's own badge is the llama one, and collapsing them invites a future
// edit that lets an MCP drift silently colour the whole card.
function authFor(entry, kind) {
    const auth = entry.auth || {};
    return (kind === "mcp" ? auth.mcp : auth) || {};
}

function kindLive(entry, kind) {
    const live = state.live[entry.node_id] || {};
    return (kind === "mcp" ? live.mcp : live) || {};
}

function kindBadge(entry, kind) {
    return kindLive(entry, kind).badge || authFor(entry, kind).state || "unknown";
}

function kindMessage(entry, kind) {
    return kindLive(entry, kind).message || authFor(entry, kind).message || "";
}

function pendingEntries() {
    // Either service being out of date makes the NODE out of date — one apply
    // pushes both keys, and a node whose MCP key is staged but unapplied is
    // exactly what "Apply all pending" is for. Checking only the llama half
    // hid a newly keyed :30900 from the button that fixes it.
    return state.entries.filter(e =>
        Object.keys(KINDS).some(kind => {
            if (!authFor(e, kind).supported) return false;
            const badge = kindBadge(e, kind);
            return badge === "pending" || badge === "never_applied";
        })
    );
}

function renderBar() {
    const el = document.getElementById("access-bar");
    if (!el) return;

    const pending = pendingEntries();
    const applyAll = pending.length
        ? `<button class="btn btn-primary" data-action="apply-all" ${state.applyRunning ? "disabled" : ""}>
               Apply all pending (${pending.length})
           </button>`
        : "";

    const checked = state.liveCheckedAt
        ? `Live check ${escapeHtml(new Date(state.liveCheckedAt).toLocaleTimeString())}`
        : "Live check pending…";

    const liveErr = state.liveError
        ? `<span class="access-note">Live probe unavailable: ${escapeHtml(state.liveError)}</span>`
        : "";

    const running = state.applyRunning
        ? '<span class="access-note">Apply running — pods are restarting…</span>'
        : "";

    // Only while a run is tracked: the mount-time probe polls once with nothing
    // in flight, and a banner from that has no later poll to clear it.
    const pollErr = state.applyRunning && state.applyPollError
        ? `<span class="access-note">Apply status unavailable: ${escapeHtml(state.applyPollError)}</span>`
        : "";

    el.innerHTML = `
        ${applyAll}
        <span class="access-muted">${checked}</span>
        ${running}
        ${pollErr}
        ${liveErr}
        <span class="spacer"></span>
        <span class="access-muted">Keys are stored on the orchestrator and never shown in list responses.</span>
    `;
}

function renderCards() {
    const el = document.getElementById("access-cards");
    if (!el) return;

    let html;
    if (state.loadError) {
        html = `<div class="card"><div class="empty-state">Access data unavailable: ${escapeHtml(state.loadError)}</div></div>`;
    } else if (!state.entries.length) {
        html = '<div class="card"><div class="empty-state">No enabled nodes.</div></div>';
    } else {
        html = state.entries.map(cardHtml).join("");
    }

    // Skipped when it would rewrite what is already there: the 20s live poll
    // and the 2s apply poll both land here, normally with nothing but one badge
    // string changed, and a rewrite drops a text selection, the focus in a
    // readonly copy field and the 1.5s "Copied ✓" label. The memo lives in
    // utils.js because Overview needs the same thing.
    setHtmlIfChanged(el, html);
}

function headerHtml(entry) {
    const badge = badgeFor(entry);
    const meta = BADGE[badge] || BADGE.unknown;
    const parts = [entry.node_id];
    if (entry.role) parts.push(entry.role);
    if (entry.group) {
        const n = (entry.group_members || []).length;
        parts.push(n ? `${entry.group} (${n} nodes)` : entry.group);
    }
    return `
        <div class="access-head">
            <span class="access-title">${escapeHtml(entry.display_name)}</span>
            <span class="access-sub">${escapeHtml(parts.join(" · "))}</span>
            <span class="spacer"></span>
            <span class="badge ${meta.cls}">${escapeHtml(meta.label)}</span>
        </div>
    `;
}

function endpointRow(label, value, extra = "") {
    return `
        <div class="endpoint-row">
            <span class="endpoint-label">${escapeHtml(label)}</span>
            <input class="copy-field" readonly value="${escapeHtml(value)}" spellcheck="false">
            <button class="btn" data-action="copy-field">Copy</button>
            ${extra}
        </div>
    `;
}

// `withMessage: false` on worker cards: their only auth.message is the
// "no API key" line the card already states verbatim as its muted note, so
// rendering both prints it twice. Apply outcomes still show — a group apply
// pushes configs to every member, workers included.
function statusHtml(entry, withMessage = true) {
    const lines = [];
    const outcome = state.applyOutcomes[entry.node_id];
    if (outcome) lines.push(`<div class="access-outcome">Apply: ${escapeHtml(outcome)}</div>`);
    const note = state.notes[entry.node_id];
    if (note) lines.push(`<div class="access-note">${escapeHtml(note)}</div>`);
    const msg = withMessage ? messageFor(entry) : "";
    if (msg) lines.push(`<div>${escapeHtml(msg)}</div>`);
    // The MCP verdict has its own badge in the auth section but no text of its
    // own there — this is where the reason for it is written. Prefixed, and
    // skipped when it repeats the llama line: the two share one desired hash,
    // so "Saved but not applied" is normally the same sentence twice.
    const mcpMsg = withMessage && (entry.auth || {}).mcp?.supported
        ? kindMessage(entry, "mcp") : "";
    if (mcpMsg && mcpMsg !== msg) lines.push(`<div>MCP: ${escapeHtml(mcpMsg)}</div>`);
    if (!lines.length) return "";
    return `<div class="access-status">${lines.join("")}</div>`;
}

function cardHtml(entry) {
    return entry.serves_llama ? serverCardHtml(entry) : workerCardHtml(entry);
}

// Rendered on worker cards as well as serving ones: a worker's snippet is built
// from its HEAD's base URL, and its key line names the head whose key field the
// operator has to copy from (access_service._env_snippet). That is the useful
// thing to hand someone looking at a worker.
// The OpenAI-compatible half and the MCP half of the block, in one <pre>. Both
// come from the backend (access_service._env_snippet / _mcp_snippet) so the
// variable names cannot drift from what the harness actually reads.
function snippetLines(entry) {
    return [
        ...(Array.isArray(entry.env_snippet) ? entry.env_snippet : []),
        ...(Array.isArray(entry.mcp_snippet) ? entry.mcp_snippet : []),
    ];
}

function snippetHtml(entry) {
    const lines = snippetLines(entry);
    if (!lines.length) return "";
    return `
        <div class="snippet-block">
            <div class="endpoint-label">For your harness</div>
            <pre class="snippet">${escapeHtml(lines.join("\n"))}</pre>
            <button class="btn" data-action="copy-snippet" data-node="${escapeHtml(entry.node_id)}">Copy block</button>
        </div>
    `;
}

// A shard worker serves nothing itself: every URL field comes back null so a
// copy button structurally cannot hand out a dead address. Show the head's
// endpoints instead, and no auth or apply controls at all.
function workerCardHtml(entry) {
    const head = entry.served_by;
    const servedBy = head
        ? `<div class="access-muted">Served by ${escapeHtml(head.display_name)} (${escapeHtml(head.node_id)})</div>`
        : '<div class="access-muted">Not attached to a group head.</div>';

    // served_by mirrors the head's own entry (access_service's second pass), so
    // chat_ui_url is already gated on the head's webui setting — a worker card
    // cannot advertise a UI its head has switched off.
    const rows = head
        ? endpointRow("OpenAI-compatible API", head.api_base_url) +
          (head.chat_ui_url ? endpointRow("Browser chat UI", head.chat_ui_url) : "")
        : "";

    return `
        <div class="card access-card access-card-worker${highlightClass(entry)}" data-card="${escapeHtml(entry.node_id)}">
            ${headerHtml(entry)}
            ${servedBy}
            ${rows}
            ${snippetHtml(entry)}
            <div class="access-muted access-worker-note">Shard workers run rpc-server, which has no API key.</div>
            ${statusHtml(entry, false)}
        </div>
    `;
}

function serverCardHtml(entry) {
    const auth = entry.auth || {};

    const apiRow = entry.api_base_url
        ? endpointRow("OpenAI-compatible API", entry.api_base_url)
        : '<div class="endpoint-row access-muted">No API endpoint for this node.</div>';

    const modelRow = entry.model_id
        ? `<div class="endpoint-row access-muted">Model id: <code>${escapeHtml(entry.model_id)}</code></div>`
        : "";

    // A link, not a copy field: this one is meant to be opened.
    let uiRow;
    if (entry.chat_ui_url) {
        uiRow = `
            <div class="endpoint-row">
                <span class="endpoint-label">Browser chat UI</span>
                <a class="access-link" href="${escapeHtml(entry.chat_ui_url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(entry.chat_ui_url)}</a>
            </div>
        `;
    } else if (!entry.webui_enabled) {
        uiRow = `
            <div class="endpoint-row">
                <span class="endpoint-label">Browser chat UI</span>
                <span class="access-muted">web UI disabled on this node</span>
            </div>
        `;
    } else {
        uiRow = "";
    }

    const mcpRow = entry.mcp_url ? endpointRow("MCP", entry.mcp_url) : "";

    const applyDisabled = state.applyRunning || !auth.supported ? "disabled" : "";

    return `
        <div class="card access-card${highlightClass(entry)}" data-card="${escapeHtml(entry.node_id)}">
            ${headerHtml(entry)}
            ${apiRow}
            ${modelRow}
            ${uiRow}
            ${mcpRow}
            ${snippetHtml(entry)}
            ${authHtml(entry)}
            <div class="access-actions">
                <button class="btn btn-primary" data-action="apply" data-node="${escapeHtml(entry.node_id)}" ${applyDisabled}>Apply to this node</button>
            </div>
            ${statusHtml(entry)}
        </div>
    `;
}

function authHtml(entry) {
    const auth = entry.auth || {};
    if (!auth.supported && !(auth.mcp || {}).supported) return "";

    const sharers = entry.shared_template_with || [];
    // One note for the whole section: both toggles edit the SAME template file,
    // so a shared file affects the sharers whichever service is toggled.
    const sharedNote = sharers.length
        ? `<div class="access-muted">Template shared with: ${escapeHtml(sharers.join(", "))} — toggling auth changes them too.</div>`
        : "";

    return `
        <div class="access-auth">
            <div class="endpoint-label">Authentication</div>
            ${sharedNote}
            ${authKindHtml(entry, "llama", KINDS.llama.label)}
            ${authKindHtml(entry, "mcp", KINDS.mcp.label)}
        </div>
    `;
}

function authKindHtml(entry, kind, label) {
    const auth = authFor(entry, kind);
    if (!auth.supported) return "";
    const nodeId = escapeHtml(entry.node_id);
    const k = escapeHtml(kind);
    const data = `data-node="${nodeId}" data-kind="${k}"`;

    const revealed = revealedKey(entry.node_id, kind);
    let keyBlock;
    if (auth.has_key) {
        const value = revealed || auth.key_masked || "";
        // Rotate is offered only while auth is staged ON — see the rule in the
        // final branch below. Reveal/Copy stay unconditional: nothing deletes a
        // .key file, so the running pod may still be enforcing this key until
        // the next Apply, and the operator needs to be able to read it.
        const rotateBtn = auth.desired_enabled
            ? `<button class="btn btn-ghost" data-action="rotate" ${data}>Rotate</button>`
            : "";
        keyBlock = `
            <div class="key-field">
                <input class="copy-field" readonly value="${escapeHtml(value)}" spellcheck="false">
                <button class="btn" data-action="${revealed ? "hide-key" : "reveal-key"}" ${data}>${revealed ? "Hide" : "Reveal"}</button>
                <button class="btn" data-action="copy-key" ${data}>Copy</button>
                ${rotateBtn}
            </div>
        `;
    } else if (auth.key_state === "unreadable") {
        // A key file that EXISTS but cannot be read — a .key left root-owned by
        // a sudo run is the usual cause. Distinct from "no key staged yet"
        // because the two call for opposite actions: this one is not fixed by
        // generating another, and a Generate button here would fail on the
        // write. access_service._kind_block carries the fix in `message`, which
        // statusHtml renders.
        keyBlock = `
            <div class="key-field">
                <span class="access-muted">A key file exists but cannot be read — fix its ownership.</span>
            </div>
        `;
    } else if (auth.desired_enabled) {
        keyBlock = `
            <div class="key-field">
                <span class="access-muted">No key staged yet.</span>
                <button class="btn" data-action="rotate" ${data}>Generate key</button>
            </div>
        `;
    } else {
        // The rule, for this branch and the has_key one above: no rotate/generate
        // button while auth is off. POST .../auth/rotate answers 409 for a node
        // whose template carries no api_key placeholder line, so the button
        // could only ever produce an error alert. Tick the box first.
        keyBlock = `
            <div class="key-field">
                <span class="access-muted">No key needed — auth is off for this port.</span>
            </div>
        `;
    }

    // The pre-rotation key is what a still-running harness is holding: it keeps
    // working until the Apply that installs the new one.
    const prev = auth.has_previous_key
        ? `<div class="access-muted">Previous key still live on this node —
               <button class="btn btn-link" data-action="copy-previous" ${data}>Copy previous key</button>
           </div>`
        : "";

    // Its own badge, from the live route's per-port probe: a staged-but-
    // unapplied MCP key is invisible to the llama probe, so one badge for the
    // card would report a node green while its :30900 is still open. The
    // explanation behind the badge is rendered once, by statusHtml.
    const meta = BADGE[kindBadge(entry, kind)] || BADGE.unknown;

    return `
        <div class="access-auth-kind" data-auth-kind="${k}">
            <div class="access-auth-head">
                <span class="endpoint-label">${escapeHtml(label)}</span>
                <span class="badge ${meta.cls}">${escapeHtml(meta.label)}</span>
            </div>
            <label class="auth-toggle">
                <input type="checkbox" data-action="toggle-auth" ${data} ${auth.desired_enabled ? "checked" : ""}>
                Require an API key
            </label>
            ${keyBlock}
            ${prev}
        </div>
    `;
}

// Deep link: scroll the named card into view and flash its border. Consumed
// once, so a later re-render (a live poll, an apply outcome) does not yank the
// page back while the operator is reading a different card.
//
// The highlight lives in state, not only on the node: the 20s live poll rewrites
// the card list, so highlightClass() has to be able to put the class back on the
// replacement element.
function consumeFocus() {
    const nodeId = state.focusNode;
    if (!nodeId) return;
    state.focusNode = null;
    const card = document.querySelector(`[data-card="${CSS.escape(nodeId)}"]`);
    if (!card) return;
    state.highlightNode = nodeId;
    card.classList.add("access-card-focus");
    // Smooth scrolling is driven by requestAnimationFrame, which a hidden tab
    // throttles to a standstill, so a deep link opened in a background tab
    // never arrives. Jump instantly in that case.
    card.scrollIntoView({
        behavior: document.hidden ? "auto" : "smooth",
        block: "center",
    });
    window.setTimeout(() => {
        if (state.highlightNode !== nodeId) return;
        state.highlightNode = null;
        const el = document.querySelector(`[data-card="${CSS.escape(nodeId)}"]`);
        if (el) el.classList.remove("access-card-focus");
    }, 2500);
}

function highlightClass(entry) {
    return state.highlightNode === entry.node_id ? " access-card-focus" : "";
}

function setNote(nodeId, text) {
    if (text) state.notes[nodeId] = text;
    else delete state.notes[nodeId];
}

// ---------------------------------------------------------------------------
// Copy handling
// ---------------------------------------------------------------------------

async function runCopy(btn, text, target) {
    // Remember the ORIGINAL label: rapid clicks would otherwise capture
    // "Copied ✓" as the label to revert to.
    const original = btn.dataset.label || btn.textContent.trim();
    btn.dataset.label = original;
    const outcome = await copyText(text, target);
    if (!btn.isConnected) return outcome;
    btn.textContent = COPY_LABEL[outcome] || COPY_LABEL.failed;
    btn.classList.add("copy-feedback");
    window.setTimeout(() => {
        // The card may have been re-rendered while the feedback was showing.
        if (!btn.isConnected) return;
        btn.textContent = original;
        btn.classList.remove("copy-feedback");
    }, COPY_FEEDBACK_MS);
    return outcome;
}

async function copyFieldOf(btn) {
    const input = btn.closest(".endpoint-row, .key-field")?.querySelector(".copy-field");
    if (!input) return;
    await runCopy(btn, input.value, input);
}

// The displayed snippet carries a placeholder line per keyed service (see
// _env_snippet / _mcp_snippet in access_service.py). When a service has auth on
// and a key staged, the COPIED text substitutes the real one, fetched on demand
// from that service's plaintext route; with auth off the copied text is what is
// on screen.
async function copySnippet(btn, entry) {
    const pre = btn.closest(".snippet-block")?.querySelector(".snippet");
    const lines = snippetLines(entry).slice();
    const failed = [];
    let substituted = false;
    let note = "";

    for (const kind of Object.keys(KINDS)) {
        const auth = authFor(entry, kind);
        if (!auth.supported || !auth.desired_enabled || !auth.has_key) continue;
        const data = await fetchJSON(keyUrl(entry.node_id, kind));
        if (!isCurrent(mountToken)) return;
        if (!isError(data) && data.api_key) {
            substituted = true;
            replaceKeyLine(lines, KINDS[kind].envVar, data.api_key);
        } else {
            replaceKeyLine(lines, KINDS[kind].envVar, "<click Reveal to copy>");
            failed.push(`${KINDS[kind].label}: ${errorText(data)}`);
        }
    }
    if (failed.length) {
        note = `Could not fetch ${failed.join("; ")} — copied with a placeholder instead.`;
    }

    const outcome = await runCopy(btn, lines.join("\n"), pre);
    if (!isCurrent(mountToken)) return;
    if (substituted && outcome !== "copied") {
        note = "Copy failed — the block on screen still shows the placeholder line. " +
               "Use Reveal on the key field to get the real one.";
    }
    // Render only when a note appears or clears. render() replaces the button
    // runCopy has just relabelled "Copied ✓" and drops copyText's last-resort
    // selection of the <pre>, which on a failed copy is the only way left to
    // get the text.
    const had = state.notes[entry.node_id];
    setNote(entry.node_id, note);
    if (note || had) render();
}

function replaceKeyLine(lines, envVar, value) {
    for (let i = 0; i < lines.length; i++) {
        if (String(lines[i]).startsWith(`${envVar}=`)) {
            lines[i] = `${envVar}=${value}`;
        }
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

async function onCardChange(e) {
    const input = e.target.closest('input[data-action="toggle-auth"]');
    if (!input) return;
    const entry = entryById(input.dataset.node);
    if (!entry) return;
    const kind = input.dataset.kind || "llama";
    const wanted = input.checked;

    const sharers = entry.shared_template_with || [];
    // Turning auth OFF names what actually opens, per service: "the endpoint"
    // is accurate for llama-server and useless for MCP, where the open port
    // reaches run_inference — which forwards to llama-server WITH this node's
    // llama key attached. An operator unticking that box is opening keyed
    // inference to the LAN, and the confirm has to say so.
    const opens = kind === "mcp"
        ? "Anyone who can reach :30900 can call this node's MCP tools, including " +
          "run_inference — which forwards to llama-server with the node's own API key."
        : "The endpoint becomes open to anyone who can reach it on the LAN.";
    let msg = wanted
        ? `Require an API key on ${entry.display_name} — ${KINDS[kind].label}?\n\n` +
          "A key is generated (if none exists) and written into the node's config " +
          "template. Nothing reaches the node until you Apply."
        : `Stop requiring an API key on ${entry.display_name} — ${KINDS[kind].label}?\n\n` +
          opens + " Nothing reaches the node until you Apply.";
    if (sharers.length) {
        msg += "\n\nThis edits the config TEMPLATE FILE, which is shared with:\n  " +
               sharers.join(", ") + "\nThose nodes change too.";
    }
    msg += "\n\nContinue?";

    if (!window.confirm(msg)) {
        input.checked = !wanted;
        return;
    }

    const resp = await fetchJSON(`/api/access/${encodeURIComponent(entry.node_id)}/auth`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: wanted, kind }),
    });
    if (!isCurrent(mountToken)) return;

    if (resp._error === 400 && Array.isArray(resp.errors)) {
        window.alert(
            "The edited template failed validation, so nothing was written:\n\n" +
            resp.errors.map(x => `${x.loc}: ${x.msg}`).join("\n")
        );
        input.checked = !wanted;
        return;
    }
    if (isError(resp)) {
        window.alert(`Could not change auth: ${errorText(resp)}`);
        input.checked = !wanted;
        return;
    }

    const also = (resp.also_affects || []).length
        ? ` Also changed: ${resp.also_affects.join(", ")}.`
        : "";
    setNote(entry.node_id, `${resp.note || "Staged."}${also}`);

    await loadNodes();
    if (!isCurrent(mountToken)) return;
    render();
}

async function onCardClick(e) {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;

    if (action === "copy-field") {
        await copyFieldOf(btn);
        return;
    }

    const entry = entryById(btn.dataset.node);
    if (!entry) return;
    // Read off the button, never inferred from the DOM position: an inference
    // that goes wrong reveals or rotates the OTHER service's credential.
    const kind = btn.dataset.kind || "llama";

    if (action === "copy-snippet") await copySnippet(btn, entry);
    else if (action === "reveal-key") await revealKey(entry, kind);
    else if (action === "hide-key") hideKey(entry, kind);
    else if (action === "copy-key") await copyKey(btn, entry, kind);
    else if (action === "copy-previous") await copyPreviousKey(btn, entry, kind);
    else if (action === "rotate") await rotateKey(entry, kind);
    else if (action === "apply") await applyOne(entry);
}

async function onBarClick(e) {
    const btn = e.target.closest("button[data-action]");
    if (!btn || btn.dataset.action !== "apply-all") return;
    await applyAllPending();
}

// ---------------------------------------------------------------------------
// Key actions
// ---------------------------------------------------------------------------

// The kind decides the route segment, so /key and /mcp-key are named in
// exactly one place.
function keyUrl(nodeId, kind, previous = false) {
    const base = `/api/access/${encodeURIComponent(nodeId)}/${KINDS[kind].path}`;
    return previous ? `${base}/previous` : base;
}

function revealedKey(nodeId, kind) {
    return (state.revealed[nodeId] || {})[kind] || null;
}

async function revealKey(entry, kind) {
    const data = await fetchJSON(keyUrl(entry.node_id, kind));
    if (!isCurrent(mountToken)) return;
    if (isError(data) || !data.api_key) {
        setNote(entry.node_id, `Could not reveal the ${KINDS[kind].label} key: ${errorText(data)}`);
        render();
        return;
    }
    // Per kind, not per node: revealing one service's key must not put the
    // other service's key on screen under the wrong label.
    state.revealed[entry.node_id] = { ...state.revealed[entry.node_id], [kind]: data.api_key };
    setNote(entry.node_id, "");
    render();
}

function hideKey(entry, kind) {
    const held = { ...state.revealed[entry.node_id] };
    delete held[kind];
    state.revealed[entry.node_id] = held;
    render();
}

async function copyKey(btn, entry, kind) {
    const input = btn.closest(".key-field")?.querySelector(".copy-field");
    const cached = revealedKey(entry.node_id, kind);
    if (cached) {
        await runCopy(btn, cached, input);
        return;
    }
    // Not revealed: fetch the plaintext for the clipboard without putting it
    // on screen.
    const data = await fetchJSON(keyUrl(entry.node_id, kind));
    if (!isCurrent(mountToken)) return;
    if (isError(data) || !data.api_key) {
        setNote(entry.node_id, `Could not copy the ${KINDS[kind].label} key: ${errorText(data)}`);
        render();
        return;
    }
    const outcome = await runCopy(btn, data.api_key, null);
    if (!isCurrent(mountToken)) return;
    if (outcome !== "copied") {
        setNote(entry.node_id, "Clipboard unavailable — press Reveal, then select the key by hand.");
        render();
    }
}

async function copyPreviousKey(btn, entry, kind) {
    const data = await fetchJSON(keyUrl(entry.node_id, kind, true));
    if (!isCurrent(mountToken)) return;
    if (isError(data) || !data.api_key) {
        setNote(entry.node_id, `Could not read the previous key: ${errorText(data)}`);
        render();
        return;
    }
    const outcome = await runCopy(btn, data.api_key, null);
    if (!isCurrent(mountToken)) return;
    if (outcome !== "copied") {
        setNote(entry.node_id, "Clipboard unavailable — the previous key could not be copied.");
        render();
    }
}

async function rotateKey(entry, kind) {
    const auth = authFor(entry, kind);
    const label = KINDS[kind].label;
    const msg = auth.has_key
        ? `Rotate the ${label} API key for ${entry.display_name}?\n\n` +
          "A new key is generated and staged on the orchestrator now. It takes " +
          "effect on the NEXT Apply — until then the node keeps enforcing the OLD " +
          "key, which stays available here as \"Copy previous key\".\n\nContinue?"
        : `Generate a ${label} API key for ${entry.display_name}?\n\n` +
          "The key is staged on the orchestrator and takes effect on the next " +
          "Apply.\n\nContinue?";
    if (!window.confirm(msg)) return;

    const resp = await fetchJSON(`/api/access/${encodeURIComponent(entry.node_id)}/auth/rotate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind }),
    });
    if (!isCurrent(mountToken)) return;
    if (isError(resp)) {
        window.alert(`Rotate failed: ${errorText(resp)}`);
        return;
    }
    // The revealed plaintext is now the previous key — drop it rather than
    // showing a stale value next to a fresh fingerprint. Only this kind's:
    // the other service's key was not touched.
    hideKey(entry, kind);
    setNote(entry.node_id, `${label} key ${resp.action || "rotated"} (staged) — Apply to activate.`);

    await loadNodes();
    if (!isCurrent(mountToken)) return;
    render();
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

// The push set is WIDER than the node named on the button, in two ways the
// operator has to consent to before the run starts: the server pushes to the
// whole group a node belongs to, and it also pushes to every node whose own
// config embeds this node's MCP key (a planner's ${mcp_endpoints} list carries
// each peer's key), restarting those too when that key moved. Naming only the
// node here would have an operator confirm "edge-01 only" and then find the
// planner reloaded.
function applyConfirmText(entry) {
    const members = entry.group_members || [];
    const scope = members.length
        ? `(whole group: ${members.join(", ")})`
        : `(${entry.node_id})`;
    return (
        `Apply settings to ${entry.display_name}?\n\n` +
        "1. regenerate configs on the Mac\n" +
        `2. push over SSH to /etc/aureax/aureax-edge.yaml ${scope}, plus any node ` +
        "whose config embeds this node's MCP key (a planner)\n" +
        `3. RESTART the llama-server pod on ${entry.node_id} — and on those ` +
        "key-carrying nodes, if their key changed\n\n" +
        "The model reloads from scratch — minutes for a sharded head; the " +
        "endpoint errors for that whole window. Workers keep running (restarting " +
        "them would force a cold reload). Harnesses using an OLD key get 401s " +
        "once the new server is up.\n\n" +
        "Continue?"
    );
}

async function applyOne(entry) {
    if (!window.confirm(applyConfirmText(entry))) return;
    const resp = await fetchJSON(`/api/access/${encodeURIComponent(entry.node_id)}/apply`, {
        method: "POST",
    });
    if (!isCurrent(mountToken)) return;
    await handleApplyResponse(resp, [entry.node_id]);
}

async function applyAllPending() {
    const pending = pendingEntries();
    if (!pending.length) return;

    const groups = [...new Set(pending.map(e => e.group).filter(Boolean))];
    const msg =
        `Apply to ${pending.length} node(s)?\n\n` +
        pending.map(e => `  ${e.display_name} (${e.node_id})`).join("\n") + "\n\n" +
        (groups.length ? `Groups touched: ${groups.join(", ")}\n\n` : "") +
        "Each node is applied SEQUENTIALLY: configs are regenerated, pushed over " +
        "SSH, and the llama-server pod is restarted. Every one of them reloads its " +
        "model from scratch — minutes for a sharded head — and its endpoint errors " +
        "for that whole window. Harnesses using an OLD key get 401s once the new " +
        "server is up.\n\nContinue?";
    if (!window.confirm(msg)) return;

    const nodeIds = pending.map(e => e.node_id);
    const resp = await fetchJSON("/api/access/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_ids: nodeIds }),
    });
    if (!isCurrent(mountToken)) return;
    await handleApplyResponse(resp, nodeIds);
}

async function handleApplyResponse(resp, nodeIds) {
    if (resp._error === 409) {
        // Inline per-node notes rather than the window.alert() lifecycle.js uses
        // for its 409: one 409 covers every node in the batch, so a note per
        // card beats a single alert, and each row otherwise looks like its
        // button did nothing.
        const detail = resp.detail || "another operation is already running";
        state.busyNodes = nodeIds.slice();
        for (const id of nodeIds) setNote(id, `Busy — ${detail}. Wait for it to finish.`);
        // `running` comes with the apply-vs-apply 409 only; the lifecycle-busy
        // 409 (access_api._start_apply) carries no snapshot, so there is no
        // apply run whose completion would clear these notes. Watch the
        // lifecycle state instead. Either way something must clear them, or
        // they stick until the tab is remounted.
        if (resp.running) startApplyPolling();
        else { startLifecycleWait(); render(); }
        return;
    }
    // A transport failure means the answer was lost, not that nothing started:
    // _start_apply detects the orchestrator IP and builds the push set — shell
    // calls with no timeout of their own — BEFORE it latches runner.running and
    // answers 202, and the server does not abandon the run when the browser
    // gives up at FETCH_TIMEOUT_MS. Ask the server before declaring a failure.
    if (resp._error === "timeout" || resp._error === "network") {
        await pollApply();
        if (!isCurrent(mountToken)) return;
        if (state.applyRunning) {
            for (const id of nodeIds) {
                setNote(id, `No answer to the Apply request (${errorText(resp)}), but a run is in progress — tracking it.`);
            }
            render();
            return;
        }
    }
    if (isError(resp)) {
        for (const id of nodeIds) setNote(id, `Apply failed to start: ${errorText(resp)}`);
        render();
        return;
    }
    state.applyOutcomes = {};
    state.applyError = null;
    state.busyNodes = [];
    for (const id of nodeIds) setNote(id, "Apply started…");
    startApplyPolling();
}

function startApplyPolling() {
    // Cleared on both edges of a tracked run: the banner must only ever report
    // a poll that failed inside the run on screen, never one left over from the
    // mount-time probe, which polls with nothing in flight.
    state.applyPollError = null;
    state.applyRunning = true;
    render();
    if (applyTimer) return;
    applyTimer = setInterval(pollApply, APPLY_POLL_MS);
}

function stopApplyPolling() {
    if (applyTimer) { clearInterval(applyTimer); applyTimer = null; }
    state.applyRunning = false;
    state.applyPollError = null;
}

// ---------------------------------------------------------------------------
// Waiting out a lifecycle run
// ---------------------------------------------------------------------------

// Only reached from the lifecycle-busy 409 above. Clearing the "Busy" notes on
// the 20s live tick instead would wipe "wait for it to finish" while the
// lifecycle run is still going, so wait for the state machine to leave
// starting/stopping and clear then.
function startLifecycleWait() {
    if (lifecycleTimer) return;
    lifecycleTimer = setInterval(pollLifecycleBusy, LIFECYCLE_POLL_MS);
}

function stopLifecycleWait() {
    if (lifecycleTimer) { clearInterval(lifecycleTimer); lifecycleTimer = null; }
}

async function pollLifecycleBusy() {
    // One guard covers every other path that empties busyNodes (a later apply,
    // a completed run), so none of them has to remember to stop this timer.
    if (!state.busyNodes.length) { stopLifecycleWait(); return; }
    const snap = await fetchJSON("/api/lifecycle/state");
    if (!isCurrent(mountToken)) return;
    // Keep the interval and the notes on a failed poll: the next tick recovers,
    // and an unreadable state is not evidence the run has finished.
    if (isError(snap)) return;
    if (snap.state === "starting" || snap.state === "stopping") return;
    stopLifecycleWait();
    for (const id of state.busyNodes) setNote(id, "");
    state.busyNodes = [];
    render();
}

async function pollApply() {
    const snap = await fetchJSON("/api/access/apply/state");
    if (!isCurrent(mountToken)) return;
    if (isError(snap)) {
        // Keep the interval so the next tick recovers. Stopping on one transient
        // failure would clear applyRunning mid-apply — the "Apply running" note
        // gone, every Apply button live again, and nothing left to poll for the
        // completion that clears the per-node notes.
        state.applyPollError = errorText(snap);
        render();
        return;
    }
    state.applyPollError = null;

    // ApplyRunner keeps its `nodes` dict after a run ends, for the life of the
    // dashboard process, so a snapshot alone says nothing about when its
    // outcomes were produced. Adopt them only for a run this tab is tracking,
    // or the mount-time probe prints "Apply: pushed + restarted" on every card
    // from a run that finished hours ago. The final completion poll still
    // qualifies, since applyRunning is cleared below this.
    if (snap.running || state.applyRunning) {
        state.applyOutcomes = snap.nodes || {};
        state.applyError = snap.error || null;
    }

    if (snap.running) {
        if (!applyTimer) startApplyPolling();
        else {
            state.applyRunning = true;
            render();
        }
        return;
    }

    const wasRunning = state.applyRunning;
    stopApplyPolling();
    if (!wasRunning) {
        // Nothing was in flight (the mount-time probe): no outcomes were
        // adopted above, so the cards keep whatever they already show.
        render();
        return;
    }

    // Both note sets: the nodes this run reported on, and any this tab was told
    // were "Busy" by a 409. A busy node appears in no other run's outcomes, so
    // clearing only the outcomes leaves its note stuck until remount. Nothing
    // else is wiped: state.notes also holds staged "Key rotated" messages.
    for (const id of [...Object.keys(state.applyOutcomes), ...state.busyNodes]) setNote(id, "");
    state.busyNodes = [];
    stopLifecycleWait();
    if (state.applyError) {
        window.alert(`Apply failed: ${state.applyError}`);
    }

    await loadNodes();
    if (!isCurrent(mountToken)) return;
    await loadLive();
    if (!isCurrent(mountToken)) return;
    render();
}
