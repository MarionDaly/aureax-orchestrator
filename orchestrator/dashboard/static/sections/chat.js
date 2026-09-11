// Chat section — session-based agentic conversation.
//
// Hash: #chat                                — last-used target, latest session
//       #chat/<target>                       — this target, latest session
//       #chat/<target>/<session_id>          — specific session
//
// State is driven by the SSE stream: the server is the source of truth,
// so opening the same session in two tabs shows live activity in both.

import { errorText, escapeHtml, fetchJSON, isError, readLocal, stickyScroll, writeLocal } from "/utils.js";
import { isCurrent } from "/main.js";

const NEW_SESSION_VALUE = "__new__";

let state = {
    targets: [],
    targetId: null,
    sessions: [],
    session: null,
    messages: [],         // from stream (dedupe by id)
    pending: false,       // true while a run is active
    stream: null,         // current EventSource
    replaying: false,     // buffering the stream's history replay
    loadError: null,      // why the dropdowns are empty, if they are
};

export async function mount(root, arg, token) {
    root.innerHTML = `
        <section>
            <h2>Chat</h2>
            <div class="card chat-panel">
                <div class="chat-header">
                    <label>Target:
                        <select id="chat-target" class="select-control"></select>
                    </label>
                    <label>Session:
                        <select id="chat-session" class="select-control"></select>
                    </label>
                    <span id="chat-status" class="chat-status"></span>
                    <span class="spacer"></span>
                    <button class="btn" id="chat-reset" title="Clear history in this session">Reset</button>
                    <button class="btn btn-ghost" id="chat-delete" title="Delete this session">Delete</button>
                </div>
                <div class="chat-messages" id="chat-messages"></div>
                <form class="chat-input" id="chat-form">
                    <textarea id="chat-prompt" rows="2" placeholder="Type a message — Shift+Enter for newline, Enter to send"></textarea>
                    <button class="btn btn-primary" id="chat-send" type="submit">Send</button>
                </form>
            </div>
        </section>
    `;

    document.getElementById("chat-form").addEventListener("submit", onSubmit);
    document.getElementById("chat-prompt").addEventListener("keydown", onKeyDown);
    document.getElementById("chat-target").addEventListener("change", onTargetChange);
    document.getElementById("chat-session").addEventListener("change", onSessionChange);
    document.getElementById("chat-reset").addEventListener("click", onReset);
    document.getElementById("chat-delete").addEventListener("click", onDelete);

    await loadTargets();
    // Each await is a chance for the user to switch tabs. Continuing past one
    // would open an EventSource on an unmounted section that nothing can ever
    // close — unmount() has already run and the next mount starts fresh.
    if (!isCurrent(token)) return;

    // Parse hash: arg may be "<target>" or "<target>/<session_id>"
    const [wantTarget, wantSession] = (arg || "").split("/").filter(Boolean);
    // Guarded (readLocal): a browser configured to block site data throws on
    // localStorage itself, and a throw here aborts mount() before the selects
    // are built or a session is opened — main.js only console.errors it, so the
    // Chat tab would sit permanently blank.
    const remembered = readLocal("aureax.chat.target");
    state.targetId = pickTarget(wantTarget || remembered);
    renderTargetSelect();

    await loadSessions();
    if (!isCurrent(token)) return;
    if (wantSession && state.sessions.find(s => s.id === wantSession)) {
        await selectSession(wantSession);
    } else {
        // If there's already a session for this target, open it; otherwise
        // leave things empty — sending a prompt will create one on demand.
        await selectSession(state.sessions[0]?.id || null);
    }
}

export function unmount() {
    closeStream();
    state = {
        targets: [], targetId: null, sessions: [], session: null,
        messages: [], pending: false, stream: null,
        replaying: false, loadError: null,
    };
}

// ---- Data loaders ----------------------------------------------------------

// fetchJSON never throws, so a failed load arrives as a non-array: check
// isError() or it silently becomes [] — two empty dropdowns, a null target, and
// no reason on screen. The reason lives in state because the status span is
// cleared on every session switch, and idleStatus() re-states it from there
// instead of blanking the line.
async function loadTargets() {
    const data = await fetchJSON("/api/chat/targets");
    if (isError(data)) {
        state.loadError = `Chat unavailable: ${errorText(data)}`;
        state.targets = [];
        return;
    }
    state.loadError = null;
    state.targets = Array.isArray(data) ? data : [];
}

async function loadSessions() {
    // No target means the request would be `target_id=null`, which lists every
    // session of every target rather than none.
    if (!state.targetId) {
        state.sessions = [];
        renderSessionSelect();
        return;
    }
    const data = await fetchJSON(`/api/chat/sessions?target_id=${encodeURIComponent(state.targetId)}`);
    if (isError(data)) {
        state.loadError = `Sessions unavailable: ${errorText(data)}`;
        state.sessions = [];
        renderSessionSelect();
        return;
    }
    state.loadError = null;
    state.sessions = Array.isArray(data) ? data : [];
    renderSessionSelect();
}

// An empty fleet is not an error: chat_service excludes shard workers, so a
// fleet of workers legitimately offers no target.
function idleStatus() {
    if (state.loadError) return state.loadError;
    if (!state.targets.length) return "No chat targets configured.";
    return "";
}

function pickTarget(wanted) {
    if (wanted && state.targets.find(t => t.id === wanted)) return wanted;
    return state.targets[0]?.id || null;
}

// ---- Select handlers -------------------------------------------------------

async function onTargetChange(e) {
    state.targetId = e.target.value;
    // Guarded for the same reason as the read, and it matters more here: a
    // throw would skip closeStream() below and leave the previous target's
    // EventSource open while the dropdown shows the new one.
    writeLocal("aureax.chat.target", state.targetId);
    closeStream();
    state.session = null;
    state.messages = [];
    state.pending = false;
    renderMessages();
    await loadSessions();
    await selectSession(state.sessions[0]?.id || null);
}

async function onSessionChange(e) {
    const value = e.target.value;
    if (value === NEW_SESSION_VALUE) {
        await createNewSession();
    } else {
        await selectSession(value);
    }
}

async function createNewSession() {
    const resp = await fetchJSON("/api/chat/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_id: state.targetId }),
    });
    if (isError(resp)) {
        setStatus(`Create failed: ${errorText(resp)}`);
        // Put the dropdown back on the session that is actually open —
        // otherwise it reads "+ New session" while Reset/Delete still act on
        // the old one.
        renderSessionSelect();
        return;
    }
    await loadSessions();
    await selectSession(resp.id);
}

async function selectSession(sessionId) {
    closeStream();
    state.session = sessionId ? state.sessions.find(s => s.id === sessionId) : null;
    state.messages = [];
    state.pending = false;
    renderSessionSelect();
    renderMessages();
    if (!sessionId) {
        updateUrl();
        setStatus(idleStatus());
        return;
    }
    updateUrl();
    openStream(sessionId);
}

function updateUrl() {
    const parts = ["#chat"];
    // A null target would render as the literal "#chat/null".
    if (state.targetId) parts.push(state.targetId);
    if (state.session) parts.push(state.session.id);
    const hash = parts.join("/");
    if (window.location.hash !== hash) {
        // replaceState so we don't fire hashchange + re-mount the section
        history.replaceState(null, "", hash);
    }
}

// ---- Streaming -------------------------------------------------------------

function openStream(sessionId) {
    state.stream = new EventSource(
        `/api/chat/sessions/${encodeURIComponent(sessionId)}/stream`,
    );
    state.stream.addEventListener("session", (e) => {
        // Start of replay. chat_service.stream() re-sends the whole snapshot on
        // EVERY connect, and EventSource reconnects by itself — after a sleep or
        // a dropped link — without openStream() running again, so the arming
        // belongs on the frame rather than in the enclosing function. `session`
        // is the frame the server emits exactly once up front and never live,
        // which makes it the one reliable marker. Dropping the transcript here
        // too means the rebuild is the server's history rather than a merge with
        // ours, so messages another tab deleted while we were away do not
        // survive in this one.
        state.messages = [];
        state.replaying = true;
        const sess = safeParse(e.data);
        if (sess?.session) state.session = sess.session;
        renderSessionSelect();
    });
    state.stream.addEventListener("message", (e) => {
        const payload = safeParse(e.data);
        if (payload?.message) applyMessage(payload.message);
    });
    state.stream.addEventListener("status", (e) => {
        const payload = safeParse(e.data);
        if (!payload) return;
        // chat_service.stream() emits the whole history as `message` frames and
        // then exactly one `status` frame — that frame is the end-of-replay
        // marker. Draw the buffered transcript in a single pass here: patching
        // each replayed message costs one forced layout per message, up to
        // MAX_MESSAGES of them on a reconnect.
        if (state.replaying) {
            state.replaying = false;
            renderMessages();
        }
        state.pending = payload.status === "active";
        setStatus(state.pending ? "Thinking…" : "");
        if (state.session) state.session.status = payload.status;
        renderSessionSelect();
        setSending(state.pending);
    });
    state.stream.addEventListener("reset", () => {
        state.messages = [];
        renderMessages();
    });
    state.stream.addEventListener("session_updated", (e) => {
        const payload = safeParse(e.data);
        if (payload?.title && state.session) {
            state.session.title = payload.title;
            loadSessions().catch(e => console.error("session refresh failed", e));
        }
    });
    // tool_call / tool_result / final come from chat_service's event passthrough.
    // The messages themselves arrive separately as `message` frames, so these
    // three drive transient UI only: what the agent is doing right now.
    state.stream.addEventListener("tool_call", (e) => {
        const p = safeParse(e.data);
        if (p) setStatus(`Calling ${p.node_id || "?"}.${p.tool || "?"}…`);
    });
    state.stream.addEventListener("tool_result", (e) => {
        const p = safeParse(e.data);
        if (p) setStatus("Thinking…");
    });
    state.stream.addEventListener("final", () => {
        setStatus("");
    });
    // Both a server `event: error` frame (a MessageEvent, carrying .data) and a
    // transport failure (a bare Event, no .data) dispatch to this same listener,
    // so `e.data` is the discriminator. The no-data case is ignored here because
    // EventSource reconnects on its own; onerror below reports the one case it
    // does not, a stream closed for good.
    state.stream.addEventListener("error", (e) => {
        if (e.data) {
            const payload = safeParse(e.data);
            appendError(payload?.message || "unknown error");
        }
    });
    state.stream.onerror = () => {
        const es = state.stream;
        // onerror is the same hook on the same event type, so a server `event:
        // error` frame lands here too, with the connection still OPEN — the
        // listener above already handled it.
        if (!es || es.readyState === EventSource.OPEN) return;
        if (es.readyState === EventSource.CONNECTING) {
            setStatus("Reconnecting…");  // EventSource retries by itself
            return;
        }
        // CLOSED is permanent: a non-2xx response (the 404 for a session another
        // tab deleted) stops the stream for good. The composer is re-enabled by
        // the stream's own `status` frame, which will now never arrive, so
        // release it here or the panel sits disabled on "Sending…" with no way
        // back short of a reload.
        setStatus("Stream closed — reload to reconnect");
        state.pending = false;
        setSending(false);
    };
}

function closeStream() {
    state.replaying = false;
    if (state.stream) {
        try { state.stream.close(); } catch {}
        state.stream = null;
    }
}

// Cap on retained messages. The stream replays a session's full history on
// connect and then appends indefinitely, so without a cap an afternoon's
// conversation holds every message plus every tool result in memory, and every
// full render walks all of them.
const MAX_MESSAGES = 500;

// Drop the oldest entries past the cap, and their DOM nodes with them. Every
// writer to state.messages goes through this, errors included — a stream that
// keeps failing appends as surely as one that succeeds.
function trimMessages() {
    if (state.messages.length <= MAX_MESSAGES) return;
    const dropped = state.messages.splice(0, state.messages.length - MAX_MESSAGES);
    const el = document.getElementById("chat-messages");
    for (const d of dropped) el?.querySelector(`[data-msg-id="${CSS.escape(d.id)}"]`)?.remove();
}

function applyMessage(msg) {
    const idx = state.messages.findIndex(m => m.id === msg.id);
    if (idx >= 0) state.messages[idx] = msg;
    else state.messages.push(msg);

    trimMessages();

    // History replay: collected here, drawn in one pass by the `status` frame
    // that closes the replay block.
    if (state.replaying) return;

    // Patch just this message's node rather than rebuilding the transcript.
    // A full innerHTML rewrite per streamed frame is O(n^2) over a session AND
    // discards per-node UI state: every <details class="chat-tool"> the operator
    // has expanded snaps shut on the next token, and any text selection goes
    // with it.
    upsertMessageNode(msg);
}

function appendError(text) {
    const msg = {
        id: `err-${Date.now()}-${Math.random()}`,
        role: "error",
        content: text,
    };
    state.messages.push(msg);
    trimMessages();
    upsertMessageNode(msg);
}

function toolMapFor() {
    const toolMap = new Map();
    for (const m of state.messages) {
        if (m.role === "tool" && m.tool_call_id) toolMap.set(m.tool_call_id, m);
    }
    return toolMap;
}

function upsertMessageNode(msg) {
    const el = document.getElementById("chat-messages");
    if (!el) return;

    // A tool result is not rendered on its own — it is folded into the
    // assistant message that requested it, so that node is what must refresh.
    if (msg.role === "tool" || msg.role === "system") {
        if (msg.role === "tool" && msg.tool_call_id) refreshAssistantFor(msg);
        return;
    }

    el.querySelector(".empty-state")?.remove();

    const restoreScroll = stickyScroll(el);
    const html = renderMessage(msg, toolMapFor());
    const existing = el.querySelector(`[data-msg-id="${CSS.escape(msg.id)}"]`);
    if (existing) {
        existing.outerHTML = html;
    } else {
        el.insertAdjacentHTML("beforeend", html);
    }
    restoreScroll();
}

function refreshAssistantFor(toolMsg) {
    const toolCallId = toolMsg.tool_call_id;
    const owner = state.messages.find(
        m => m.role === "assistant" && (m.tool_calls || []).some(tc => tc.id === toolCallId),
    );
    if (!owner) return;

    // Patch only this call's <details>. Re-rendering the whole assistant message
    // re-serialises every OTHER call's args and result — MCP results are
    // arbitrarily large JSON, and a turn with k calls would pay that k times —
    // and reparsing the node snaps shut every block the operator has expanded.
    const el = document.getElementById("chat-messages");
    const block = el?.querySelector(
        `[data-msg-id="${CSS.escape(owner.id)}"] details[data-tool-call-id="${CSS.escape(toolCallId)}"]`,
    );
    if (!block) {
        upsertMessageNode(owner);
        return;
    }
    const tc = (owner.tool_calls || []).find(t => t.id === toolCallId);
    const restoreScroll = stickyScroll(el);
    // toolDetailsHtml emits `open` only while a call is pending, so a completed
    // block renders collapsed.
    block.outerHTML = toolDetailsHtml(tc, toolMsg);
    restoreScroll();
}

// ---- Ask + reset + delete --------------------------------------------------

async function onSubmit(e) {
    e.preventDefault();
    if (state.pending) return;
    const input = document.getElementById("chat-prompt");
    const prompt = input.value.trim();
    if (!prompt) return;

    // The re-entry guard above only holds if this is set before the first await
    // below — the send is otherwise re-enterable for the whole session-creation
    // round trip.
    state.pending = true;

    // No active session? Create one on demand so a fresh prompt just works.
    if (!state.session) {
        if (!state.targetId) { state.pending = false; return; }
        await createNewSession();
        if (!state.session) { state.pending = false; return; }  // error already flashed
    }

    state.pending = true;  // selectSession() cleared it while creating
    setSending(true);
    setStatus("Sending…");

    const resp = await fetchJSON(`/api/chat/sessions/${encodeURIComponent(state.session.id)}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
    });
    if (isError(resp)) {
        // The textarea is cleared only on success, below. Clearing it before the
        // request would destroy whatever was typed whenever the send fails, and
        // nothing else holds a copy.
        appendError(`Ask failed: ${errorText(resp)}`);
        state.pending = false;
        setSending(false);
        setStatus("");
        return;
    }
    input.value = "";
    // Live updates arrive via the stream — including the `status` frame that
    // releases the composer. If the stream is gone (closed for good after a
    // 404), nothing will ever send that frame, so release it here instead of
    // leaving a dead input on "Sending…".
    if (!state.stream || state.stream.readyState === EventSource.CLOSED) {
        state.pending = false;
        setSending(false);
        setStatus("Stream disconnected — reload to see the reply.");
    }
}

function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        onSubmit(e).catch(err => console.error("send failed", err));
    }
}

async function onReset() {
    if (!state.session) return;
    if (!window.confirm("Clear all messages in this session?")) return;
    // A reset that 404s (the session is gone) emits no 'reset' frame, so nothing
    // else would tell the operator the button did not work. appendError, not
    // setStatus: the status span is overwritten by the stream's own frames
    // within a second.
    const resp = await fetchJSON(`/api/chat/sessions/${encodeURIComponent(state.session.id)}/reset`, { method: "POST" });
    if (isError(resp)) {
        appendError(`Reset failed: ${errorText(resp)}`);
        return;
    }
    // On success the stream's 'reset' event clears state.messages client-side.
}

async function onDelete() {
    if (!state.session) return;
    if (!window.confirm("Delete this session permanently?")) return;
    const id = state.session.id;
    closeStream();
    const resp = await fetchJSON(`/api/chat/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (isError(resp)) {
        // alert(), as elsewhere for a failed destructive action: the recovery
        // below reopens the stream, whose first `status` frame overwrites the
        // status span, and whose replay would drop an appended error node.
        window.alert(`Delete failed: ${errorText(resp)}`);
        // The row is still there; reopen the stream closeStream() just dropped
        // rather than switching away as though it were gone.
        await selectSession(id);
        return;
    }
    await loadSessions();
    await selectSession(state.sessions[0]?.id || null);
}

// ---- Rendering -------------------------------------------------------------

function renderTargetSelect() {
    const el = document.getElementById("chat-target");
    if (!el) return;
    el.innerHTML = state.targets.map(t => `
        <option value="${escapeHtml(t.id)}" ${t.id === state.targetId ? "selected" : ""}>
            ${escapeHtml(t.label)}
        </option>
    `).join("");
}

function renderSessionSelect() {
    const el = document.getElementById("chat-session");
    if (!el) return;
    const items = state.sessions.map(s => {
        const dot = s.status === "active" ? "●" : "";
        const title = s.title || "Untitled";
        const label = dot ? `${dot} ${title}` : title;
        const selected = state.session && s.id === state.session.id ? "selected" : "";
        return `<option value="${escapeHtml(s.id)}" ${selected}>${escapeHtml(label)}</option>`;
    });
    el.innerHTML = `<option value="${NEW_SESSION_VALUE}">+ New session</option>${items.join("")}`;
}

function renderMessages() {
    const el = document.getElementById("chat-messages");
    if (!el) return;

    // Built before the filter below, so it still covers the tool messages the
    // filter removes.
    const toolMap = toolMapFor();

    const visible = state.messages.filter(m => m.role !== "system" && m.role !== "tool");
    if (!visible.length) {
        el.innerHTML = '<div class="empty-state">No messages yet. Say hello.</div>';
        return;
    }
    // Full rebuild — only for a session switch or reset. Streaming updates go
    // through upsertMessageNode, which patches a single node.
    const restoreScroll = stickyScroll(el);
    el.innerHTML = visible.map(m => renderMessage(m, toolMap)).join("");
    restoreScroll();
}

function renderMessage(m, toolMap) {
    // data-msg-id is the anchor upsertMessageNode patches against, so every
    // branch below must carry it.
    const anchor = `data-msg-id="${escapeHtml(m.id)}"`;
    if (m.role === "user") {
        return `<div class="chat-msg chat-user" ${anchor}><div class="chat-bubble">${escapeHtml(m.content || "")}</div></div>`;
    }
    if (m.role === "error") {
        return `<div class="chat-msg chat-assistant" ${anchor}><div class="pod-error">${escapeHtml(m.content)}</div></div>`;
    }
    // role === "assistant"
    const toolsHtml = (m.tool_calls || []).map(tc => toolDetailsHtml(tc, toolMap.get(tc.id))).join("");
    const content = (m.content || "").trim();
    const textHtml = content
        ? `<div class="chat-bubble">${escapeHtml(content)}</div>`
        : "";
    return `<div class="chat-msg chat-assistant" ${anchor}>${toolsHtml}${textHtml}</div>`;
}

// One tool call's block. Shared by renderMessage and refreshAssistantFor's
// targeted patch, so both spell the markup — and the pending-only `open` — once.
// data-tool-call-id is the anchor that patch selects on.
function toolDetailsHtml(tc, tool) {
    const fn = (tc || {}).function || {};
    const name = fn.name || "unknown";
    // indexOf, not split("__", 2), splitting once the way MCPPool.parse_tool_call
    // does: JS's second argument to split() caps the result length and discards
    // the remainder, and a tool name may itself contain "__".
    const i = name.indexOf("__");
    const [nodeId, toolName] = i < 0 ? ["", name] : [name.slice(0, i), name.slice(i + 2)];
    let args = fn.arguments || "{}";
    try { args = JSON.stringify(JSON.parse(args), null, 2); } catch {}
    const result = tool
        ? `<pre>${escapeHtml(pretty(tool.content))}</pre>`
        : '<span class="chat-tool-pending">…</span>';
    return `
        <details class="chat-tool" data-tool-call-id="${escapeHtml((tc || {}).id)}" ${tool ? "" : "open"}>
            <summary>
                <span class="chat-tool-label">${escapeHtml(nodeId || "local")} · ${escapeHtml(toolName)}</span>
            </summary>
            <div class="chat-tool-body">
                <div class="chat-tool-section"><span>args</span><pre>${escapeHtml(args)}</pre></div>
                <div class="chat-tool-section"><span>result</span>${result}</div>
            </div>
        </details>
    `;
}

function pretty(v) {
    if (v == null) return "";
    if (typeof v === "string") {
        try { return JSON.stringify(JSON.parse(v), null, 2); } catch { return v; }
    }
    try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}

function setStatus(text) {
    const el = document.getElementById("chat-status");
    if (el) el.textContent = text;
}

function setSending(sending) {
    const btn = document.getElementById("chat-send");
    const input = document.getElementById("chat-prompt");
    if (btn) btn.disabled = sending;
    if (input) input.disabled = sending;
}

function safeParse(s) {
    try { return JSON.parse(s); } catch { return null; }
}
