// Configs section — sidebar of YAML files + editor pane with validation on save.

import { errorText, escapeHtml, fetchJSON, isError, readLocal, writeLocal } from "/utils.js";
import { isCurrent } from "/main.js";

const GROUPS = [
    { kind: "network",       label: "Network" },
    { kind: "edge-template", label: "Edge (template)" },
    { kind: "edge-node",     label: "Edge (per-node)" },
    { kind: "planner",       label: "Planner" },
    { kind: "generated",     label: "Generated (read-only)" },
];

let state = {
    files: [],
    selected: null,     // { name, kind, editable, original, content, dirty }
    templateStale: false,
    statusMessage: "",
    // "error" | "saved" | "" — what statusMessage is, so statusText can rank it
    // against the dirty flag instead of guessing from the text.
    statusKind: "",
    errors: [],
    // Why the sidebar is empty, when it is. A failed list is not "no configs":
    // that reading tells the operator the tree is empty when the request never
    // arrived.
    loadError: null,
};

// Registered once at module load, not in mount(): unmount() returns early to
// preserve a dirty buffer, so a listener removed there would go missing for the
// rest of the page. Nothing else in static/ guards a close or reload.
window.addEventListener("beforeunload", (e) => {
    if (state.selected?.dirty) e.preventDefault();
});

export async function mount(root, _arg, token) {
    root.innerHTML = `
        <section>
            <h2>Configuration</h2>
            <div id="config-banner"></div>
            <div class="configs-layout card">
                <aside id="config-sidebar" class="config-sidebar"></aside>
                <div id="config-editor" class="config-editor"></div>
            </div>
        </section>
    `;
    await loadFileList();
    // The section may have been swapped out during that fetch; root has been
    // emptied, so every getElementById below would be null.
    if (!isCurrent(token)) return;
    render();
    document.getElementById("config-sidebar").addEventListener("click", onSidebarClick);
    document.getElementById("config-editor").addEventListener("click", onEditorClick);
    document.getElementById("config-editor").addEventListener("input", onEditorInput);
    document.getElementById("config-editor").addEventListener("keydown", onEditorKeyDown);
    document.getElementById("config-banner").addEventListener("click", onBannerClick);
}

export function unmount() {
    // Keep an unsaved draft across a tab switch. main.js unmounts on every
    // hashchange, so bailing here is what stops a click on Overview mid-edit
    // from discarding the buffer with no prompt — a strictly larger loss than
    // the ones this section already confirms. mount() re-runs loadFileList()
    // and render(), which restores the editor from state.selected.
    if (state.selected?.dirty) return;
    state = {
        files: [], selected: null, templateStale: false,
        statusMessage: "", statusKind: "", errors: [], loadError: null,
    };
}

async function loadFileList() {
    // fetchJSON never throws, so a failed load arrives as a non-array: without
    // the isError() check a transport error or the 10s timeout collapses to []
    // and the sidebar reports "No configs found." — a claim about the repo,
    // made when nothing was read at all.
    const files = await fetchJSON("/api/config");
    if (isError(files)) {
        state.loadError = errorText(files);
        state.files = [];
        return;
    }
    state.loadError = null;
    state.files = Array.isArray(files) ? files : [];
}

function render() {
    renderSidebar();
    renderEditor();
    renderBanner();
}

function renderSidebar() {
    const el = document.getElementById("config-sidebar");
    if (!el) return;
    const byKind = {};
    for (const f of state.files) (byKind[f.kind] ||= []).push(f);

    const groupHtml = GROUPS.map(group => {
        const files = byKind[group.kind] || [];
        if (!files.length) return "";
        const items = files.map(f => {
            const active = state.selected && state.selected.name === f.name ? "active" : "";
            // The "network" group holds two entries: the operator's own fleet
            // file and the shipped example beside it. On a fresh clone only the
            // example exists, so it is badged rather than left looking like the
            // live fleet definition.
            const example = f.name.endsWith(".example.yaml")
                ? '<span class="badge">example</span>' : "";
            return `
                <li class="config-item ${active}" data-name="${escapeHtml(f.name)}">
                    <span class="config-name">${escapeHtml(f.name)}</span>
                    ${example}
                    ${f.editable ? "" : '<span class="badge">view</span>'}
                </li>
            `;
        }).join("");
        return `
            <div class="config-group">
                <div class="config-group-label">${escapeHtml(group.label)}</div>
                <ul class="config-list">${items}</ul>
            </div>
        `;
    }).join("");
    const empty = state.loadError
        ? `<div class="empty-state">Configs unavailable: ${escapeHtml(state.loadError)}</div>`
        : '<div class="empty-state">No configs found.</div>';
    el.innerHTML = groupHtml || empty;
}

// One expression for the status line, shared by the full render and the
// incremental update so the two cannot disagree about precedence.
//
// A failure outranks the dirty flag: a rejected save, or validation errors
// describing the buffer as it stands, is the only report the operator gets and
// "Unsaved changes" would swallow it. Everything else loses to dirty, because
// statusMessage is set on save and never cleared — a "Saved 14:02:11" stamp
// must not sit over edits typed since.
function statusText(sel) {
    if (!sel) return "";
    if (state.statusKind === "error" || state.errors?.length) {
        return state.statusMessage || "Unsaved changes";
    }
    return sel.dirty ? "Unsaved changes" : (state.statusMessage || "Clean");
}

// What the redaction marker in the buffer means, which differs by kind: an
// editable file round-trips (config_api.save_config puts the on-disk values
// back), while a view-only generated config simply never shows the key.
function redactionNote(sel) {
    return sel.editable
        ? "Secrets are shown as a marker. Leave a marked line as it is and the "
          + "on-disk value is restored on save; replace it to set a new one."
        : "Secrets are shown as a marker. This file is view-only — read the key "
          + "from the Access tab.";
}

function renderEditor() {
    const el = document.getElementById("config-editor");
    if (!el) return;
    if (!state.selected) {
        // The status line is rendered here too. A read that fails on the first
        // file click leaves state.selected null, and this branch is the only
        // thing the pane shows — without it the failure is reported nowhere.
        const status = state.statusMessage
            ? `<div class="editor-status">${escapeHtml(state.statusMessage)}</div>`
            : "";
        el.innerHTML =
            '<div class="empty-state">Select a config file to view or edit.</div>' + status;
        return;
    }
    const sel = state.selected;
    const readonly = sel.editable ? "" : "readonly";
    const saveDisabled = (!sel.editable || !sel.dirty) ? "disabled" : "";
    const reloadDisabled = sel.editable && sel.dirty ? "" : "disabled";
    const noteHtml = sel.redacted ? `
        <div class="editor-note">${escapeHtml(redactionNote(sel))}</div>
    ` : "";
    const errorsHtml = state.errors.map(e => `
        <div class="pod-error"><strong>${escapeHtml(e.loc)}</strong>: ${escapeHtml(e.msg)}</div>
    `).join("");
    el.innerHTML = `
        <div class="editor-header">
            <span class="editor-title">${escapeHtml(sel.name)}</span>
            <span class="editor-meta">${escapeHtml(sel.kind)}${sel.editable ? "" : " · view-only"}</span>
            <span class="spacer"></span>
            <button class="btn" data-action="reload" ${reloadDisabled}>Reload</button>
            <button class="btn btn-primary" data-action="save" ${saveDisabled}>Save</button>
        </div>
        ${noteHtml}
        <textarea class="editor-textarea" id="editor-textarea" spellcheck="false" ${readonly}>${escapeHtml(sel.content)}</textarea>
        <div class="editor-status">${escapeHtml(statusText(sel))}</div>
        ${errorsHtml}
    `;
}

function renderBanner() {
    const el = document.getElementById("config-banner");
    if (!el) return;
    if (state.templateStale) {
        el.innerHTML = `
            <div class="banner">
                <span>Template changed — per-node configs in <code>~/.aureax/configs/</code> are stale.</span>
                <button class="btn btn-primary" data-action="regenerate">Regenerate</button>
                <button class="btn btn-ghost" data-action="dismiss">Dismiss</button>
            </div>
        `;
    } else {
        el.innerHTML = "";
    }
}

// In-flight status text, written straight into the DOM rather than through
// render(): a full render rewrites the editor including the <textarea>, so
// announcing "Saving…" that way would destroy and recreate the element being
// edited — focus lost, caret back to offset 0, scroll reset on a
// several-hundred-line YAML and the native undo stack discarded. Deliberately
// not statusText(): mid-save the buffer is still dirty, and "Unsaved changes"
// is not the message to show. `kind` is what statusText ranks the message by.
function setStatus(text, kind = "") {
    state.statusMessage = text;
    state.statusKind = kind;
    const el = document.getElementById("config-editor");
    if (!el) return;
    const status = el.querySelector(".editor-status");
    if (status) status.textContent = text;
    el.querySelectorAll(".pod-error").forEach(node => node.remove());
}

async function onSidebarClick(e) {
    const item = e.target.closest(".config-item");
    if (!item) return;
    const name = item.dataset.name;
    if (!name || (state.selected && state.selected.name === name && !state.selected.dirty)) return;
    if (state.selected?.dirty && !window.confirm("Discard unsaved changes?")) return;
    await selectFile(name);
}

// Bumped per load so a slower earlier response cannot overwrite a later one.
// Each sidebar click starts its own load and the handler does a disk read per
// call, so two quick clicks race; without this, the loser can win and leave the
// operator editing the file clicked first, with the sidebar highlight moved
// back to it so nothing looks wrong.
let loadSeq = 0;

async function selectFile(name) {
    const seq = ++loadSeq;
    state.errors = [];
    setStatus("Loading…");
    const data = await fetchJSON(`/api/config/${encodeURIComponent(name)}`);
    if (seq !== loadSeq) return;
    if (isError(data)) {
        setStatus(`Failed to load: ${errorText(data)}`, "error");
        render();
        return;
    }
    state.selected = {
        name: data.name,
        kind: data.kind,
        editable: data.editable,
        original: data.content,
        content: data.content,
        dirty: false,
        // GET /api/config/{name} reports whether it replaced any secret with
        // the redaction marker; renderEditor turns that into the note above
        // the buffer.
        redacted: Boolean(data.redacted),
    };
    state.statusMessage = "";
    state.statusKind = "";
    render();
}

function onEditorInput(e) {
    if (e.target.id !== "editor-textarea") return;
    if (!state.selected) return;
    state.selected.content = e.target.value;
    const wasDirty = state.selected.dirty;
    state.selected.dirty = state.selected.content !== state.selected.original;
    if (wasDirty !== state.selected.dirty) renderEditorMeta();
}

function renderEditorMeta() {
    // Lightweight update — toggle disabled on save/reload + status text
    const sel = state.selected;
    if (!sel) return;
    const saveBtn = document.querySelector('#config-editor button[data-action="save"]');
    const reloadBtn = document.querySelector('#config-editor button[data-action="reload"]');
    if (saveBtn) saveBtn.disabled = !sel.editable || !sel.dirty;
    if (reloadBtn) reloadBtn.disabled = !(sel.editable && sel.dirty);
    const status = document.querySelector(".editor-status");
    if (status) status.textContent = statusText(sel);
}

function onEditorKeyDown(e) {
    if (e.target.id !== "editor-textarea") return;
    if (e.key === "Tab") {
        // A readonly <textarea> is still focusable and still fires keydown, and
        // assigning .value bypasses the readonly guard — so without this bail
        // Tab injects two spaces into a view-only generated config and marks it
        // dirty, on a file that cannot be saved and whose Reload stays disabled.
        // The guard sits inside this branch so Cmd+S below still preventDefaults
        // on a view-only file.
        if (e.target.readOnly) return;
        e.preventDefault();
        const ta = e.target;
        const start = ta.selectionStart;
        const end = ta.selectionEnd;
        ta.value = ta.value.slice(0, start) + "  " + ta.value.slice(end);
        ta.selectionStart = ta.selectionEnd = start + 2;
        ta.dispatchEvent(new Event("input", { bubbles: true }));
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        // toLowerCase: KeyboardEvent.key carries the PRODUCED character, so with
        // Caps Lock on or Shift held it is "S". Without the fold the comparison
        // misses, nothing is prevented, and the browser opens its own Save-Page
        // dialog instead.
        e.preventDefault();
        save();
    }
}

async function onEditorClick(e) {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    if (btn.dataset.action === "save") await save();
    else if (btn.dataset.action === "reload") await reload();
}

async function save() {
    const sel = state.selected;
    if (!sel || !sel.editable || !sel.dirty) return;
    state.errors = [];
    setStatus("Saving…");
    // Snapshot what is actually sent: the input listener keeps writing into
    // sel.content while the PUT is in flight, so only `sent` may be treated as
    // saved. Re-reading sel.content afterwards would mark characters that never
    // left the browser as clean, and the next tab switch drops them.
    const sent = sel.content;
    const resp = await fetchJSON(`/api/config/${encodeURIComponent(sel.name)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: sent }),
    });
    if (resp.ok) {
        sel.original = sent;
        sel.dirty = sel.content !== sent;
        state.statusMessage = `Saved ${new Date().toLocaleTimeString()}`;
        state.statusKind = "saved";
        state.errors = [];
        // Every saveable file is a source template: config_api.save_config 403s
        // the only non-source kind (generated), and each SSH target's
        // config_file is resolved per target at generate time. So any save
        // leaves ~/.aureax/configs/ stale.
        state.templateStale = true;
        await loadFileList();
    } else if (resp._error === 400 && Array.isArray(resp.errors)) {
        state.errors = resp.errors;
        state.statusMessage = `Validation failed (${resp.errors.length})`;
        state.statusKind = "error";
    } else {
        // Carries the 400 detail from _restore_redacted_passwords, which is the
        // only instruction the operator gets when the redacted password lines no
        // longer line up.
        state.statusMessage = `Save failed: ${errorText(resp)}`;
        state.statusKind = "error";
    }
    render();
}

async function reload() {
    const sel = state.selected;
    if (!sel) return;
    if (sel.dirty && !window.confirm("Discard unsaved changes?")) return;
    await selectFile(sel.name);
}

async function onBannerClick(e) {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    if (btn.dataset.action === "dismiss") {
        state.templateStale = false;
        renderBanner();
        return;
    }
    if (btn.dataset.action === "regenerate") {
        // The remembered IP is a convenience, not a requirement — readLocal
        // absorbs the throw a browser configured to block site data raises, so
        // the prompt still opens with no prefill.
        const remembered = readLocal("aureax.orchestrator_ip");
        // Trimmed BEFORE the guard: " " is truthy, so a whitespace-only prompt
        // would otherwise pass it, store "" over the remembered IP and POST an
        // empty orchestrator_ip that config_api.regenerate rejects — losing the
        // prefill for the next attempt on the way to a failure alert.
        const ip = (window.prompt("Orchestrator IP (for per-node configs):", remembered) || "").trim();
        if (!ip) return;
        // Not remembered for next time if the store is blocked; the regenerate
        // still runs.
        writeLocal("aureax.orchestrator_ip", ip);
        btn.disabled = true;
        const resp = await fetchJSON("/api/config/regenerate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ orchestrator_ip: ip }),
        });
        if (resp.generated) {
            state.templateStale = false;
            state.statusMessage = `Regenerated ${resp.generated.length} config(s)`;
            state.statusKind = "saved";
            await loadFileList();
            render();
        } else {
            window.alert(`Regenerate failed: ${errorText(resp)}`);
            btn.disabled = false;
        }
    }
}
