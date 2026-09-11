// Shared helpers used by all dashboard sections.

// Every request is bounded. Without a signal a stalled connection lets the
// interval-driven pollers stack pending requests until the browser's six-socket
// per-origin cap starves the rest of the dashboard — long enough here for the
// slowest route to answer, short enough that the pollers cannot stack that deep.
const FETCH_TIMEOUT_MS = 10000;

// Never throws. Callers must check `_error` before touching any other field —
// see isError() below; skipping that check turns a transport blip into a
// TypeError deep inside a render function.
export async function fetchJSON(url, options = {}) {
    const signal = options.signal ?? AbortSignal.timeout(FETCH_TIMEOUT_MS);
    try {
        const resp = await fetch(url, { ...options, signal });
        const isJson = (resp.headers.get("content-type") || "").includes("json");
        const body = isJson ? await resp.json().catch(() => ({})) : {};
        if (!resp.ok) {
            // `detail` is FastAPI's error field. Surfacing it as _message means
            // callers can render one thing instead of each picking a field, and
            // "Save failed: 403" becomes "Save failed: config is view-only".
            return { _error: resp.status, _message: body.detail || resp.statusText, ...body };
        }
        return body;
    } catch (e) {
        // Chrome's DOMException messages are terse, so name the cause here.
        if (e.name === "TimeoutError") {
            return { _error: "timeout", _message: `no response after ${FETCH_TIMEOUT_MS / 1000}s` };
        }
        if (e.name === "AbortError") {
            return { _error: "aborted", _message: "request cancelled" };
        }
        return { _error: "network", _message: e.message };
    }
}

// True for both failure shapes fetchJSON can return (HTTP status and network).
export function isError(data) {
    return !data || data._error !== undefined;
}

// Human-readable reason from a fetchJSON failure.
export function errorText(data) {
    if (!data) return "no response";
    return data._message || data.detail || `HTTP ${data._error}`;
}

// Escapes &, <, > AND quotes. The quotes matter: this is used in attribute
// position (data-node-dot="...", href="#chat/node-...", <option value="...">),
// where a value containing a double quote would otherwise close the attribute
// early. Node ids and config filenames are operator-supplied, so that is
// reachable from configuration alone.
export function escapeHtml(str) {
    return String(str ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

// Copy `text` to the clipboard, degrading twice. Returns "copied" | "select" |
// "failed" so the caller can label its button honestly — a copy button that
// silently does nothing is worse than one that says it failed.
//
// The dashboard now serves https://<lan-ip>:7001, which IS a potentially
// trustworthy origin, so `navigator.clipboard` is defined and the async
// Clipboard API is the path that normally runs. The deprecated execCommand
// fallback stays: the browser still refuses writeText when the document is not
// focused or the user denied the permission, and a page loaded over plain http
// (someone forcing the scheme, or a future non-TLS bind) has no Clipboard API
// at all — in both cases execCommand is what makes the button work rather than
// silently do nothing.
//
// `el` is the element to fall back to selecting (an <input>/<textarea>, or any
// node whose text contents we can range-select) when both copy paths fail.
export async function copyText(text, el) {
    const value = String(text ?? "");

    if (window.isSecureContext && navigator.clipboard && navigator.clipboard.writeText) {
        try {
            await navigator.clipboard.writeText(value);
            return "copied";
        } catch (e) {
            // Permission denied / not focused — fall through rather than
            // reporting a failure the next path can still fix.
        }
    }

    // Off-screen textarea + execCommand. Deprecated, still universally
    // implemented, and the only thing that works over plain http.
    const ta = document.createElement("textarea");
    try {
        ta.value = value;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.top = "-1000px";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        ta.setSelectionRange(0, value.length);
        const ok = document.execCommand("copy");
        if (ok) return "copied";
    } catch (e) {
        // fall through to the selection fallback
    } finally {
        // In a finally, and unconditional: `value` here is a plaintext
        // aureax_sk_ key for three Access-tab callers, so the textarea holding
        // it must never stay attached to the document — and execCommand can
        // throw (SecurityError in a restricted context) on the way out.
        // remove() is a no-op when the append never happened.
        ta.value = "";
        ta.remove();
    }

    // Last resort: leave something selected so the user can press Cmd/Ctrl+C.
    if (el) {
        try {
            if (typeof el.select === "function") {
                el.focus();
                el.select();
            } else {
                const range = document.createRange();
                range.selectNodeContents(el);
                const sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(range);
            }
            return "select";
        } catch (e) {
            // fall through
        }
    }
    return "failed";
}

export function timeAgo(isoString) {
    if (!isoString) return "never";
    const diff = Date.now() - new Date(isoString).getTime();
    // Nodes stamp their own metrics and are re-provisioned regularly, so a node
    // whose clock runs ahead of the orchestrator's yields a NEGATIVE age and an
    // unparseable stamp yields NaN. NaN passes neither comparison below, so it
    // is answered here rather than rendering as "NaNh ago"; the clamp on the
    // next line covers the negative case.
    if (Number.isNaN(diff)) return "unknown";
    const seconds = Math.max(0, Math.floor(diff / 1000));
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
}

// Two missed samples. The collector reports every 60s, so one late sample is
// normal jitter and two is a node that stopped talking.
//
// One rule, shared by the Overview grid and the node detail view: separate
// copies drift, and the fleet then reads online on one screen and offline on
// the other with nothing saying which is right.
export const FRESH_WINDOW_MS = 120000;

export function isFresh(iso) {
    if (!iso) return false;
    // An unparseable stamp gives NaN, which fails the comparison — the same
    // answer as "too old", and the honest one for a value we cannot read. A
    // node whose clock runs ahead gives a negative age and stays fresh.
    return Date.now() - new Date(iso).getTime() < FRESH_WINDOW_MS;
}

// The metrics both the Overview card and the node detail view lead with. The
// card shows these and counts the rest, so a new collector metric stops
// lengthening every card in the grid; the detail view sorts them to the top and
// follows with everything else.
export const HEADLINE_METRICS = [
    "cpu_percent",
    "memory_percent",
    "temperature_c",
    "load_1m",
    "tokens_per_second",
    "kv_cache_usage_ratio",
];

export function formatMetricName(name) {
    return name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

export function formatMetricValue(name, value) {
    // Coerced before use. metric_value arrives via POST /api/v1/telemetry,
    // which is unauthenticated and does not check the type, and SQLite's REAL
    // affinity stores a non-numeric value as TEXT — so `value` can be a string
    // here, and every branch below calls .toFixed() on it.
    if (typeof value !== "number" || !Number.isFinite(value)) {
        const coerced = Number(value);
        if (!Number.isFinite(coerced)) return String(value ?? "?");
        value = coerced;
    }
    if (name.includes("percent") || name.includes("pct")) return `${value.toFixed(1)}%`;
    // endsWith, not includes: these are unit SUFFIXES. As a substring test "_c"
    // also matches kv_cache_usage_ratio (collector.py), which would render
    // KV-cache occupancy as a temperature; "_ms" and "_gb" have the same shape
    // and are one metric name away from the same collision.
    if (name.includes("temperature") || name.endsWith("_c")) return `${value.toFixed(1)}°C`;
    if (name.endsWith("_ms")) return `${value.toFixed(0)} ms`;
    if (name.endsWith("_gb")) return `${value.toFixed(2)} GB`;
    // A node's uptime is the reason this branch exists, and raw seconds is the
    // one unit that never answers the question asked of it: "1209600" says
    // nothing, "14.0 d" says the node has not rebooted since the last deploy.
    if (name.endsWith("_seconds")) {
        if (value >= 86400) return `${(value / 86400).toFixed(1)} d`;
        if (value >= 3600) return `${(value / 3600).toFixed(1)} h`;
        return `${Math.round(value)} s`;
    }
    if (Number.isInteger(value)) return `${value}`;
    return `${value.toFixed(2)}`;
}

// The dot class for one container of a pod, from the two fields kubectl
// reports. A running container that has not passed its readiness probe is the
// case worth separating: it is a model still loading or a head still waiting on
// its peers, which is neither healthy nor dead.
//
// One copy, shared by the Overview grid and the node detail view: separate
// copies drift, and the same container then reads green on one screen and
// amber on the other with nothing saying which is right.
export function containerStateClass(c) {
    const state = c.state || "unknown";
    if (state === "running") return c.ready ? "online" : "degraded";
    if (state.startsWith("exited") || state === "CrashLoopBackOff") return "offline";
    return "unknown";
}

// The restart count as a badge, or "" for a container that has never restarted
// — a zero beside every healthy container is noise, and the badge is there to
// be noticed.
//
// Escaped like every other server value: api.py proxies restartCount straight
// out of the kubectl JSON with no int coercion, so anything non-numeric
// arriving in that field would land in innerHTML as markup.
export function restartBadge(c) {
    return c.restarts > 0
        ? `<span class="badge ${c.restarts >= 5 ? "crash" : "warn"}">${escapeHtml(c.restarts)} restarts</span>`
        : "";
}

// Keep a scrollable pane pinned to the bottom ACROSS a DOM mutation: read the
// geometry before, restore after.
//
//     const restore = stickyScroll(pane);
//     pane.appendChild(frag);
//     restore();
//
// Two steps because the test has to happen before the content changes —
// afterwards scrollHeight has already grown and every pane reads as
// "scrolled up". `slack` is how close to the bottom still counts as pinned,
// in pixels; sub-pixel layout and fractional device pixels mean the sum is
// rarely exactly scrollHeight.
//
// One helper, shared by the log pane and the chat transcript: separate copies
// drift to different slack values, and a reader then cannot tell whether the
// difference means anything. It does not.
export function stickyScroll(el, slack = 8) {
    const pinned = el.scrollTop + el.clientHeight >= el.scrollHeight - slack;
    return () => {
        if (pinned) el.scrollTop = el.scrollHeight;
    };
}

// What each container was last given. A WeakMap on the element, not a map keyed
// by id: a section that rebuilds its containers on mount (`root.innerHTML = …`,
// which both callers below do) then starts from an empty memo for free, so a
// re-mounted grid can never sit blank behind a cached identical string.
const lastHtml = new WeakMap();

// Write `html` into `el` only when it differs from what this helper last wrote
// there.
//
// Writing innerHTML replaces every input, <pre> and button under `el`,
// destroying a text selection, the focus in a readonly copy field and any
// transient button label — and the polled sections land here several times a
// minute with nothing but one badge string changed. The whole rendered string
// is compared rather than a hand-picked signature: a subset goes stale the
// moment a renderer reads one more field.
//
// One helper, shared by the Overview grids and the Access cards: separate
// copies drift, and a reader then cannot tell whether the difference means
// anything. It does not.
export function setHtmlIfChanged(el, html) {
    if (lastHtml.get(el) === html) return;
    lastHtml.set(el, html);
    el.innerHTML = html;
}

// localStorage, for values that are a convenience and never a requirement. A
// browser configured to block site data throws on `window.localStorage` itself,
// and in an async listener that throw becomes an unhandled rejection the
// operator never sees — the feature simply stops working. Both helpers absorb
// it: a failed read is "nothing remembered", a failed write is "not remembered
// next time", and the caller carries on either way.
//
// One pair, shared by the sections that remember a preference: a section that
// guards only its read still aborts mid-handler on the write.
export function readLocal(key, fallback = "") {
    try {
        return localStorage.getItem(key) ?? fallback;
    } catch (e) {
        return fallback;
    }
}

export function writeLocal(key, value) {
    try {
        localStorage.setItem(key, value);
    } catch (e) {
        // Not remembered; every caller treats that as acceptable.
    }
}
