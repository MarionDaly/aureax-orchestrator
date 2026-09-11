// Hash router — swaps sections. Hash may include a sub-arg: "#chat/node-edge-01".

import * as overview from "/sections/overview.js";
import * as lifecycle from "/sections/lifecycle.js";
import * as access from "/sections/access.js";
import * as chat from "/sections/chat.js";
import * as configs from "/sections/configs.js";
import * as node from "/sections/node.js";

const sections = {
    "#overview":  overview,
    "#lifecycle": lifecycle,
    "#access":    access,
    "#chat":      chat,
    "#configs":   configs,
    "#node":      node,
};

let current = null;

// Incremented on every route change. mount() is async — chat and configs both
// await a fetch before opening their stream — and unmount() runs before a
// superseded mount resumes. Handlers therefore re-check isCurrent(token) after
// each await, so a mount that has lost the race never opens an EventSource or
// an interval nothing will close.
let routeToken = 0;

export function isCurrent(token) {
    return token === routeToken;
}

async function route() {
    const raw = window.location.hash || "#overview";
    const [tab, ...rest] = raw.split("/");
    const arg = rest.join("/") || null;
    const handler = sections[tab] || sections["#overview"];
    const root = document.getElementById("app-main");

    const token = ++routeToken;

    if (current && current.unmount) {
        try { current.unmount(); } catch (e) { console.error("unmount failed", e); }
    }
    root.innerHTML = "";
    current = handler;

    // #node has no tab of its own — it is reached by clicking an Overview card
    // or a Lifecycle row, so Overview stays lit while a detail view is open
    // rather than the whole nav going dark.
    const lit = tab === "#node" ? "#overview" : tab;
    document.querySelectorAll(".tab").forEach(t => {
        t.classList.toggle("active", t.getAttribute("href") === lit);
    });

    try {
        await handler.mount(root, arg, token);
    } catch (e) {
        console.error("mount failed", e);
    }

    // A mount that lost the race must not leave anything running — but only
    // when a DIFFERENT section took over. Sections keep their state in module
    // scope, so if this same handler was re-mounted meanwhile, unmount() here
    // would tear down the new mount's stream and interval rather than the
    // abandoned one's.
    if (!isCurrent(token) && current !== handler && handler.unmount) {
        try { handler.unmount(); } catch (e) { console.error("late unmount failed", e); }
    }
}

window.addEventListener("hashchange", () => { route(); });
route();
