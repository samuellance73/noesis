// ── State ────────────────────────────────────────────────────────────────────
let messages = [];
let isGenerating = false;
let currentMode = "chat"; // "chat" | "agent" | "goal"
let goalReader = null;    // active SSE reader for the autonomous loop (stop support)

const SYSTEM_PROMPT = "You are Noesis, a multi-step reasoning agent. Think carefully before responding.";

// ── DOM ───────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const dom = {
    messages: $("messages"),
    modelSelect: $("modelSelect"),
    chatInput: $("chatInput"),
    sendBtn: $("sendBtn"),
    stopBtn: $("stopBtn"),
    statusText: $("statusText"),
    chatModeBtn: $("chatModeBtn"),
    agentModeBtn: $("agentModeBtn"),
    goalModeBtn: $("goalModeBtn"),
};

function setMode(mode) {
    // Always allow mode switching — if a previous run left isGenerating stuck
    // (e.g. a crash before finally ran), reset it so the UI isn't permanently locked.
    if (isGenerating) {
        isGenerating = false;
        setSendState(false);
        if (goalReader) { try { goalReader.cancel(); } catch { } goalReader = null; }
        dom.stopBtn.style.display = "none";
        setStatus("");
    }
    currentMode = mode;
    const btns = { chat: dom.chatModeBtn, agent: dom.agentModeBtn, goal: dom.goalModeBtn };
    Object.entries(btns).forEach(([k, b]) => {
        b.classList.toggle("active", k === mode);
        b.setAttribute("aria-checked", k === mode ? "true" : "false");
    });
    const placeholders = {
        chat: "Ask anything…",
        agent: "Enter a request for the single-turn agent…",
        goal: "Set an ultimate goal — the agent will work autonomously until done…",
    };
    dom.chatInput.placeholder = placeholders[mode];
}

// ── Markdown ──────────────────────────────────────────────────────────────────
const renderer = new marked.Renderer();
renderer.code = function ({ text, lang }) {
    const escaped = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    const id = "c_" + Math.random().toString(36).slice(2, 9);
    return `<div class="code-block">
        <div class="code-meta"><span>${lang || "code"}</span>
            <button onclick="copyCode('${id}')">copy</button></div>
        <pre><code id="${id}" class="${lang ? "language-" + lang : ""}">${escaped}</code></pre>
    </div>`;
};
marked.use({ renderer });

window.copyCode = function (id) {
    const el = document.getElementById(id);
    if (el) navigator.clipboard.writeText(el.textContent);
};

// ── Models ────────────────────────────────────────────────────────────────────
async function loadModels() {
    try {
        const res = await fetch("/api/models");
        if (!res.ok) throw new Error(res.status);
        const { data } = await res.json();
        dom.modelSelect.innerHTML = "";
        data.map(m => m.id).sort().forEach(id => {
            const opt = document.createElement("option");
            opt.value = id; opt.textContent = id;
            dom.modelSelect.appendChild(opt);
        });
    } catch (e) { setStatus("⚠ Could not load models."); }
}

// ── Rendering helpers ─────────────────────────────────────────────────────────
function escapeHTML(s) {
    if (!s) return "";
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
function renderMessage(role, content, index) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.dataset.index = index;
    el.innerHTML = role === "user"
        ? `<p>${escapeHTML(content).replace(/\n/g, "<br>")}</p>`
        : marked.parse(content);
    el.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));
    dom.messages.appendChild(el);
    scrollDown();
    return el;
}
function updateMessage(index, content) {
    const el = dom.messages.querySelector(`.msg[data-index="${index}"]`);
    if (!el) return;
    el.innerHTML = marked.parse(content);
    el.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));
    scrollDown();
}
function updateMessageHTML(index, html) {
    const el = dom.messages.querySelector(`.msg[data-index="${index}"]`);
    if (!el) return;
    el.innerHTML = html;
    el.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));
    scrollDown();
}
function setSendState(active) {
    dom.sendBtn.disabled = active;
    dom.sendBtn.textContent = active ? "…" : "▶";
}
function setStatus(text) { dom.statusText.textContent = text; }
function scrollDown() { dom.messages.scrollTop = dom.messages.scrollHeight; }
function resizeInput() {
    dom.chatInput.style.height = "auto";
    dom.chatInput.style.height = dom.chatInput.scrollHeight + "px";
}

// ── SSE stream reader ─────────────────────────────────────────────────────────
async function* readSSE(response) {
    const reader = response.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
            const t = line.trim();
            if (!t || !t.startsWith("data: ")) continue;
            try { yield JSON.parse(t.slice(6)); } catch { }
        }
    }
    // store reader ref for cancellation
    goalReader = reader;
}

// ── Send dispatcher ───────────────────────────────────────────────────────────
async function send() {
    if (isGenerating) return;
    const text = dom.chatInput.value.trim();
    if (!text) return;
    const model = dom.modelSelect.value;
    if (!model) { setStatus("Select a model first."); return; }

    isGenerating = true;
    setSendState(true);
    messages.push({ role: "user", content: text });
    dom.chatInput.value = "";
    resizeInput();
    renderMessage("user", text, messages.length - 1);
    const assistantIndex = messages.length;
    messages.push({ role: "assistant", content: "" });

    if (currentMode === "goal") {
        await runGoalMode(text, model, assistantIndex);
    } else if (currentMode === "agent") {
        await runAgentMode(text, model, assistantIndex);
    } else {
        await runChatMode(model, assistantIndex);
    }
}

// ── Stop (autonomous loop) ────────────────────────────────────────────────────
async function stopGoal() {
    if (goalReader) {
        try { goalReader.cancel(); } catch { }
        goalReader = null;
    }
    dom.stopBtn.style.display = "none";
    isGenerating = false;
    setSendState(false);
    setStatus("Stopped.");
}

// ═══════════════════════════════════════════════════════════════════════════════
// GOAL MODE — autonomous loop
// ═══════════════════════════════════════════════════════════════════════════════
async function runGoalMode(goal, model, assistantIndex) {
    dom.stopBtn.style.display = "flex";
    setStatus("Autonomous loop running…");

    // State for rendering
    const state = {
        cycles: [],       // cycles[n] = { tasks:[], thought:"", progress:"", complete:false }
        finalAnswer: null,
        stopped: false,
        goalText: goal,
    };

    renderMessage("assistant", "", assistantIndex);
    updateMessageHTML(assistantIndex, buildGoalHTML(state));

    try {
        const res = await fetch("/api/agent/goal", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model, goal }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        for await (const ev of readSSE(res)) {
            console.log("GOAL EVENT:", ev);
            handleGoalEvent(ev, state, assistantIndex);
            if (ev.event === "goal_complete" || ev.event === "stopped" || ev.event === "error") break;
        }
    } catch (e) {
        if (e.name !== "AbortError") {
            const msg = `\n\n*Error: ${e.message}*`;
            messages[assistantIndex].content = msg;
            updateMessage(assistantIndex, msg);
            setStatus(`Error: ${e.message}`);
        }
    } finally {
        goalReader = null;
        dom.stopBtn.style.display = "none";
        isGenerating = false;
        setSendState(false);
        setStatus("");
    }
}

function handleGoalEvent(ev, state, assistantIndex) {
    const cycle = ev.cycle ? ev.cycle - 1 : 0; // 0-indexed

    switch (ev.event) {
        case "goal_set":
            state.goalText = ev.goal;
            break;

        case "cycle_start":
            while (state.cycles.length < ev.cycle) {
                state.cycles.push({ tasks: [], thought: "", progress: "", subtaskResults: [], running: true });
            }
            break;

        case "manager_thought":
            if (state.cycles[cycle]) state.cycles[cycle].thought = ev.thought;
            setStatus(`Cycle ${ev.cycle} — thinking…`);
            break;

        case "spawning_tasks":
            if (state.cycles[cycle]) {
                state.cycles[cycle].tasks = ev.tasks || [];
            }
            setStatus(`Cycle ${ev.cycle} — running ${ev.count} task(s)…`);
            break;

        case "final_answer": {
            // A sub-task completed
            const taskGoal = ev.task_goal || "";
            const answer = ev.answer || "";
            if (state.cycles[cycle]) {
                state.cycles[cycle].subtaskResults.push({ goal: taskGoal, answer });
            }
            break;
        }

        case "cycle_complete":
            if (state.cycles[cycle]) {
                state.cycles[cycle].progress = ev.progress_update || "";
                state.cycles[cycle].running = false;
            }
            setStatus(`Cycle ${ev.cycle} complete.`);
            break;

        case "goal_complete":
            state.finalAnswer = ev.final_answer || "";
            state.stopped = false;
            setStatus("Goal complete!");
            break;

        case "stopped":
            state.stopped = true;
            setStatus("Stopped.");
            break;

        case "error":
            state.error = ev.message || "Unknown error";
            state.stopped = true;
            break;
    }

    updateMessageHTML(assistantIndex, buildGoalHTML(state));
}

function buildGoalHTML(state) {
    let html = `<div class="goal-container">`;

    // Goal header
    html += `<div class="goal-header">
        <span class="goal-icon">🎯</span>
        <span class="goal-title">${escapeHTML(state.goalText)}</span>
    </div>`;

    // Cycles
    state.cycles.forEach((cyc, i) => {
        const cycNum = i + 1;
        const isRunning = cyc.running;
        html += `<div class="goal-cycle ${isRunning ? "cycle-running" : "cycle-done"}">
            <div class="cycle-header">
                <span class="cycle-badge">${isRunning ? '<span class="spinner"></span>' : "✓"} Cycle ${cycNum}</span>
                ${cyc.progress ? `<span class="cycle-progress">${escapeHTML(cyc.progress)}</span>` : ""}
            </div>`;

        if (cyc.thought) {
            html += `<div class="cycle-thought"><span class="label">🧠</span>${escapeHTML(cyc.thought)}</div>`;
        }

        if (cyc.tasks && cyc.tasks.length > 0) {
            html += `<div class="cycle-tasks">`;
            cyc.tasks.forEach(t => {
                const result = cyc.subtaskResults.find(r => r.goal === t);
                html += `<div class="task-row ${result ? "task-done" : "task-running"}">
                    <span class="task-status">${result ? "✓" : '<span class="spinner-sm"></span>'}</span>
                    <span class="task-goal">${escapeHTML(t)}</span>
                </div>`;
                if (result && result.answer) {
                    html += `<div class="task-answer">${marked.parse(result.answer)}</div>`;
                }
            });
            html += `</div>`;
        }

        html += `</div>`; // cycle
    });

    // Final answer
    if (state.finalAnswer) {
        html += `<div class="goal-final">
            <div class="goal-final-header">✅ Goal Complete</div>
            <div class="goal-final-content">${marked.parse(state.finalAnswer)}</div>
        </div>`;
    }

    if (state.stopped && !state.finalAnswer) {
        html += `<div class="goal-stopped">⏹ Loop stopped.</div>`;
    }
    if (state.error) {
        html += `<div class="goal-error">❌ ${escapeHTML(state.error)}</div>`;
    }

    html += `</div>`;
    return html;
}

// ═══════════════════════════════════════════════════════════════════════════════
// AGENT MODE — single-turn with parallel tools
// ═══════════════════════════════════════════════════════════════════════════════
async function runAgentMode(text, model, assistantIndex) {
    setStatus("Running agent…");
    const assistantEl = renderMessage("assistant", "", assistantIndex);
    assistantEl.innerHTML = `<div class="agent-thinking">
        <span class="spinner"></span>
        <span id="agent-loop-status">Starting…</span>
    </div>`;

    const steps = [];   // steps[iIdx] = { thought, toolCalls:[], observation, finalAnswer }

    function ensureStep(iIdx) {
        while (steps.length <= iIdx) steps.push({ thought: "", toolCalls: [], observation: null, finalAnswer: null });
    }

    try {
        const res = await fetch("/api/agent/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model, user_input: text, stream: true }),
        });
        if (!res.ok) { const e = await res.json().catch(() => { }); throw new Error(e.detail || `HTTP ${res.status}`); }

        for await (const ev of readSSE(res)) {
            console.log("AGENT EVENT:", ev);
            const iIdx = ev.step_index ?? 0;
            switch (ev.event) {
                case "plan_ready": setStatus("Executing…"); break;
                case "thought":
                    ensureStep(iIdx);
                    steps[iIdx].thought = ev.thought;
                    break;
                case "tool_start":
                    ensureStep(iIdx);
                    steps[iIdx].toolCalls.push({ tool_name: ev.tool_name, tool_input: ev.tool_input });
                    break;
                case "tool_observation":
                    ensureStep(iIdx);
                    steps[iIdx].observation = ev.observation;
                    break;
                case "final_answer":
                    ensureStep(iIdx);
                    steps[iIdx].finalAnswer = ev.answer;
                    break;
                case "error": throw new Error(ev.message);
            }
            updateMessageHTML(assistantIndex, buildAgentHTML(steps));
        }
    } catch (e) {
        const msg = `\n\n*Error: ${e.message}*`;
        messages[assistantIndex].content = msg;
        updateMessage(assistantIndex, msg);
        setStatus(`Error: ${e.message}`);
    } finally {
        isGenerating = false;
        setSendState(false);
        setStatus("");
    }
}

function buildAgentHTML(steps) {
    if (!steps.length) return `<div class="agent-thinking"><span class="spinner"></span><span>Starting…</span></div>`;
    let html = `<div class="agent-steps">`;
    steps.forEach((s, i) => {
        html += `<div class="agent-step">
            <div class="step-header"><span class="step-num">Iteration ${i + 1}</span></div>`;
        if (s.thought) html += `<div class="step-thought">${escapeHTML(s.thought)}</div>`;
        s.toolCalls.forEach(tc => {
            html += `<div class="step-action">
                <span class="action-icon">⚙️</span>
                <span class="action-desc">Tool <code>${escapeHTML(tc.tool_name)}</code> ← <code>${escapeHTML(JSON.stringify(tc.tool_input))}</code></span>
            </div>`;
        });
        if (s.observation != null) {
            html += `<div class="step-observation">
                <div class="obs-header">Observation</div>
                <pre><code>${escapeHTML(s.observation)}</code></pre>
            </div>`;
        }
        if (s.finalAnswer) {
            html += `<div class="agent-final-answer">
                <div class="final-header">💡 Final Answer</div>
                <div class="final-content">${marked.parse(s.finalAnswer)}</div>
            </div>`;
        }
        html += `</div>`;
    });
    html += `</div>`;
    return html;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CHAT MODE
// ═══════════════════════════════════════════════════════════════════════════════
async function runChatMode(model, assistantIndex) {
    setStatus("Thinking…");
    renderMessage("assistant", "▋", assistantIndex);
    const payload = {
        model,
        messages: [{ role: "system", content: SYSTEM_PROMPT }, ...messages.slice(0, -1)],
        temperature: 0.6,
        stream: true,
    };
    let accumulated = "";
    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) { const e = await res.json().catch(() => { }); throw new Error(e.detail || `HTTP ${res.status}`); }

        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            const lines = buf.split("\n"); buf = lines.pop();
            for (const line of lines) {
                const t = line.trim();
                if (!t || t === "data: [DONE]" || !t.startsWith("data: ")) continue;
                try {
                    const data = JSON.parse(t.slice(6));
                    const chunk = data.choices?.[0]?.delta?.content || "";
                    if (chunk) { accumulated += chunk; messages[assistantIndex].content = accumulated; updateMessage(assistantIndex, accumulated); }
                } catch { }
            }
        }
    } catch (e) {
        accumulated += `\n\n*Error: ${e.message}*`;
        messages[assistantIndex].content = accumulated;
        updateMessage(assistantIndex, accumulated);
        setStatus(`Error: ${e.message}`);
    } finally {
        isGenerating = false;
        setSendState(false);
        setStatus("");
    }
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadModels();
    dom.chatInput.addEventListener("input", resizeInput);
    dom.chatInput.addEventListener("keydown", e => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
    });
    dom.sendBtn.addEventListener("click", send);
    dom.stopBtn.addEventListener("click", stopGoal);
    dom.chatModeBtn.addEventListener("click", () => setMode("chat"));
    dom.agentModeBtn.addEventListener("click", () => setMode("agent"));
    dom.goalModeBtn.addEventListener("click", () => setMode("goal"));
});
