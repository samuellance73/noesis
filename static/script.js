// ── State ────────────────────────────────────────────────────────────────────
let messages = [];          // [{role, content}]
let isGenerating = false;
let currentMode = "chat";   // "chat" or "agent"

const SYSTEM_PROMPT = "You are Noesis, a multi-step reasoning agent. Think carefully before responding. Break complex problems into steps and reason through each one before giving a final answer.";

// ── DOM ───────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const dom = {
    messages: $("messages"),
    modelSelect: $("modelSelect"),
    chatInput: $("chatInput"),
    sendBtn: $("sendBtn"),
    statusText: $("statusText"),
    chatModeBtn: $("chatModeBtn"),
    agentModeBtn: $("agentModeBtn"),
};

function setMode(mode) {
    if (isGenerating) return;
    currentMode = mode;
    if (mode === "chat") {
        dom.chatModeBtn.classList.add("active");
        dom.chatModeBtn.setAttribute("aria-checked", "true");
        dom.agentModeBtn.classList.remove("active");
        dom.agentModeBtn.setAttribute("aria-checked", "false");
        dom.chatInput.placeholder = "Ask anything…";
    } else {
        dom.agentModeBtn.classList.add("active");
        dom.agentModeBtn.setAttribute("aria-checked", "true");
        dom.chatModeBtn.classList.remove("active");
        dom.chatModeBtn.setAttribute("aria-checked", "false");
        dom.chatInput.placeholder = "Enter query for the Agent (will run steps and tools)…";
    }
}

// ── Markdown ──────────────────────────────────────────────────────────────────
const renderer = new marked.Renderer();
renderer.code = function ({ text, lang }) {
    const escaped = text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;")
        .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
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
            opt.value = id;
            opt.textContent = id;
            dom.modelSelect.appendChild(opt);
        });
    } catch (e) {
        setStatus("⚠ Could not load models — check API connection.");
    }
}

// ── Rendering ─────────────────────────────────────────────────────────────────
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

// ── Sending ───────────────────────────────────────────────────────────────────
async function send() {
    if (isGenerating) return;
    const text = dom.chatInput.value.trim();
    if (!text) return;

    const model = dom.modelSelect.value;
    if (!model) { setStatus("Select a model first."); return; }

    isGenerating = true;
    setStatus("Reasoning…");
    setSendState(true);

    messages.push({ role: "user", content: text });
    dom.chatInput.value = "";
    resizeInput();

    renderMessage("user", text, messages.length - 1);

    // placeholder assistant message
    const assistantIndex = messages.length;
    messages.push({ role: "assistant", content: "" });
    const assistantEl = renderMessage("assistant", "▋", assistantIndex);

    if (currentMode === "agent") {
        // Render thinking loader
        assistantEl.innerHTML = `
            <div class="agent-thinking" style="display:flex; align-items:center; gap:8px; color:var(--dim); font-size:13.5px; font-family:var(--font-mono)">
                <span class="spinner"></span>
                <span>Agent is executing ReAct loop (thinking, selecting tools)...</span>
            </div>
        `;
        try {
            const res = await fetch("/api/agent/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model, user_input: text }),
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            messages[assistantIndex].content = data.result;

            const html = buildAgentStepsHTML(data.result, data.steps);
            updateMessageHTML(assistantIndex, html);
        } catch (e) {
            const errText = `\n\n*Error: ${e.message}*`;
            messages[assistantIndex].content = errText;
            updateMessage(assistantIndex, errText);
            setStatus(`Error: ${e.message}`);
        } finally {
            isGenerating = false;
            setSendState(false);
            setStatus("");
        }
        return;
    }

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

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const reader = res.body.getReader();
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
                if (!t || t === "data: [DONE]") continue;
                if (!t.startsWith("data: ")) continue;
                try {
                    const data = JSON.parse(t.slice(6));
                    if (data.error) throw new Error(data.error);
                    const chunk = data.choices?.[0]?.delta?.content || "";
                    if (chunk) {
                        accumulated += chunk;
                        messages[assistantIndex].content = accumulated;
                        updateMessage(assistantIndex, accumulated);
                    }
                } catch { /* non-fatal parse glitch */ }
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

// ── UI Helpers ────────────────────────────────────────────────────────────────
function setSendState(active) {
    dom.sendBtn.disabled = active;
    dom.sendBtn.textContent = active ? "…" : "▶";
}

function setStatus(text) {
    dom.statusText.textContent = text;
}

function scrollDown() {
    dom.messages.scrollTop = dom.messages.scrollHeight;
}

function resizeInput() {
    dom.chatInput.style.height = "auto";
    dom.chatInput.style.height = dom.chatInput.scrollHeight + "px";
}

function escapeHTML(s) {
    if (!s) return "";
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function updateMessageHTML(index, html) {
    const el = dom.messages.querySelector(`.msg[data-index="${index}"]`);
    if (!el) return;
    el.innerHTML = html;
    el.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));
    scrollDown();
}

function buildAgentStepsHTML(result, steps) {
    if (!steps || steps.length === 0) {
        return marked.parse(result);
    }

    let html = `<div class="agent-steps">`;
    steps.forEach((s, idx) => {
        const stepNum = idx + 1;
        const stepData = s.step || {};
        const thought = stepData.thought || "";
        const toolCall = stepData.tool_call;
        const observation = s.observation;

        html += `
        <div class="agent-step">
            <div class="step-header">
                <span class="step-num">Step ${stepNum}</span>
                <span class="step-title">Reasoning</span>
            </div>
            <div class="step-thought">${escapeHTML(thought)}</div>
        `;

        if (toolCall) {
            html += `
            <div class="step-action">
                <span class="action-icon">⚙️</span>
                <span class="action-desc">Executing tool <code>${escapeHTML(toolCall.tool_name)}</code> with input: <code>${escapeHTML(JSON.stringify(toolCall.tool_input))}</code></span>
            </div>
            `;
        }

        if (observation !== undefined && observation !== null) {
            html += `
            <div class="step-observation">
                <div class="obs-header">Observation</div>
                <pre><code>${escapeHTML(observation)}</code></pre>
            </div>
            `;
        }

        html += `</div>`; // Close agent-step
    });

    // Final Answer section
    html += `
    <div class="agent-final-answer">
        <div class="final-header">💡 Final Answer</div>
        <div class="final-content">${marked.parse(result)}</div>
    </div>
    `;

    html += `</div>`; // Close agent-steps
    return html;
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadModels();

    dom.chatInput.addEventListener("input", resizeInput);
    dom.chatInput.addEventListener("keydown", e => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
    });
    dom.sendBtn.addEventListener("click", send);

    dom.chatModeBtn.addEventListener("click", () => setMode("chat"));
    dom.agentModeBtn.addEventListener("click", () => setMode("agent"));
});
