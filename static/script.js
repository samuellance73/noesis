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
        // 1. Initial visual state for starting the loop
        assistantEl.innerHTML = `
            <div class="agent-thinking" style="display:flex; align-items:center; gap:8px; color:var(--dim); font-size:13.5px; font-family:var(--font-mono)">
                <span class="spinner"></span>
                <span id="agent-loop-status">Starting ReAct Loop...</span>
            </div>
        `;

        // Local state to keep track of steps as they stream in
        let streamingSteps = [];
        let finalAnswer = "";

        try {
            const res = await fetch("/api/agent/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model, user_input: text, stream: true }),
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }

            const reader = res.body.getReader();
            const dec = new TextDecoder();
            let buf = "";

            let planData = null;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buf += dec.decode(value, { stream: true });
                const lines = buf.split("\n");
                buf = lines.pop(); // Keep incomplete line in the buffer

                for (const line of lines) {
                    const t = line.trim();
                    if (t) {
                        if (t.startsWith("data: ")) {
                            try {
                                const eventData = JSON.parse(t.slice(6));
                                console.log("STREAM EVENT:", eventData);
                            } catch (e) {
                                console.log("RAW STREAM LINE:", t);
                            }
                        } else {
                            console.log("RAW STREAM LINE:", t);
                        }
                    }
                    if (!t || !t.startsWith("data: ")) continue;

                    try {
                        const eventData = JSON.parse(t.slice(6));

                        if (eventData.event === "planning_start") {
                            const statusText = document.getElementById("agent-loop-status");
                            if (statusText) statusText.textContent = "Generating Plan...";
                            continue;
                        }

                        if (eventData.event === "plan_ready") {
                            planData = eventData.milestones;
                            const statusText = document.getElementById("agent-loop-status");
                            if (statusText) statusText.textContent = "Executing Plan...";
                            // Immediately render the full plan so all milestones are visible
                            updateMessageHTML(assistantIndex, buildAgentStepsHTML("", streamingSteps, planData));
                            continue;
                        }

                        if (eventData.event === "done") {
                            // Reuse the already-populated streamingSteps so thoughts/tools/observations are preserved.
                            // Collect each milestone's final answer from eventData.results if available.
                            if (eventData.results) {
                                eventData.results.forEach((r, idx) => {
                                    if (streamingSteps[idx]) {
                                        streamingSteps[idx].final_answer = r.result || "";
                                    }
                                });
                            }
                            updateMessageHTML(assistantIndex, buildAgentStepsHTML("", streamingSteps, planData));
                            continue;
                        }

                        if (eventData.event === "error") {
                            throw new Error(eventData.message);
                        }

                        const stepIdx = eventData.step_index;
                        if (stepIdx !== undefined && !streamingSteps[stepIdx]) {
                            streamingSteps[stepIdx] = { step: {}, observation: null, milestone_index: null, milestone_goal: null };
                        }

                        switch (eventData.event) {
                            case "step_start":
                                const statusText = document.getElementById("agent-loop-status");
                                if (statusText) statusText.textContent = `Executing Step ${stepIdx + 1}: ${eventData.step_goal}...`;
                                // Store milestone info for rendering
                                if (streamingSteps[stepIdx]) {
                                    streamingSteps[stepIdx].milestone_index = eventData.milestone_index ?? null;
                                    streamingSteps[stepIdx].milestone_goal = eventData.milestone_goal ?? eventData.step_goal ?? null;
                                }
                                break;
                            case "iteration_start":
                                break;
                            case "thought":
                                streamingSteps[stepIdx].step.thought = eventData.thought;
                                updateMessageHTML(assistantIndex, buildAgentStepsHTML(finalAnswer, streamingSteps, planData));
                                break;
                            case "tool_start":
                                streamingSteps[stepIdx].step.tool_call = {
                                    tool_name: eventData.tool_name,
                                    tool_input: eventData.tool_input
                                };
                                updateMessageHTML(assistantIndex, buildAgentStepsHTML(finalAnswer, streamingSteps, planData));
                                break;
                            case "tool_observation":
                                streamingSteps[stepIdx].observation = eventData.observation;
                                updateMessageHTML(assistantIndex, buildAgentStepsHTML(finalAnswer, streamingSteps, planData));
                                break;
                            case "final_answer":
                                streamingSteps[stepIdx].final_answer = eventData.answer;
                                updateMessageHTML(assistantIndex, buildAgentStepsHTML("", streamingSteps, planData));
                                break;
                        }
                    } catch (err) {
                        console.error("Parse error in stream line:", err, t);
                    }
                }
            }
        } catch (e) {
            const errText = `\\n\\n*Error: ${e.message}*`;
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

function buildAgentStepsHTML(result, steps, planMilestones) {
    const hasSteps = steps && steps.length > 0;
    const hasPlan = planMilestones && planMilestones.length > 0;

    if (!hasSteps && !hasPlan) {
        return marked.parse(result || "");
    }

    let html = `<div class="agent-steps">`;

    // ── Plan Overview (all milestones, with live status badges) ──────────────
    if (hasPlan) {
        html += `<div class="plan-overview">
            <div class="plan-overview-header">📋 Plan</div>
            <div class="plan-overview-list">`;
        planMilestones.forEach((m, i) => {
            const stepData = steps[i];
            let statusClass = "ms-pending";
            let statusIcon = "○";
            if (stepData) {
                if (stepData.final_answer) {
                    statusClass = "ms-done";
                    statusIcon = "✓";
                } else {
                    statusClass = "ms-running";
                    statusIcon = "◉";
                }
            }
            html += `
            <div class="plan-milestone-row ${statusClass}">
                <span class="milestone-status">${statusIcon}</span>
                <span class="milestone-label">M${i + 1}</span>
                <span class="milestone-text">${escapeHTML(m.goal)}</span>
            </div>`;
        });
        html += `</div></div>`;
    }

    // ── Step Detail Cards (only for milestones that have started) ────────────
    if (hasSteps) {
        steps.forEach((s, idx) => {
            const stepNum = idx + 1;
            const stepData = s.step || {};
            const thought = stepData.thought || "";
            const toolCall = stepData.tool_call;
            const observation = s.observation;
            const finalAnswer = s.final_answer;
            // Prefer authoritative planMilestones source; fall back to streamed data
            const milestoneGoal = (planMilestones && planMilestones[idx])
                ? planMilestones[idx].goal
                : (s.milestone_goal || null);

            html += `
            <div class="agent-step">
                <div class="step-header">
                    <span class="step-num">Step ${stepNum}</span>
                    ${milestoneGoal
                    ? `<span class="step-milestone" title="${escapeHTML(milestoneGoal)}">🎯 Milestone ${stepNum}: ${escapeHTML(milestoneGoal)}</span>`
                    : `<span class="step-title">Reasoning</span>`
                }
                </div>
            `;

            if (thought) {
                html += `<div class="step-thought">${escapeHTML(thought)}</div>`;
            }

            if (toolCall) {
                html += `
                <div class="step-action">
                    <span class="action-icon">⚙️</span>
                    <span class="action-desc">Executing tool <code>${escapeHTML(toolCall.tool_name)}</code> with input: <code>${escapeHTML(JSON.stringify(toolCall.tool_input))}</code></span>
                </div>`;
            }

            if (observation !== undefined && observation !== null) {
                let displayObs = observation;
                let isJson = false;
                if (typeof observation === "string") {
                    try {
                        displayObs = JSON.stringify(JSON.parse(observation), null, 2);
                        isJson = true;
                    } catch (e) { /* raw string */ }
                } else if (typeof observation === "object") {
                    displayObs = JSON.stringify(observation, null, 2);
                    isJson = true;
                }
                const codeClass = isJson ? ' class="language-json"' : "";
                html += `
                <div class="step-observation">
                    <div class="obs-header">Observation</div>
                    <pre><code${codeClass}>${escapeHTML(displayObs)}</code></pre>
                </div>`;
            }

            if (finalAnswer) {
                html += `
                <div class="agent-final-answer">
                    <div class="final-header">💡 Final Answer</div>
                    <div class="final-content">${marked.parse(finalAnswer)}</div>
                </div>`;
            }

            html += `</div>`; // Close agent-step
        });
    }

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
