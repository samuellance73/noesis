// ==========================================================================
// State Management
// ==========================================================================
let conversations = [];
let activeChatId = null;
let models = [];
let isGenerating = false;
let activeStreamController = null;

// Default configuration
const DEFAULT_SYSTEM_PROMPT = "You are Noesis, a premium and highly capable AI coding assistant. Provide clear, accurate, structured answers with well-commented code blocks where applicable.";
const DEFAULT_MODEL = "qwen/qwen3-32b";

// ==========================================================================
// Markdown Custom Renderer
// ==========================================================================
const renderer = new marked.Renderer();
renderer.code = function (code, infostring, escaped) {
    let text = "";
    let lang = "";
    if (typeof code === 'object' && code !== null) {
        text = code.text || "";
        lang = code.lang || "";
    } else {
        text = code || "";
        lang = infostring || "";
    }

    // Escape HTML tags to prevent execution/malformation in the code block
    const escapedCode = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");

    const langClass = lang ? `language-${lang}` : '';
    const uniqueId = 'code_' + Math.random().toString(36).substr(2, 9);

    return `
    <div class="code-block-wrapper">
        <div class="code-header">
            <span class="code-lang">${lang || 'code'}</span>
            <button class="copy-code-btn" onclick="copyCodeToClipboard(this, '${uniqueId}')">
                <i data-lucide="copy"></i>
                <span>Copy</span>
            </button>
        </div>
        <pre><code id="${uniqueId}" class="${langClass}">${escapedCode}</code></pre>
    </div>`;
};
marked.use({ renderer });

// Global function to copy code block content
window.copyCodeToClipboard = function (btn, codeId) {
    const codeEl = document.getElementById(codeId);
    if (!codeEl) return;

    const text = codeEl.textContent;
    navigator.clipboard.writeText(text).then(() => {
        const span = btn.querySelector('span');
        const icon = btn.querySelector('i');
        const originalText = span.textContent;

        span.textContent = 'Copied!';
        btn.style.color = '#10b981';

        if (window.lucide) {
            btn.innerHTML = `<i data-lucide="check"></i><span>Copied!</span>`;
            lucide.createIcons({ attrs: { class: ['copy-success-icon'] } });
            btn.querySelector('.copy-success-icon').style.width = '12px';
            btn.querySelector('.copy-success-icon').style.height = '12px';
        }

        setTimeout(() => {
            btn.style.color = '';
            btn.innerHTML = `<i data-lucide="copy"></i><span>Copy</span>`;
            lucide.createIcons();
            // Re-adjust size
            btn.querySelector('i').style.width = '12px';
            btn.querySelector('i').style.height = '12px';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
};

// ==========================================================================
// Initialization & Lifecycle
// ==========================================================================
document.addEventListener("DOMContentLoaded", async () => {
    initDOMReferences();
    loadFromLocalStorage();
    initEventListeners();
    await fetchModels();

    // Create initial chat if none exists
    if (conversations.length === 0) {
        createNewChat();
    } else {
        renderConversationsList();
        loadChat(activeChatId);
    }

    // Initialize Lucide Icons
    if (window.lucide) {
        lucide.createIcons();
    }
});

// DOM Elements cache
let dom = {};
function initDOMReferences() {
    dom = {
        sidebar: document.getElementById("sidebar"),
        menuToggleBtn: document.getElementById("menuToggleBtn"),
        mobileCloseBtn: document.getElementById("mobileCloseBtn"),
        newChatBtn: document.getElementById("newChatBtn"),
        conversationsList: document.getElementById("conversationsList"),
        modelSelect: document.getElementById("modelSelect"),
        temperatureSlider: document.getElementById("temperatureSlider"),
        temperatureVal: document.getElementById("temperatureVal"),
        maxTokensInput: document.getElementById("maxTokensInput"),
        maxTokensVal: document.getElementById("maxTokensVal"),
        unlimitedTokens: document.getElementById("unlimitedTokens"),
        systemPrompt: document.getElementById("systemPrompt"),
        systemPromptToggle: document.getElementById("systemPromptToggle"),
        clearAllBtn: document.getElementById("clearAllBtn"),
        activeChatTitle: document.getElementById("activeChatTitle"),
        activeModelBadge: document.getElementById("activeModelBadge"),
        exportChatBtn: document.getElementById("exportChatBtn"),
        clearChatBtn: document.getElementById("clearChatBtn"),
        messagesContainer: document.getElementById("messagesContainer"),
        emptyState: document.getElementById("emptyState"),
        typingIndicator: document.getElementById("typingIndicator"),
        charCount: document.getElementById("charCount"),
        chatInput: document.getElementById("chatInput"),
        sendBtn: document.getElementById("sendBtn")
    };
}

// ==========================================================================
// API Operations
// ==========================================================================
async function fetchModels() {
    try {
        const response = await fetch('/api/models');
        if (!response.ok) throw new Error("Could not retrieve models");
        const result = await response.json();

        if (result && result.data) {
            models = result.data.map(m => m.id).sort((a, b) => {
                // Keep GPT models at the top
                const aGpt = a.includes('gpt');
                const bGpt = b.includes('gpt');
                if (aGpt && !bGpt) return -1;
                if (!aGpt && bGpt) return 1;
                return a.localeCompare(b);
            });
            populateModelsSelect();
        }
    } catch (e) {
        console.error("Error loading models from backend:", e);
        // Robust fallback models
        models = [
            "openrouter/openai/gpt-3.5-turbo",
            "openrouter/openai/gpt-4o",
            "openrouter/google/gemini-2.5-flash",
            "openrouter/deepseek/deepseek-chat",
            "openrouter/anthropic/claude-3.5-sonnet",
            "openrouter/meta-llama/llama-3-70b-instruct"
        ];
        populateModelsSelect();
    }
}

function populateModelsSelect() {
    if (!dom.modelSelect) return;

    dom.modelSelect.innerHTML = "";
    models.forEach(modelId => {
        const option = document.createElement("option");
        option.value = modelId;
        // Format label nicely
        const cleanName = modelId.replace("openrouter/", "");
        option.textContent = cleanName;
        dom.modelSelect.appendChild(option);
    });

    // Match selected option with active chat model
    const chat = getActiveChat();
    if (chat && models.includes(chat.model)) {
        dom.modelSelect.value = chat.model;
        updateActiveModelBadge(chat.model);
    } else if (chat) {
        // Default to first loaded model or GPT-3.5
        const defaultModelToUse = models.includes(DEFAULT_MODEL) ? DEFAULT_MODEL : models[0];
        chat.model = defaultModelToUse;
        dom.modelSelect.value = defaultModelToUse;
        updateActiveModelBadge(defaultModelToUse);
        saveToLocalStorage();
    }
}

// ==========================================================================
// Event Listeners Setup
// ==========================================================================
function initEventListeners() {
    // Sidebar Mobile Toggles
    dom.menuToggleBtn.addEventListener("click", () => dom.sidebar.classList.add("open"));
    dom.mobileCloseBtn.addEventListener("click", () => dom.sidebar.classList.remove("open"));

    // New Chat
    dom.newChatBtn.addEventListener("click", () => {
        createNewChat();
        dom.sidebar.classList.remove("open");
    });

    // Reset All Data
    dom.clearAllBtn.addEventListener("click", () => {
        if (confirm("Reset application? All conversation histories will be permanently deleted.")) {
            localStorage.removeItem('noesis_chats');
            localStorage.removeItem('noesis_active_chat_id');
            window.location.reload();
        }
    });

    // Model Select Changes
    dom.modelSelect.addEventListener("change", (e) => {
        const chat = getActiveChat();
        if (chat) {
            chat.model = e.target.value;
            updateActiveModelBadge(chat.model);
            saveToLocalStorage();
        }
    });

    // Temperature Slider
    dom.temperatureSlider.addEventListener("input", (e) => {
        const val = e.target.value;
        dom.temperatureVal.textContent = val;
        const chat = getActiveChat();
        if (chat) {
            chat.temperature = parseFloat(val);
            saveToLocalStorage();
        }
    });

    // Max Tokens Slider
    dom.maxTokensInput.addEventListener("input", (e) => {
        const val = e.target.value;
        dom.maxTokensVal.textContent = val;
        const chat = getActiveChat();
        if (chat) {
            chat.max_tokens = parseInt(val);
            dom.unlimitedTokens.checked = false;
            saveToLocalStorage();
        }
    });

    // Max Tokens Unlimited Checkbox
    dom.unlimitedTokens.addEventListener("change", (e) => {
        const chat = getActiveChat();
        if (chat) {
            if (e.target.checked) {
                chat.max_tokens = null;
                dom.maxTokensVal.textContent = "Default";
                dom.maxTokensInput.disabled = true;
            } else {
                chat.max_tokens = parseInt(dom.maxTokensInput.value);
                dom.maxTokensVal.textContent = dom.maxTokensInput.value;
                dom.maxTokensInput.disabled = false;
            }
            saveToLocalStorage();
        }
    });

    // System Prompt toggle visbility
    dom.systemPromptToggle.addEventListener("click", () => {
        dom.systemPrompt.classList.toggle("visible");
        const isVisible = dom.systemPrompt.classList.contains("visible");
        dom.systemPromptToggle.innerHTML = isVisible ? '<i data-lucide="eye-off"></i>' : '<i data-lucide="eye"></i>';
        if (window.lucide) lucide.createIcons();
    });

    // System Prompt changes
    dom.systemPrompt.addEventListener("input", (e) => {
        const chat = getActiveChat();
        if (chat) {
            chat.systemPrompt = e.target.value;
            saveToLocalStorage();
        }
    });

    // Input area resizing & keybindings
    dom.chatInput.addEventListener("input", () => {
        dom.charCount.textContent = `${dom.chatInput.value.length} chars`;
        dom.chatInput.style.height = "auto";
        dom.chatInput.style.height = `${dom.chatInput.scrollHeight}px`;
    });

    dom.chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Send Button
    dom.sendBtn.addEventListener("click", sendMessage);

    // Export Chat
    dom.exportChatBtn.addEventListener("click", exportCurrentChat);

    // Clear Chat
    dom.clearChatBtn.addEventListener("click", () => {
        const chat = getActiveChat();
        if (chat && chat.messages.length > 0) {
            if (confirm("Clear all messages in this conversation? This cannot be undone.")) {
                chat.messages = [];
                saveToLocalStorage();
                renderMessages();
            }
        }
    });

    // Suggestions click
    document.querySelectorAll(".suggestion-card").forEach(card => {
        card.addEventListener("click", () => {
            const prompt = card.getAttribute("data-prompt");
            dom.chatInput.value = prompt;
            dom.chatInput.dispatchEvent(new Event("input"));
            dom.chatInput.focus();
        });
    });
}

// ==========================================================================
// Conversation State Helpers
// ==========================================================================
function getActiveChat() {
    return conversations.find(c => c.id === activeChatId);
}

function createNewChat() {
    const modelToUse = dom.modelSelect.value || DEFAULT_MODEL;
    const newChat = {
        id: "chat_" + Date.now(),
        title: "New Conversation",
        messages: [],
        model: modelToUse,
        temperature: 0.7,
        max_tokens: null,
        systemPrompt: DEFAULT_SYSTEM_PROMPT
    };

    conversations.unshift(newChat);
    activeChatId = newChat.id;
    saveToLocalStorage();
    renderConversationsList();
    loadChat(newChat.id);
}

function loadChat(chatId) {
    activeChatId = chatId;
    localStorage.setItem("noesis_active_chat_id", chatId);

    // Highlight list item
    document.querySelectorAll(".conversation-item").forEach(item => {
        item.classList.toggle("active", item.getAttribute("data-id") === chatId);
    });

    const chat = getActiveChat();
    if (!chat) return;

    // Load config states into inputs
    dom.activeChatTitle.textContent = chat.title;

    if (models.includes(chat.model)) {
        dom.modelSelect.value = chat.model;
    }
    updateActiveModelBadge(chat.model);

    dom.temperatureSlider.value = chat.temperature;
    dom.temperatureVal.textContent = chat.temperature;

    if (chat.max_tokens === null) {
        dom.unlimitedTokens.checked = true;
        dom.maxTokensVal.textContent = "Default";
        dom.maxTokensInput.disabled = true;
    } else {
        dom.unlimitedTokens.checked = false;
        dom.maxTokensInput.value = chat.max_tokens;
        dom.maxTokensVal.textContent = chat.max_tokens;
        dom.maxTokensInput.disabled = false;
    }

    dom.systemPrompt.value = chat.systemPrompt || "";

    renderMessages();
}

function renameChat(chatId) {
    const chat = conversations.find(c => c.id === chatId);
    if (!chat) return;

    const newTitle = prompt("Rename conversation:", chat.title);
    if (newTitle && newTitle.trim()) {
        chat.title = newTitle.trim();
        saveToLocalStorage();
        renderConversationsList();
        if (chatId === activeChatId) {
            dom.activeChatTitle.textContent = chat.title;
        }
    }
}

function deleteChat(chatId, event) {
    if (event) event.stopPropagation();

    if (conversations.length <= 1) {
        alert("You must keep at least one conversation history.");
        return;
    }

    if (confirm("Delete this conversation?")) {
        const index = conversations.findIndex(c => c.id === chatId);
        if (index > -1) {
            conversations.splice(index, 1);

            // If deleting current, select another
            if (chatId === activeChatId) {
                activeChatId = conversations[0].id;
            }

            saveToLocalStorage();
            renderConversationsList();
            loadChat(activeChatId);
        }
    }
}

function updateActiveModelBadge(modelId) {
    if (!dom.activeModelBadge) return;
    const cleanName = modelId.replace("openrouter/", "");
    dom.activeModelBadge.textContent = cleanName;
}

// ==========================================================================
// Rendering Elements
// ==========================================================================
function renderConversationsList() {
    if (!dom.conversationsList) return;

    dom.conversationsList.innerHTML = "";

    conversations.forEach(chat => {
        const isActive = chat.id === activeChatId;
        const item = document.createElement("div");
        item.className = `conversation-item ${isActive ? 'active' : ''}`;
        item.setAttribute("data-id", chat.id);

        item.innerHTML = `
            <div class="chat-item-left">
                <i data-lucide="message-square"></i>
                <span class="chat-title-span">${escapeHTML(chat.title)}</span>
            </div>
            <div class="chat-actions">
                <button class="chat-action-btn edit-btn" title="Rename Conversation">
                    <i data-lucide="edit-3"></i>
                </button>
                <button class="chat-action-btn delete-btn" title="Delete Conversation">
                    <i data-lucide="trash"></i>
                </button>
            </div>
        `;

        // Listeners for item click, edit, delete
        item.addEventListener("click", () => loadChat(chat.id));
        item.querySelector(".edit-btn").addEventListener("click", (e) => {
            e.stopPropagation();
            renameChat(chat.id);
        });
        item.querySelector(".delete-btn").addEventListener("click", (e) => {
            e.stopPropagation();
            deleteChat(chat.id, e);
        });

        dom.conversationsList.appendChild(item);
    });

    if (window.lucide) lucide.createIcons();
}

function renderMessages() {
    if (!dom.messagesContainer) return;

    const chat = getActiveChat();
    if (!chat || chat.messages.length === 0) {
        dom.messagesContainer.querySelectorAll(".message-wrapper").forEach(el => el.remove());
        dom.emptyState.style.display = "flex";
        return;
    }

    dom.emptyState.style.display = "none";

    // Clear old message elements
    dom.messagesContainer.querySelectorAll(".message-wrapper").forEach(el => el.remove());

    chat.messages.forEach((msg, idx) => {
        appendMessageDOM(msg.role, msg.content, idx);
    });

    scrollToBottom();
}

function appendMessageDOM(role, content, index) {
    if (role === 'system') return; // Hide system messages from actual list

    const wrapper = document.createElement("div");
    wrapper.className = `message-wrapper ${role}`;
    wrapper.setAttribute("data-index", index);

    const isUser = role === 'user';
    const avatarIcon = isUser ? 'user' : 'cpu';

    // Parse Markdown safely
    let parsedHTML = content;
    if (!isUser) {
        try {
            parsedHTML = marked.parse(content);
        } catch (e) {
            console.error("Markdown parse error:", e);
            parsedHTML = escapeHTML(content);
        }
    } else {
        parsedHTML = `<p>${escapeHTML(content).replace(/\n/g, "<br>")}</p>`;
    }

    wrapper.innerHTML = `
        <div class="avatar-wrapper">
            <i data-lucide="${avatarIcon}"></i>
        </div>
        <div class="message-content-container">
            <div class="message-bubble">${parsedHTML}</div>
            <div class="message-meta">
                <span>${role === 'user' ? 'You' : 'Noesis'}</span>
                <button class="meta-action-btn" onclick="copyMessageText(this)" title="Copy Message Text">
                    <i data-lucide="copy"></i>
                    <span>Copy</span>
                </button>
            </div>
        </div>
    `;

    dom.messagesContainer.appendChild(wrapper);

    // Apply Highlight.js to newly added code blocks
    wrapper.querySelectorAll("pre code").forEach(block => {
        try {
            hljs.highlightElement(block);
        } catch (e) {
            console.error(e);
        }
    });

    if (window.lucide) lucide.createIcons();
}

function updateAssistantMessageDOM(index, content) {
    const wrapper = dom.messagesContainer.querySelector(`.message-wrapper[data-index="${index}"]`);
    if (!wrapper) return;

    const bubble = wrapper.querySelector(".message-bubble");
    if (!bubble) return;

    try {
        bubble.innerHTML = marked.parse(content);
    } catch (e) {
        bubble.textContent = content;
    }

    // Re-apply highlight.js
    wrapper.querySelectorAll("pre code").forEach(block => {
        try {
            hljs.highlightElement(block);
        } catch (e) {
            console.error(e);
        }
    });

    if (window.lucide) lucide.createIcons();
}

// Global function to copy entire message text
window.copyMessageText = function (btn) {
    const container = btn.closest('.message-content-container');
    const bubble = container.querySelector('.message-bubble');

    // To preserve newlines and get clean text, we can read the raw message from our state
    const wrapper = btn.closest('.message-wrapper');
    const idx = parseInt(wrapper.getAttribute("data-id") || wrapper.getAttribute("data-index"));

    const chat = getActiveChat();
    let text = "";
    if (chat && chat.messages[idx]) {
        text = chat.messages[idx].content;
    } else {
        text = bubble.innerText;
    }

    navigator.clipboard.writeText(text).then(() => {
        const span = btn.querySelector('span');
        const originalText = span.textContent;
        span.textContent = 'Copied!';
        btn.style.color = '#10b981';
        setTimeout(() => {
            span.textContent = originalText;
            btn.style.color = '';
        }, 2000);
    });
};

// ==========================================================================
// Message Handling & Streaming API
// ==========================================================================
async function sendMessage() {
    if (isGenerating) return;

    const text = dom.chatInput.value.trim();
    if (!text) return;

    const chat = getActiveChat();
    if (!chat) return;

    isGenerating = true;
    toggleSendingState(true);

    // Add user message to state
    chat.messages.push({ role: 'user', content: text });

    // Set chat title if it's the first message
    if (chat.title === "New Conversation" && chat.messages.length === 1) {
        chat.title = text.length > 28 ? text.substring(0, 25) + "..." : text;
        renderConversationsList();
    }

    saveToLocalStorage();

    // Clear input
    dom.chatInput.value = "";
    dom.chatInput.dispatchEvent(new Event("input"));

    // Render user message
    dom.emptyState.style.display = "none";
    appendMessageDOM('user', text, chat.messages.length - 1);
    scrollToBottom();

    // Add placeholder assistant message to state and DOM
    const assistantMsgIndex = chat.messages.length;
    chat.messages.push({ role: 'assistant', content: "" });
    appendMessageDOM('assistant', "", assistantMsgIndex);
    scrollToBottom();

    // Compile messages payload (including system prompt)
    const apiMessages = [];
    if (chat.systemPrompt) {
        apiMessages.push({ role: 'system', content: chat.systemPrompt });
    }
    // Append previous dialogue
    chat.messages.slice(0, -1).forEach(m => {
        apiMessages.push({ role: m.role, content: m.content });
    });

    const payload = {
        model: chat.model,
        messages: apiMessages,
        temperature: chat.temperature,
        max_tokens: chat.max_tokens,
        stream: true
    };

    let assistantMsgContent = "";

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errJson = await response.json().catch(() => ({}));
            throw new Error(errJson.detail || `Server error ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop(); // Keep incomplete line

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;

                if (trimmed === "data: [DONE]") {
                    break;
                }

                if (trimmed.startsWith("data: ")) {
                    const dataStr = trimmed.slice(6);
                    try {
                        const data = JSON.parse(dataStr);
                        if (data.error) {
                            throw new Error(data.error);
                        }

                        const content = data.choices?.[0]?.delta?.content || "";
                        if (content) {
                            assistantMsgContent += content;
                            chat.messages[assistantMsgIndex].content = assistantMsgContent;
                            updateAssistantMessageDOM(assistantMsgIndex, assistantMsgContent);
                            scrollToBottom();
                        }
                    } catch (err) {
                        console.warn("Non-critical JSON parse error from SSE:", err, trimmed);
                    }
                }
            }
        }

    } catch (e) {
        console.error("Transmission error:", e);
        assistantMsgContent += `\n\n*Error: Could not generate response. (${e.message})*`;
        chat.messages[assistantMsgIndex].content = assistantMsgContent;
        updateAssistantMessageDOM(assistantMsgIndex, assistantMsgContent);
        scrollToBottom();
    } finally {
        isGenerating = false;
        toggleSendingState(false);
        saveToLocalStorage();
    }
}

function toggleSendingState(active) {
    if (active) {
        dom.typingIndicator.style.display = "flex";
        dom.sendBtn.disabled = true;
        dom.sendBtn.style.opacity = "0.5";
        dom.sendBtn.innerHTML = '<i data-lucide="loader" class="loader-spin"></i>';
    } else {
        dom.typingIndicator.style.display = "none";
        dom.sendBtn.disabled = false;
        dom.sendBtn.style.opacity = "1";
        dom.sendBtn.innerHTML = '<i data-lucide="send"></i>';
    }
    if (window.lucide) lucide.createIcons();
}

// ==========================================================================
// Storage & Utilities
// ==========================================================================
function saveToLocalStorage() {
    localStorage.setItem("noesis_chats", JSON.stringify(conversations));
    localStorage.setItem("noesis_active_chat_id", activeChatId);
}

function loadFromLocalStorage() {
    const stored = localStorage.getItem("noesis_chats");
    const activeId = localStorage.getItem("noesis_active_chat_id");

    if (stored) {
        try {
            conversations = JSON.parse(stored);
            activeChatId = activeId;
        } catch (e) {
            console.error("Failed to parse local storage conversations:", e);
            conversations = [];
        }
    }
}

function exportCurrentChat() {
    const chat = getActiveChat();
    if (!chat) return;

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(chat, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `${chat.title.toLowerCase().replace(/[^a-z0-9]/g, '_')}_export.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}

function scrollToBottom() {
    if (!dom.messagesContainer) return;
    // Scroll height minus client height gives the exact bottom scroll top
    dom.messagesContainer.scrollTop = dom.messagesContainer.scrollHeight;
}

function escapeHTML(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
