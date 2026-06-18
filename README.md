# Noesis — Autonomous Agent Framework

Noesis is a self-directed AI agent that can pursue complex, multi-step goals autonomously. Give it an ultimate goal and it will plan, delegate, execute tools in parallel, synthesise results, and loop until the goal is achieved — all observable in real time through a web UI, terminal, or Discord.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Layer 1 — Integration](#layer-1--integration)
  - [Layer 2 — Agent Core](#layer-2--agent-core)
  - [Layer 3 — Interfaces](#layer-3--interfaces)
  - [Layer 4 — Observability](#layer-4--observability)
- [How It Works](#how-it-works)
  - [Single-Turn Mode](#single-turn-mode)
  - [Autonomous Goal Mode](#autonomous-goal-mode)
- [Tools](#tools)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running](#running)
- [Configuration](#configuration)
- [Logs](#logs)
- [Tests](#tests)

---

## Overview

Noesis has two execution modes:

| Mode | What it does |
|---|---|
| **Single-Turn** | Takes one user message, runs a ReAct loop (reason → tool → observe → repeat), returns a final answer. Good for focused, self-contained questions. |
| **Autonomous Goal** | Takes an ultimate goal, breaks it into sub-tasks each cycle, spawns parallel executors, accumulates findings, and loops until the goal is declared complete. Good for open-ended research or multi-stage work. |

Both modes stream their progress as Server-Sent Events (SSE) so the UI can show thoughts, tool calls, and results as they happen rather than waiting for a final response.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Interface Layer                           │
│                                                                    │
│   Web UI (FastAPI + SSE)  │  Terminal (Rich)  │  Discord (bot)   │
└───────────────┬───────────┴────────┬──────────┴──────────────────┘
                │                    │
                ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                          Agent Layer                              │
│                                                                    │
│   GoalManager  ←→  AgentExecutor (×N, in parallel)               │
│   (strategic)        (tactical ReAct loop)                        │
│                                                                    │
│   Schemas: GoalState, CompletedTask, AgentStep, ToolCall          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
┌──────────────────────┐   ┌─────────────────────────────────────┐
│     Tool Registry     │   │         Integration Layer            │
│                       │   │                                      │
│  web_search           │   │  UpstreamService  (httpx + retry)   │
│  python_execute       │   │  config.py        (pydantic-settings)│
│  run_command          │   │  client.py        (client factory)   │
└──────────────────────┘   └─────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Observability Layer                         │
│                                                                    │
│  tracer.py       — ContextVar span tree, @traced_tool decorator   │
│  run_logger.py   — Per-run + per-task human-readable file logs    │
│  logging_setup.py — agent.log / trace.log / goal_manager.log      │
└──────────────────────────────────────────────────────────────────┘
```

### Layer 1 — Integration

**`integrations/llm/`** owns all upstream LLM communication.

- **`config.py`** — Reads `UPSTREAM_API_URL`, `API_KEY`, and `TAVILY_API_KEY` from environment / `.env` via `pydantic-settings`. Fails fast at startup if the URL is missing.
- **`client.py`** — Factory that builds a shared `httpx.AsyncClient` with the base URL, auth header, and a configurable timeout. Both the web app and CLI call this once and reuse the connection pool.
- **`service.py`** — `UpstreamService` wraps the client with three methods:
  - `fetch_models()` — GET `/models`
  - `get_chat_completion(payload)` — POST `/chat/completions`, returns parsed JSON
  - `stream_chat_completion(payload)` — streams raw SSE lines for live token streaming to the browser

  Every method is decorated with a `tenacity` retry policy: up to 4 attempts with exponential backoff, retrying only on `429`, `5xx`, timeout, and network errors.

### Layer 2 — Agent Core

**`agents/`** contains all agent logic, isolated from both transport and UI concerns.

#### Schemas (`schemas.py`)

Pydantic models that define the data contract between every component:

| Model | Purpose |
|---|---|
| `ToolCall` | A single tool invocation: `tool_name` + `tool_input` |
| `AgentStep` | One LLM response: `thought`, list of `tool_calls`, optional `final_answer`. Includes a validator that normalises legacy `tool_call` (singular) → `tool_calls` (list) and aliased field names. |
| `AgentState` | Per-run state for an executor: user input, step history, max iterations |
| `SubTask` | A unit of work handed to an executor: `goal` + `context` |
| `CompletedTask` | A finished sub-task: `goal` + the executor's verified `answer` |
| `GoalState` | Cross-cycle state owned by the GoalManager: the ultimate goal, progress summary, completed tasks, failed tasks, open questions, cycle counter |
| `ManagerDecision` | What the GoalManager LLM decides each cycle: tasks to spawn, progress update, whether goal is complete, optional final answer |

#### Tool Registry (`tools.py`)

`ToolRegistry` is a lightweight registry that maps tool names to async (or sync) callables. Each tool is registered with a `@tools_registry.register(name, description=...)` decorator — the description is injected verbatim into the agent's system prompt so the LLM knows what tools exist and when to use them.

**Built-in tools:**

| Tool | What it does |
|---|---|
| `web_search` | Calls the Tavily Search API, returns top 5 results as formatted text |
| `python_execute` | Runs arbitrary Python 3 in a subprocess; returns stdout/stderr. 10s timeout, killed on expiry. |
| `run_command` | Runs a shell command; returns stdout/stderr. 15s timeout. |

All three tools are wrapped with `@traced_tool(...)` so every invocation appears in the trace tree automatically.

#### Agent Executor (`executor.py`)

`AgentExecutor` implements a **ReAct** (Reason + Act) loop for a single task. It is intentionally stateless across calls — you give it a task, it runs to completion (or the iteration limit), and returns.

**Loop:**

```
for i in range(max_iterations):
    1. Build messages list (system prompt + conversation history)
    2. Inject budget-pressure reminder when 2 iterations remain
    3. POST to LLM → get assistant message
    4. Parse JSON response → AgentStep
       - if final_answer is set → yield final_answer, stop
       - if tool_calls is non-empty → run all tools concurrently (asyncio.gather)
         → append OBSERVATIONS to messages → continue loop
       - if neither → inject a corrective nudge → continue loop
```

**JSON parsing is resilient:**  The parser strips `<think>` tags, extracts from markdown code fences, locates the outermost `{}`, removes trailing commas (invalid JSON), and has a fallback that converts single-quote dicts to double-quote JSON. This handles the full range of model output sloppiness.

**Parallel tool execution:** All tool calls from a single LLM response are executed concurrently with `asyncio.gather`. Their observations are combined into one user message fed back into the conversation.

**Observation truncation:** Observations longer than 2,000 characters are truncated before being fed back to the LLM, with a note on how many characters were omitted. This prevents large tool results from blowing up the context window.

**Events yielded:**

| Event | When |
|---|---|
| `iteration_start` | Each loop iteration begins |
| `thought` | LLM responded with a thought |
| `tool_start` | Each tool is about to run (one per tool call) |
| `tool_observation` | All tools in this iteration completed |
| `final_answer` | LLM produced a final answer |
| `error` | Upstream failure, parse failure, or iteration limit hit |

#### Goal Manager (`goal_manager.py`)

`GoalManager` is the autonomous orchestrator. It runs a multi-cycle loop where each cycle:

1. **Drains** any user-injected refinements from a queue
2. **Asks** a manager LLM: *"Given everything we know, what should be done next?"* → parses a `ManagerDecision`
3. **Spawns** N `AgentExecutor` instances in parallel (one per `SubTask`) via `asyncio.gather`
4. **Collects** results → appends successful answers to `GoalState.completed`, failed tasks to `GoalState.failed_tasks`
5. **Updates** `GoalState.progress_summary` with verified findings
6. **Streams** a `cycle_complete` event to the caller
7. **Checks** `is_goal_complete` → if true, streams `goal_complete` and exits

The manager LLM sees the full current state at each cycle: goal, progress summary, completed tasks with their answers, failed tasks (with a hint to break them into smaller chunks), and open questions. It decides what to delegate next — or declares the goal complete and writes a final answer.

**Stopping:** The loop stops on `request_stop()` (programmatic), `inject_input("stop")` (from stdin or Discord), or a hard cap of 5 cycles. Mid-run input is accepted via `inject_input()` — anything that isn't a stop command becomes a goal refinement injected into the next cycle.

**Events yielded** (superset of executor events):

| Event | When |
|---|---|
| `goal_set` | Loop starts |
| `cycle_start` | Each cycle begins |
| `user_input_received` | A mid-run refinement was injected |
| `manager_thought` | Manager LLM responded |
| `spawning_tasks` | Executors about to be launched |
| *(all executor events)* | Pass-through from each sub-task |
| `cycle_complete` | Cycle finished, includes progress update |
| `goal_complete` | Goal achieved, includes final answer |
| `stopped` | Stop signal received |
| `error` | Any failure, includes progress summary so far |

### Layer 3 — Interfaces

All interfaces consume the same `GoalManager` and `AgentExecutor` APIs. They are pure presentation — no agent logic lives here.

#### Web (`interfaces/web/`)

FastAPI app (`main.py`) with three API endpoints:

| Endpoint | What it does |
|---|---|
| `GET /api/models` | Proxies model list from upstream |
| `POST /api/chat` | Direct LLM passthrough (streaming or not) — used by the chat tab |
| `POST /api/agent/run` | Single-turn agent; streams SSE events |
| `POST /api/agent/goal` | Autonomous goal loop; streams SSE events until complete |

The `UpstreamService` is created **once at startup** in the FastAPI lifespan context and stored in `app.state` — all requests share the same connection pool. Endpoints receive it via a `Depends` dependency.

The frontend (`static/`) is vanilla HTML + CSS + JS with `marked.js` for Markdown rendering. It connects to SSE streams and renders events as they arrive — thoughts, tool calls, and final answers show up progressively without page reloads.

#### Terminal (`interfaces/cli/`)

A Rich-based terminal UI (`run_cli.py`). Runs the `GoalManager` loop in a background async task while a foreground `_input_listener` reads stdin and pipes lines into `manager.inject_input()`. This lets you refine the goal or type `stop` while the agent is working. Logging is configured at `WARNING` level so the console stays clean — the Rich UI is the primary output channel.

#### Discord (`interfaces/discord/`)

A Discord bot that exposes the same agent capabilities as a chat interface (`run_discord.py`).

### Layer 4 — Observability

Three complementary observation systems:

#### Tracer (`utils/tracer.py`)

A zero-dependency span tree built on `contextvars.ContextVar`. A `Trace` is created at the start of each run; spans are pushed/popped as async context managers. The tree renders with box-drawing characters for easy visual parsing:

```
┌─[TRACE] id=a3f9b2c1
│
│  Query: Research the top 5 AI papers from 2025
│
├──[SPAN] ▶  goal_manager[cycle=1]  model=groq/openai/gpt-oss-120b
│    └──[RESULT]  [ok]  ▸ 2.41s
├──[SPAN] ▶  tool:web_search  input=top AI papers 2025
│    └──[RESULT]  [ok]  ▸ 890ms  result_len=1842 chars
└─[DONE]  id=a3f9b2c1  total ▸ 18.3s
```

Any function can be traced with `@traced(...)` or `@traced_tool(...)` decorators. If no trace is active, these are no-ops — tools and executors work identically in tests without a trace.

#### Run Logger (`utils/run_logger.py`)

For autonomous goal runs, `RunLogger` creates a directory under `logs/runs/<timestamp>_<run_id>/` containing:

- **`_manager.log`** — Manager thoughts, cycle decisions, spawned tasks, final answer
- **`task-N_<slug>.log`** — One file per executor sub-task, containing every thought, tool call, tool result, and final answer for that task

Because each sub-task gets its own file, parallel execution never interleaves. You can `tail -f` any task file to watch it in real time.

#### Structured Logging (`utils/logging_setup.py`)

Four simultaneous logging destinations:

| Destination | Content | Tail command |
|---|---|---|
| `logs/agent.log` | Everything at INFO+, full metadata | `tail -f logs/agent.log` |
| `logs/trace.log` | Trace tree only, clean box-drawing format | `tail -f logs/trace.log` |
| `logs/goal_manager.log` | Autonomous loop events only | `tail -f logs/goal_manager.log` |
| Console | Configurable level (WARNING in CLI, INFO in Discord) | — |

---

## How It Works

### Single-Turn Mode

```
User: "What is the current price of Bitcoin?"

  AgentExecutor
  ├─ iter 1: thought="I need to search for the current price."
  │          tool_calls=[{web_search: "Bitcoin price today"}]
  │          → runs web_search concurrently
  │          → observation: "Bitcoin: $67,432 (Coinbase, 2 mins ago)..."
  │
  └─ iter 2: thought="I have the price from a live source."
             final_answer="The current Bitcoin price is approximately $67,432."

→ returns "The current Bitcoin price is approximately $67,432."
```

### Autonomous Goal Mode

```
User goal: "Research the top 5 AI papers of 2025 and summarize each one."

Cycle 1
  GoalManager thinks: "I need to find the papers first."
  Spawns 1 executor: "Search for the top 5 most cited AI papers published in 2025"
    └─ AgentExecutor runs web_search → finds 5 paper titles + URLs
  GoalState.completed: [{"goal": "find papers", "answer": "Paper 1: ..., Paper 2: ..."}]
  progress_update: "Found 5 candidate papers. Will now summarize each."

Cycle 2
  GoalManager thinks: "I have the list. Now summarize all 5 in parallel."
  Spawns 5 executors (one per paper):
    ├─ AgentExecutor: "Summarize: Attention Is All You Need (2025 follow-up)"
    ├─ AgentExecutor: "Summarize: ..."
    ├─ AgentExecutor: "Summarize: ..."
    ├─ AgentExecutor: "Summarize: ..."
    └─ AgentExecutor: "Summarize: ..."
  All 5 run concurrently → 5 summaries collected
  GoalState.completed: 6 total tasks

Cycle 3
  GoalManager: "I have summaries for all 5 papers. Goal complete."
  is_goal_complete: true
  final_answer: "Here are the top 5 AI papers of 2025: ..."

→ goal_complete event emitted with final answer
```

---

## Tools

### `web_search`

Uses the [Tavily](https://tavily.com/) Search API. Returns the title, URL, and content snippet for the top 5 results.

- Requires: `TAVILY_API_KEY` in `.env`
- Timeout: 10 seconds
- Input: plain search query string

### `python_execute`

Runs Python 3 code in a sandboxed subprocess (same interpreter, separate process). The agent writes code that prints to stdout; the tool captures and returns that output.

- Timeout: 10 seconds (process is killed on expiry)
- Input: a string of Python source code
- Use cases: calculations, data transformations, text processing

### `run_command`

Runs a shell command and returns stdout + stderr.

- Timeout: 15 seconds
- Input: a shell command string
- Use cases: file operations, system info, calling CLIs

---

## Project Structure

```
Agent/
├── main.py                      # FastAPI app entrypoint (web server)
├── run_cli.py                   # Terminal UI entrypoint
├── run_discord.py               # Discord bot entrypoint
├── pyproject.toml               # Dependencies + Python version
├── .env                         # Your local credentials (not committed)
├── .env.example                 # Template for .env
│
├── agents/
│   ├── executor.py              # Single-turn ReAct loop
│   ├── goal_manager.py          # Multi-cycle autonomous loop
│   ├── schemas.py               # All Pydantic data models
│   └── tools.py                 # ToolRegistry + built-in tools
│
├── integrations/
│   └── llm/
│       ├── client.py            # httpx client factory
│       ├── config.py            # Settings (pydantic-settings, reads .env)
│       ├── schemas.py           # ChatMessage, ChatPayload
│       └── service.py           # UpstreamService with retry logic
│
├── interfaces/
│   ├── web/
│   │   ├── router.py            # FastAPI routes (/api/*)
│   │   └── static/              # index.html, style.css, script.js
│   ├── cli/
│   │   └── main.py              # Rich terminal UI + stdin listener
│   └── discord/
│       └── bot.py               # Discord bot
│
├── utils/
│   ├── tracer.py                # ContextVar span tree + @traced decorators
│   ├── run_logger.py            # Per-run/per-task human-readable file logs
│   └── logging_setup.py        # Global logging config (4 destinations)
│
├── tests/
│   ├── test_agent.py
│   ├── test_command_tool.py
│   ├── test_llm_router.py
│   ├── test_python_tool.py
│   └── test_tavily.py
│
└── logs/                        # Created at runtime
    ├── agent.log                # Full verbose log
    ├── trace.log                # Span tree
    ├── goal_manager.log         # Autonomous loop events
    └── runs/                    # Per-run directories
        └── 20260617_213045_a3f9b2c1/
            ├── _manager.log
            ├── task-0_find-top-5-ai-papers.log
            └── task-1_summarize-attention-paper.log
```

---

## Setup

**Requirements:** Python 3.12+, [uv](https://github.com/astral-sh/uv) (recommended) or pip.

```bash
# Clone and enter the project
cd Agent

# Install dependencies
uv sync

# Copy and fill in credentials
cp .env.example .env
```

Edit `.env`:

```env
# Your OpenAI-compatible LLM endpoint (any provider works)
UPSTREAM_API_URL=https://your-llm-provider.com/v1

# API key for the upstream (can be a placeholder if the endpoint is open)
API_KEY=sk-your-key-here

# Optional: enables the web_search tool
TAVILY_API_KEY=tvly-your-key-here
```

The agent works with **any OpenAI-compatible API** — OpenAI, Groq, Ollama, a HuggingFace Space running vLLM, etc. Point `UPSTREAM_API_URL` at any `/v1`-compatible endpoint.

---

## Running

### Web Interface

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# or
python main.py
```

Open `http://localhost:8000`. The UI has two tabs:

- **Chat** — Direct LLM chat or single-turn agent mode
- **Autonomous Goal** — Set a goal and watch the multi-cycle loop execute in real time

### Terminal (CLI)

```bash
uv run python run_cli.py
# or
python run_cli.py
```

Enter your goal at the prompt. While the agent runs, you can type at any time:

- Any text → injected as a goal refinement into the next cycle
- `stop` / `quit` / `exit` / `Ctrl-C` → halts the loop gracefully

### Discord Bot

```bash
# Add to .env first:
# DISCORD_BOT_TOKEN=your-bot-token

python run_discord.py
```

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `UPSTREAM_API_URL` | ✓ | `https://alisaajer-newrepo18.hf.space/v1` | OpenAI-compatible LLM base URL |
| `API_KEY` | | `""` | Bearer token for the upstream |
| `TAVILY_API_KEY` | | `""` | Enables `web_search` tool |
| `DISCORD_BOT_TOKEN` | Discord only | — | Discord bot token |
| `AGENT_MODEL` | | `groq/openai/gpt-oss-120b` | Default model for CLI |

The model can also be selected per-request from the web UI's model dropdown (populated live from `/api/models`).

---

## Logs

All log files are written to `logs/` at the project root (created automatically).

| File | Contents | Best for |
|---|---|---|
| `logs/agent.log` | Everything: all modules, timestamps, file+line | Deep debugging |
| `logs/trace.log` | Span tree only — clean box-drawing visualization | Tracing LLM call timing and nesting |
| `logs/goal_manager.log` | Cycle-level events: decisions, spawns, completions | Watching the autonomous loop at a high level |
| `logs/runs/<id>/_manager.log` | Manager thoughts + cycle summaries for one run | Reviewing a completed autonomous run |
| `logs/runs/<id>/task-N_<slug>.log` | Every thought/tool/result for one sub-task | Debugging why a specific sub-task failed |

Tail any file in real time:

```bash
tail -f logs/trace.log          # live span tree
tail -f logs/goal_manager.log   # live autonomous loop events
tail -f logs/agent.log          # everything
```

---

## Tests

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_agent.py -v
```

Tests use `pytest-asyncio` for async test support. The test suite covers the LLM router, individual tools (`web_search`, `python_execute`, `run_command`), and basic agent execution flows.
