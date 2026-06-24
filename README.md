# Noesis — Autonomous Agent Framework

Noesis is a modular, event-driven autonomous AI agent framework designed to break down and execute complex, multi-step goals. Built around a **background daemon scheduler**, Noesis coordinates strategic planning, parallel sub-task execution, qualitative evaluation, and episodic memory persistence to achieve user-defined missions.

---

## Table of Contents

- [Core Features](#core-features)
- [System Architecture](#system-architecture)
  - [Layer 1 — Integration & Transport](#layer-1--integration--transport)
  - [Layer 2 — Scheduler & Triggers](#layer-2--scheduler--triggers)
  - [Layer 3 — Agent Core](#layer-3--agent-core)
  - [Layer 4 — Interfaces](#layer-4--interfaces)
  - [Layer 5 — Observability & Utilities](#layer-5--observability--utilities)
- [How It Works (Execution & Routing)](#how-it-works-execution--routing)
  - [The Trigger Lifecycle](#the-trigger-lifecycle)
  - [Specialized Executor Profiles](#specialized-executor-profiles)
- [Project Structure](#project-structure)
- [Setup & Configuration](#setup--configuration)
- [Running Noesis](#running-noesis)
- [Observability & Debugging](#observability--debugging)

---

## Core Features

- **Daemon-Driven Trigger Architecture**: An in-memory, thread-safe scheduling queue (`TriggerStore`) that supports parallel execution, batching, and a programmatic "fast-lane" for immediate human operator triggers.
- **Strategic Goal Orchestration**: The `GoalManager` coordinates high-level planning, generates structured world model updates, defines objectives, handles failures with localized sub-task decomposition, and directs parallel execution.
- **Resilient ReAct Workers**: The `AgentExecutor` manages single-turn tactical execution. It features automated budget-pressure overrides near iteration limits, resilient multi-format JSON recovery, and automatic truncation of excessively long observations.
- **Episodic Memory Tier**: On startup, past runs are indexed for semantic keyword overlap to prime the agent's current working memory with historical domain maps and beliefs. Summaries are automatically written back to disk on cycle updates.
- **Output Quality Criticism**: A dedicated LLM-based `Critic` grades sub-task answers on a quantitative scale ($0.0$ to $1.0$), logging scores and reasoning directly into the world model findings.
- **Discord Selfbot Integration**: Built on `discord.py-self`, the platform handles dual-mode routing: fast-laning direct commands from a designated operator while replying to neutral channel participants as background processes run.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Presentation Layer                       │
│                                                                  │
│   Web UI (FastAPI)     │    Terminal CLI (Rich)   │  Discord Bot │
└─────────────────┬────────────────────────────────────────────────┘
                  │ Submits Triggers
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                          Scheduler Layer                         │
│                                                                  │
│   TriggerStore (In-Memory Queue)  ◄──►  Background Daemon        │
│   EventBus (SSE Pub/Sub Multiplexer)                             │
└─────────────────┬────────────────────────────────────────────────┘
                  │ Routes to strategic / tactical runners
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                            Agent Layer                           │
│                                                                  │
│   GoalManager (Strategic)     ◄──►   AgentExecutor (Tactical)    │
│   Critic (Score Evaluation)   ◄──►   Episodic Memory (Store/     │
│                                                       Writer)    │
└─────────────────┬────────────────────────────────────────────────┘
                  │ Invokes tools or upstream calls
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Integration Layer                        │
│                                                                  │
│   UpstreamService (httpx2 + retries)  │  Tool Registry           │
└──────────────────────────────────────────────────────────────────┘
```

### Layer 1 — Integration & Transport

All upstream LLM and external API communication is managed through standard clients and retry schemas:

- **`integrations/llm/config.py`**: Validates parameters and handles fast-failing initialization via `pydantic-settings`.
- **`integrations/llm/client.py`**: Exposes a shared, connection-pooled async `httpx2` client.
- **`integrations/llm/service.py`**: The `UpstreamService` wraps chat and streaming completion methods with a `tenacity` policy configured for exponential backoff on network errors, timeouts, or transient HTTP codes ($429$, $5xx$).

### Layer 2 — Scheduler & Triggers

Rather than starting execution on direct UI threads, Noesis decouples input sources using a scheduler pattern:

- **`triggers/store.py` (`TriggerStore`)**: Holds in-memory `Trigger` models. It supports "bunching" (merging rapid-fire consecutive inputs from the same channel into a single execution context) and exposes a fast-lane event loop flag (`human_ready`).
- **`triggers/daemon.py`**: A persistent background service. It polls at a regular interval or wakes up instantly when operator inputs are flagged, routing tasks based on their origin.
- **`utils/event_bus.py`**: An async pub/sub multiplexer. Subscribed endpoints (such as SSE streams or Discord) listen to this bus to broadcast raw thought blocks, tool starts, and final answers.

### Layer 3 — Agent Core

The planning and processing engines operate under strict structural guidelines:

- **`agents/schemas.py`**: Houses Pydantic models for the system state (e.g., `GoalState`, `SubTask`, `Objective`, `WorldModel`). It implements a resilient `@model_validator` that maps varying model field structures (like mapping single `tool_call` payloads into plural `tool_calls` lists).
- **`agents/executor.py` (`AgentExecutor`)**: Resolves concrete, isolated objectives. Features include:
  - Parallel execution of tool calls utilizing `asyncio.gather`.
  - Step budget reminders injected system-side when iteration limits approach.
  - Hard character-truncation thresholds on tool results to prevent context overflow.
- **`agents/goal_manager.py` (`GoalManager`)**: Manages the multi-cycle, global objective loop. It maintains a structured world model (mapping domains, tracking gaps, and weighting beliefs) and decomposes failed or blocked sub-tasks into simpler, smaller actions.
- **`agents/critic.py`**: A validation step that prompts an independent LLM evaluation flow to assign a qualitative rating and reasoning block to completed sub-task findings.
- **`agents/memory/`**:
  - `episodic_store.py`: Performs keyword-based retrieval over run summaries stored in historical log directories on startup to seed the current agent's world model.
  - `episodic_writer.py`: Serializes current state, objective statuses, beliefs, and cycle summaries into a `summary.json` file on each loop update.

### Layer 4 — Interfaces

Interfaces act strictly as visual presentation layers and trigger submitters:

- **`interfaces/cli/`**: Uses `rich` elements to display live thoughts, tool executions, and state tables. It runs an asynchronous task in the background to listen for keyboard inputs, allowing mid-run objective refinements or graceful shutdown commands (`stop`, `halt`).
- **`interfaces/discord/`**: Exposes a user-account bot utilizing `discord.py-self`. It fast-lanes direct prompts from a designated administrator while processing low-priority context queues for other channel participants.

---

## How It Works (Execution & Routing)

### The Trigger Lifecycle

When a query is submitted via the CLI, Discord, or the Web API, it follows a structured path to completion:

```
[User Input]
     │
     ▼
TriggerStore.submit() ──(If Source: human/executor)──► Set human_ready Event
     │                                                      │
     ▼                                                      ▼
[Daemon Poll Wakeup] ◄───────────────────────────────── Daemon Instant Wakeup
     │
     ├─► Trigger.source in ("human", "discord") ──► Spawn GoalManager (Multi-Cycle)
     │                                                ├─ Retreive Episodic Memory
     │                                                ├─ Ask LLM for ManagerDecision
     │                                                ├─ Spawn parallel AgentExecutors
     │                                                └─ Critic Evaluation & Summary Write
     │
     └─► Trigger.source == "executor" (or other) ──► Spawn AgentExecutor (Single-Turn)
                                                      └─ Fast ReAct Loop
```

1. **Submission**: The user input is converted into a `Trigger` and placed in the in-memory queue.
2. **Scheduling**: If the trigger source is labeled `"human"` or `"executor"`, the daemon's wait condition is bypassed.
3. **Routing**:
   - Complex prompts (`"human"`, `"discord"`) initiate a **`GoalManager`** run. The manager performs memory alignment, plans objectives, delegates tasks to concurrent executors, and runs critic reviews.
   - Simplified queries or fast-path operations (e.g., direct Discord command prefixes) bypass the manager entirely and route to a single-turn **`AgentExecutor`** to return an immediate reply.
4. **Broadcasting**: Internal states (thoughts, actions, and observations) are captured by decorators and sent to the `EventBus` to update all active interfaces.

### Specialized Executor Profiles

To minimize the execution of unnecessary commands and prevent the agent from using conflicting tools, the system implements specialized `ExecutorType` profiles. When the `GoalManager` spawns sub-tasks, it assigns a specific profile that limits the available tool registry:

| Executor Type | Permitted Tools | Best Used For |
| :--- | :--- | :--- |
| **`RESEARCH`** | `web_search` | Target-specific online information retrieval and document discovery. |
| **`CODE`** | `python_execute`, `run_command` | Sandboxed mathematical calculations, local scripting, and system queries. |
| **`SYNTHESIS`** | *No tools* | Pure logical reasoning, context compilation, formatting, and drafting. |
| **`FULL`** | All registered tools | Open-ended agent tasks requiring general access to all capabilities. |

---

## Project Structure

```
Agent/
├── main.py                      # FastAPI web server entrypoint
├── run_cli.py                   # Rich-based Terminal UI entrypoint
├── run_discord.py               # Discord bot entrypoint
├── pyproject.toml               # Poetry/pip/uv dependencies config
├── .env.example                 # Environment configuration template
│
├── agents/
│   ├── executor.py              # Tactical single-turn ReAct worker
│   ├── goal_manager.py          # Strategic multi-cycle orchestrator
│   ├── critic.py                # Quantitative sub-task evaluation
│   ├── schemas.py               # Pydantic data schemas
│   ├── tools.py                 # Tool registration & specialized profiles
│   └── memory/                  # Episodic memory utilities
│       ├── episodic_store.py    # Reads historical runs for context seeding
│       └── episodic_writer.py   # Serializes run summaries to disk
│
├── integrations/
│   └── llm/
│       ├── client.py            # httpx2 shared connection factory
│       ├── config.py            # Pydantic Settings validator
│       ├── schemas.py           # Core payload schemas
│       └── service.py           # Upstream API client with tenacity policies
│
├── triggers/
│   ├── store.py                 # In-memory queue storage & buncher
│   └── daemon.py                # Background polling & routing loop
│
├── utils/
│   ├── tracer.py                # ContextVar trace tree & decorator logs
│   ├── ssl_patch.py             # Global SSL verification bypass helper
│   ├── json_parser.py           # Resilient LLM JSON cleaner
│   ├── llm_log_formatter.py     # Request/response log formatting
│   └── logging_setup.py         # Global logging outputs (agent/trace/gm/llm)
│
└── logs/                        # Automatically created at runtime
    ├── agent.log                # Verbose metadata system log
    ├── trace.log                # Text-rendered trace trees
    ├── goal_manager.log         # Manager decisions & cycle statuses
    ├── daemon.log               # Daemon poll ticks & scheduling logs
    └── runs/                    # Ephemeral historical summaries
        └── <timestamp>_<id>/
            ├── summary.json     # Episodic memory snapshot
            └── task-N_<name>.log # Individual worker activity
```

---

## Setup & Configuration

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

### 1. Install Dependencies

Using `uv` (recommended):
```bash
uv sync
```

Or using standard `pip`:
```bash
pip install -r pyproject.toml
```

### 2. Configure Environment Variables

Copy the template configuration:
```bash
cp .env.example .env
```

Open `.env` and fill in the required parameters:
```env
# Upstream OpenAI-compatible LLM endpoint
UPSTREAM_API_URL=https://your-llm-provider.com/v1
API_KEY=sk-your-provider-key-here

# Optional: Enables web_search tool
TAVILY_API_KEY=tvly-your-tavily-key-here

# Optional: Enables Discord Interface
DISCORD_BOT_TOKEN=your-discord-user-token
DISCORD_HUMAN_USER=your_username

# Optional: GitHub Tool Credentials
GITHUB_TOKEN=your-github-token-here

# Global CLI/Console Model selection
AGENT_MODEL=groq/openai/gpt-oss-120b
```

---

## Running Noesis

### Terminal (CLI) Mode

Start the interactive terminal UI:
```bash
uv run python run_cli.py
```
- Enter your goal to start the autonomous cycle.
- While the agent is running, type any text and press Enter to inject a real-time goal refinement, or type `stop` to gracefully halt the loop.

### Web Server & Daemon

To run the web interface alongside the background scheduler, launch the FastAPI app:
```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
The FastAPI lifespan handles starting the background daemon (`start_daemon`) automatically.

### Discord Selfbot Mode

Ensure the `DISCORD_BOT_TOKEN` (the user token) and `DISCORD_HUMAN_USER` environment variables are set, then run:
```bash
uv run python run_discord.py
```
- The bot will monitor DMs and Group DMs.
- Direct prompts from the designated human operator are automatically queued in the `TriggerStore` as strategic runs. Prefixing commands with `!` bypasses the manager for a fast, single-turn response.

---

## Observability & Debugging

Noesis splits runtime logs into specialized destinations inside the `logs/` directory to simplify debugging:

### Log Files Reference

| Target Log File | Logging Scope | Best Used For |
| :--- | :--- | :--- |
| **`logs/agent.log`** | Standard metadata formatting (`INFO` and up) across all files. | Deep debugging of internal errors, lines, and system exceptions. |
| **`logs/trace.log`** | Raw trace outputs. Uses box-drawing characters without standard timestamps. | Visualizing nesting patterns, execution times, and sequential tool chains. |
| **`logs/goal_manager.log`** | High-level status updates of the strategic loop. | Checking cycle boundaries, planning decisions, and task completions. |
| **`logs/daemon.log`** | Background scheduler activity. | Monitoring queue additions, polling triggers, and routing. |
| **`logs/llm.log`** | Verbatim raw request and response completion details. | Evaluating raw system prompts, chat history, and generated outputs. |

### Visual Trace Output

The system formats execution steps hierarchically inside `logs/trace.log` using standard ASCII box-drawing characters:

```
┌─[TRACE] id=4a2f8c1b
│
│  Query: Find the current definition of gravity
│
├──[SPAN] ▶  goal_manager[cycle=1]  model=gpt-oss-120b
│    └──[RESULT]  [ok]  ▸ 1.25s
├──[SPAN] ▶  tool:web_search  input=definition of gravity
│    └──[RESULT]  [ok]  ▸ 890ms  result_len=1420 chars
└─[DONE]  id=4a2f8c1b  total ▸ 2.14s  cycles=1  tasks=1
```

You can tail any log destination in real-time to monitor the system:
```bash
tail -f logs/trace.log
```