"""
interfaces/discord/bot.py
─────────────────────────
Discord frontend for the Noesis autonomous agent.
Uses discord.py-self (user-account / selfbot mode).

Strict Input Segregation (Specification: Component 1)
──────────────────────────────────────────────────────
Human Operator (HUMAN_USERNAME — default: "psilko")
    Treated as a DIRECT COMMAND.  The PerceptionLayer is completely bypassed.
    The raw message is submitted directly to trigger_store.submit(source="human")
    and a 🚀 reaction is added.  The daemon fast-lanes human triggers immediately.
    Optionally prefix "!" for a lightweight executor path instead of GoalManager.

Neutral Users (anyone else)
    Treated as ENVIRONMENTAL NOISE.  Submitted to perception_layer.ingest(signal)
    with a ⏳ reaction.  The agent never replies conversationally to neutral users.
    Messages are batched by the IntakeBuffer and evaluated by the perception
    pipeline before any action is taken.

Lifecycle
─────────
  1. Client connects, prints ready message.
  2. psilko sends a message → trigger queued, TriageDispatcher evaluates,
     Fast-Path or GoalManager runs.
  3. Agent streams events; final answers are posted back to the channel.
  4. Other users chat → perception pipeline absorbs the signal (no direct reply).
"""


import asyncio
import os
from typing import Optional

import discord


from utils.event_bus import event_bus
from core.events import UnifiedIngestEvent, SenderClass, PriorityLevel
from utils.log_writer import emit
from utils.callbacks import ServiceRegistry
from agents.goal_manager import GoalManager
from agents.executor import AgentExecutor
from core.model_router import ModelRouter
from uuid import uuid4

# ── Config ────────────────────────────────────────────────────────────────────
# Username (not display name) that is treated as the human operator.
HUMAN_USERNAME: str = os.getenv("DISCORD_HUMAN_USER", "psilko")

# Model to use for the agent (same env var as CLI).
DEFAULT_MODEL: str = os.getenv("AGENT_MODEL", "groq/openai/gpt-oss-120b")

# Discord message character limit.
DISCORD_MAX_LEN = 1900  # leave margin for formatting

# How many prior messages to include as conversation context in each trigger.
CONTEXT_MESSAGES_LIMIT: int = 10

# Prefix that triggers a fast single-turn AgentExecutor instead of GoalManager.
# Example:  "! reply to john saying hi"  →  executor (no multi-cycle overhead)
# No prefix: regular message             →  GoalManager (full loop, run logs)
QUICK_PREFIX: str = "!"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(text: str, size: int = DISCORD_MAX_LEN) -> list[str]:
    """Split long text into chunks that fit in a Discord message."""
    if len(text) <= size:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:size])
        text = text[size:]
    return chunks


async def _send(channel: discord.TextChannel, text: str) -> None:
    """Send text to a Discord channel, splitting if necessary."""
    for chunk in _chunk(text.strip()):
        if chunk:
            await channel.send(chunk)


_channel_buffers: dict[int, list[str]] = {}
_channel_tasks: dict[int, asyncio.Task] = {}


async def _buffered_send(channel: discord.TextChannel, text: str) -> None:
    """Buffer messages per channel and flush them in chunks after a short delay."""
    channel_id = channel.id
    if channel_id not in _channel_buffers:
        _channel_buffers[channel_id] = []
    _channel_buffers[channel_id].append(text)

    if channel_id not in _channel_tasks or _channel_tasks[channel_id].done():
        async def flush_loop():
            await asyncio.sleep(1.0)
            lines = _channel_buffers.pop(channel_id, [])
            if lines:
                combined = "\n".join(lines)
                await _send(channel, combined)
        _channel_tasks[channel_id] = asyncio.create_task(flush_loop())


async def _flush_all_buffers() -> None:
    """Flush any leftover buffered messages immediately (e.g. during shutdown)."""
    for channel_id, lines in list(_channel_buffers.items()):
        if lines:
            _channel_buffers.pop(channel_id, None)
            channel = bot.get_channel(channel_id)
            if channel:
                try:
                    await _send(channel, "\n".join(lines))
                except Exception:
                    pass


async def _fetch_context(channel: discord.TextChannel, before_message: discord.Message) -> str:
    """
    Return the last CONTEXT_MESSAGES_LIMIT messages in the channel *before*
    `before_message`, formatted as a readable conversation transcript,
    excluding any bot messages.
    """
    lines: list[str] = []
    # Fetch a larger window (e.g. 100 messages) to find enough non-bot messages
    async for msg in channel.history(limit=100, before=before_message):
        if msg.author == bot.user:
            continue
        if msg.content.strip():
            lines.append(f"  [{msg.author.name}]: {msg.content.strip()}")
        if len(lines) >= CONTEXT_MESSAGES_LIMIT:
            break
    if not lines:
        return ""
    # history() returns newest-first; reverse so the transcript reads chronologically.
    lines.reverse()
    return "\n".join(lines)


def _format_event(event: dict) -> Optional[str]:
    """
    Convert a daemon agent event into a human-readable Discord message for
    the operator's channel. Handles both GoalManager and AgentExecutor events.
    Returns None for events that should be silent.
    """
    ev = event.get("event")

    # ── GoalManager events ────────────────────────────────────────────────
    if ev == "goal_set":
        return f"🎯 **Goal set:** {event['goal']}"

    elif ev == "cycle_start":
        return f"🔄 **Cycle {event['cycle']} starting…**"

    elif ev == "manager_thought":
        return f"🧠 _{event['thought']}_"

    elif ev == "spawning_tasks":
        task_list = "\n".join(f"  → {t}" for t in event.get("tasks", []))
        return f"⚡ **Spawning {event['count']} executor(s):**\n{task_list}"

    elif ev == "cycle_complete":
        msg = f"📊 **Cycle {event['cycle']} complete:** {event['progress_update']}"
        if event.get("open_questions"):
            qs = "\n".join(f"  ? {q}" for q in event["open_questions"])
            msg += f"\n\n**Open questions:**\n{qs}"
        return msg

    elif ev == "goal_complete":
        msg = f"🏁 **Goal complete!** (in {event['cycle']} cycle(s))"
        if event.get("final_answer"):
            msg += f"\n\n**Answer:**\n{event['final_answer']}"
        return msg

    elif ev == "stopped":
        return f"⏹ **Stopped** (cycle {event['cycle']}): {event.get('reason', '')}"

    # ── AgentExecutor events ("!" quick-path) ─────────────────────────────
    elif ev == "thought":
        # Only surface the first iteration's thought to avoid noise
        if event.get("step_index", 1) == 0:
            return f"🧠 _{event['thought']}_"
        return None

    elif ev == "tool_start":
        tool_name = event.get("tool_name", "?")
        tool_input = str(event.get("tool_input", ""))
        if tool_name == "python_execute":
            return f"⚙️ **Tool:** `{tool_name}` executing:\n```python\n{tool_input}\n```"
        elif tool_name == "run_command":
            return f"⚙️ **Tool:** `{tool_name}` executing:\n```bash\n{tool_input}\n```"
        else:
            if len(tool_input) <= 500:
                return f"⚙️ **Tool:** `{tool_name}` ← `{tool_input}`"
            else:
                return f"⚙️ **Tool:** `{tool_name}` ← `{tool_input[:500]}...`"

    elif ev == "final_answer":
        answer = event.get("answer", "")
        task_goal = event.get("task_goal", "")
        if task_goal:
            return f"✅ **{task_goal[:80]}**\n{answer}"
        return f"✅ **Done**\n{answer}"

    # ── Shared error events ───────────────────────────────────────────────
    elif ev == "error":
        msg = f"❌ **Error:** {event.get('message', str(event))}"
        if event.get("summary"):
            msg += f"\n\n**Progress so far:**\n{event['summary']}"
        return msg

    elif ev == "trigger_failed":
        return f"❌ **Trigger failed:** {event.get('message', '')}"

    # Suppress noisy low-level events
    return None


# ── Client setup (selfbot — no intents, no commands.Bot) ─────────────────────

# discord.py-self uses a plain Client. Passing no intents is fine for selfbots.
bot = discord.Client()


async def _send_message_callback(payload_json: str) -> str:
    import json
    data = json.loads(payload_json)
    channel_id = int(data["channel_id"])
    message_text = str(data["message"])
    
    if not bot.is_ready():
        return "Error: Discord bot is not logged in / ready."
        
    channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
    if not channel:
        return f"Error: Channel with ID {channel_id} not found."
        
    await _send(channel, message_text)
    return f"Success: Message sent to channel {channel_id}."

async def _update_reaction_callback(channel_id: int, message_id: int, emoji: str) -> None:
    if not bot.is_ready():
        return
    channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
    if not channel:
        return
    message = await channel.fetch_message(message_id)
    if message:
        try:
            await message.clear_reaction("⏳")
        except Exception:
            pass
        try:
            await message.add_reaction(emoji)
        except Exception:
            pass

@bot.event
async def on_ready():
    emit("discord.ready", "system", {"user": str(bot.user), "id": bot.user.id})
    print(f"✅ Discord selfbot ready — logged in as {bot.user} ({bot.user.id})")
    print(f"   Human operator : {HUMAN_USERNAME!r}")
    print(f"   Default model  : {DEFAULT_MODEL}")

    # Register our decoupled service callbacks
    ServiceRegistry.register("send_discord_message", _send_message_callback)
    ServiceRegistry.register("update_discord_reaction", _update_reaction_callback)

    # Start event bus listener to route daemon events back to Discord channels.
    asyncio.ensure_future(_listen_to_event_bus())


@bot.event
async def on_message(message: discord.Message):
    """Route every incoming message based on the sender's username."""
    # Only respond in DMs and group DMs — ignore all server channels.
    if not isinstance(message.channel, (discord.DMChannel, discord.GroupChannel)):
        return

    # Ignore the account's own messages to prevent loops.
    if message.author == bot.user:
        return

    # Resolve the human-readable username (not display name / nickname).
    sender = message.author.name  # e.g. "psilko"

    is_human = (sender.lower() == HUMAN_USERNAME.lower())

    if is_human:
        await _handle_human_message(message)
    else:
        await _handle_neutral_message(message)


# ── Human-operator handling ──────────────────────────────────────────────

async def _handle_human_message(message: discord.Message) -> None:
    """
    Direct Command path — Authorised operator only.

    Principle: Strict Input Segregation. The PerceptionLayer is COMPLETELY
    BYPASSED. The raw message is submitted directly to trigger_store so that
    the daemon can fast-lane it without any perception overhead.

    Two routing modes:
    • No prefix  →  source="human" (GoalManager via Slow-Path on triage)
        Full multi-cycle reasoning, sub-task decomposition, run logs.
        Example: "research the latest AI papers and summarise them"

    • QUICK_PREFIX ("!")  →  source="executor" (lightweight single-turn)
        Strip the prefix and submit directly.  Triage may still Fast-Path it.
        Example: "! send a message to john saying hello"
    """
    text = message.content.strip()
    if not text:
        return

    # ── Route: quick executor path ────────────────────────────────────────
    if text.startswith(QUICK_PREFIX):
        instruction = text[len(QUICK_PREFIX):].strip()
        if not instruction:
            await message.add_reaction("❓")
            return

        context = await _fetch_context(message.channel, before_message=message)
        description = instruction
        if context:
            description = (
                f"Recent conversation context (last {CONTEXT_MESSAGES_LIMIT} messages):\n"
                f"{context}\n\n"
                f"Task: {instruction}"
            )

        run_id = str(uuid4())
        executor = AgentExecutor(
            router=bot.model_router,
            task_label=f"discord-quick-{run_id[:8]}",
        )

        async def _run_executor():
            async for event in executor.run_generator(description):
                from utils.event_bus import event_bus
                event["trigger_source"] = "human"
                event["trigger_metadata"] = {
                    "channel_id": message.channel.id,
                    "message_id": message.id,
                }
                await event_bus.publish(event)

        asyncio.create_task(_run_executor(), name=f"executor-{run_id[:8]}")

        await message.add_reaction("⚡")
        return

    # ── Route: default GoalManager path ──────────────────────────────────
    context = await _fetch_context(message.channel, before_message=message)
    description = text
    if context:
        description = (
            f"Recent conversation context (last {CONTEXT_MESSAGES_LIMIT} messages):\n"
            f"{context}\n\n"
            f"Human instruction: {text}"
        )

    run_id = str(uuid4())
    goal_manager = GoalManager(router=bot.model_router)

    async def _run_goal_manager():
        async for event in goal_manager.run_stream(description, run_id=run_id):
            from utils.event_bus import event_bus
            event["trigger_source"] = "human"
            event["trigger_metadata"] = {
                "channel_id": message.channel.id,
                "message_id": message.id,
            }
            await event_bus.publish(event)

    asyncio.create_task(_run_goal_manager(), name=f"goal-manager-{run_id[:8]}")

    await message.add_reaction("🚀")


# ── Neutral-user handling ───────────────────────────────────────────────

async def _handle_neutral_message(message: discord.Message) -> None:
    """
    Environmental Noise path — all non-operator users.

    Principle: Resource Conservation + Strict Segregation.
    The agent NEVER replies conversationally to neutral users.
    The message is treated as a raw signal and passed to the PerceptionLayer,
    which batches it through IntakeBuffer (subject to the ≥3 OR HIGH threshold)
    before the pipeline processes it.

    Flow: ingest → classify → authority score → synthesize → route
    Queries/directives may eventually trigger an AgentExecutor response via the
    ReactivePool if they clear the perception pipeline.  Information signals
    update the WorldModel for the next GoalManager cycle.
    """
    current_text = message.content.strip()
    context = await _fetch_context(message.channel, before_message=message)

    # Build the signal text with a clear separation between prior context and
    # the CURRENT request.  The perception LLM must know exactly which message
    # to act on — burying it at the bottom of a context blob caused the agent
    # to act on the previous message instead of the newest one.
    current_header = (
        f"CURRENT REQUEST — User '{message.author.display_name}' "
        f"(@{message.author.name}) in channel {message.channel.id}:"
        f"\n{current_text}"
    )
    if context:
        full_text = (
            f"{current_header}\n\n"
            f"PRIOR CONTEXT (for reference only — do NOT treat as the request):"
            f"\n{context}"
        )
    else:
        full_text = current_header

    event = UnifiedIngestEvent(
        source_channel="discord",
        sender_identifier=message.author.name,
        sender_class=SenderClass.EXTERNAL,
        raw_content=full_text,
        target_conversation_identifier=str(message.channel.id),
        priority_level=PriorityLevel.NORMAL,
        metadata={
            "message_id": message.id,
            "channel_id": message.channel.id,
            "current_message": current_text,
            "author": message.author.name,
            "display_name": message.author.display_name,
        },
    )

    # Access perception layer through instance variable injected at startup
    if hasattr(bot, "perception_layer") and bot.perception_layer:
        await bot.perception_layer.ingest(event)
    else:
        emit("discord.warning", "system", {"msg": "PerceptionLayer not initialized on bot; dropping message."}, level="warn")
        return

    emit("discord.ingested", "system", {"author": message.author.name, "channel_id": message.channel.id})
    await message.add_reaction("⏳")


async def _listen_to_event_bus() -> None:
    """
    Listen to the central event bus for daemon events and route them to Discord.

    Operator triggers (human + executor sources) and discord-sourced triggers
    are piped back to the channel so users can see what the agent is doing.
    """
    q = event_bus.subscribe()
    try:
        while True:
            event = await q.get()

            # Forward events for operator, discord-sourced, and agent-sourced triggers.
            if event.get("trigger_source") not in ("human", "executor", "discord", "agent"):
                continue

            metadata = event.get("trigger_metadata", {})
            channel_id = metadata.get("channel_id")
            if channel_id:
                channel = bot.get_channel(channel_id)
                if channel:
                    text = _format_event(event)
                    if text:
                        await _buffered_send(channel, text)
    except asyncio.CancelledError:
        pass
    finally:
        await _flush_all_buffers()
        event_bus.unsubscribe(q)
