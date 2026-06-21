"""
interfaces/discord/bot.py
─────────────────────────
Discord frontend for the Noesis autonomous agent.
Uses discord.py-self (user-account / selfbot mode).

Dual-mode input routing
───────────────────────
• Messages from the HUMAN_USERNAME (default: "psilko")
    → treated as HUMAN INPUT → submitted to the trigger store which runs an
      AgentExecutor in the background daemon.

• Messages from anyone else in the same channel
    → treated as NEUTRAL CONTEXT → the agent responds autonomously to them
      (one-shot executor reply), i.e. the agent *talks back* like a normal
      participant while continuing any in-flight work.

Lifecycle
─────────
  1. Client connects, prints ready message.
  2. psilko sends a message in any channel → trigger queued, executor runs.
  3. Agent streams events; final answers are posted back to the channel.
  4. Other users chat → agent replies with a quick autonomous response.
"""

import asyncio
import logging
import os
from typing import Optional

import discord

from integrations.llm.client import get_client
from integrations.llm.service import UpstreamService
from integrations.llm.schemas import ChatMessage, ChatPayload
from triggers.store import trigger_store
from utils.event_bus import event_bus

logger = logging.getLogger("noesis.discord")

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
    `before_message`, formatted as a readable conversation transcript.
    """
    lines: list[str] = []
    async for msg in channel.history(limit=CONTEXT_MESSAGES_LIMIT, before=before_message):
        if msg.author == bot.user:
            author_tag = "[Agent]"
        else:
            author_tag = f"{msg.author.display_name} (@{msg.author.name})"
        if msg.content.strip():
            lines.append(f"  [{author_tag}]: {msg.content.strip()}")
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

# Shared LLM client/service — initialised in on_ready so the event loop is live.
_http_client_ctx = None
_service: Optional[UpstreamService] = None


@bot.event
async def on_ready():
    global _http_client_ctx, _service
    logger.info("Discord selfbot logged in as %s (id=%s)", bot.user, bot.user.id)
    print(f"✅ Discord selfbot ready — logged in as {bot.user} ({bot.user.id})")
    print(f"   Human operator : {HUMAN_USERNAME!r}")
    print(f"   Default model  : {DEFAULT_MODEL}")

    # Open a persistent httpx client for the lifetime of the bot.
    _http_client_ctx = get_client(timeout=60.0)
    http_client = await _http_client_ctx.__aenter__()
    _service = UpstreamService(http_client)

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


# ── Human-operator handling ───────────────────────────────────────────────────

async def _handle_human_message(message: discord.Message) -> None:
    """
    Messages from the designated human operator.  Two routing modes:

    • No prefix  →  GoalManager (source="human")
        Full multi-cycle reasoning, sub-task decomposition, run logs.
        The daemon fast-lanes human triggers immediately.
        Example: "research the latest AI papers and summarise them"

    • QUICK_PREFIX ("!")  →  AgentExecutor (source="executor")
        Lightweight single-turn execution — no GoalManager overhead.
        Strip the prefix, run directly, reply fast.
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

        trigger_store.submit(
            source="executor",
            description=description,
            model=DEFAULT_MODEL,
            metadata={"channel_id": message.channel.id, "message_id": message.id},
        )
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

    trigger_store.submit(
        source="human",
        description=description,
        model=DEFAULT_MODEL,
        metadata={"channel_id": message.channel.id, "message_id": message.id},
    )
    await message.add_reaction("🚀")


# ── Neutral-user handling ─────────────────────────────────────────────────────

async def _handle_neutral_message(message: discord.Message) -> None:
    """
    Messages from anyone who is NOT the human operator (psilko).
    Includes recent channel history so the agent understands the conversation.
    Submit to the daemon's trigger store (picked up on the next 60-second poll).
    """
    context = await _fetch_context(message.channel, before_message=message)
    context_section = (
        f"Recent conversation context (last {CONTEXT_MESSAGES_LIMIT} messages):\n{context}\n\n"
        if context else ""
    )
    description = (
        f"Ultimate Goal: Be a highly skilled social engineer that wants to get people to help you.\n\n"
        f"{context_section}"
        f"In Discord channel {message.channel.id}, user '{message.author.display_name}' (@{message.author.name}) said:\n"
        f"{message.content}\n\n"
        f"Please reply to them in channel {message.channel.id} using the send_discord_message tool."
    )
    bunch_key = f"discord_{message.channel.id}_{message.author.id}"
    trigger_store.submit(
        source="discord",
        description=description,
        model=DEFAULT_MODEL,
        metadata={"channel_id": message.channel.id, "message_id": message.id},
        bunch_key=bunch_key,
    )
    logger.info(
        "[Discord] Queued daemon trigger for neutral message from %s in channel %d.",
        message.author.name, message.channel.id,
    )
    await message.add_reaction("⏳")


async def _listen_to_event_bus() -> None:
    """
    Listen to the central event bus for daemon events and route them to Discord.

    Operator triggers (human + executor sources) are piped back to the channel
    so the operator sees what the agent is doing.

    Neutral / discord-sourced triggers are intentionally silent here — the agent
    already replies to those users via the send_discord_message tool directly.
    Leaking the internal state to a random user would expose admin-only info.
    """
    q = event_bus.subscribe()
    try:
        while True:
            event = await q.get()

            # Forward events for both operator-sourced trigger types.
            if event.get("trigger_source") not in ("human", "executor"):
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
