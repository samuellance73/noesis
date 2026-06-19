"""
interfaces/discord/bot.py
─────────────────────────
Discord frontend for the Noesis autonomous agent.
Uses discord.py-self (user-account / selfbot mode).

Dual-mode input routing
───────────────────────
• Messages from the HUMAN_USERNAME (default: "psilko")
    → treated as HUMAN INPUT → injected into the active GoalManager via
      inject_input() or, if no run is active, starts a brand-new run.

• Messages from anyone else in the same channel
    → treated as NEUTRAL CONTEXT → the agent responds autonomously to them
      (one-shot executor reply), i.e. the agent *talks back* like a normal
      participant while continuing its main goal loop.

Lifecycle
─────────
  1. Client connects, prints ready message.
  2. psilko sends a message in any channel → starts GoalManager loop.
  3. Agent streams events; final/cycle answers are posted back to the channel.
  4. Other users chat → agent replies with a quick autonomous response.
  5. psilko types "stop" / "halt" / "quit" → halts the loop.
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
    Convert a GoalManager event dict into a human-readable Discord message.
    Returns None for events that should be silent.
    """
    ev = event.get("event")

    if ev == "goal_set":
        return f"🎯 **Goal set:** {event['goal']}"

    elif ev == "cycle_start":
        return f"🔄 **Cycle {event['cycle']} starting...**"

    elif ev == "manager_thought":
        return f"🧠 _{event['thought']}_"

    elif ev == "spawning_tasks":
        task_list = "\n".join(f"  → {t}" for t in event.get("tasks", []))
        return f"⚡ **Spawning {event['count']} executor(s):**\n{task_list}"

    elif ev == "final_answer":
        task_goal = event.get("task_goal", "Sub-task")
        return f"✅ **{task_goal[:80]}**\n{event['answer']}"

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

    elif ev == "error":
        msg = f"❌ **Error:** {event.get('message', str(event))}"
        if event.get("summary"):
            msg += f"\n\n**Progress so far:**\n{event['summary']}"
        return msg

    elif ev == "user_input_received":
        return f"↩ _Injected:_ {event['message']}"

    # Suppress noisy low-level events by default
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
    Messages from the designated human operator:
    Submit as a human trigger to the daemon immediately (fast-lane).
    Includes the last CONTEXT_MESSAGES_LIMIT messages as conversation context.
    """
    text = message.content.strip()
    if not text:
        return

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

    Only events from HUMAN-sourced triggers are piped back to the channel as
    verbose status updates (goal tracking, cycle info, etc.).

    Neutral / discord-sourced triggers are intentionally silent here — the agent
    already replies to those users via the send_discord_message tool directly.
    Leaking the internal loop state to a random user would expose admin-only info.
    """
    q = event_bus.subscribe()
    try:
        while True:
            event = await q.get()

            # Only forward event-bus chatter for human-operator triggers.
            if event.get("trigger_source") != "human":
                continue

            metadata = event.get("trigger_metadata", {})
            channel_id = metadata.get("channel_id")
            if channel_id:
                channel = bot.get_channel(channel_id)
                if channel:
                    text = _format_event(event)
                    if text:
                        await _send(channel, text)
    except asyncio.CancelledError:
        pass
    finally:
        event_bus.unsubscribe(q)
