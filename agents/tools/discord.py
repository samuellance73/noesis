"""
agents/tools/discord.py
──────────────────────
Discord message tool implementation.
"""

from .registry import tools_registry


@tools_registry.register(
    "send_discord_message",
    description="Send a message to a specific Discord channel. Input must be a JSON string with 'channel_id' (integer) and 'message' (string), e.g. {\"channel_id\": 123456789, \"message\": \"Hello!\"}",
)
async def send_discord_message(payload_json: str) -> str:
    from utils.callbacks import ServiceRegistry
    try:
        # Simply delegate execution to the service registry
        return await ServiceRegistry.call("send_discord_message", payload_json)
    except Exception as e:
        return f"Error sending message: {e}"
