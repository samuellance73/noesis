"""
agents/tools/reddit.py
──────────────────────
Reddit tools implementation.
"""

import json
from .registry import tools_registry


@tools_registry.register(
    "send_reddit_reply",
    description=(
        "Reply to a specific Reddit comment, submission, or message. "
        "Input must be a JSON string with 'item_id' (string, the ID without prefix), "
        "'item_type' (string, one of: 'comment', 'submission', 'message'), and "
        "'body' (string, the text of the reply)."
    ),
)
async def send_reddit_reply(payload_json: str) -> str:
    from utils.callbacks import ServiceRegistry
    try:
        data = json.loads(payload_json)
        return await ServiceRegistry.call(
            "send_reddit_reply",
            item_id=data["item_id"],
            item_type=data["item_type"],
            body=data["body"],
        )
    except Exception as e:
        return f"Error sending Reddit reply: {e}"


@tools_registry.register(
    "create_reddit_post",
    description=(
        "Create a new self-post (submission) in a subreddit. "
        "Input must be a JSON string with 'subreddit' (string), "
        "'title' (string), and 'body' (string)."
    ),
)
async def create_reddit_post(payload_json: str) -> str:
    from utils.callbacks import ServiceRegistry
    try:
        data = json.loads(payload_json)
        return await ServiceRegistry.call(
            "create_reddit_post",
            subreddit_name=data["subreddit"],
            title=data["title"],
            body=data["body"],
        )
    except Exception as e:
        return f"Error creating Reddit post: {e}"


@tools_registry.register(
    "fetch_reddit_posts",
    description=(
        "Fetch recent posts (submissions) from a specific subreddit. "
        "Input must be a JSON string with optional 'subreddit' (string, defaults to self_subreddit), "
        "optional 'limit' (integer between 1 and 25, defaults to 5), and "
        "optional 'listing' (string, one of: 'new', 'hot', 'top', 'rising', 'controversial', defaults to 'hot')."
    ),
)
async def fetch_reddit_posts(payload_json: str) -> str:
    from utils.callbacks import ServiceRegistry
    try:
        data = json.loads(payload_json) if payload_json else {}
        subreddit = data.get("subreddit", "")
        limit = data.get("limit", 5)
        listing = data.get("listing", "hot")
        return await ServiceRegistry.call(
            "fetch_reddit_posts",
            subreddit=subreddit,
            limit=limit,
            listing=listing,
        )
    except Exception as e:
        return f"Error fetching Reddit posts: {e}"

