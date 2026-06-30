"""
integrations/reddit/poller.py
──────────────────────────────
RedditPoller — async polling loop for Reddit signals.

Runs as a long-lived asyncio task (started in app/lifespan.py).

Supports two modes:
  1. Standard OAuth (PRAW):
     • Gated on client_id/client_secret/refresh_token/username.
     • Streams subreddit posts/comments and polls inbox.
  2. Selfbot (HTTPX):
     • Gated on session_token (token_v2).
     • Polls configured self_subreddit and unread inbox.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import asyncpraw  # type: ignore[import-untyped]
import httpx2

from integrations.reddit.adapter import RedditAdapter
from integrations.reddit.config import RedditConfig
from utils.log_writer import emit
from utils.callbacks import ServiceRegistry
from utils.event_bus import event_bus

if TYPE_CHECKING:
    from perception.layer import PerceptionLayer


_USER_AGENT = "python:noesis-agent:v1.0 (by /u/{username})"

# Global references to allow service callbacks to access active session
_reddit_client: asyncpraw.Reddit | None = None
_selfbot_client: httpx2.AsyncClient | None = None
_selfbot_config: RedditConfig | None = None


# ── Selfbot Helper Stubs ──────────────────────────────────────────────────────

class RedditCommentStub:
    def __init__(self, data: dict):
        self.author = data.get("author")
        self.body = data.get("body", "")
        self.id = data.get("id", "")
        self.permalink = data.get("permalink", "")


class RedditInboxStub:
    def __init__(self, data: dict):
        self.author = data.get("author")
        self.body = data.get("body", "")
        self.id = data.get("id", "")
        self.subject = data.get("subject", "")
        self.subreddit = data.get("subreddit")
        
        subj = self.subject.lower()
        if data.get("was_comment", False):
            if "username mention" in subj:
                self.type = "username_mention"
            else:
                self.type = "comment_reply"
        else:
            self.type = "message"


def _get_auth_header(token: str) -> str:
    if token.startswith("Bearer "):
        return token
    return f"Bearer {token}"


def _encode_form(fields: dict) -> bytes:
    """URL-encode a dict as UTF-8 form bytes (application/x-www-form-urlencoded).

    httpx2's ``data=`` kwarg falls back to ASCII encoding on some builds,
    which chokes on characters like U+2026 (…).  This helper encodes every
    value explicitly as UTF-8 and percent-encodes the result so the wire
    bytes are always safe regardless of the underlying HTTP client version.
    """
    from urllib.parse import urlencode
    return urlencode(
        {k: v.encode("utf-8") if isinstance(v, str) else v for k, v in fields.items()},
        encoding="utf-8",
    ).encode("ascii")  # urlencode output is always ASCII-safe after encoding values


# ── Formatter & Callbacks ─────────────────────────────────────────────────────

def _format_reddit_event(event: dict) -> str | None:
    """Format daemon events into a clean reply text suitable for Reddit."""
    ev = event.get("event")
    if ev == "final_answer":
        return event.get("answer")
    elif ev == "goal_complete":
        return event.get("final_answer")
    elif ev == "error":
        return f"An error occurred while processing the request: {event.get('message', 'Unknown error')}"
    return None


async def _selfbot_send_reply_callback(
    client: httpx2.AsyncClient,
    config: RedditConfig,
    item_id: str,
    item_type: str,
    body: str,
) -> str:
    """Service registry callback to reply in selfbot mode."""
    try:
        thing_id = item_id
        if not (thing_id.startswith("t1_") or thing_id.startswith("t3_") or thing_id.startswith("t4_")):
            if item_type in ("message", "subreddit_message"):
                thing_id = f"t4_{item_id}"
            elif item_type in ("comment", "comment_reply", "username_mention"):
                thing_id = f"t1_{item_id}"
            elif item_type == "submission":
                thing_id = f"t3_{item_id}"

        headers = {
            "Authorization": _get_auth_header(config.session_token),
            "User-Agent": config.user_agent,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }

        payload = _encode_form({
            "api_type": "json",
            "thing_id": thing_id,
            "text": body,
        })

        resp = await client.post("https://oauth.reddit.com/api/comment", content=payload, headers=headers)
        resp.raise_for_status()
        res_json = resp.json()
        
        try:
            new_id = res_json["json"]["data"]["things"][0]["data"]["id"]
            return f"Success: Replied to {item_id} (reply ID: {new_id})."
        except Exception:
            return f"Success: Replied to {item_id}."
    except Exception as e:
        emit("reddit.reply_error", "reddit", {"error": str(e), "item_id": item_id}, level="error")
        return f"Error: {e}"


async def _selfbot_create_post_callback(
    client: httpx2.AsyncClient,
    config: RedditConfig,
    subreddit_name: str,
    title: str,
    body: str,
) -> str:
    """Service registry callback to submit a text post in selfbot mode."""
    try:
        headers = {
            "Authorization": _get_auth_header(config.session_token),
            "User-Agent": config.user_agent,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }

        payload = _encode_form({
            "api_type": "json",
            "kind": "self",
            "sr": subreddit_name,
            "title": title,
            "text": body,
        })

        resp = await client.post("https://oauth.reddit.com/api/submit", content=payload, headers=headers)
        resp.raise_for_status()
        res_json = resp.json()
        
        try:
            things = res_json["json"]["data"]["things"]
            new_id = things[0]["data"]["id"]
            url = things[0]["data"]["url"]
            return f"Success: Created post in r/{subreddit_name} (ID: {new_id}, URL: {url})."
        except Exception:
            return f"Success: Created post in r/{subreddit_name}."
    except Exception as e:
        emit("reddit.post_error", "reddit", {"error": str(e), "subreddit": subreddit_name}, level="error")
        return f"Error: {e}"


async def _send_reply_callback(item_id: str, item_type: str, body: str) -> str:
    """Service registry callback to reply to comments, submissions, or inbox messages."""
    global _reddit_client, _selfbot_client, _selfbot_config
    if _selfbot_client is not None and _selfbot_config is not None:
        return await _selfbot_send_reply_callback(_selfbot_client, _selfbot_config, item_id, item_type, body)
        
    if _reddit_client is None:
        return "Error: Reddit client is not running."
    try:
        if item_type in ("message", "subreddit_message"):
            item = await _reddit_client.inbox.message(item_id)
        elif item_type in ("comment", "comment_reply", "username_mention"):
            item = await _reddit_client.comment(item_id)
        elif item_type == "submission":
            item = await _reddit_client.submission(item_id)
        else:
            return f"Error: Unknown item type '{item_type}'."
        
        reply_obj = await item.reply(body)
        return f"Success: Replied to {item_id} (reply ID: {reply_obj.id})."
    except Exception as e:
        emit("reddit.reply_error", "reddit", {"error": str(e), "item_id": item_id}, level="error")
        return f"Error: {e}"


async def _create_post_callback(subreddit_name: str, title: str, body: str) -> str:
    """Service registry callback to submit a text post to a subreddit."""
    global _reddit_client, _selfbot_client, _selfbot_config
    if _selfbot_client is not None and _selfbot_config is not None:
        return await _selfbot_create_post_callback(_selfbot_client, _selfbot_config, subreddit_name, title, body)

    if _reddit_client is None:
        return "Error: Reddit client is not running."
    try:
        subreddit = await _reddit_client.subreddit(subreddit_name)
        submission = await subreddit.submit(title=title, selftext=body)
        return f"Success: Created post in r/{subreddit_name} (ID: {submission.id}, URL: {submission.url})."
    except Exception as e:
        emit("reddit.post_error", "reddit", {"error": str(e), "subreddit": subreddit_name}, level="error")
        return f"Error: {e}"


async def _selfbot_fetch_posts_callback(
    client: httpx2.AsyncClient,
    config: RedditConfig,
    subreddit: str,
    limit: int,
    listing: str,
) -> str:
    """Service registry callback to fetch recent posts in selfbot mode."""
    import json
    try:
        headers = {
            "Authorization": _get_auth_header(config.session_token),
            "User-Agent": config.user_agent,
        }
        url = f"https://oauth.reddit.com/r/{subreddit}/{listing}.json?limit={limit}"
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        children = data.get("data", {}).get("children", [])
        posts = []
        for child in children:
            d = child.get("data", {})
            if d.get("stickied"):
                continue
            posts.append({
                "id": d.get("id", ""),
                "title": d.get("title", ""),
                "author": d.get("author", "deleted"),
                "subreddit": d.get("subreddit", subreddit),
                "url": f"https://reddit.com{d.get('permalink', '')}",
                "is_self": d.get("is_self", True),
                "score": d.get("score", 0),
                "num_comments": d.get("num_comments", 0),
                "created_utc": int(d.get("created_utc", 0)),
            })
        return json.dumps({"subreddit": subreddit, "listing": listing, "posts": posts})
    except Exception as e:
        emit("reddit.fetch_error", "reddit", {"error": str(e), "subreddit": subreddit}, level="error")
        return f"Error: {e}"


async def _fetch_posts_callback(subreddit: str = "", limit: int = 5, listing: str = "hot") -> str:
    """Service registry callback to fetch recent posts from a subreddit."""
    import json
    global _reddit_client, _selfbot_client, _selfbot_config

    # Default to the configured self-subreddit when the caller leaves it blank
    if not subreddit:
        subreddit = _selfbot_config.self_subreddit if _selfbot_config else "test"

    limit = max(1, min(limit, 25))  # clamp to [1, 25]
    listing = listing if listing in ("new", "hot", "top", "rising", "controversial") else "hot"

    if _selfbot_client is not None and _selfbot_config is not None:
        return await _selfbot_fetch_posts_callback(_selfbot_client, _selfbot_config, subreddit, limit, listing)

    if _reddit_client is None:
        return "Error: Reddit client is not running."
    try:
        sub = await _reddit_client.subreddit(subreddit)
        if listing == "hot":
            feed = sub.hot(limit=limit)
        elif listing == "top":
            feed = sub.top(limit=limit)
        elif listing == "rising":
            feed = sub.rising(limit=limit)
        elif listing == "controversial":
            feed = sub.controversial(limit=limit)
        else:
            feed = sub.new(limit=limit)
        posts = []

        async for submission in feed:
            if getattr(submission, "stickied", False):
                continue
            posts.append({
                "id": str(submission.id),
                "title": str(submission.title),
                "author": str(submission.author) if submission.author else "deleted",
                "subreddit": subreddit,
                "url": f"https://reddit.com{submission.permalink}",
                "is_self": submission.is_self,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "created_utc": int(submission.created_utc),
            })
        return json.dumps({"subreddit": subreddit, "listing": listing, "posts": posts})
    except Exception as e:
        emit("reddit.fetch_error", "reddit", {"error": str(e), "subreddit": subreddit}, level="error")
        return f"Error: {e}"


# ── Event Bus & Streams Loops ────────────────────────────────────────────────

async def _listen_to_event_bus() -> None:
    """
    Listen to the central event bus for daemon events and route them back to Reddit.
    """
    q = event_bus.subscribe()
    try:
        while True:
            event = await q.get()
            # Only handle events related to Reddit triggers
            metadata = event.get("trigger_metadata", {})
            reddit_item_id = metadata.get("reddit_item_id")
            if not reddit_item_id:
                continue

            # Format event (only reply on final answers or completion)
            text = _format_reddit_event(event)
            if text:
                await _send_reply_callback(
                    item_id=reddit_item_id,
                    item_type=metadata.get("reddit_item_type", "comment"),
                    body=text,
                )
    except asyncio.CancelledError:
        pass
    finally:
        event_bus.unsubscribe(q)


# ── Standard PRAW Polling Loops ───────────────────────────────────────────────

async def _inbox_loop(
    reddit: asyncpraw.Reddit,
    config: RedditConfig,
    perception: "PerceptionLayer",
) -> None:
    """Poll the Reddit inbox on a fixed interval (for standard PRAW mode)."""
    emit("reddit.inbox_loop_started", "reddit", {})
    while True:
        try:
            await reddit.user.me()
            inbox = reddit.inbox.unread(limit=25)
            async for item in inbox:
                try:
                    event = RedditAdapter.inbox_item_to_event(item, config)
                    await perception.ingest(event)
                    emit(
                        "reddit.item_ingested",
                        "reddit",
                        {
                            "type": event.metadata.get("reddit_item_type"),
                            "author": event.metadata.get("reddit_author"),
                            "channel": event.source_channel,
                        },
                    )
                    await item.mark_read()
                except Exception as exc:
                    emit(
                        "reddit.item_error",
                        "reddit",
                        {"error": str(exc)},
                        level="error",
                    )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            emit(
                "reddit.inbox_error",
                "reddit",
                {"error": str(exc)},
                level="error",
            )
        await asyncio.sleep(config.poll_interval_seconds)


async def _subreddit_stream(
    reddit: asyncpraw.Reddit,
    subreddit_name: str,
    config: RedditConfig,
    perception: "PerceptionLayer",
) -> None:
    """Stream new submissions and comments from a single subreddit (for standard PRAW mode)."""
    emit("reddit.stream_started", "reddit", {"subreddit": subreddit_name})
    subreddit = await reddit.subreddit(subreddit_name)

    async def _submission_stream() -> None:
        async for submission in subreddit.stream.submissions(skip_existing=True):
            try:
                event = RedditAdapter.submission_to_event(submission, subreddit_name, config)
                await perception.ingest(event)
                emit(
                    "reddit.submission_ingested",
                    "reddit",
                    {"subreddit": subreddit_name, "id": submission.id},
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                emit(
                    "reddit.stream_error",
                    "reddit",
                    {"subreddit": subreddit_name, "error": str(exc)},
                    level="error",
                )

    async def _comment_stream() -> None:
        async for comment in subreddit.stream.comments(skip_existing=True):
            try:
                event = RedditAdapter.comment_to_event(comment, subreddit_name, config)
                await perception.ingest(event)
                emit(
                    "reddit.comment_ingested",
                    "reddit",
                    {"subreddit": subreddit_name, "id": comment.id},
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                emit(
                    "reddit.stream_error",
                    "reddit",
                    {"subreddit": subreddit_name, "error": str(exc)},
                    level="error",
                )

    sub_task = asyncio.create_task(_submission_stream(), name=f"reddit-sub-{subreddit_name}")
    cmt_task = asyncio.create_task(_comment_stream(), name=f"reddit-cmt-{subreddit_name}")
    try:
        await asyncio.gather(sub_task, cmt_task)
    except asyncio.CancelledError:
        sub_task.cancel()
        cmt_task.cancel()
        await asyncio.gather(sub_task, cmt_task, return_exceptions=True)
        raise


# ── Selfbot Polling Loops ────────────────────────────────────────────────────

async def _selfbot_comments_loop(
    client: httpx2.AsyncClient,
    config: RedditConfig,
    perception: "PerceptionLayer",
) -> None:
    """Poll the self_subreddit for new comments on a fixed interval using selfbot token."""
    emit("reddit.selfbot_comments_loop_started", "reddit", {"subreddit": config.self_subreddit})
    
    seen_comment_ids: set[str] = set()
    first_poll = True
    headers = {
        "Authorization": _get_auth_header(config.session_token),
        "User-Agent": config.user_agent,
    }
    
    poll_interval = max(45.0, config.poll_interval_seconds)

    while True:
        try:
            url = f"https://oauth.reddit.com/r/{config.self_subreddit}/comments.json?limit=25"
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            
            data = resp.json()
            children = data.get("data", {}).get("children", [])
            
            new_comments = []
            for child in children:
                c_data = child.get("data", {})
                c_id = c_data.get("id")
                if not c_id:
                    continue
                if c_id not in seen_comment_ids:
                    seen_comment_ids.add(c_id)
                    if not first_poll:
                        new_comments.append(c_data)
            
            first_poll = False
            
            # Process new comments in chronological order (oldest first)
            for c_data in reversed(new_comments):
                try:
                    comment_stub = RedditCommentStub(c_data)
                    event = RedditAdapter.comment_to_event(comment_stub, config.self_subreddit, config)
                    await perception.ingest(event)
                    emit(
                        "reddit.comment_ingested",
                        "reddit",
                        {"subreddit": config.self_subreddit, "id": comment_stub.id},
                    )
                except Exception as exc:
                    emit(
                        "reddit.item_error",
                        "reddit",
                        {"error": f"Failed to ingest selfbot comment: {exc}"},
                        level="error",
                    )
                    
        except asyncio.CancelledError:
            break
        except Exception as exc:
            emit(
                "reddit.comments_error",
                "reddit",
                {"error": f"Selfbot comments poll failed: {exc}"},
                level="error",
            )
        await asyncio.sleep(poll_interval)


async def _selfbot_inbox_loop(
    client: httpx2.AsyncClient,
    config: RedditConfig,
    perception: "PerceptionLayer",
) -> None:
    """Poll the unread inbox messages on a fixed interval using selfbot token."""
    emit("reddit.selfbot_inbox_loop_started", "reddit", {})
    
    headers = {
        "Authorization": _get_auth_header(config.session_token),
        "User-Agent": config.user_agent,
    }
    
    poll_interval = max(45.0, config.poll_interval_seconds)

    while True:
        try:
            url = "https://oauth.reddit.com/message/unread.json?limit=25"
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            
            data = resp.json()
            children = data.get("data", {}).get("children", [])
            
            for child in children:
                c_data = child.get("data", {})
                c_id = c_data.get("id")
                c_fullname = c_data.get("name") # e.g. t4_xxxx or t1_xxxx
                if not c_id:
                    continue
                
                try:
                    inbox_stub = RedditInboxStub(c_data)
                    event = RedditAdapter.inbox_item_to_event(inbox_stub, config)
                    await perception.ingest(event)
                    emit(
                        "reddit.item_ingested",
                        "reddit",
                        {
                            "type": event.metadata.get("reddit_item_type"),
                            "author": event.metadata.get("reddit_author"),
                            "channel": event.source_channel,
                        },
                    )
                    
                    # Mark the message as read immediately
                    mark_url = "https://oauth.reddit.com/api/read_message"
                    mark_payload = _encode_form({"id": c_fullname or f"t4_{c_id}"})
                    mark_headers = {**headers, "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"}
                    mark_resp = await client.post(mark_url, content=mark_payload, headers=mark_headers)
                    mark_resp.raise_for_status()
                except Exception as exc:
                    emit(
                        "reddit.item_error",
                        "reddit",
                        {"error": f"Failed to process selfbot inbox item: {exc}"},
                        level="error",
                    )
                    
        except asyncio.CancelledError:
            break
        except Exception as exc:
            emit(
                "reddit.inbox_error",
                "reddit",
                {"error": f"Selfbot inbox poll failed: {exc}"},
                level="error",
            )
        await asyncio.sleep(poll_interval)


# ── Top-Level Runner ─────────────────────────────────────────────────────────

async def run_reddit_poller(
    config: RedditConfig,
    perception: "PerceptionLayer",
) -> None:
    """
    Top-level entry-point. Starts either the PRAW Reddit client loop or
    the Selfbot HTTPX polling loops concurrently.
    """
    global _reddit_client, _selfbot_client, _selfbot_config
    if not config.enabled:
        emit(
            "reddit.disabled",
            "reddit",
            {"msg": "Reddit credentials not configured — Reddit interface disabled."},
            level="warn",
        )
        return

    # Register callbacks with the ServiceRegistry
    ServiceRegistry.register("send_reddit_reply", _send_reply_callback)
    ServiceRegistry.register("create_reddit_post", _create_post_callback)
    ServiceRegistry.register("fetch_reddit_posts", _fetch_posts_callback)

    if config.is_selfbot:
        async with httpx2.AsyncClient(timeout=30.0) as client:
            _selfbot_client = client
            _selfbot_config = config
            emit(
                "reddit.selfbot.started",
                "reddit",
                {
                    "self_subreddit": config.self_subreddit,
                    "poll_interval": max(45.0, config.poll_interval_seconds),
                },
            )

            tasks: list[asyncio.Task] = [
                asyncio.create_task(
                    _selfbot_inbox_loop(client, config, perception),
                    name="reddit-selfbot-inbox",
                ),
                asyncio.create_task(
                    _selfbot_comments_loop(client, config, perception),
                    name="reddit-selfbot-comments",
                ),
                asyncio.create_task(
                    _listen_to_event_bus(),
                    name="reddit-selfbot-event-listener",
                )
            ]

            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                emit("reddit.selfbot.cancelled", "reddit", {"msg": "Reddit selfbot poller cancelled."})
            finally:
                _selfbot_client = None
                _selfbot_config = None
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                emit("reddit.selfbot.stopped", "reddit", {"msg": "Reddit selfbot poller stopped."})
        return

    # Standard PRAW Mode
    user_agent = _USER_AGENT.format(username=config.username)

    async with asyncpraw.Reddit(
        client_id=config.client_id,
        client_secret=config.client_secret,
        refresh_token=config.refresh_token,
        user_agent=user_agent,
    ) as reddit:
        _reddit_client = reddit
        emit(
            "reddit.started",
            "reddit",
            {
                "username": config.username,
                "subreddits": config.subreddits,
                "poll_interval": config.poll_interval_seconds,
            },
        )

        tasks: list[asyncio.Task] = [
            asyncio.create_task(
                _inbox_loop(reddit, config, perception),
                name="reddit-inbox",
            ),
            asyncio.create_task(
                _listen_to_event_bus(),
                name="reddit-event-listener",
            )
        ]

        for sub in config.subreddits:
            tasks.append(
                asyncio.create_task(
                    _subreddit_stream(reddit, sub, config, perception),
                    name=f"reddit-stream-{sub}",
                )
            )

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            emit("reddit.cancelled", "reddit", {"msg": "Reddit poller cancelled."})
        finally:
            _reddit_client = None
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            emit("reddit.stopped", "reddit", {"msg": "Reddit poller stopped."})
