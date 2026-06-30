"""
integrations/reddit/adapter.py
───────────────────────────────
RedditAdapter — converts asyncpraw objects into UnifiedIngestEvent.

Handles three Reddit signal types:
  • inbox items  (DMs / comment replies / username mentions)
  • subreddit submissions (new posts, optional)
  • subreddit comments   (new comments, optional)

Source-channel convention:
  "reddit:dm"              — private messages
  "reddit:mention"         — username mentions / comment replies
  "reddit:r/<subreddit>"   — subreddit post/comment monitoring
"""

from __future__ import annotations

import re

from core.events import PriorityLevel, SenderClass, UnifiedIngestEvent
from integrations.reddit.config import RedditConfig


def _classify_sender(author: str, config: RedditConfig) -> SenderClass:
    """Map a Reddit username to a SenderClass based on config lists."""
    lower = author.lower()
    if config.human_user and lower == config.human_user.lower():
        return SenderClass.OPERATOR
    if lower in {u.lower() for u in config.operator_usernames}:
        return SenderClass.OPERATOR
    if lower in {u.lower() for u in config.trusted_usernames}:
        return SenderClass.TRUSTED
    return SenderClass.EXTERNAL


def _strip_markdown(text: str) -> str:
    """Light cleanup — collapse whitespace and trim."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class RedditAdapter:
    """Stateless converter — all methods are static."""

    @staticmethod
    def inbox_item_to_event(item, config: RedditConfig) -> UnifiedIngestEvent:
        """
        Convert an asyncpraw inbox item (Message, Comment, or SubredditMessage)
        to a UnifiedIngestEvent.

        asyncpraw inbox types:
          item.type == "message"  → private DM
          item.type == "username_mention"  → comment that mentions the bot
          item.type == "comment_reply"     → reply to a comment the bot made
        """
        author = str(item.author) if item.author else "deleted"
        item_type = getattr(item, "type", "message")

        if item_type == "message":
            source_channel = "reddit:dm"
            conversation_id = f"reddit:dm:{author}"
            raw = _strip_markdown(str(item.body))
        else:
            # mention or comment_reply — attach subreddit context
            subreddit = str(getattr(item, "subreddit", "unknown"))
            source_channel = "reddit:mention"
            conversation_id = f"reddit:mention:{subreddit}"
            body = _strip_markdown(str(item.body))
            subject = getattr(item, "subject", "")
            raw = f"[{item_type} in r/{subreddit}] {subject}: {body}" if subject else f"[{item_type} in r/{subreddit}] {body}"

        is_human = bool(config.human_user and author.lower() == config.human_user.lower())
        priority = PriorityLevel.HIGH if is_human else PriorityLevel.NORMAL

        return UnifiedIngestEvent(
            source_channel=source_channel,
            sender_identifier=f"reddit:{author}",
            sender_class=_classify_sender(author, config),
            raw_content=raw,
            target_conversation_identifier=conversation_id,
            priority_level=priority,
            metadata={
                "reddit_item_id": str(getattr(item, "id", "")),
                "reddit_item_type": item_type,
                "reddit_author": author,
            },
        )

    @staticmethod
    def submission_to_event(submission, subreddit_name: str, config: RedditConfig) -> UnifiedIngestEvent:
        """Convert a new subreddit submission (post) to a UnifiedIngestEvent."""
        author = str(submission.author) if submission.author else "deleted"
        title = _strip_markdown(str(submission.title))
        selftext = _strip_markdown(str(submission.selftext or ""))
        raw = f"[post in r/{subreddit_name}] {title}" + (f": {selftext[:800]}" if selftext else "")

        is_human = bool(config.human_user and author.lower() == config.human_user.lower())
        priority = PriorityLevel.HIGH if is_human else PriorityLevel.NORMAL

        return UnifiedIngestEvent(
            source_channel=f"reddit:r/{subreddit_name}",
            sender_identifier=f"reddit:{author}",
            sender_class=_classify_sender(author, config),
            raw_content=raw,
            target_conversation_identifier=f"reddit:r/{subreddit_name}",
            priority_level=priority,
            metadata={
                "reddit_item_id": str(submission.id),
                "reddit_item_type": "submission",
                "reddit_author": author,
                "reddit_subreddit": subreddit_name,
                "reddit_url": f"https://reddit.com{submission.permalink}",
            },
        )

    @staticmethod
    def comment_to_event(comment, subreddit_name: str, config: RedditConfig) -> UnifiedIngestEvent:
        """Convert a new subreddit comment to a UnifiedIngestEvent."""
        author = str(comment.author) if comment.author else "deleted"
        body = _strip_markdown(str(comment.body))
        raw = f"[comment in r/{subreddit_name}] {body[:800]}"

        is_human = bool(config.human_user and author.lower() == config.human_user.lower())
        priority = PriorityLevel.HIGH if is_human else PriorityLevel.NORMAL

        return UnifiedIngestEvent(
            source_channel=f"reddit:r/{subreddit_name}",
            sender_identifier=f"reddit:{author}",
            sender_class=_classify_sender(author, config),
            raw_content=raw,
            target_conversation_identifier=f"reddit:r/{subreddit_name}",
            priority_level=priority,
            metadata={
                "reddit_item_id": str(comment.id),
                "reddit_item_type": "comment",
                "reddit_author": author,
                "reddit_subreddit": subreddit_name,
                "reddit_url": f"https://reddit.com{comment.permalink}",
            },
        )

