import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from core.events import UnifiedIngestEvent, SenderClass, PriorityLevel
from integrations.reddit.config import RedditConfig
from integrations.reddit.adapter import RedditAdapter
from integrations.reddit.poller import run_reddit_poller, _send_reply_callback, _create_post_callback
from agents.tools import tools_registry
from utils.callbacks import ServiceRegistry


@pytest.fixture
def mock_reddit_config():
    return RedditConfig(
        client_id="mock_id",
        client_secret="mock_secret",
        refresh_token="mock_token",
        username="mock_bot",
        poll_interval_seconds=10.0,
        subreddits=["test_sub"],
        operator_usernames=["op_user"],
        trusted_usernames=["trusted_user"],
    )


def test_reddit_adapter_classification(mock_reddit_config):
    # Mock comment authors such that str(author) returns the expected string
    op_author = MagicMock()
    op_author.__str__.return_value = "op_user"
    
    trusted_author = MagicMock()
    trusted_author.__str__.return_value = "trusted_user"
    
    external_author = MagicMock()
    external_author.__str__.return_value = "someone_else"

    # Test operator classification
    op_event = RedditAdapter.comment_to_event(
        comment=MagicMock(author=op_author, body="hello", permalink="/r/test_sub/comments/abc"),
        subreddit_name="test_sub",
        config=mock_reddit_config
    )
    assert op_event.sender_class == SenderClass.OPERATOR

    # Test trusted classification
    trusted_event = RedditAdapter.comment_to_event(
        comment=MagicMock(author=trusted_author, body="hello", permalink="/r/test_sub/comments/abc"),
        subreddit_name="test_sub",
        config=mock_reddit_config
    )
    assert trusted_event.sender_class == SenderClass.TRUSTED

    # Test external classification
    external_event = RedditAdapter.comment_to_event(
        comment=MagicMock(author=external_author, body="hello", permalink="/r/test_sub/comments/abc"),
        subreddit_name="test_sub",
        config=mock_reddit_config
    )
    assert external_event.sender_class == SenderClass.EXTERNAL


@pytest.mark.asyncio
async def test_reddit_tools_registered():
    assert "send_reddit_reply" in tools_registry.tools
    assert "create_reddit_post" in tools_registry.tools


@pytest.mark.asyncio
async def test_reddit_callbacks(mock_reddit_config):
    # Manually register the callbacks in the ServiceRegistry (mimicking poller initialization)
    ServiceRegistry.register("send_reddit_reply", _send_reply_callback)
    ServiceRegistry.register("create_reddit_post", _create_post_callback)

    # Mock asyncpraw.Reddit client
    mock_reddit = MagicMock()
    
    mock_subreddit = MagicMock()
    mock_submission = MagicMock()
    mock_submission.id = "sub123"
    mock_submission.url = "http://mock_url"
    # submit is async in asyncpraw
    mock_subreddit.submit = AsyncMock(return_value=mock_submission)
    
    # subreddit() is async in asyncpraw
    mock_reddit.subreddit = AsyncMock(return_value=mock_subreddit)

    mock_comment = MagicMock()
    mock_comment.id = "reply123"
    # reply is async in asyncpraw
    mock_comment.reply = AsyncMock(return_value=mock_comment)
    
    # comment() is async in asyncpraw
    mock_reddit.comment = AsyncMock(return_value=mock_comment)

    mock_message = MagicMock()
    mock_message.id = "msg123"
    # reply is async in asyncpraw
    mock_message.reply = AsyncMock(return_value=mock_message)
    
    mock_inbox = MagicMock()
    # inbox.message() is async in asyncpraw
    mock_inbox.message = AsyncMock(return_value=mock_message)
    mock_reddit.inbox = mock_inbox

    with patch("integrations.reddit.poller._reddit_client", mock_reddit):
        # Test create_reddit_post
        res_post = await ServiceRegistry.call(
            "create_reddit_post",
            subreddit_name="test_sub",
            title="Hello World",
            body="Post body",
        )
        assert "Success: Created post in r/test_sub" in res_post
        assert "sub123" in res_post
        mock_subreddit.submit.assert_called_once_with(title="Hello World", selftext="Post body")

        # Test send_reddit_reply for comment
        res_reply_cmt = await ServiceRegistry.call(
            "send_reddit_reply",
            item_id="cmt123",
            item_type="comment",
            body="Comment reply text",
        )
        assert "Success: Replied to cmt123" in res_reply_cmt
        assert "reply123" in res_reply_cmt
        mock_comment.reply.assert_called_once_with("Comment reply text")

        # Test send_reddit_reply for message
        res_reply_msg = await ServiceRegistry.call(
            "send_reddit_reply",
            item_id="msg123",
            item_type="message",
            body="Message reply text",
        )
        assert "Success: Replied to msg123" in res_reply_msg
        assert "msg123" in res_reply_msg
        mock_message.reply.assert_called_once_with("Message reply text")


@pytest.fixture
def mock_selfbot_config():
    return RedditConfig(
        session_token="Bearer test_token",
        user_agent="test_ua",
        self_subreddit="test_self_sub",
        human_user="test_human",
        poll_interval_seconds=45.0,
    )


def test_reddit_selfbot_adapter_classification(mock_selfbot_config):
    # Test human user classification
    human_comment = MagicMock(author="test_human", body="hello", permalink="/r/test_sub/comments/abc")
    human_comment.id = "cmt_human"
    
    event = RedditAdapter.comment_to_event(
        comment=human_comment,
        subreddit_name="test_self_sub",
        config=mock_selfbot_config
    )
    assert event.sender_class == SenderClass.OPERATOR
    assert event.priority_level == PriorityLevel.HIGH

    # Test other user classification
    other_comment = MagicMock(author="other_user", body="hello", permalink="/r/test_sub/comments/abc")
    other_comment.id = "cmt_other"
    
    event_other = RedditAdapter.comment_to_event(
        comment=other_comment,
        subreddit_name="test_self_sub",
        config=mock_selfbot_config
    )
    assert event_other.sender_class == SenderClass.EXTERNAL
    assert event_other.priority_level == PriorityLevel.NORMAL


@pytest.mark.asyncio
async def test_reddit_selfbot_callbacks(mock_selfbot_config):
    from integrations.reddit.poller import _selfbot_send_reply_callback, _selfbot_create_post_callback
    
    # Mock httpx2.AsyncClient response
    mock_client = AsyncMock()
    
    mock_reply_response = MagicMock()
    mock_reply_response.json.return_value = {
        "json": {
            "data": {
                "things": [
                    {
                        "data": {
                            "id": "t1_reply_id"
                        }
                    }
                ]
            }
        }
    }
    mock_client.post.return_value = mock_reply_response

    # Test send reply
    res_reply = await _selfbot_send_reply_callback(
        client=mock_client,
        config=mock_selfbot_config,
        item_id="cmt123",
        item_type="comment",
        body="reply content",
    )
    assert "Success: Replied to cmt123" in res_reply
    assert "t1_reply_id" in res_reply
    
    # Test create post
    mock_post_response = MagicMock()
    mock_post_response.json.return_value = {
        "json": {
            "data": {
                "things": [
                    {
                        "data": {
                            "id": "t3_post_id",
                            "url": "http://reddit.com/r/test_sub/post"
                        }
                    }
                ]
            }
        }
    }
    mock_client.post.return_value = mock_post_response

    res_post = await _selfbot_create_post_callback(
        client=mock_client,
        config=mock_selfbot_config,
        subreddit_name="test_self_sub",
        title="post title",
        body="post body",
    )
    assert "Success: Created post in r/test_self_sub" in res_post
    assert "t3_post_id" in res_post


@pytest.mark.asyncio
async def test_reddit_selfbot_polling(mock_selfbot_config):
    from integrations.reddit.poller import _selfbot_comments_loop, _selfbot_inbox_loop
    
    mock_client = AsyncMock()
    mock_perception = AsyncMock()
    
    # Set up mock response for comments
    mock_comments_response = MagicMock()
    mock_comments_response.json.return_value = {
        "data": {
            "children": [
                {
                    "kind": "t1",
                    "data": {
                        "id": "c1",
                        "author": "test_human",
                        "body": "hello human",
                        "permalink": "/r/test_self_sub/comments/c1",
                    }
                },
                {
                    "kind": "t1",
                    "data": {
                        "id": "c2",
                        "author": "someone_else",
                        "body": "hello external",
                        "permalink": "/r/test_self_sub/comments/c2",
                    }
                }
            ]
        }
    }
    mock_client.get.return_value = mock_comments_response

    # Run one step of the comments loop by calling it and letting it run, but since it's an infinite loop,
    # we can patch asyncio.sleep to raise a CancelledError to break the loop after one iteration.
    with patch("asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError)):
        try:
            await _selfbot_comments_loop(mock_client, mock_selfbot_config, mock_perception)
        except asyncio.CancelledError:
            pass
            
    # During the first poll, comments are only marked as seen but not ingested to avoid backlogs.
    # So ingest should not have been called.
    assert mock_perception.ingest.call_count == 0

    # Set up mock response for inbox
    mock_inbox_response = MagicMock()
    mock_inbox_response.json.return_value = {
        "data": {
            "children": [
                {
                    "kind": "t1",
                    "data": {
                        "id": "m1",
                        "name": "t1_m1",
                        "author": "someone",
                        "body": "message body",
                        "subject": "comment reply",
                        "was_comment": True,
                    }
                }
            ]
        }
    }
    mock_client.get.return_value = mock_inbox_response
    mock_client.post.return_value = MagicMock(status_code=200)

    with patch("asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError)):
        try:
            await _selfbot_inbox_loop(mock_client, mock_selfbot_config, mock_perception)
        except asyncio.CancelledError:
            pass

    # Inbox items are ingested immediately.
    assert mock_perception.ingest.call_count == 1
    call_args = mock_perception.ingest.call_args[0][0]
    assert call_args.sender_identifier == "reddit:someone"
    assert call_args.priority_level == PriorityLevel.NORMAL


@pytest.mark.asyncio
async def test_reddit_fetch_posts_tool_registered():
    assert "fetch_reddit_posts" in tools_registry.tools


@pytest.mark.asyncio
async def test_reddit_fetch_posts_callbacks(mock_reddit_config):
    from integrations.reddit.poller import _fetch_posts_callback
    ServiceRegistry.register("fetch_reddit_posts", _fetch_posts_callback)

    # 1. Test standard OAuth (PRAW) Mode
    mock_reddit = MagicMock()
    mock_subreddit = MagicMock()
    mock_submission = MagicMock(
        id="sub123",
        title="Sample Post",
        author=MagicMock(__str__=lambda s: "author123"),
        permalink="/r/test_sub/comments/sub123",
        is_self=True,
        score=10,
        num_comments=5,
        created_utc=1600000000.0,
        stickied=False
    )

    # Mock submission stream (async generator helper for feed)
    async def mock_hot_stream(*args, **kwargs):
        yield mock_submission

    mock_subreddit.hot = mock_hot_stream
    mock_reddit.subreddit = AsyncMock(return_value=mock_subreddit)

    with patch("integrations.reddit.poller._reddit_client", mock_reddit):
        res = await ServiceRegistry.call(
            "fetch_reddit_posts",
            subreddit="test_sub",
            limit=5,
            listing="hot"
        )
        data = json.loads(res)
        assert data["subreddit"] == "test_sub"
        assert data["listing"] == "hot"
        assert len(data["posts"]) == 1
        assert data["posts"][0]["id"] == "sub123"
        assert data["posts"][0]["title"] == "Sample Post"


@pytest.mark.asyncio
async def test_reddit_fetch_posts_selfbot_callback(mock_selfbot_config):
    from integrations.reddit.poller import _selfbot_fetch_posts_callback

    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "children": [
                {
                    "data": {
                        "id": "selfbot123",
                        "title": "Selfbot Title",
                        "author": "selfbot_author",
                        "permalink": "/r/test_self_sub/comments/selfbot123",
                        "is_self": True,
                        "score": 42,
                        "num_comments": 2,
                        "created_utc": 1700000000.0,
                        "stickied": False
                    }
                }
            ]
        }
    }
    mock_client.get.return_value = mock_response

    res = await _selfbot_fetch_posts_callback(
        client=mock_client,
        config=mock_selfbot_config,
        subreddit="test_self_sub",
        limit=5,
        listing="hot"
    )
    data = json.loads(res)
    assert data["subreddit"] == "test_self_sub"
    assert len(data["posts"]) == 1
    assert data["posts"][0]["id"] == "selfbot123"
    assert data["posts"][0]["score"] == 42


