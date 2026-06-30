"""
integrations/reddit/config.py
──────────────────────────────
Pydantic-settings config for the Reddit integration.

Selfbot mode .env entries (no API keys required):
    REDDIT_SESSION_TOKEN="Bearer eyJ..."   # Full token_v2 bearer — must be pure ASCII
    REDDIT_USER_AGENT="Mozilla/5.0 ..."
    REDDIT_SELF_SUBREDDIT=noesis_agent
    REDDIT_HUMAN_USER=psilko

Optional:
    REDDIT_POLL_INTERVAL_SECONDS=30
    REDDIT_SUBREDDITS=wallstreetbets,python        # comma-separated; leave empty to skip
    REDDIT_OPERATOR_USERNAMES=friend1,friend2      # DMs/mentions from these → OPERATOR class
    REDDIT_TRUSTED_USERNAMES=user3,user4           # → TRUSTED class; all others → EXTERNAL

WARNING: If REDDIT_SESSION_TOKEN was copy-pasted from a terminal that truncated the
output with '…' (U+2026), that character will be present in the token and will cause
every HTTP request to crash with an 'ascii' codec error. Always paste the full token.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedditConfig(BaseSettings):
    # ── OAuth2 credentials (required) ─────────────────────────────────────────
    client_id: str = ""
    client_secret: str = ""
    refresh_token: str = ""
    username: str = ""

    # ── Selfbot Configuration (alternative to OAuth2) ─────────────────────────
    session_token: str = ""
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    self_subreddit: str = "noesis_agent"
    human_user: str = "psilko"

    # ── Polling ───────────────────────────────────────────────────────────────
    # How often to poll the Reddit inbox (seconds).
    poll_interval_seconds: float = 30.0

    # ── Optional subreddit monitoring ─────────────────────────────────────────
    # If non-empty, new posts/comments in these subreddits are also ingested.
    subreddits: list[str] = []

    # ── Sender classification ─────────────────────────────────────────────────
    operator_usernames: list[str] = []   # DMs/mentions from these → OPERATOR
    trusted_usernames: list[str] = []    # → TRUSTED; everything else → EXTERNAL

    model_config = SettingsConfigDict(
        env_prefix="REDDIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("session_token", mode="before")
    @classmethod
    def _validate_session_token(cls, v: object) -> object:
        """
        Reject tokens containing non-ASCII characters.

        The most common cause is copy-pasting from a terminal that truncated the
        token with a Unicode ellipsis '\u2026' (…).  HTTP headers require pure
        ASCII; httpx will crash at request time with a cryptic codec error.
        Catching it here gives a clear, actionable message at startup instead.
        """
        if not isinstance(v, str) or not v:
            return v
        try:
            v.encode("ascii")
        except UnicodeEncodeError as exc:
            bad_char = v[exc.start]
            raise ValueError(
                f"REDDIT_SESSION_TOKEN contains a non-ASCII character "
                f"(U+{ord(bad_char):04X} '{bad_char}') at position {exc.start}. "
                f"Your token was almost certainly truncated with a '\u2026' when "
                f"you copied it. Please re-copy the full, untruncated token from "
                f"Reddit (Settings → Privacy & Security → Manage third-party "
                f"authorizations, or grab token_v2 fresh from your browser "
                f"DevTools → Application → Cookies)."
            ) from exc
        return v

    @field_validator("subreddits", "operator_usernames", "trusted_usernames", mode="before")
    @classmethod
    def _parse_csv(cls, v: object) -> list[str]:
        """Accept a comma-separated string or a plain Python list."""
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v  # type: ignore[return-value]

    @property
    def enabled(self) -> bool:
        """True when either standard credentials or selfbot token is present."""
        standard_enabled = bool(self.client_id and self.client_secret and self.refresh_token and self.username)
        selfbot_enabled = bool(self.session_token)
        return standard_enabled or selfbot_enabled

    @property
    def is_selfbot(self) -> bool:
        """True if selfbot mode is active."""
        return bool(self.session_token)


settings = RedditConfig()

