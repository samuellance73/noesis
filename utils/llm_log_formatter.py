"""
utils/llm_log_formatter.py
──────────────────────────
Formatting helpers for LLM request/response log blocks.

Extracted from integrations/llm/service.py so that the transport layer
(UpstreamService) has a single responsibility: making HTTP calls.
"""

import datetime
import json

_SEP  = "═" * 80
_DASH = "─" * 80


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_messages(payload: dict) -> str:
    """Render all messages in a payload as a readable indented string."""
    lines = []
    for m in payload.get("messages", []):
        role    = str(m.get("role", "unknown")).upper()
        content = str(m.get("content", "")).strip()
        lines.append(f"  [{role}]")
        for line in content.splitlines():
            lines.append(f"  {line}")
        lines.append("")
    return "\n".join(lines)


def format_usage(usage: dict | None) -> str:
    """Return a compact token-usage string, or empty string if unavailable."""
    if not usage:
        return ""
    prompt     = usage.get("prompt_tokens", "?")
    completion = usage.get("completion_tokens", "?")
    total      = usage.get("total_tokens", "?")
    return f"tokens → prompt={prompt}  completion={completion}  total={total}"


def decode_sse_text(raw_lines: list[str]) -> str:
    """Extract the concatenated text content from SSE 'data: {...}' lines."""
    parts = []
    for line in raw_lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj   = json.loads(payload)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            chunk = delta.get("content") or ""
            if chunk:
                parts.append(chunk)
        except (json.JSONDecodeError, IndexError, KeyError):
            pass
    return "".join(parts)


def format_chat_block(payload: dict, data: dict, elapsed: float) -> str:
    """Format a full non-streaming request+response as a single log block."""
    model = payload.get("model", "unknown")

    res_lines = []
    for c in data.get("choices", []):
        msg     = c.get("message", {})
        role    = str(msg.get("role", "assistant")).upper()
        content = str(msg.get("content") or "").strip()
        res_lines.append(f"  [{role}]")
        for line in content.splitlines():
            res_lines.append(f"  {line}")
        res_lines.append("")

    usage_str = format_usage(data.get("usage"))
    meta      = f"  model={model}  elapsed={elapsed:.2f}s"
    if usage_str:
        meta += f"  {usage_str}"

    return (
        f"{_SEP}\n"
        f"  ▼ REQUEST   {now()}\n"
        f"{_DASH}\n"
        f"{format_messages(payload)}"
        f"{_DASH}\n"
        f"  ▲ RESPONSE\n"
        f"{_DASH}\n"
        + "\n".join(res_lines) +
        f"\n{_DASH}\n"
        f"{meta}\n"
        f"{_SEP}\n"
    )


def format_stream_block(
    payload: dict,
    raw_lines: list[str],
    elapsed: float,
) -> str:
    """Format a completed streaming request+response as a single log block."""
    model    = payload.get("model", "unknown")
    text_out = decode_sse_text(raw_lines)
    resp_display = text_out if text_out else "(no text content decoded from stream)"

    return (
        f"{_SEP}\n"
        f"  ▼ STREAM REQUEST   {now()}\n"
        f"{_DASH}\n"
        f"{format_messages(payload)}"
        f"{_DASH}\n"
        f"  ▲ STREAMED RESPONSE\n"
        f"{_DASH}\n"
        f"  {resp_display}\n"
        f"{_DASH}\n"
        f"  model={model}  elapsed={elapsed:.2f}s  chunks={len(raw_lines)}\n"
        f"{_SEP}\n"
    )
