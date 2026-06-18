"""
utils/tracer.py
───────────────
Zero-dependency LLM trace tree with decorator-based instrumentation.

## Usage

### 1. Start a trace (orchestrator only)
    from utils.tracer import Trace, set_current_trace

    trace = Trace(query=user_input)
    set_current_trace(trace)          # stored in a ContextVar — flows automatically
    ...
    trace.done()

### 2. Instrument any function (no parameter threading needed)
    from utils.tracer import traced

    @traced("planner", log_args=["goal"])
    async def plan(goal: str, service) -> list[dict]:
        ...  # pure business logic — decorator wraps the span

### 3. Instrument tool functions
    from utils.tracer import traced_tool

    @traced_tool("web_search")
    async def web_search(query: str) -> str:
        ...  # decorator logs input, output preview, timing

### 4. Ad-hoc span inside a function (escape hatch)
    from utils.tracer import current_span

    async def something():
        with current_span("sub-step") as span:
            span.log_event("custom event", key=value)
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
import uuid
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger("noesis.tracer")

F = TypeVar("F", bound=Callable)

# ─── Tree-drawing characters ────────────────────────────────────────────────
_INDENT  = "│    "
_FMT_TRACE  = "┌─[TRACE]"
_FMT_SPAN   = "├──[SPAN]"
_FMT_RESULT = "│    └──[RESULT]"
_FMT_ERROR  = "│    └──[ERROR] "
_FMT_DONE   = "└─[DONE] "


def _ms(seconds: float) -> str:
    return f"{seconds:.2f}s" if seconds >= 1 else f"{seconds * 1000:.0f}ms"


def _clip(value: Any, max_len: int = 120) -> str:
    s = str(value).replace("\n", " ").strip()
    return s if len(s) <= max_len else s[:max_len] + f"… [+{len(s) - max_len}]"


# ─── ContextVars: trace state for the current async task context ─────────────
_active_trace: ContextVar["Trace | None"] = ContextVar("_active_trace", default=None)
_depth_stack_var: ContextVar[list[Span] | None] = ContextVar("_depth_stack_var", default=None)


def set_current_trace(trace: "Trace") -> None:
    """Call once per request in the orchestrator to make the trace available everywhere."""
    _active_trace.set(trace)
    _depth_stack_var.set([])


def get_current_trace() -> "Trace | None":
    return _active_trace.get()


# ─── Span ───────────────────────────────────────────────────────────────────

@dataclass
class Span:
    name: str
    depth: int = 0
    start: float = field(default_factory=time.perf_counter)

    def _indent(self) -> str:
        return _INDENT * self.depth

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def log_open(self, **kwargs: Any) -> None:
        extras = "  ".join(f"{k}={_clip(v)}" for k, v in kwargs.items())
        logger.info(f"{self._indent()}{_FMT_SPAN} ▶  {self.name}  {extras}")

    def log_close(self, status: str = "ok", **kwargs: Any) -> None:
        extras = "  ".join(f"{k}={_clip(v)}" for k, v in kwargs.items())
        logger.info(f"{self._indent()}{_FMT_RESULT}  [{status}]  ▸ {_ms(self.elapsed())}  {extras}")

    def log_error(self, message: str) -> None:
        logger.error(f"{self._indent()}{_FMT_ERROR} {_clip(message, 160)}  ▸ {_ms(self.elapsed())}")

    def log_event(self, label: str, **kwargs: Any) -> None:
        extras = "  ".join(f"{k}={_clip(v)}" for k, v in kwargs.items())
        logger.info(f"{self._indent()}{_FMT_SPAN}   ▷ {label}  {extras}")

    def log_warn(self, message: str) -> None:
        logger.warning(f"{self._indent()}{_FMT_ERROR} ⚠ {_clip(message, 160)}")


# ─── Trace ──────────────────────────────────────────────────────────────────

class Trace:
    def __init__(self, query: str, trace_id: str | None = None):
        self.query    = query
        self.id       = trace_id or uuid.uuid4().hex[:8]
        self.start    = time.perf_counter()
        _depth_stack_var.set([])
        logger.info(
            f"{_FMT_TRACE} id={self.id}\n│\n│  Query: {_clip(query, 160)}\n│"
        )

    def _get_stack(self) -> list[Span]:
        stack = _depth_stack_var.get()
        if stack is None:
            stack = []
            _depth_stack_var.set(stack)
        return stack

    def _depth(self) -> int:
        return len(self._get_stack())

    @contextmanager
    def span(self, name: str, **open_kwargs):
        stack = list(self._get_stack())
        s = Span(name=name, depth=len(stack))
        stack.append(s)
        token = _depth_stack_var.set(stack)
        s.log_open(**open_kwargs)
        try:
            yield s
        except Exception as exc:
            s.log_error(str(exc))
            raise
        finally:
            _depth_stack_var.reset(token)

    @asynccontextmanager
    async def aspan(self, name: str, **open_kwargs):
        stack = list(self._get_stack())
        s = Span(name=name, depth=len(stack))
        stack.append(s)
        token = _depth_stack_var.set(stack)
        s.log_open(**open_kwargs)
        try:
            yield s
        except Exception as exc:
            s.log_error(str(exc))
            raise
        finally:
            _depth_stack_var.reset(token)

    def done(self, **kwargs: Any) -> None:
        extras = "  ".join(f"{k}={_clip(v)}" for k, v in kwargs.items())
        logger.info(f"│\n{_FMT_DONE} id={self.id}  total ▸ {_ms(time.perf_counter() - self.start)}  {extras}\n{'─'*72}")

    def error(self, message: str) -> None:
        logger.error(f"│\n{_FMT_ERROR} {_clip(message, 200)}  ▸ {_ms(time.perf_counter() - self.start)}\n{'─'*72}")


# ─── @traced decorator ──────────────────────────────────────────────────────

def traced(
    span_name: str,
    *,
    log_args: list[str] | None = None,
    log_result: bool = True,
    result_clip: int = 120,
) -> Callable[[F], F]:
    """
    Decorator that wraps a function in a trace span automatically.

    The span is opened on call and closed (with timing) on return/exception.
    If no active Trace exists the function runs normally with no overhead.

    Args:
        span_name:   Name shown in the trace tree.
        log_args:    List of argument names whose values to log in the span header.
        log_result:  Whether to log the return value on span close.
        result_clip: Max chars for the return value preview.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace = get_current_trace()
            if trace is None:
                return await func(*args, **kwargs)

            # Collect whitelisted arg values for the span header
            open_kwargs = _collect_args(func, args, kwargs, log_args or [])

            async with trace.aspan(span_name, **open_kwargs) as span:
                t0 = time.perf_counter()
                result = await func(*args, **kwargs)
                close_kwargs = {}
                if log_result:
                    close_kwargs["result"] = _clip(result, result_clip)
                span.log_close(elapsed=_ms(time.perf_counter() - t0), **close_kwargs)
                return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            trace = get_current_trace()
            if trace is None:
                return func(*args, **kwargs)

            open_kwargs = _collect_args(func, args, kwargs, log_args or [])
            with trace.span(span_name, **open_kwargs) as span:
                t0 = time.perf_counter()
                result = func(*args, **kwargs)
                close_kwargs = {}
                if log_result:
                    close_kwargs["result"] = _clip(result, result_clip)
                span.log_close(elapsed=_ms(time.perf_counter() - t0), **close_kwargs)
                return result

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper  # type: ignore[return-value]

    return decorator


def traced_tool(
    tool_name: str,
    *,
    input_arg: str = "query",
    result_clip: int = 200,
) -> Callable[[F], F]:
    """
    Specialised decorator for tool functions.  Logs input, HTTP-style status,
    elapsed time, and a result preview.  Keeps tool functions pure.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            trace = get_current_trace()
            tool_input = _first_arg(func, args, kwargs, input_arg)

            if trace is None:
                return await func(*args, **kwargs)

            async with trace.aspan(f"tool:{tool_name}", input=_clip(tool_input)) as span:
                t0 = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                ok = not str(result).lower().startswith("error")
                span.log_close(
                    status="ok" if ok else "error",
                    elapsed=_ms(elapsed),
                    result_len=f"{len(str(result))} chars",
                    preview=_clip(result, result_clip),
                )
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ─── Escape hatch: ad-hoc span inside a function ────────────────────────────

@contextmanager
def current_span(name: str, **open_kwargs):
    """
    Open a span on the current trace without needing a reference to it.
    Safe to call even when no trace is active (becomes a no-op).
    """
    trace = get_current_trace()
    if trace is None:
        yield _NoopSpan()
        return
    with trace.span(name, **open_kwargs) as span:
        yield span


@asynccontextmanager
async def current_aspan(name: str, **open_kwargs):
    """Async version of current_span."""
    trace = get_current_trace()
    if trace is None:
        yield _NoopSpan()
        return
    async with trace.aspan(name, **open_kwargs) as span:
        yield span


class _NoopSpan:
    """Returned when no trace is active — all methods are silent no-ops."""
    def log_open(self, *a, **k): pass
    def log_close(self, *a, **k): pass
    def log_error(self, *a, **k): pass
    def log_event(self, *a, **k): pass
    def log_warn(self, *a, **k): pass


# ─── Private helpers ─────────────────────────────────────────────────────────

def _collect_args(
    func: Callable,
    args: tuple,
    kwargs: dict,
    names: list[str],
) -> dict[str, Any]:
    """Extract named arguments from a call for span metadata."""
    if not names:
        return {}
    sig    = inspect.signature(func)
    params = list(sig.parameters.keys())
    bound  = {}
    for i, val in enumerate(args):
        if i < len(params):
            bound[params[i]] = val
    bound.update(kwargs)
    return {n: _clip(bound[n]) for n in names if n in bound}


def _first_arg(
    func: Callable,
    args: tuple,
    kwargs: dict,
    name: str,
) -> Any:
    """Return the value of the first meaningful argument (for tools)."""
    sig    = inspect.signature(func)
    params = list(sig.parameters.keys())
    if name in kwargs:
        return kwargs[name]
    if name in params:
        idx = params.index(name)
        if idx < len(args):
            return args[idx]
    return args[0] if args else ""
