import logging
import os


def setup_global_logging(console_level=logging.WARNING):
    """
    Configures three logging destinations:

      logs/agent.log  — Every INFO+ message from every module, with full
                        timestamp + module + line metadata. Useful for
                        debugging individual lines of code.

      logs/trace.log  — The clean trace tree only (noesis.tracer logger),
                        message-only format so the box-drawing characters
                        remain perfectly aligned. Tail this file to watch
                        the agent think in real-time:
                            tail -f logs/trace.log

      console         — Mirrors the trace tree + INFO from all modules at
                        the configured level (INFO for verbose, WARNING to
                        silence day-to-day noise).
    """
    os.makedirs("logs", exist_ok=True)

    # ── Root logger ────────────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Clear handlers to avoid duplicates on reload

    # ── 1. agent.log — Full verbose log with metadata ─────────────────────────
    agent_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    )
    agent_handler = logging.FileHandler("logs/agent.log", encoding="utf-8")
    agent_handler.setLevel(logging.INFO)
    agent_handler.setFormatter(agent_formatter)
    root_logger.addHandler(agent_handler)

    # ── 2. trace.log — Clean trace tree (message only, no prefix clutter) ─────
    #    Only captures the dedicated tracer logger so the tree characters
    #    line up perfectly and the file reads like a timeline of the run.
    #
    #    IMPORTANT: clear the child logger's handlers first.  Uvicorn reload
    #    calls this function more than once; without the reset every line
    #    gets written once per setup call (visible as doubled/tripled output).
    tracer_logger = logging.getLogger("noesis.tracer")
    tracer_logger.handlers = []          # ← prevents duplicate lines on reload

    trace_formatter = logging.Formatter("%(message)s")
    trace_handler = logging.FileHandler("logs/trace.log", encoding="utf-8")
    trace_handler.setLevel(logging.INFO)
    trace_handler.setFormatter(trace_formatter)

    tracer_logger.addHandler(trace_handler)
    tracer_logger.propagate = True       # still flows to root → agent.log (with prefix)

    # ── 3. Console — Configurable level, clean short format ───────────────────
    console_formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

