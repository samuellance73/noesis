import logging
import os


def setup_global_logging(console_level=logging.WARNING):
    """
    Configures logging destinations:

      logs/agent.log        — Every INFO+ message from every module, with full
                              timestamp + module + line metadata.

      logs/trace.log        — Clean trace tree only (noesis.tracer logger),
                              message-only so box-drawing characters stay aligned.
                              Tail with:  tail -f logs/trace.log

      logs/goal_manager.log — Autonomous loop events only (noesis.goal_manager):
                              cycle starts, manager decisions, task spawning,
                              goal completion, stop signals.
                              Tail with:  tail -f logs/goal_manager.log

      logs/llm.log          — Full LLM request + response pairs in readable blocks.
                              Each block shows model, timing, tokens, and content.
                              Tail with:  tail -f logs/llm.log

      console               — Mirrors all loggers at the configured level.
    """
    os.makedirs("logs", exist_ok=True)

    # ── Root logger ────────────────────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Clear handlers to avoid duplicates on reload

    # ── 1. agent.log — Full verbose log with metadata ─────────────────────────
    agent_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    )
    agent_handler = logging.FileHandler("logs/agent.log", mode="w", encoding="utf-8")
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
    trace_handler = logging.FileHandler("logs/trace.log", mode="w", encoding="utf-8")
    trace_handler.setLevel(logging.INFO)
    trace_handler.setFormatter(trace_formatter)

    tracer_logger.addHandler(trace_handler)
    tracer_logger.propagate = True       # still flows to root → agent.log (with prefix)

    # ── 3. goal_manager.log — High-level autonomous loop events ───────────────
    gm_logger = logging.getLogger("noesis.goal_manager")
    gm_logger.handlers = []              # prevent duplicate lines on reload

    gm_formatter = logging.Formatter(
        "%(asctime)s  [%(levelname)-5s]  %(message)s", datefmt="%H:%M:%S"
    )
    gm_handler = logging.FileHandler("logs/goal_manager.log", mode="w", encoding="utf-8")
    gm_handler.setLevel(logging.INFO)
    gm_handler.setFormatter(gm_formatter)

    gm_logger.addHandler(gm_handler)
    gm_logger.propagate = True           # still flows to root → agent.log

    # ── 4. daemon.log — Background daemon lifecycle events ────────────────────
    daemon_logger = logging.getLogger("noesis.daemon")
    daemon_logger.handlers = []          # prevent duplicate lines on reload

    daemon_formatter = logging.Formatter(
        "%(asctime)s  [%(levelname)-5s]  %(message)s", datefmt="%H:%M:%S"
    )
    daemon_handler = logging.FileHandler("logs/daemon.log", mode="w", encoding="utf-8")
    daemon_handler.setLevel(logging.DEBUG)   # DEBUG so poll ticks are visible
    daemon_handler.setFormatter(daemon_formatter)

    daemon_logger.addHandler(daemon_handler)
    daemon_logger.propagate = True       # still flows to root → agent.log

    # ── 5. perception.log — Perception layer signal processing events ──────────
    #    Captures every batch: raw count, dedup count, per-event routing decisions,
    #    synthesizer latency, and any fallbacks triggered.
    #    Tail with:  tail -f logs/perception.log
    perception_logger = logging.getLogger("noesis.perception")
    perception_logger.handlers = []     # prevent duplicate lines on reload

    perception_formatter = logging.Formatter(
        "%(asctime)s  [%(levelname)-5s]  %(message)s", datefmt="%H:%M:%S"
    )
    perception_handler = logging.FileHandler("logs/perception.log", mode="w", encoding="utf-8")
    perception_handler.setLevel(logging.DEBUG)
    perception_handler.setFormatter(perception_formatter)

    perception_logger.addHandler(perception_handler)
    perception_logger.propagate = True  # still flows to root → agent.log

    # ── 6. llm.log — Full request/response pairs, one clean block each ────────
    #    The service.py code builds a complete block (request + response +
    #    timing/tokens) and passes it as a single logger.info() call.  We use
    #    a plain %(message)s formatter here so the block is written verbatim —
    #    no per-record timestamp prefix or separator that would double up.
    llm_logger = logging.getLogger("noesis.llm")
    llm_logger.handlers = []             # prevent duplicate lines on reload

    llm_formatter = logging.Formatter("%(message)s")
    llm_handler = logging.FileHandler("logs/llm.log", mode="w", encoding="utf-8")
    llm_handler.setLevel(logging.INFO)
    llm_handler.setFormatter(llm_formatter)

    llm_logger.addHandler(llm_handler)
    llm_logger.propagate = False         # do not spam root logger

    # ── 7. Console — Configurable level, clean short format ───────────────────
    console_formatter = logging.Formatter("[%(levelname)-8s] %(name)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

