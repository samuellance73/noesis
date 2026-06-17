import logging
import os

def setup_global_logging(console_level=logging.WARNING):
    """
    Configures logging to write all detailed logs to logs/agent.log,
    while allowing the console output to be set independently.
    """
    # 1. Create a logs/ directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Capture everything at INFO level or higher
    
    # Remove existing handlers to avoid duplicates during restarts
    root_logger.handlers = []

    # 2. FILE HANDLER (Writes everything to logs/agent.log)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    )
    file_handler = logging.FileHandler("logs/agent.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 3. CONSOLE HANDLER (Prints to terminal screen)
    console_formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level) # Configured per-interface
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
