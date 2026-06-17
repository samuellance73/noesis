import asyncio
import logging
from dotenv import load_dotenv
from utils.logging_setup import setup_global_logging
from interfaces.cli.main import run_terminal_interface

# Configures File logs to run at INFO level in background,
# but keeps the terminal console silent so rich UI remains clean.
setup_global_logging(console_level=logging.WARNING)

# Load credentials
load_dotenv(override=True)

if __name__ == "__main__":
    try:
        asyncio.run(run_terminal_interface())
    except KeyboardInterrupt:
        print("\nExiting CLI...")

