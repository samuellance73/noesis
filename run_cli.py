import asyncio
from dotenv import load_dotenv
from interfaces.cli.main import run_terminal_interface
from utils.log_writer import emit

# System startup logging
emit("system.startup", "system", {})

# Load credentials
load_dotenv(override=True)

if __name__ == "__main__":
    try:
        asyncio.run(run_terminal_interface())
    except KeyboardInterrupt:
        print("\nExiting CLI...")

