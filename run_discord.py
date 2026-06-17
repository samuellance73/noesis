import os
import logging
from dotenv import load_dotenv
from utils.logging_setup import setup_global_logging
from interfaces.discord.bot import bot

# Both the console terminal and logs/agent.log will print INFO operations
setup_global_logging(console_level=logging.INFO)

# Load environment variables
load_dotenv(override=True)

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("Error: DISCORD_BOT_TOKEN not found in .env")
        exit(1)
        
    print("Starting Discord Agent Interface...")
    bot.run(DISCORD_TOKEN)

