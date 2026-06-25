# test_ingest.py
import asyncio
import os
import sys

# Patch SSL
import utils.ssl_patch as _ssl
_ssl.apply()

from dotenv import load_dotenv
load_dotenv(override=True)

from main import app
from perception.schemas import RawSignal, RawSignalSource, SourceType, Priority

async def main():
    # Wait for app state to be ready (fastapi app is not running via server, but we can start perception layer)
    from app.lifespan import lifespan
    async with lifespan(app):
        print("Lifespan started, ingesting signal...")
        
        text = """Recent conversation context:
  [carma0972]: try agian
  [carma0972]: hiii

User 'carma' (@carma0972) in channel 1517315368925003966 said:
can you make a github site with some test code and reply in this discord channel with the url? Reply to THIS channel using your tool"""

        signal = RawSignal(
            source=RawSignalSource(
                type=SourceType.USER,
                identifier="carma0972",
                display_name="carma",
            ),
            text=text,
            priority=Priority.NORMAL,
            channel_id="1517315368925003966",
        )
        
        await app.state.perception.ingest(signal)
        print("Signal ingested. Waiting 15 seconds for processing...")
        await asyncio.sleep(15.0)
        print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
