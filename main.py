import os

from dotenv import load_dotenv

# Must happen before any network imports — patches global SSL context so
# restricted environments (NixOS, missing CA bundles) can reach HTTPS APIs.
import utils.ssl_patch as _ssl
_ssl.apply()

load_dotenv(override=True)

from utils.log_writer import clear_log
clear_log()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
