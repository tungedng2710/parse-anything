"""Development entry point for the MonkeyOCR FastAPI application."""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
