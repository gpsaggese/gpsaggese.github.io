"""FastAPI application for AgenticEDA."""
from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import session, notebook, run, kernel, story, history, stream, chat

SESSIONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "sessions"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create sessions directory on startup."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="AgenticEDA", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router, prefix="/api")
app.include_router(notebook.router, prefix="/api")
app.include_router(run.router, prefix="/api")
app.include_router(story.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(stream.router, prefix="/api")
# kernel router added separately (has WebSocket)
app.include_router(kernel.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

# Serve session files statically
app.mount("/files", StaticFiles(directory=str(SESSIONS_DIR), check_dir=False), name="session_files")
