"""WebSocket endpoint for agent streaming events."""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

_LOG = logging.getLogger(__name__)
router = APIRouter(tags=["stream"])

# In-memory dict of session_id -> list of events
# Buffers persist so late-connecting clients can replay all events
_progress_buffers: dict[str, list[dict]] = {}
_buffer_timestamps: dict[str, float] = {}

# How long to keep buffers after completion (seconds)
_BUFFER_TTL = 300


def push_event(session_id: str, event: dict):
    """Called by the agent runner to push streaming events."""
    if session_id not in _progress_buffers:
        _progress_buffers[session_id] = []
    _progress_buffers[session_id].append(event)
    _buffer_timestamps[session_id] = time.time()


def _cleanup_old_buffers():
    """Remove buffers older than TTL."""
    now = time.time()
    expired = [
        sid for sid, ts in _buffer_timestamps.items()
        if now - ts > _BUFFER_TTL
    ]
    for sid in expired:
        _progress_buffers.pop(sid, None)
        _buffer_timestamps.pop(sid, None)


@router.websocket("/stream/{session_id}")
async def stream_events(websocket: WebSocket, session_id: str):
    """Stream agent events over WebSocket. Late clients replay from the start."""
    await websocket.accept()
    cursor = 0
    _cleanup_old_buffers()

    try:
        while True:
            events = _progress_buffers.get(session_id, [])

            # Send any new events since our cursor
            while cursor < len(events):
                await websocket.send_json(events[cursor])
                cursor += 1

            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        _LOG.info("Stream client disconnected for session %s", session_id)
    # Buffer is NOT cleaned up — other clients may reconnect
