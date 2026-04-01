import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay, MediaRecorder
import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-poc")

app = FastAPI(title="WebRTC Camera PoC")
relay = MediaRelay()


class Offer(BaseModel):
    sdp: str
    type: str


@dataclass
class CameraStream:
    camera_id: str
    source_track: object  # MediaStreamTrack
    publisher_pc: RTCPeerConnection
    recorder: Optional[MediaRecorder] = None
    recording_path: Optional[str] = None
    recording_id: Optional[str] = None


# One active publisher track per camera_id
camera_streams: Dict[str, CameraStream] = {}

# Track all peer connections so we can close them on shutdown
pcs: Set[RTCPeerConnection] = set()
recordings_root = Path("recordings")
db_pool: Optional[asyncpg.Pool] = None


app.mount("/static", StaticFiles(directory="static"), name="static")


SCHEMA_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS pgcrypto",
    """
    CREATE TABLE IF NOT EXISTS recordings (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        camera TEXT NOT NULL,
        file_path TEXT NOT NULL UNIQUE,
        started_at TIMESTAMPTZ NOT NULL,
        ended_at TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_recordings_started_at ON recordings (started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_recordings_ended_at ON recordings (ended_at DESC)",
]


async def init_db():
    global db_pool
    db_url = os.getenv("DATABASE_URL", "postgresql://webrtc:webrtcpass@localhost:5433/recordings")
    db_pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=5)
    async with db_pool.acquire() as conn:
        for statement in SCHEMA_STATEMENTS:
            await conn.execute(statement)
    logger.info("Connected to PostgreSQL and ensured recordings schema exists")


async def close_db():
    global db_pool
    if db_pool is not None:
        await db_pool.close()
        db_pool = None


async def insert_recording(camera: str, file_path: str, started_at: datetime) -> Optional[str]:
    if db_pool is None:
        return None
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO recordings (camera, file_path, started_at)
            VALUES ($1, $2, $3)
            RETURNING id::text
            """,
            camera,
            file_path,
            started_at,
        )
    return row["id"] if row else None


async def set_recording_ended(recording_id: str, ended_at: datetime):
    if db_pool is None:
        return
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE recordings
            SET ended_at = $1
            WHERE id = $2::uuid
            """,
            ended_at,
            recording_id,
        )


@app.get("/")
async def root():
    return {
        "message": "WebRTC camera PoC backend is running",
        "usage": {
            "viewer_page": "/camera/{camera_id}",
            "publish_offer": "/publish/{camera_id}/offer",
            "view_offer": "/view/{camera_id}/offer",
        },
    }


@app.get("/camera/{camera_id}")
async def camera_page(camera_id: str):
    return FileResponse("static/viewer.html")


@app.get("/api/cameras")
async def list_cameras():
    return {
        "cameras": sorted(list(camera_streams.keys()))
    }


@app.post("/publish/{camera_id}/offer")
async def publish_offer(camera_id: str, offer: Offer):
    """
    Raspberry Pi / publisher calls this endpoint.

    It sends a WebRTC offer with a video track.
    The backend stores the incoming video track under camera_id.
    """
    pc = RTCPeerConnection()
    pcs.add(pc)

    logger.info("Publisher connected for camera_id=%s", camera_id)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(
            "Publisher PC state for %s: %s",
            camera_id,
            pc.connectionState,
        )
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await cleanup_publisher(camera_id, pc)

    @pc.on("track")
    def on_track(track):
        logger.info(
            "Received track kind=%s for camera_id=%s",
            track.kind,
            camera_id,
        )

        if track.kind == "video":
            recordings_root.mkdir(parents=True, exist_ok=True)
            camera_dir = recordings_root / camera_id
            camera_dir.mkdir(parents=True, exist_ok=True)
            started_at = datetime.now(timezone.utc)
            filename = f"{started_at.strftime('%Y%m%d_%H%M%S')}.mp4"
            recording_path = camera_dir / filename
            recorder = MediaRecorder(str(recording_path))
            recorder.addTrack(track)
            relative_recording_path = recording_path.as_posix()

            async def start_stream():
                recording_id = None
                try:
                    recording_id = await insert_recording(
                        camera=camera_id,
                        file_path=relative_recording_path,
                        started_at=started_at,
                    )
                except Exception:
                    logger.exception(
                        "Failed to insert recording metadata for camera_id=%s",
                        camera_id,
                    )

                camera_streams[camera_id] = CameraStream(
                    camera_id=camera_id,
                    source_track=track,
                    publisher_pc=pc,
                    recorder=recorder,
                    recording_path=str(recording_path),
                    recording_id=recording_id,
                )
                logger.info("Stored live video track for camera_id=%s", camera_id)
                await start_recording(camera_id, pc, recorder)

            asyncio.create_task(start_stream())

        @track.on("ended")
        async def on_ended():
            logger.info("Track ended for camera_id=%s", camera_id)
            if camera_id in camera_streams:
                existing = camera_streams[camera_id]
                if existing.publisher_pc == pc:
                    await stop_recording(camera_id, pc)
                    del camera_streams[camera_id]

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )


@app.post("/view/{camera_id}/offer")
async def view_offer(camera_id: str, offer: Offer):
    """
    Browser viewer calls this endpoint.

    It sends a recvonly offer.
    The backend answers and forwards the current live track for camera_id.
    """
    if camera_id not in camera_streams:
        raise HTTPException(status_code=404, detail=f"No live stream for '{camera_id}'")

    pc = RTCPeerConnection()
    pcs.add(pc)

    logger.info("Viewer connected for camera_id=%s", camera_id)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(
            "Viewer PC state for %s: %s",
            camera_id,
            pc.connectionState,
        )
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await cleanup_pc(pc)

    stream = camera_streams[camera_id]
    relayed_track = relay.subscribe(stream.source_track)
    pc.addTrack(relayed_track)

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )


async def cleanup_publisher(camera_id: str, pc: RTCPeerConnection):
    if camera_id in camera_streams:
        existing = camera_streams[camera_id]
        if existing.publisher_pc == pc:
            await stop_recording(camera_id, pc)
            logger.info("Cleaning up publisher for camera_id=%s", camera_id)
            del camera_streams[camera_id]
    await cleanup_pc(pc)


async def cleanup_pc(pc: RTCPeerConnection):
    if pc in pcs:
        pcs.discard(pc)
    if pc.connectionState != "closed":
        await pc.close()


async def start_recording(camera_id: str, pc: RTCPeerConnection, recorder: MediaRecorder):
    try:
        await recorder.start()
        logger.info("Started recording for camera_id=%s", camera_id)
    except Exception:
        logger.exception("Failed to start recording for camera_id=%s", camera_id)
        if camera_id in camera_streams:
            existing = camera_streams[camera_id]
            if existing.publisher_pc == pc:
                existing.recorder = None


async def stop_recording(camera_id: str, pc: RTCPeerConnection):
    if camera_id not in camera_streams:
        return

    existing = camera_streams[camera_id]
    if existing.publisher_pc != pc:
        return
    if existing.recorder is None:
        return

    recorder = existing.recorder
    existing.recorder = None

    try:
        await recorder.stop()
        if existing.recording_id:
            try:
                await set_recording_ended(existing.recording_id, datetime.now(timezone.utc))
            except Exception:
                logger.exception(
                    "Failed to set ended_at for recording_id=%s",
                    existing.recording_id,
                )
        logger.info(
            "Saved recording for camera_id=%s to %s",
            camera_id,
            existing.recording_path,
        )
    except Exception:
        logger.exception("Failed to stop recording for camera_id=%s", camera_id)


@app.on_event("startup")
async def on_startup():
    await init_db()


@app.on_event("shutdown")
async def on_shutdown():
    await asyncio.gather(
        *(stop_recording(camera_id, stream.publisher_pc) for camera_id, stream in list(camera_streams.items())),
        return_exceptions=True,
    )
    await asyncio.gather(*(cleanup_pc(pc) for pc in list(pcs)), return_exceptions=True)
    await close_db()
    pcs.clear()
    camera_streams.clear()
