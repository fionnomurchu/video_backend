import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay, MediaRecorder

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


# One active publisher track per camera_id
camera_streams: Dict[str, CameraStream] = {}

# Track all peer connections so we can close them on shutdown
pcs: Set[RTCPeerConnection] = set()
recordings_root = Path("recordings")


app.mount("/static", StaticFiles(directory="static"), name="static")


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
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            recording_path = camera_dir / filename
            recorder = MediaRecorder(str(recording_path))
            recorder.addTrack(track)

            camera_streams[camera_id] = CameraStream(
                camera_id=camera_id,
                source_track=track,
                publisher_pc=pc,
                recorder=recorder,
                recording_path=str(recording_path),
            )
            logger.info("Stored live video track for camera_id=%s", camera_id)
            asyncio.create_task(start_recording(camera_id, pc, recorder))

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
        logger.info(
            "Saved recording for camera_id=%s to %s",
            camera_id,
            existing.recording_path,
        )
    except Exception:
        logger.exception("Failed to stop recording for camera_id=%s", camera_id)


@app.on_event("shutdown")
async def on_shutdown():
    await asyncio.gather(
        *(stop_recording(camera_id, stream.publisher_pc) for camera_id, stream in list(camera_streams.items())),
        return_exceptions=True,
    )
    await asyncio.gather(*(cleanup_pc(pc) for pc in list(pcs)), return_exceptions=True)
    pcs.clear()
    camera_streams.clear()
