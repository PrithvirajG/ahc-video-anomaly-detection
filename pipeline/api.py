"""
Detector service the dashboard talks to.

    uv run uvicorn pipeline.api:app --reload --port 8010
    cd web && npm run dev          # http://localhost:5173

Right now the detector itself is a stub: `_demo_stream` emits synthetic scores and
alerts so the whole path - websocket, timeline, alert feed, clip seeking - is
verified end to end before any model exists. Replace `run_detector` with the real
cascade and nothing on the frontend has to change.

Shape of the cascade this is built around (Cerberus, arXiv 2510.16290):
  stage 1  always-on CLIP/SigLIP frame scoring, tuned for >95% recall, cheap
  stage 2  a small VLM that only sees the ~5% of frames stage 1 escalates
Both stages emit into the same queue; `stage` on each alert says which fired.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_ROOT = Path(os.getenv("DATA_ROOT", ROOT / "data"))
if not DATA_ROOT.is_absolute():
    DATA_ROOT = ROOT / DATA_ROOT

CLASSES = [
    "normal", "traffic_accident", "traffic_congestion",
    "stalled_or_broken_down_vehicle", "vehicle_blocking_traffic",
    "wrong_way_driving", "road_spill_or_debris", "waterlogging_or_flood",
    "fire", "smoke", "fighting_or_violence", "loitering_or_suspicious_presence",
]

# Every connected dashboard gets the same broadcast.
CLIENTS: set[WebSocket] = set()
STATE = {"frames_seen": 0, "frames_escalated": 0, "alerts": 0, "fps": 0.0, "gpu": "—"}


async def broadcast(kind: str, data: dict) -> None:
    msg = json.dumps({"type": kind, "data": data})
    for ws in list(CLIENTS):
        try:
            await ws.send_text(msg)
        except Exception:
            CLIENTS.discard(ws)


async def _demo_stream() -> None:
    """Synthetic traffic so the UI can be built and demoed before the model lands.

    Deliberately imitates the real statistics: mostly-normal frames, occasional
    bursts, and a stage-1 alert that stage 2 either confirms or silently drops -
    the false-alarm behaviour the problem statement says matters as much as recall.
    """
    t = 0.0
    while True:
        await asyncio.sleep(0.1)
        t += 0.1
        STATE["frames_seen"] += 1

        # Slow drift with rare spikes; anomalies are ~1-5% of frames in real data.
        base = 0.12 + 0.06 * random.random()
        spike = random.random() < 0.01
        anomaly = min(1.0, base + (0.7 * random.random() if spike else 0.0))

        await broadcast("score", {"t": round(t, 1), "health": round(1 - anomaly, 3),
                                  "anomaly": round(anomaly, 3)})

        if anomaly > 0.5:
            STATE["frames_escalated"] += 1
            # Stage 2 rejects most escalations - that is the whole point of the cascade.
            confirmed = random.random() < 0.35
            STATE["alerts"] += 1
            cls = random.choice([c for c in CLASSES if c != "normal"])
            await broadcast("alert", {
                "id": str(uuid.uuid4()),
                "video_id": "T001",
                "class_name": cls,
                "confidence": round(anomaly, 3),
                "start_time_sec": round(t, 1),
                "end_time_sec": round(t + random.uniform(1, 4), 1),
                "description_summary": (
                    f"[stub] {cls.replace('_', ' ')} consistent with the learned norm being broken"
                    if confirmed else ""
                ),
                "stage": "vlm" if confirmed else "filter",
                "wall_clock": time.strftime("%H:%M:%S"),
            })

        if STATE["frames_seen"] % 10 == 0:
            seen = STATE["frames_seen"]
            await broadcast("stats", {
                "fps": 10.0,
                "frames_seen": seen,
                "frames_escalated": STATE["frames_escalated"],
                "alerts": STATE["alerts"],
                "escalation_rate": round(STATE["frames_escalated"] / max(seen, 1), 4),
                "gpu": STATE["gpu"],
            })


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import torch

        STATE["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        STATE["gpu"] = "cpu"

    task = asyncio.create_task(_demo_stream())
    yield
    task.cancel()


app = FastAPI(title="AHC anomaly detector", lifespan=lifespan)


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "classes": CLASSES, **STATE})


@app.get("/api/videos")
def videos() -> JSONResponse:
    """Whatever is on disk under data/test - the dashboard's clip picker."""
    test = DATA_ROOT / "test" / "videos"
    names = sorted(p.stem for p in test.glob("*.mp4")) if test.exists() else []
    return JSONResponse({"root": str(test), "videos": names})


@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket) -> None:
    await ws.accept()
    CLIENTS.add(ws)
    try:
        while True:
            await ws.receive_text()  # keeps the socket open; no client commands yet
    except WebSocketDisconnect:
        pass
    finally:
        CLIENTS.discard(ws)


# Serve the clips so the player can seek into an event interval. Mounted last so
# it cannot shadow the /api routes.
_clips = DATA_ROOT / "test" / "videos"
if _clips.exists():
    app.mount("/clips", StaticFiles(directory=str(_clips)), name="clips")
