"""
Dashboard Server — FastAPI backend for Public Website & Mobile PWA.
Authentication, MJPEG streaming, WebSocket alerts, ngrok tunnel.
"""

import asyncio
import base64
import json
import logging
import os
import secrets
import socket
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse, StreamingResponse

log = logging.getLogger("dashboard.server")

# ── Config ──────────────────────────────────────────────────────────
DASHBOARD_PASSWORD = os.environ.get("VANGUARD_PASSWORD", "vanguard123")
SESSION_COOKIE = "vanguard_session"
_valid_sessions: set = set()

# ── Shared state (thread-safe) ──────────────────────────────────────
_state_lock = threading.Lock()
_connected_clients: List[WebSocket] = []
_alert_history: List[Dict[str, Any]] = []
_MAX_HISTORY = 100

# Desktop Command Center State
_latest_frame_jpg: Optional[bytes] = None
_system_status: Dict[str, Any] = {
    "status": "Starting...",
    "device": "Unknown",
    "yolo_model": "Unknown",
    "brain_model": "Unknown",
    "tts_model": "Unknown",
}


# ── Auth Helpers ────────────────────────────────────────────────────

def _create_session() -> str:
    token = secrets.token_urlsafe(32)
    _valid_sessions.add(token)
    return token


def _is_authenticated(request: Request) -> bool:
    token = request.cookies.get(SESSION_COOKIE)
    return token in _valid_sessions


def _ws_is_authenticated(ws: WebSocket) -> bool:
    """Check if a WebSocket connection has a valid session cookie."""
    token = ws.cookies.get(SESSION_COOKIE)
    return token in _valid_sessions


# ── Frame / Status API (called from main.py) ───────────────────────

def update_system_status(status_updates: Dict[str, Any]):
    _system_status.update(status_updates)


def update_frame(frame: np.ndarray):
    global _latest_frame_jpg
    try:
        success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if success:
            _latest_frame_jpg = buf.tobytes()
    except Exception as e:
        log.warning("Failed to encode live frame: %s", e)


def push_log(message: str, level: str = "info"):
    _broadcast({
        "type": "log",
        "level": level,
        "message": message,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })


def alert_clients(verdict_data: Dict[str, Any], frame: Optional[np.ndarray] = None):
    alert = {
        "type": "alert",
        "timestamp": datetime.now().isoformat(),
        "threat_score": verdict_data.get("threat_score", 0.0),
        "chain_of_thought": verdict_data.get("chain_of_thought", ""),
        "actions": verdict_data.get("actions", []),
        "frame": None,
    }
    if frame is not None:
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            alert["frame"] = base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception:
            pass

    with _state_lock:
        _alert_history.append(alert)
        if len(_alert_history) > _MAX_HISTORY:
            _alert_history.pop(0)

    _broadcast(alert)


def _broadcast(data: dict):
    if _loop is None:
        return
    message = json.dumps(data)
    disconnected = []

    with _state_lock:
        clients_snapshot = list(_connected_clients)

    for ws in clients_snapshot:
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(message), _loop)
        except Exception:
            disconnected.append(ws)

    if disconnected:
        with _state_lock:
            for ws in disconnected:
                if ws in _connected_clients:
                    _connected_clients.remove(ws)


# ── FastAPI app ─────────────────────────────────────────────────────

app = FastAPI(title="Vanguard AI Dashboard")
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Auth Routes ─────────────────────────────────────────────────────

@app.get("/login")
async def login_page():
    html = open(os.path.join(_static_dir, "login.html"), encoding="utf-8").read()
    html = html.replace("{{ error }}", "")
    return HTMLResponse(html)


@app.post("/login")
async def login_submit(password: str = Form(...)):
    if password == DASHBOARD_PASSWORD:
        token = _create_session()
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(SESSION_COOKIE, token, httponly=True, max_age=86400)
        return response
    else:
        html = open(os.path.join(_static_dir, "login.html"), encoding="utf-8").read()
        html = html.replace("{{ error }}", "Incorrect password. Try again.")
        return HTMLResponse(html)


# ── Protected Routes ────────────────────────────────────────────────

@app.get("/")
async def root(request: Request):
    if not _is_authenticated(request):
        return RedirectResponse(url="/login")
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.get("/manifest.json")
async def manifest():
    return FileResponse(os.path.join(_static_dir, "manifest.json"))


@app.get("/sw.js")
async def service_worker():
    return FileResponse(os.path.join(_static_dir, "sw.js"), media_type="application/javascript")


@app.get("/api/system_status")
async def api_system_status(request: Request):
    if not _is_authenticated(request):
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)
    return JSONResponse(content=_system_status)


@app.get("/api/network")
async def get_network(request: Request):
    if not _is_authenticated(request):
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    return JSONResponse(content={"ip": ip})


@app.get("/api/history")
async def get_history(request: Request):
    if not _is_authenticated(request):
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)
    with _state_lock:
        history = []
        for a in _alert_history[-50:]:
            entry = {k: v for k, v in a.items() if k != "frame"}
            entry["has_frame"] = a.get("frame") is not None
            history.append(entry)
    return JSONResponse(content=history)


async def _frame_generator():
    while True:
        if _latest_frame_jpg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + _latest_frame_jpg + b"\r\n"
            )
        await asyncio.sleep(0.066)  # ~15fps streaming


@app.get("/api/video_feed")
async def video_feed(request: Request):
    if not _is_authenticated(request):
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Auth check
    if not _ws_is_authenticated(ws):
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    with _state_lock:
        _connected_clients.append(ws)

    await ws.send_text(json.dumps({
        "type": "connected",
        "message": "Vanguard AI Dashboard Connected",
        "alerts_total": len(_alert_history),
    }))
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with _state_lock:
            if ws in _connected_clients:
                _connected_clients.remove(ws)


# ── Chat Console (two-way Brain interaction) ───────────────────────

_brain_ref = None
_tts_ref = None
_stream_ref = None


def set_chat_refs(brain, tts, stream):
    """Called from main.py to wire Brain/TTS/Stream into the chat endpoint."""
    global _brain_ref, _tts_ref, _stream_ref
    _brain_ref = brain
    _tts_ref = tts
    _stream_ref = stream


@app.websocket("/ws/chat")
async def chat_endpoint(ws: WebSocket):
    """Two-way chat: user types -> Brain thinks -> response streams back -> TTS speaks."""
    # Auth check
    if not _ws_is_authenticated(ws):
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            user_text = msg.get("message", "").strip()

            if not user_text:
                continue

            if not _brain_ref:
                await ws.send_text(json.dumps({
                    "type": "chat_response",
                    "message": "Brain not loaded yet.",
                }))
                continue

            # Grab latest frame for context
            frame = None
            if _stream_ref:
                try:
                    frame = _stream_ref.read()
                except Exception:
                    pass

            # Run Brain chat in a thread to not block the event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, _brain_ref.chat, user_text, frame
            )

            await ws.send_text(json.dumps({
                "type": "chat_response",
                "message": response,
            }))

            # Auto-trigger TTS if available
            if _tts_ref and _tts_ref.available:
                threading.Thread(
                    target=_tts_ref.speak, args=(response[:200],), daemon=True
                ).start()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error("Chat WebSocket error: %s", e)


# ── Server runner ───────────────────────────────────────────────────

_loop: Optional[asyncio.AbstractEventLoop] = None


class DashboardServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Dashboard starting at http://%s:%d", self.host, self.port)

    def _run(self):
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="error",
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        _loop.run_until_complete(server.serve())
