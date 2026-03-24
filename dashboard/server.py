"""
Dashboard Server - FastAPI backend for the browser command center.
Authentication, MJPEG streaming, chat, and live browser speech updates.
"""

import asyncio
import base64
import json
import logging
import os
import secrets
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("dashboard.server")


DASHBOARD_PASSWORD = os.environ.get("VANGUARD_PASSWORD", "vanguard123")
SESSION_COOKIE = "vanguard_session"
_valid_sessions: set[str] = set()


_state_lock = threading.Lock()
_connected_clients: List[WebSocket] = []
_alert_history: List[Dict[str, Any]] = []
_log_history: List[Dict[str, Any]] = []
_chat_history: Dict[str, List[Dict[str, str]]] = {}
_MAX_HISTORY = 100
_MAX_CHAT_TURNS = 12

_latest_frame_jpg: Optional[bytes] = None
_system_status: Dict[str, Any] = {
    "status": "Starting...",
    "device": "Unknown",
    "yolo_model": "Unknown",
    "brain_model": "Unknown",
    "tts_model": "Unknown",
    "stt_backend": "Unknown",
}

_live_reply_condition = threading.Condition()
_pending_live_reply: Optional[Dict[str, Any]] = None


def _create_session() -> str:
    token = secrets.token_urlsafe(32)
    _valid_sessions.add(token)
    return token


def _is_authenticated(request: Request) -> bool:
    token = request.cookies.get(SESSION_COOKIE)
    return token in _valid_sessions


def _ws_is_authenticated(ws: WebSocket) -> bool:
    token = ws.cookies.get(SESSION_COOKIE)
    return token in _valid_sessions


def update_system_status(status_updates: Dict[str, Any]):
    with _state_lock:
        _system_status.update(status_updates)


def update_frame(frame: np.ndarray):
    global _latest_frame_jpg
    try:
        success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if success:
            _latest_frame_jpg = buf.tobytes()
    except Exception as exc:
        log.warning("Failed to encode live frame: %s", exc)


def push_log(message: str, level: str = "info"):
    entry = {
        "type": "log",
        "level": level,
        "message": message,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    with _state_lock:
        _log_history.append(entry)
        if len(_log_history) > _MAX_HISTORY:
            _log_history.pop(0)
    _broadcast(entry)


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


def _broadcast(data: Dict[str, Any]):
    if _loop is None:
        return

    message = json.dumps(data)
    disconnected: List[WebSocket] = []

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


def request_browser_live_reply(timeout_seconds: float = 2.0) -> str:
    global _pending_live_reply

    with _state_lock:
        has_clients = bool(_connected_clients)

    if not has_clients:
        return ""

    request_payload = {
        "id": secrets.token_urlsafe(12),
        "transcript": "",
        "done": False,
    }

    with _live_reply_condition:
        _pending_live_reply = request_payload

    _broadcast(
        {
            "type": "listen_request",
            "request_id": request_payload["id"],
            "timeout": max(1.0, float(timeout_seconds)),
        }
    )

    deadline = time.time() + max(1.5, float(timeout_seconds) + 1.0)

    with _live_reply_condition:
        while time.time() < deadline:
            if request_payload["done"]:
                transcript = str(request_payload["transcript"]).strip()
                if _pending_live_reply is request_payload:
                    _pending_live_reply = None
                return transcript
            _live_reply_condition.wait(timeout=max(0.1, deadline - time.time()))

        if _pending_live_reply is request_payload:
            _pending_live_reply = None

    return ""


app = FastAPI(title="Vanguard AI Dashboard")
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/login")
async def login_page():
    with open(os.path.join(_static_dir, "login.html"), encoding="utf-8") as handle:
        html = handle.read()
    html = html.replace("{{ error }}", "")
    return HTMLResponse(html)


@app.post("/login")
async def login_submit(password: str = Form(...)):
    if password == DASHBOARD_PASSWORD:
        token = _create_session()
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(SESSION_COOKIE, token, httponly=True, max_age=86400)
        return response

    with open(os.path.join(_static_dir, "login.html"), encoding="utf-8") as handle:
        html = handle.read()
    html = html.replace("{{ error }}", "Incorrect password. Try again.")
    return HTMLResponse(html)


@app.get("/")
async def root(request: Request):
    if not _is_authenticated(request):
        return RedirectResponse(url="/login")
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.get("/api/system_status")
async def api_system_status(request: Request):
    if not _is_authenticated(request):
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)
    with _state_lock:
        return JSONResponse(content=dict(_system_status))


@app.get("/api/history")
async def get_history(request: Request):
    if not _is_authenticated(request):
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)

    with _state_lock:
        history = []
        for alert in _alert_history[-50:]:
            entry = {key: value for key, value in alert.items() if key != "frame"}
            entry["has_frame"] = alert.get("frame") is not None
            history.append(entry)
    return JSONResponse(content=history)


@app.get("/api/logs")
async def get_logs(request: Request):
    if not _is_authenticated(request):
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)

    with _state_lock:
        return JSONResponse(content=list(_log_history[-100:]))


@app.post("/api/live_reply")
async def live_reply_api(request: Request):
    global _pending_live_reply

    if not _is_authenticated(request):
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    request_id = str(payload.get("request_id", "")).strip()
    transcript = str(payload.get("transcript", "")).strip()

    if not request_id:
        return JSONResponse(content={"error": "missing_request_id"}, status_code=400)

    with _live_reply_condition:
        pending = _pending_live_reply
        if pending and pending.get("id") == request_id:
            pending["transcript"] = transcript[:500]
            pending["done"] = True
            _live_reply_condition.notify_all()

    return JSONResponse(content={"ok": True})


async def _frame_generator():
    while True:
        if _latest_frame_jpg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + _latest_frame_jpg + b"\r\n"
            )
        await asyncio.sleep(0.066)


@app.get("/api/video_feed")
async def video_feed(request: Request):
    if not _is_authenticated(request):
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)

    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if not _ws_is_authenticated(ws):
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    with _state_lock:
        _connected_clients.append(ws)

    await ws.send_text(
        json.dumps(
            {
                "type": "connected",
                "message": "Vanguard AI Dashboard Connected",
                "alerts_total": len(_alert_history),
            }
        )
    )

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


_brain_ref = None
_tts_ref = None
_stream_ref = None
_chat_handler_ref = None


def set_chat_refs(brain, tts, stream, chat_handler=None):
    global _brain_ref, _tts_ref, _stream_ref
    _brain_ref = brain
    _tts_ref = tts
    _stream_ref = stream
    global _chat_handler_ref
    _chat_handler_ref = chat_handler


def _append_chat_turn(session_id: str, role: str, message: str):
    text = str(message or "").strip()
    if not text:
        return

    with _state_lock:
        history = _chat_history.setdefault(session_id, [])
        history.append({"role": role, "message": text[:500]})
        if len(history) > _MAX_CHAT_TURNS:
            del history[:-_MAX_CHAT_TURNS]


def _build_chat_context(session_id: str) -> str:
    parts: List[str] = []

    with _state_lock:
        chat_turns = list(_chat_history.get(session_id, []))[-_MAX_CHAT_TURNS:]
        live_logs = list(_log_history[-6:])

    if chat_turns:
        chat_lines = []
        for turn in chat_turns:
            speaker = "User" if turn.get("role") == "user" else "Vanguard"
            chat_lines.append(f"{speaker}: {turn.get('message', '')}")
        parts.append("Recent browser chat:\n" + "\n".join(chat_lines))

    relevant_logs = []
    for entry in live_logs:
        message = str(entry.get("message", "")).strip()
        if message:
            relevant_logs.append(f"[{entry.get('timestamp', '--:--:--')}] {message}")
    if relevant_logs:
        parts.append("Recent live activity:\n" + "\n".join(relevant_logs))

    return "\n\n".join(parts)


def _run_chat_response(user_text: str, session_id: str = "browser") -> str:
    if not _brain_ref:
        return "Brain not loaded yet."

    frame = None
    if _stream_ref:
        try:
            frame = _stream_ref.read()
        except Exception:
            frame = None

    _append_chat_turn(session_id, "user", user_text)
    push_log(f'Console heard: "{user_text[:120]}"', "info")

    context = _build_chat_context(session_id)
    if _chat_handler_ref is not None:
        response = _chat_handler_ref(user_text, frame, context)
    else:
        response = _brain_ref.chat(user_text, frame, context)
    response = str(response or "").strip() or "I heard you, but I do not have a solid answer yet."

    _append_chat_turn(session_id, "assistant", response)
    push_log(f'Console reply: "{response[:120]}"', "info")

    if _tts_ref and _tts_ref.available:
        threading.Thread(
            target=_tts_ref.speak,
            args=(response[:200],),
            daemon=True,
        ).start()

    return response


@app.post("/api/chat")
async def chat_api(request: Request):
    if not _is_authenticated(request):
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    user_text = str(payload.get("message", "")).strip()
    if not user_text:
        return JSONResponse(content={"error": "empty_message"}, status_code=400)

    loop = asyncio.get_event_loop()
    session_id = request.cookies.get(SESSION_COOKIE, "browser")
    response = await loop.run_in_executor(None, _run_chat_response, user_text, session_id)
    return JSONResponse(content={"message": response})


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
