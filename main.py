"""
Vanguard AI — Main Entry Point
===============================
Root orchestrator. Ties together:
  1. Sentinel (YOLO Vision)
  2. Brain (Qwen2.5-VL Reasoning via llama.cpp)
  3. Actions (TTS, STT, Telegram, Dashboard)
  4. Desktop GUI + Web Dashboard
"""

import os
import sys
import signal
from pathlib import Path

# ── CUDA DLL Fix ─────────────────────────────────────────────────────
# llama-cpp-python needs CUDA runtime DLLs (cudart64_12.dll, cublas64_12.dll).
# PyTorch bundles these. Register PyTorch's lib dir so Windows can find them.
try:
    import torch
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_torch_lib)
except Exception:
    pass

import argparse
import logging
import time
import threading
from datetime import datetime

import cv2
import numpy as np


def setup_logging(level: str = "INFO", log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(log_dir, f"vanguard_{timestamp}.log"),
                encoding="utf-8",
            ),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Vanguard AI Security Pipeline")
    parser.add_argument("--source", default=None, help="Camera source: 0 for webcam, or RTSP URL.")
    parser.add_argument("--dry-run", action="store_true", help="Test without loading heavy models.")
    parser.add_argument("--no-gui", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-ngrok", action="store_true", help="Skip ngrok tunnel.")
    parser.add_argument("--no-stt", action="store_true", help="Disable microphone listening.")
    parser.add_argument("--port", type=int, default=None, help="Dashboard server port.")
    parser.add_argument("--password", default=None, help="Set dashboard access password.")
    parser.add_argument("--telegram-token", default=None, help="Telegram Bot API token.")
    parser.add_argument("--telegram-chat", default=None, help="Telegram chat ID for alerts.")
    return parser.parse_args()


def start_ngrok(port: int, log) -> str:
    """Start ngrok tunnel and return the public URL."""
    try:
        from pyngrok import ngrok
        tunnel = ngrok.connect(port, "http")
        url = tunnel.public_url
        log.info("[NGROK] Tunnel active: %s", url)
        return url
    except Exception as e:
        log.warning("Ngrok failed: %s", e)
        return ""


# ── Graceful Shutdown ────────────────────────────────────────────────

_shutdown_event = threading.Event()


def _signal_handler(sig, frame):
    logging.getLogger("main").info("Shutdown signal received.")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)
try:
    signal.signal(signal.SIGTERM, _signal_handler)
except (OSError, AttributeError):
    pass  # SIGTERM not available on Windows


def dry_run(args):
    """Quick pipeline test without loading models."""
    from config import settings
    setup_logging(settings.LOG_LEVEL, settings.LOG_DIR)
    log = logging.getLogger("main")
    log.info("=" * 60)
    log.info("DRY RUN — Testing pipeline flow")
    log.info("=" * 60)
    log.info("Config loaded [OK]")
    log.info("  YOLO:   %s", settings.YOLO_MODEL)
    log.info("  Brain:  %s", settings.BRAIN_MODEL)
    log.info("  TTS:    %s", settings.KOKORO_MODEL)
    log.info("  STT:    %s", settings.STT_MODEL)
    log.info("  Device: %s", settings.YOLO_DEVICE)

    # Start dashboard
    from dashboard.server import DashboardServer, update_system_status
    port = args.port or settings.DASHBOARD_PORT
    if args.password:
        import dashboard.server as ds
        ds.DASHBOARD_PASSWORD = args.password

    dashboard_server = DashboardServer(port=port)
    dashboard_server.start()

    update_system_status({
        "status": "Dry Run",
        "device": settings.YOLO_DEVICE,
        "yolo_model": settings.YOLO_MODEL,
        "brain_model": settings.BRAIN_MODEL,
        "tts_model": settings.KOKORO_MODEL,
    })

    # Start ngrok
    ngrok_url = ""
    if not args.no_ngrok:
        ngrok_url = start_ngrok(port, log)

    log.info("")
    log.info("Dashboard: http://localhost:%d", port)
    if ngrok_url:
        log.info("Public:    %s", ngrok_url)
    log.info("Password:  %s", args.password or "vanguard123")
    log.info("Dry run complete. Press Ctrl+C to stop.")

    _shutdown_event.wait()


def main():
    args = parse_args()
    from config import settings

    setup_logging(settings.LOG_LEVEL, settings.LOG_DIR)
    log = logging.getLogger("main")

    if args.source is not None:
        try:
            settings.CAMERA_SOURCE = int(args.source)
        except ValueError:
            settings.CAMERA_SOURCE = args.source

    if args.dry_run:
        dry_run(args)
        return

    # ── Banner ───────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  VANGUARD AI — Autonomous Security Pipeline")
    log.info("  Device: %s | Camera: %s", settings.YOLO_DEVICE, settings.CAMERA_SOURCE)
    log.info("=" * 60)

    # ── Start Dashboard Server ───────────────────────────────────────
    from dashboard.server import DashboardServer, update_system_status, update_frame, push_log
    import dashboard.server as ds

    port = args.port or settings.DASHBOARD_PORT
    if args.password:
        ds.DASHBOARD_PASSWORD = args.password

    dashboard_server = DashboardServer(port=port)
    dashboard_server.start()
    log.info("Dashboard at http://localhost:%d (password: %s)", port, ds.DASHBOARD_PASSWORD)

    # ── Start Ngrok ──────────────────────────────────────────────────
    ngrok_url = ""
    if not args.no_ngrok:
        ngrok_url = start_ngrok(port, log)

    # ── Initialize Components ────────────────────────────────────────
    log.info("Initializing Sentinel (YOLO)...")
    push_log("Loading YOLO detector...", "info")
    from sentinel.stream import VideoStream
    from sentinel.detector import SentinelDetector

    stream = VideoStream(
        source=settings.CAMERA_SOURCE,
        width=settings.FRAME_WIDTH,
        height=settings.FRAME_HEIGHT,
    )
    detector = SentinelDetector(
        model_path=settings.YOLO_MODEL,
        confidence=settings.YOLO_CONFIDENCE,
        device=settings.YOLO_DEVICE,
        imgsz=settings.YOLO_IMGSZ,
        trigger_classes=settings.TRIGGER_CLASSES,
    )

    log.info("Initializing Brain...")
    push_log("Loading Brain model...", "info")
    from brain.reasoner import BrainReasoner

    brain = BrainReasoner(
        model_path=settings.BRAIN_MODEL,
        mmproj_path=settings.BRAIN_MMPROJ,
        n_gpu_layers=settings.BRAIN_GPU_LAYERS,
        n_ctx=settings.BRAIN_N_CTX,
        max_tokens=settings.BRAIN_MAX_TOKENS,
    )

    log.info("Initializing Actions (TTS + Dashboard + Telegram)...")
    push_log("Loading TTS voice...", "info")
    from actions.alert_action import DashboardAlert
    from actions.dispatcher import ActionDispatcher

    class _NoOpTTS:
        available = False

        @staticmethod
        def speak(_text: str) -> bool:
            return False

        @staticmethod
        def speak_async(_text: str):
            return None

    tts = _NoOpTTS()
    if settings.TTS_BACKEND == "system":
        try:
            from actions.system_tts import SystemTTS

            tts = SystemTTS(
                rate=settings.SYSTEM_TTS_RATE,
                voice_hint=settings.SYSTEM_TTS_VOICE_HINT,
            )
            if not tts.available:
                log.warning("System TTS unavailable, staying on no-op TTS")
                tts = _NoOpTTS()
        except BaseException as e:
            log.error("System TTS init crashed, staying on no-op TTS: %s", e)
            tts = _NoOpTTS()
    elif settings.TTS_BACKEND == "kokoro":
        try:
            from actions.tts_action import KokoroTTS
            tts = KokoroTTS(
                model_id=settings.KOKORO_MODEL,
                voice=settings.KOKORO_VOICE,
                speed=settings.KOKORO_SPEED,
                lang=settings.KOKORO_LANG,
            )
        except BaseException as e:
            log.error("TTS init crashed, staying on no-op TTS: %s", e)
    else:
        log.warning("TTS disabled (VANGUARD_TTS_BACKEND=%s)", settings.TTS_BACKEND)
    dashboard_alert = DashboardAlert()

    # ── STT (Speech-to-Text) ─────────────────────────────────────────
    listener = None
    if not args.no_stt:
        log.info("Initializing STT (faster-whisper)...")
        push_log("Loading speech recognition...", "info")
        from sentinel.listener import AudioListener
        try:
            listener = AudioListener(
                model_size=settings.STT_MODEL,
                device=settings.STT_DEVICE,
                language=settings.STT_LANGUAGE,
                listen_duration=settings.STT_LISTEN_DURATION,
            )
            if listener.available:
                push_log("STT ready — microphone active", "success")
            else:
                log.warning("STT initialized but not available (missing mic or model)")
                listener = None
        except Exception as e:
            log.warning("STT init failed: %s", e)
            listener = None

    # ── Wire Chat Console refs into dashboard ────────────────────────
    from dashboard.server import set_chat_refs
    set_chat_refs(brain=brain, tts=tts, stream=stream)

    # ── Telegram Uplink ──────────────────────────────────────────────
    telegram = None
    tg_token = args.telegram_token or settings.TELEGRAM_BOT_TOKEN
    tg_chat = args.telegram_chat or settings.TELEGRAM_CHAT_ID
    if tg_token:
        from actions.telegram_action import TelegramUplink
        telegram = TelegramUplink(
            bot_token=tg_token,
            chat_id=tg_chat,
            tts_callback=tts.speak,
            frame_callback=lambda: stream.read(),
        )
        telegram.start_polling()
        push_log("Telegram uplink active", "success")
    else:
        log.info("Telegram uplink disabled (no token)")

    dispatcher = ActionDispatcher(tts=tts, dashboard=dashboard_alert, telegram=telegram)

    # Update system status
    update_system_status({
        "status": "Running",
        "device": settings.YOLO_DEVICE,
        "yolo_model": settings.YOLO_MODEL,
        "brain_model": settings.BRAIN_MODEL,
        "tts_model": settings.KOKORO_MODEL,
    })
    push_log("All models loaded. System armed.", "success")

    # ── Launch Desktop GUI ───────────────────────────────────────────
    # ── Start Pipeline ─────────────────────────────────────────────────
    stream.start()
    last_brain_time = 0.0
    _brain_busy = False
    _brain_lock = threading.Lock()
    last_prealert_time = 0.0
    frame_index = 0
    last_result = None

    def brain_worker(frame_snapshot, summary, person_crops=None):
        """Run Brain analysis + STT in a separate thread."""
        nonlocal last_brain_time, _brain_busy
        try:
            push_log("Analyzing with Brain...", "info")
            # ── Listen for speech while analyzing ────────────────────
            audio_transcript = ""
            if listener and listener.available:
                push_log("Listening for speech...", "info")
                audio_transcript = listener.listen()
                if audio_transcript:
                    push_log(f"Heard: \"{audio_transcript}\"", "info")

            verdict = brain.analyze(
                frame=frame_snapshot,
                detection_summary=summary,
                audio_transcript=audio_transcript,
            )

            if verdict is None:
                log.warning("Brain returned no verdict")
                return

            threat = verdict.get("threat_score", 0.0)
            actions = verdict.get("actions", [])
            cot = verdict.get("chain_of_thought", "")

            log.info("Verdict — threat=%.2f | actions=%d", threat, len(actions))
            push_log(f"Verdict: Threat={threat:.0%}, Actions={len(actions)}", "info")

            if cot:
                push_log(f"Brain: {cot}", "info")

            # Dispatch actions
            verdict_data = {
                "threat_score": threat,
                "chain_of_thought": cot,
                "actions": actions,
            }
            dispatcher.dispatch(actions, frame=frame_snapshot, verdict_data=verdict_data)

        except Exception as e:
            log.error("Brain analysis failed: %s", e)
        finally:
            with _brain_lock:
                _brain_busy = False

    def pipeline_loop():
        """Fast loop: read frames, run YOLO, push to feeds. Never blocks."""
        nonlocal last_brain_time, _brain_busy, frame_index, last_result, last_prealert_time
        while not _shutdown_event.is_set():
            frame = stream.read()
            if frame is None:
                time.sleep(0.03)
                continue

            frame_index += 1
            run_detect = (frame_index % settings.DETECT_EVERY_N_FRAMES == 0) or (last_result is None)
            if run_detect:
                # Run heavy detection less frequently for smoother overall FPS.
                result = detector.detect(frame)
                last_result = result
            else:
                result = last_result
                if result is not None:
                    result.frame = frame

            # Draw bounding boxes
            display_frame = frame.copy()
            for det in (result.detections if result else []):
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                color = (0, 255, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det.class_name} {det.confidence:.0%}"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Always push frames to feeds (never blocked by Brain)
            update_frame(display_frame)

            if not result or not result.triggered:
                continue

            # Skip Brain for camera-blocked bypass triggers (darkness/uniformity)
            if result.bypass_only:
                continue

            # Cooldown check
            now = time.time()
            if now - last_brain_time < settings.BRAIN_COOLDOWN:
                continue

            # Only start Brain if it's not already busy
            with _brain_lock:
                if _brain_busy:
                    continue
                _brain_busy = True
                last_brain_time = now

            log.info("Trigger! %s", result.summary)
            push_log(f"Trigger: {result.summary}", "warning")

            # Instant dashboard feedback so users see immediate trigger activity
            # even before Brain inference finishes.
            now = time.time()
            if now - last_prealert_time >= 2.0:
                try:
                    dashboard_alert.alert(
                        message=f"Trigger detected: {result.summary} — analyzing...",
                        priority="low",
                        frame=frame,
                        verdict_data={"threat_score": 0.0, "actions": [], "chain_of_thought": "Analyzing..."},
                    )
                    last_prealert_time = now
                except Exception:
                    pass

            # Spawn Brain analysis in background (non-blocking)
            threading.Thread(
                target=brain_worker,
                args=(frame.copy(), result.summary),
                daemon=True
            ).start()

        # Cleanup
        stream.stop()
        if telegram:
            telegram.stop_polling()
        log.info("Pipeline stopped.")

    pipeline_thread = threading.Thread(target=pipeline_loop, daemon=True)
    pipeline_thread.start()

    # ── Main Thread: Website-only runtime ────────────────────────────
    _shutdown_event.wait()

    log.info("Vanguard AI stopped. 🔒")


if __name__ == "__main__":
    main()
