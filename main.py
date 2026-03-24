"""
Vanguard AI - Main Entry Point
==============================
Root orchestrator. Ties together:
  1. Sentinel (YOLO Vision)
  2. Brain (Qwen reasoning via llama.cpp)
  3. Actions (Speech, STT, Dashboard)
  4. Browser Dashboard
"""

import argparse
import logging
import os
import random
import signal
import sys
import threading
import time
from datetime import datetime

import cv2

# llama-cpp-python needs CUDA runtime DLLs on Windows. PyTorch bundles them.
try:
    import torch

    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_torch_lib)
except Exception:
    pass


FOLLOW_UP_LINES = [
    "No answer? That is a bold move when I can see you this clearly.",
    "You are really going silent while standing this close to a camera?",
    "If this is your robbery face, you may want a rewrite before I call the police.",
    "Still nothing? I can see you just fine, and this is getting weird fast.",
]

_shutdown_event = threading.Event()


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
    parser.add_argument("--no-ngrok", action="store_true", help="Skip ngrok tunnel.")
    parser.add_argument("--no-stt", action="store_true", help="Disable microphone listening.")
    parser.add_argument("--port", type=int, default=None, help="Dashboard server port.")
    parser.add_argument("--password", default=None, help="Set dashboard access password.")
    return parser.parse_args()


def start_ngrok(port: int, log) -> str:
    try:
        from pyngrok import ngrok

        tunnel = ngrok.connect(port, "http")
        url = tunnel.public_url
        log.info("[NGROK] Tunnel active: %s", url)
        return url
    except Exception as exc:
        log.warning("Ngrok failed: %s", exc)
        return ""


def _signal_handler(sig, frame):
    logging.getLogger("main").info("Shutdown signal received.")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)
try:
    signal.signal(signal.SIGTERM, _signal_handler)
except (OSError, AttributeError):
    pass


def dry_run(args):
    from config import settings

    setup_logging(settings.LOG_LEVEL, settings.LOG_DIR)
    log = logging.getLogger("main")
    log.info("=" * 60)
    log.info("DRY RUN - Testing pipeline flow")
    log.info("=" * 60)
    log.info("Config loaded [OK]")
    log.info("  YOLO:   %s", settings.YOLO_MODEL)
    log.info("  Brain:  %s", settings.BRAIN_MODEL)
    log.info("  TTS:    %s", settings.TTS_BACKEND)
    log.info("  STT:    %s", settings.STT_BACKEND)
    log.info("  Device: %s", settings.YOLO_DEVICE)

    from dashboard.server import DashboardServer, update_system_status

    port = args.port or settings.DASHBOARD_PORT
    if args.password:
        import dashboard.server as ds

        ds.DASHBOARD_PASSWORD = args.password

    DashboardServer(port=port).start()
    update_system_status(
        {
            "status": "Dry Run",
            "device": settings.YOLO_DEVICE,
            "yolo_model": settings.YOLO_MODEL,
            "brain_model": settings.BRAIN_MODEL,
            "tts_model": settings.TTS_BACKEND,
        }
    )

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

    log.info("=" * 60)
    log.info("  VANGUARD AI - Autonomous Security Pipeline")
    log.info("  Device: %s | Camera: %s", settings.YOLO_DEVICE, settings.CAMERA_SOURCE)
    log.info("=" * 60)

    from dashboard.server import DashboardServer, push_log, set_chat_refs, update_frame, update_system_status
    import dashboard.server as ds

    port = args.port or settings.DASHBOARD_PORT
    if args.password:
        ds.DASHBOARD_PASSWORD = args.password

    dashboard_server = DashboardServer(port=port)
    dashboard_server.start()
    log.info("Dashboard at http://localhost:%d (password: %s)", port, ds.DASHBOARD_PASSWORD)

    if not args.no_ngrok:
        start_ngrok(port, log)

    log.info("Initializing Sentinel (YOLO)...")
    push_log("Loading YOLO detector...", "info")
    from sentinel.detector import SentinelDetector
    from sentinel.stream import VideoStream

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
        n_gpu_layers=settings.BRAIN_GPU_LAYERS,
        n_ctx=settings.BRAIN_N_CTX,
        max_tokens=settings.BRAIN_MAX_TOKENS,
    )

    log.info("Initializing Actions (Speech + Dashboard)...")
    push_log("Loading speech voice...", "info")
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
    if settings.TTS_BACKEND != "none":
        try:
            from actions.speech import SpeechTTS

            tts = SpeechTTS(
                voice=settings.STREAMELEMENTS_VOICE,
                fallback_voice=settings.EDGE_TTS_VOICE,
            )
            if not tts.available:
                log.warning("Speech backend unavailable, staying on no-op TTS")
                tts = _NoOpTTS()
        except BaseException as exc:
            log.error("Speech backend init crashed, staying on no-op TTS: %s", exc)
            tts = _NoOpTTS()
    else:
        log.warning("TTS disabled (VANGUARD_TTS_BACKEND=%s)", settings.TTS_BACKEND)

    dashboard_alert = DashboardAlert()

    listener = None
    if not args.no_stt:
        log.info("Initializing STT (%s)...", settings.STT_BACKEND)
        push_log("Loading speech recognition...", "info")
        try:
            if settings.STT_BACKEND in {"browser", "website"}:
                from sentinel.browser_listener import BrowserAudioListener

                listener = BrowserAudioListener(
                    language=settings.BROWSER_STT_LANGUAGE,
                    listen_duration=settings.LIVE_REPLY_WINDOW,
                )
                push_log("Browser STT bridge ready - click Mic once in the dashboard to arm browser hearing.", "success")
            else:
                from sentinel.listener import AudioListener

                listener = AudioListener(
                    model_size=settings.STT_MODEL,
                    device=settings.STT_DEVICE,
                    language=settings.STT_LANGUAGE,
                    listen_duration=settings.STT_LISTEN_DURATION,
                )
            if listener and listener.available:
                if settings.STT_BACKEND in {"browser", "website"}:
                    push_log("STT ready - browser hearing will use the dashboard mic.", "success")
                else:
                    push_log("STT ready - microphone active", "success")
            else:
                log.warning("STT initialized but not available (missing mic or model)")
                listener = None
        except Exception as exc:
            log.warning("STT init failed: %s", exc)
            listener = None

    dispatcher = ActionDispatcher(tts=tts, dashboard=dashboard_alert)

    stream.start()
    last_brain_time = 0.0
    _brain_busy = False
    _brain_lock = threading.Lock()
    _gpu_lock = threading.Lock()
    last_prealert_time = 0.0
    frame_index = 0
    last_result = None

    interaction_lock = threading.Lock()
    interaction_state = {
        "active": False,
        "listen_thread_active": False,
        "pending_followup_at": 0.0,
        "last_spoken_at": 0.0,
        "last_spoken_message": "",
        "last_heard_at": 0.0,
        "last_heard_message": "",
        "last_person_seen_at": 0.0,
        "follow_up_count": 0,
        "exchange_count": 0,
        "last_summary": "",
    }

    def run_serialized_brain_chat(user_text: str, frame_snapshot, context: str) -> str:
        nonlocal _brain_busy

        with _brain_lock:
            _brain_busy = True
        try:
            push_log("Console request waiting for Brain...", "info")
            with _gpu_lock:
                response = brain.chat(user_text, frame_snapshot, context)
            return response
        finally:
            with _brain_lock:
                _brain_busy = False

    set_chat_refs(
        brain=brain,
        tts=tts,
        stream=stream,
        chat_handler=run_serialized_brain_chat,
    )

    update_system_status(
        {
            "status": "Running",
            "device": settings.YOLO_DEVICE,
            "yolo_model": settings.YOLO_MODEL,
            "brain_model": settings.BRAIN_MODEL,
            "tts_model": settings.TTS_BACKEND,
            "stt_backend": settings.STT_BACKEND,
        }
    )
    push_log("All models loaded. System armed.", "success")

    def summary_has_close_person(summary: str) -> bool:
        lowered = str(summary or "").lower()
        return "person (close)" in lowered or "person (very close)" in lowered

    def reset_interaction_state():
        with interaction_lock:
            interaction_state.update(
                {
                    "active": False,
                    "listen_thread_active": False,
                    "pending_followup_at": 0.0,
                    "last_spoken_at": 0.0,
                    "last_spoken_message": "",
                    "last_heard_at": 0.0,
                    "last_heard_message": "",
                    "last_person_seen_at": 0.0,
                    "follow_up_count": 0,
                    "exchange_count": 0,
                    "last_summary": "",
                }
            )

    def build_live_context() -> str:
        with interaction_lock:
            last_spoken_message = interaction_state["last_spoken_message"]
            last_heard_message = interaction_state["last_heard_message"]
            follow_up_count = interaction_state["follow_up_count"]
            exchange_count = interaction_state["exchange_count"]

        parts = []
        if last_spoken_message:
            parts.append(f"Vanguard last said: {last_spoken_message}")
        if last_heard_message:
            parts.append(f"Person last replied: {last_heard_message}")
        if follow_up_count:
            parts.append(f"Follow-up count: {follow_up_count}")
        if exchange_count:
            parts.append(f"Exchange count: {exchange_count}")
        return " | ".join(parts)

    def start_brain_job(
        frame_snapshot,
        summary,
        forced_audio: str = "",
        interaction_context: str = "",
        force: bool = False,
    ) -> bool:
        nonlocal last_brain_time, _brain_busy

        now = time.time()
        if not force and now - last_brain_time < settings.BRAIN_COOLDOWN:
            return False

        with _brain_lock:
            if _brain_busy:
                return False
            _brain_busy = True
            last_brain_time = now

        threading.Thread(
            target=brain_worker,
            args=(frame_snapshot, summary, forced_audio, interaction_context),
            daemon=True,
        ).start()
        return True

    def schedule_reply_listener(frame_snapshot, summary: str):
        if not listener or not listener.available:
            return

        with interaction_lock:
            if interaction_state["listen_thread_active"]:
                return
            if interaction_state["exchange_count"] >= settings.LIVE_MAX_EXCHANGES:
                return
            interaction_state["listen_thread_active"] = True

        def _listen_worker():
            try:
                time.sleep(settings.LIVE_LISTEN_DELAY)
                transcript = listener.listen(duration=settings.LIVE_REPLY_WINDOW)
                if transcript:
                    push_log(f'Heard live reply: "{transcript}"', "info")
                    with interaction_lock:
                        interaction_state["listen_thread_active"] = False
                        interaction_state["pending_followup_at"] = 0.0
                        interaction_state["last_heard_at"] = time.time()
                        interaction_state["last_heard_message"] = transcript
                    if not start_brain_job(
                        frame_snapshot.copy(),
                        summary,
                        forced_audio=transcript,
                        interaction_context=build_live_context(),
                        force=True,
                    ):
                        push_log("Brain busy - skipped one live reply.", "warning")
                else:
                    with interaction_lock:
                        interaction_state["listen_thread_active"] = False
                        if interaction_state["active"]:
                            interaction_state["pending_followup_at"] = time.time()
            except Exception as exc:
                log.warning("Live reply listener failed: %s", exc)
                with interaction_lock:
                    interaction_state["listen_thread_active"] = False

        threading.Thread(target=_listen_worker, daemon=True).start()

    def brain_worker(
        frame_snapshot,
        summary: str,
        forced_audio: str = "",
        interaction_context: str = "",
    ):
        nonlocal _brain_busy
        try:
            push_log("Analyzing with Brain...", "info")

            audio_transcript = str(forced_audio or "").strip()
            should_auto_listen = (
                not audio_transcript
                and listener
                and listener.available
                and settings.STT_BACKEND not in {"browser", "website"}
                and not summary_has_close_person(summary)
            )

            if should_auto_listen:
                push_log("Listening for speech...", "info")
                audio_transcript = listener.listen()
                if audio_transcript:
                    push_log(f'Heard: "{audio_transcript}"', "info")

            with _gpu_lock:
                verdict = brain.analyze(
                    frame=frame_snapshot,
                    detection_summary=summary,
                    audio_transcript=audio_transcript,
                    interaction_context=interaction_context,
                )

            if verdict is None:
                log.warning("Brain returned no verdict")
                return

            threat = verdict.get("threat_score", 0.0)
            actions = verdict.get("actions", [])
            cot = verdict.get("chain_of_thought", "")

            log.info("Verdict - threat=%.2f | actions=%d", threat, len(actions))
            push_log(f"Verdict: Threat={threat:.0%}, Actions={len(actions)}", "info")

            if cot:
                push_log(f"Brain: {cot}", "info")

            verdict_data = {
                "threat_score": threat,
                "chain_of_thought": cot,
                "actions": actions,
            }
            dispatcher.dispatch(actions, frame=frame_snapshot, verdict_data=verdict_data)

            if summary_has_close_person(summary):
                speak_message = ""
                for action in actions:
                    if action.get("function") == "speak":
                        speak_message = str(action.get("params", {}).get("message", "")).strip()
                        if speak_message:
                            break

                with interaction_lock:
                    if speak_message:
                        if not interaction_state["active"]:
                            interaction_state["follow_up_count"] = 0
                            interaction_state["exchange_count"] = 0
                        interaction_state["active"] = True
                        interaction_state["last_spoken_at"] = time.time()
                        interaction_state["last_spoken_message"] = speak_message
                        interaction_state["last_summary"] = summary
                        interaction_state["last_person_seen_at"] = time.time()
                        interaction_state["pending_followup_at"] = 0.0
                        interaction_state["exchange_count"] += 1
                    if audio_transcript:
                        interaction_state["last_heard_at"] = time.time()
                        interaction_state["last_heard_message"] = audio_transcript

                if speak_message:
                    schedule_reply_listener(frame_snapshot, summary)

        except Exception as exc:
            log.error("Brain analysis failed: %s", exc)
        finally:
            with _brain_lock:
                _brain_busy = False

    def maybe_reset_interaction(now: float, close_person_visible: bool):
        if close_person_visible:
            return
        with interaction_lock:
            if interaction_state["active"] and now - interaction_state["last_person_seen_at"] > settings.LIVE_CONVERSATION_RESET:
                interaction_state.update(
                    {
                        "active": False,
                        "listen_thread_active": False,
                        "pending_followup_at": 0.0,
                        "last_spoken_message": "",
                        "last_heard_message": "",
                        "follow_up_count": 0,
                        "exchange_count": 0,
                        "last_summary": "",
                    }
                )

    def pipeline_loop():
        nonlocal frame_index, last_prealert_time, last_result

        while not _shutdown_event.is_set():
            frame = stream.read()
            if frame is None:
                time.sleep(0.03)
                continue

            frame_index += 1
            with _brain_lock:
                brain_busy = _brain_busy

            run_detect = not brain_busy and (
                (frame_index % settings.DETECT_EVERY_N_FRAMES == 0) or (last_result is None)
            )
            if run_detect:
                with _gpu_lock:
                    result = detector.detect(frame)
                last_result = result
            else:
                result = last_result
                if result is not None:
                    result.frame = frame

            display_frame = frame.copy()
            for det in (result.detections if result else []):
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                color = (0, 255, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det.class_name} {det.confidence:.0%}"
                cv2.putText(
                    display_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            update_frame(display_frame)

            now = time.time()
            summary = result.summary if result else ""
            close_person_visible = bool(result and summary_has_close_person(summary))
            maybe_reset_interaction(now, close_person_visible)

            if close_person_visible:
                with interaction_lock:
                    interaction_state["last_person_seen_at"] = now
                    interaction_state["last_summary"] = summary
                    pending_followup_at = interaction_state["pending_followup_at"]
                    listen_thread_active = interaction_state["listen_thread_active"]
                    follow_up_count = interaction_state["follow_up_count"]
                    active = interaction_state["active"]

                if (
                    pending_followup_at
                    and now >= pending_followup_at
                    and not listen_thread_active
                    and follow_up_count < settings.LIVE_MAX_FOLLOWUPS
                ):
                    follow_up = random.choice(FOLLOW_UP_LINES)
                    actions = [
                        {"function": "speak", "params": {"message": follow_up}},
                        {"function": "log", "params": {"event": "No reply from nearby person; delivered a follow-up line."}},
                    ]
                    verdict_data = {
                        "threat_score": 0.45,
                        "chain_of_thought": "Nearby person stayed silent after being addressed.",
                        "actions": actions,
                    }
                    dispatcher.dispatch(actions, frame=frame, verdict_data=verdict_data)
                    push_log(f"Follow-up: {follow_up}", "warning")
                    with interaction_lock:
                        interaction_state["active"] = True
                        interaction_state["follow_up_count"] += 1
                        interaction_state["exchange_count"] += 1
                        interaction_state["pending_followup_at"] = 0.0
                        interaction_state["last_spoken_at"] = now
                        interaction_state["last_spoken_message"] = follow_up
                    schedule_reply_listener(frame.copy(), summary)
                    continue

                if active:
                    continue

            if not result or not result.triggered:
                continue
            if result.bypass_only:
                continue

            if not start_brain_job(
                frame.copy(),
                summary,
                interaction_context=build_live_context() if close_person_visible else "",
            ):
                continue

            log.info("Trigger! %s", summary)
            push_log(f"Trigger: {summary}", "warning")

            if now - last_prealert_time >= 2.0:
                try:
                    dashboard_alert.alert(
                        message=f"Trigger detected: {summary} - analyzing...",
                        priority="low",
                        frame=frame,
                        verdict_data={
                            "threat_score": 0.0,
                            "actions": [],
                            "chain_of_thought": "Analyzing...",
                        },
                    )
                    last_prealert_time = now
                except Exception:
                    pass

        stream.stop()
        if listener and hasattr(listener, "close"):
            try:
                listener.close()
            except Exception:
                pass
        log.info("Pipeline stopped.")

    reset_interaction_state()
    threading.Thread(target=pipeline_loop, daemon=True).start()
    _shutdown_event.wait()
    log.info("Vanguard AI stopped.")


if __name__ == "__main__":
    main()
