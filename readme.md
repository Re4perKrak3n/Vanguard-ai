# Vanguard AI: Browser Security Console

Vanguard AI is a browser-first security console that combines:

- YOLO object detection
- a local Qwen GGUF reasoning model through `llama.cpp`
- browser-native speech recognition through the dashboard page
- speaker output through StreamElements TTS with Edge fallback
- a live browser dashboard with chat, logs, alerts, and camera feed

## Architecture

1. Sentinel watches the camera feed with YOLO and detects people, vehicles, objects, and unusual motion.
2. The Brain reads the detection summary plus any recent speech transcript and decides what to say or log.
3. Actions execute directly from the Brain's JSON output:
   - `speak`
   - `alert`
   - `log`
4. The browser dashboard shows the live feed, activity logs, alerts, and the real chat console.
5. Browser hearing runs in the actual dashboard tab through the Web Speech API. When Vanguard addresses someone nearby, the page listens for a reply and sends that transcript back to the Python runtime.

## Current Runtime

- Brain model: `models/qwen2.5-3b-instruct-q4_k_m.gguf`
- Vision model: `models/yolo11n.pt`
- Speech backend: StreamElements first, Edge TTS fallback
- STT default: browser Web Speech API in the dashboard
- Optional STT fallback: `faster-whisper`
- Dashboard URL: `http://localhost:8080/login`

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Open [http://localhost:8080/login](http://localhost:8080/login), log in, and click the `Mic` button once so the browser can grant microphone access. After that:

- the command console can send spoken chat directly to the real LLM
- nearby-person interactions use the same browser mic path
- Vanguard speaks once, listens briefly, and only gives one tangy follow-up if nobody answers

Optional examples:

```bash
python main.py --dry-run
python main.py --source "rtsp://your-camera-url"
python main.py --no-stt
```

Use Whisper instead of browser hearing if you want local Python-side microphone capture:

```bash
set VANGUARD_STT_BACKEND=whisper
python main.py
```

## Notes

- The browser chat is backed by the real local LLM.
- The project no longer uses a hidden Selenium/Chrome speech recognizer.
- The live system can question nearby people if they are close to the camera or door.
- The project no longer depends on the old GUI, Telegram path, or removed multimodal projector model files.
