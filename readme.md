<p align="center">
  <img src="https://github.com/user-attachments/assets/5c40d533-db2d-44e2-8f80-ba40a455592f" alt="Project Architecture" width="800">
</p>


---


# Sentinel-Brain: Autonomous Agentic Security Pipeline

**Sentinel-Brain** is a high-performance, local-first security framework that integrates real-time computer vision with multimodal large language models (MLLMs). It transforms traditional motion alerts into an "intelligent" security agent capable of reasoning, threat assessment, and verbal intervention — with a personality that'll make intruders question their life choices.

---

## 🗝️ Core Concept

Traditional security systems suffer from high false-alarm rates. **Sentinel-Brain** solves this by using a hierarchical execution model to maximize GPU efficiency:

1.  **The Sentinel (YOLO):** A lightweight, always-on observer that detects all security-relevant objects (persons, weapons, vehicles, animals, and more) with near-zero latency.
2.  **The Brain (Qwen2.5-VL):** A multimodal specialist that performs deep contextual analysis on **every** trigger. It reasons about intent, assesses threat level, and — most importantly — **directly decides what actions to take** by outputting structured JSON function calls.
3.  **Actions (TTS + Alerts):** Executed by the Brain's JSON output — no keyword parsing, no separate "executive" layer. The Brain calls `speak()` to roast intruders through speakers and `alert()` to notify your phone.

---

## 🎭 Personality

Sentinel-Brain isn't your typical "Warning: trespasser detected" robot. It's a **sassy, trash-talking, psychologically devastating** security AI:

> *"Yo my guy, just so you know, there's 2 pitbulls in there and honestly I forgot to feed them today. Actually wait, I don't think they've eaten in like 3 days. You still wanna try that door? Be my guest 💀"*

> *"Bro you really wearing a black hoodie at 2 AM like that's not the most NPC criminal behavior I've ever seen."*

> *"I already got your face in 4K HDR uploaded to 3 different cloud servers. You're basically famous now. Congrats! 🎉"*

---

## 🛠️ System Architecture

### Phase 1: Neural Filtering (YOLO)
The system monitors a 24/7 RTSP or local video stream detecting **all** security-relevant objects:
* **Model:** `yolo11n` (or `yolo26n` when available) — NMS-Free.
* **Classes:** Persons, knives, vehicles, animals, bags, and more (full COCO set).
* **Efficiency:** Optimized for edge deployment, consuming <1GB VRAM.
* **Logic:** When any trigger class is detected with sufficient confidence, the full frame and detection context is passed to the Brain.

### Phase 2: Multimodal Reasoning (Qwen2.5-VL)
The captured frame + detection summary is processed by **Qwen2.5-VL-3B** (Int4 Quantized). The Brain:
* **Sees everything:** Full scene analysis — people, objects, context, time of day.
* **Reasons about intent:** Differentiates a delivery driver from a potential intruder.
* **Decides actions:** Outputs structured JSON with function calls (`speak`, `alert`, `log`).
* **Never skipped:** Every detection goes through full contextual reasoning.

### Phase 3: Action Execution
The Brain's JSON output directly triggers actions — no parsing, no middleware:
```json
{
    "threat_score": 0.85,
    "chain_of_thought": "Person near door at 2AM with crowbar...",
    "actions": [
        {"function": "speak", "params": {"message": "Bro, the cops are 3 mins away but the German Shepherd is 3 SECONDS away. Choose wisely."}},
        {"function": "alert", "params": {"message": "Intruder with tool at front door", "priority": "critical"}},
        {"function": "log", "params": {"event": "high-threat intrusion attempt"}}
    ]
}
```

* **`speak`** → Kokoro TTS (82M params, expressive, runs on CPU) plays audio through speakers.
* **`alert`** → Sends notification + image to Telegram.
* **`log`** → Records the event locally.

---

## 📈 Performance
*Benchmarked on RTX 3060 (6GB VRAM) / 14GB RAM*

| Component | Precision | VRAM | Latency |
| :--- | :--- | :--- | :--- |
| **YOLO11-Nano** | FP16 | ~500 MB | ~2ms |
| **Qwen2.5-VL-3B** | INT4 | ~2.8 GB | ~1.5s |
| **Kokoro TTS** | FP32 | CPU only | ~200ms |
| **System Total** | - | **~3.5 GB** | **< 2s** |

---

## 📂 Project Structure
```text
├── sentinel/           # YOLO Detection & Stream Handling
│   ├── stream.py       #   Threaded video capture (webcam / RTSP)
│   └── detector.py     #   Multi-class object detection
├── brain/              # Qwen2.5-VL Reasoning & Decision Making
│   ├── prompts.py      #   Sassy system prompt & template
│   └── reasoner.py     #   Model inference & JSON output
├── actions/            # Function Handlers (called by Brain's JSON)
│   ├── dispatcher.py   #   Routes JSON actions to handlers
│   ├── tts_action.py   #   Kokoro TTS — expressive speech
│   └── alert_action.py #   Telegram notifications
├── config/
│   └── settings.py     #   All configuration (auto-detects GPU)
├── main.py             #   Main event loop
└── requirements.txt    #   Dependencies
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the pipeline (no models loaded)
python main.py --dry-run

# 3. Run for real (with live camera + visual feed)
python main.py --show-feed

# 4. Use an RTSP stream
python main.py --source "rtsp://your-camera-url" --show-feed
```

### Telegram Setup (optional)
```bash
# Set environment variables
set TELEGRAM_BOT_TOKEN=your-bot-token
set TELEGRAM_CHAT_ID=your-chat-id
```
Then set `TELEGRAM_ENABLED = True` in `config/settings.py`.

---

## ⚙️ Configuration

All settings live in `config/settings.py` — no env files, no YAML, just Python:

| Setting | Default | Description |
| :--- | :--- | :--- |
| `CAMERA_SOURCE` | `0` | Webcam index or RTSP URL |
| `YOLO_MODEL` | `yolo11n.pt` | YOLO model file |
| `YOLO_CONFIDENCE` | `0.55` | Detection confidence threshold |
| `BRAIN_MODEL` | `Qwen2.5-VL-3B-Instruct` | Multimodal reasoning model |
| `BRAIN_QUANTIZE` | `True` | INT4 quantization |
| `KOKORO_VOICE` | `af_heart` | Kokoro voice preset |
| `BRAIN_COOLDOWN` | `10.0` | Seconds between Brain calls |
| `THREAT_THRESHOLD` | `0.6` | Minimum score for full alert |
