"""
Vanguard AI Configuration
All tuneable parameters live here. Auto-detects device capabilities.
Reads from environment variables with sensible defaults.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List

import torch

log = logging.getLogger("config")


def _detect_device() -> str:
    """Pick the best available compute device."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log.info("GPU detected: %s (%.1f GB VRAM)", name, vram)
        return "cuda"
    log.warning("No CUDA GPU found — falling back to CPU (will be slower)")
    return "cpu"


_DEVICE = _detect_device()


def _env(key: str, default: str) -> str:
    """Read env var with fallback."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


@dataclass
class Settings:
    # ── Camera / Stream ──────────────────────────────────────────────
    CAMERA_SOURCE: str | int = _env_int("VANGUARD_CAMERA", 0)
    FRAME_WIDTH: int = _env_int("VANGUARD_FRAME_WIDTH", 640)
    FRAME_HEIGHT: int = _env_int("VANGUARD_FRAME_HEIGHT", 360)

    # ── Sentinel (YOLO) ─────────────────────────────────────────────
    YOLO_MODEL: str = _env("VANGUARD_YOLO_MODEL", "models/yolo11n.pt")
    YOLO_CONFIDENCE: float = _env_float("VANGUARD_YOLO_CONF", 0.55)
    YOLO_DEVICE: str = _DEVICE
    YOLO_IMGSZ: int = _env_int("VANGUARD_YOLO_IMGSZ", 512)
    DETECT_EVERY_N_FRAMES: int = max(1, _env_int("VANGUARD_DETECT_EVERY_N_FRAMES", 1))

    # Classes that trigger the Brain (COCO class names)
    TRIGGER_CLASSES: List[str] = field(default_factory=lambda: [
        "person", "knife", "scissors", "baseball bat",
        "car", "motorcycle", "truck", "bus",
        "backpack", "handbag", "suitcase",
        "dog", "cat", "bird",
        "cell phone", "laptop",
        "fire hydrant",
    ])

    # ── Brain Config (text GGUF via llama.cpp) ─────────────────────
    BRAIN_MODEL: str = _env("VANGUARD_BRAIN_MODEL", "models/qwen2.5-3b-instruct-q4_k_m.gguf")
    # Optional multimodal projector path; ignored in current text mode.
    BRAIN_MMPROJ: str = _env("VANGUARD_BRAIN_MMPROJ", "")
    BRAIN_GPU_LAYERS: int = _env_int("VANGUARD_GPU_LAYERS", 99)  # Offload everything
    BRAIN_N_CTX: int = _env_int("VANGUARD_N_CTX", 2048)
    BRAIN_MAX_TOKENS: int = _env_int("VANGUARD_MAX_TOKENS", 160)

    # Threat threshold — Brain outputs 0.0-1.0
    THREAT_THRESHOLD: float = _env_float("VANGUARD_THREAT_THRESHOLD", 0.6)

    # ── TTS ─────────────────────────────────────────────────────────
    KOKORO_MODEL: str = _env("VANGUARD_TTS_MODEL", "hexgrad/Kokoro-82M")
    KOKORO_VOICE: str = _env("VANGUARD_TTS_VOICE", "af_heart")
    KOKORO_SPEED: float = _env_float("VANGUARD_TTS_SPEED", 1.0)
    KOKORO_LANG: str = _env("VANGUARD_TTS_LANG", "a")
    # "system" (recommended), "kokoro", or "none"
    TTS_BACKEND: str = _env("VANGUARD_TTS_BACKEND", "system").lower()
    SYSTEM_TTS_RATE: int = _env_int("VANGUARD_SYSTEM_TTS_RATE", 180)
    SYSTEM_TTS_VOICE_HINT: str = _env("VANGUARD_SYSTEM_TTS_VOICE_HINT", "")

    # ── STT (faster-whisper) ────────────────────────────────────────
    STT_MODEL: str = _env("VANGUARD_STT_MODEL", "tiny")
    STT_DEVICE: str = "cpu"  # Keep on CPU to save VRAM
    STT_LANGUAGE: str = _env("VANGUARD_STT_LANG", "en")
    STT_LISTEN_DURATION: float = _env_float("VANGUARD_STT_DURATION", 5.0)

    # ── Dashboard (PWA) ─────────────────────────────────────────────
    DASHBOARD_HOST: str = "0.0.0.0"
    DASHBOARD_PORT: int = _env_int("VANGUARD_PORT", 8080)

    # ── Pipeline ────────────────────────────────────────────────────
    BRAIN_COOLDOWN: float = _env_float("VANGUARD_COOLDOWN", 3.0)

    # ── Telegram Uplink ─────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = _env("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = _env("TELEGRAM_CHAT_ID", "")

    # ── Logging ─────────────────────────────────────────────────────
    LOG_LEVEL: str = _env("VANGUARD_LOG_LEVEL", "INFO")
    LOG_DIR: str = "logs"
