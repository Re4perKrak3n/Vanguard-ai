"""
Vanguard AI configuration.
All tunable runtime settings live here.
"""

import logging
import os
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
    log.warning("No CUDA GPU found - falling back to CPU (slower)")
    return "cpu"


_DEVICE = _detect_device()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


@dataclass
class Settings:
    # Camera / stream
    CAMERA_SOURCE: str | int = _env_int("VANGUARD_CAMERA", 0)
    FRAME_WIDTH: int = _env_int("VANGUARD_FRAME_WIDTH", 640)
    FRAME_HEIGHT: int = _env_int("VANGUARD_FRAME_HEIGHT", 360)

    # Sentinel (YOLO)
    YOLO_MODEL: str = _env("VANGUARD_YOLO_MODEL", "models/yolo11n.pt")
    YOLO_CONFIDENCE: float = _env_float("VANGUARD_YOLO_CONF", 0.55)
    YOLO_DEVICE: str = _DEVICE
    YOLO_IMGSZ: int = _env_int("VANGUARD_YOLO_IMGSZ", 512)
    DETECT_EVERY_N_FRAMES: int = max(1, _env_int("VANGUARD_DETECT_EVERY_N_FRAMES", 1))

    TRIGGER_CLASSES: List[str] = field(default_factory=lambda: [
        "person", "knife", "scissors", "baseball bat",
        "car", "motorcycle", "truck", "bus",
        "backpack", "handbag", "suitcase",
        "dog", "cat", "bird",
        "cell phone", "laptop",
        "fire hydrant",
    ])

    # Brain config
    BRAIN_MODEL: str = _env("VANGUARD_BRAIN_MODEL", "models/qwen2.5-3b-instruct-q4_k_m.gguf")
    BRAIN_GPU_LAYERS: int = _env_int("VANGUARD_GPU_LAYERS", 99)
    BRAIN_N_CTX: int = _env_int("VANGUARD_N_CTX", 2048)
    BRAIN_MAX_TOKENS: int = _env_int("VANGUARD_MAX_TOKENS", 256)
    THREAT_THRESHOLD: float = _env_float("VANGUARD_THREAT_THRESHOLD", 0.6)

    # Speech output
    TTS_BACKEND: str = _env("VANGUARD_TTS_BACKEND", "speech").lower()
    STREAMELEMENTS_VOICE: str = _env("VANGUARD_SE_VOICE", "Brian")
    EDGE_TTS_VOICE: str = _env("VANGUARD_EDGE_VOICE", "en-US-GuyNeural")

    # STT
    STT_BACKEND: str = _env("VANGUARD_STT_BACKEND", "website").lower()
    STT_MODEL: str = _env("VANGUARD_STT_MODEL", "tiny")
    STT_DEVICE: str = "cpu"
    STT_LANGUAGE: str = _env("VANGUARD_STT_LANG", "en")
    BROWSER_STT_LANGUAGE: str = _env("VANGUARD_BROWSER_STT_LANG", "en-US")
    STT_LISTEN_DURATION: float = _env_float("VANGUARD_STT_DURATION", 5.0)
    LIVE_LISTEN_DELAY: float = _env_float("VANGUARD_LIVE_LISTEN_DELAY", 0.8)
    LIVE_REPLY_WINDOW: float = _env_float("VANGUARD_LIVE_REPLY_WINDOW", 2.0)
    LIVE_CONVERSATION_RESET: float = _env_float("VANGUARD_LIVE_RESET", 20.0)
    LIVE_MAX_FOLLOWUPS: int = _env_int("VANGUARD_LIVE_MAX_FOLLOWUPS", 1)
    LIVE_MAX_EXCHANGES: int = _env_int("VANGUARD_LIVE_MAX_EXCHANGES", 3)

    # Browser dashboard
    DASHBOARD_HOST: str = "0.0.0.0"
    DASHBOARD_PORT: int = _env_int("VANGUARD_PORT", 8080)

    # Pipeline
    BRAIN_COOLDOWN: float = _env_float("VANGUARD_COOLDOWN", 3.0)

    # Logging
    LOG_LEVEL: str = _env("VANGUARD_LOG_LEVEL", "INFO")
    LOG_DIR: str = "logs"


settings = Settings()
