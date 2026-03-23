"""
SentinelDetector — YOLO-based object detection with smart bypass triggers.
Detects all security-relevant COCO classes and returns structured results.

Bypass Triggers (skip confidence gate, instant Brain trigger):
  - Motion Delta: >30% pixel change between consecutive frames
  - Coverage Block: frame goes dark/uniform (camera covered or spray-painted)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

log = logging.getLogger("sentinel.detector")


@dataclass
class Detection:
    """A single detected object."""
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)


@dataclass
class DetectionResult:
    """Full result of one inference pass."""
    triggered: bool = False
    bypass_only: bool = False  # True if trigger is camera-blocked (skip Brain)
    detections: List[Detection] = field(default_factory=list)
    frame: Optional[np.ndarray] = None
    bypass_reason: str = ""

    @property
    def summary(self) -> str:
        """Human-readable summary for the Brain's context."""
        parts = []
        if self.bypass_reason:
            parts.append(f"[BYPASS: {self.bypass_reason}]")
        if not self.detections:
            parts.append("No objects detected.")
        else:
            counts: dict[str, int] = {}
            for d in self.detections:
                counts[d.class_name] = counts.get(d.class_name, 0) + 1
            parts.extend(f"{count}x {name}" for name, count in counts.items())
        return "Detected: " + ", ".join(parts) if not self.bypass_reason else " ".join(parts)


class SentinelDetector:
    """YOLO detector with motion-delta and coverage-block bypass triggers."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.55,
        device: str = "cuda",
        imgsz: int = 512,
        trigger_classes: Optional[List[str]] = None,
        motion_threshold: float = 0.30,
        darkness_threshold: float = 15.0,
        uniformity_threshold: float = 5.0,
    ):
        self.confidence = confidence
        self.imgsz = imgsz
        self.trigger_classes = set(trigger_classes or ["person"])
        self.motion_threshold = motion_threshold
        self.darkness_threshold = darkness_threshold
        self.uniformity_threshold = uniformity_threshold

        self._prev_gray: Optional[np.ndarray] = None

        # Rate-limit bypass warnings (max 1 per type per 5 seconds)
        self._last_bypass_log: dict[str, float] = {}
        self._bypass_log_interval = 5.0

        log.info("Loading YOLO model: %s on %s", model_path, device)
        self.model = YOLO(model_path)
        self.model.to(device)
        log.info("YOLO model loaded")

    def _log_bypass(self, bypass_type: str, message: str):
        """Rate-limited bypass logging to prevent log spam."""
        now = time.time()
        last = self._last_bypass_log.get(bypass_type, 0.0)
        if now - last >= self._bypass_log_interval:
            log.warning("Bypass: %s", message)
            self._last_bypass_log[bypass_type] = now

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a single frame with bypass checks."""
        result = DetectionResult(frame=frame)

        # ── Bypass Check 1: Coverage Block ──────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

        if mean_brightness < self.darkness_threshold:
            result.triggered = True
            result.bypass_only = True  # Don't invoke Brain on black frames
            result.bypass_reason = f"CAMERA BLOCKED — darkness ({mean_brightness:.0f})"
            self._log_bypass("darkness", result.bypass_reason)
            self._prev_gray = gray.copy()
            return result

        if std_brightness < self.uniformity_threshold:
            result.triggered = True
            result.bypass_only = True  # Don't invoke Brain on uniform frames
            result.bypass_reason = f"CAMERA COVERED — uniform frame (std={std_brightness:.1f})"
            self._log_bypass("uniformity", result.bypass_reason)
            self._prev_gray = gray.copy()
            return result

        # ── Bypass Check 2: Motion Delta ────────────────────────────
        if self._prev_gray is not None:
            diff = cv2.absdiff(self._prev_gray, gray)
            motion_ratio = float(np.sum(diff > 30)) / diff.size
            if motion_ratio > self.motion_threshold:
                result.triggered = True
                # Motion bypass IS meaningful — Brain should analyze it
                result.bypass_only = False
                result.bypass_reason = f"HIGH MOTION ({motion_ratio:.0%} pixels changed)"
                self._log_bypass("motion", result.bypass_reason)
        self._prev_gray = gray.copy()

        # ── Standard YOLO Detection ─────────────────────────────────
        results = self.model(frame, conf=self.confidence, imgsz=self.imgsz, verbose=False)

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                conf = float(box.conf[0])
                bbox = tuple(box.xyxy[0].tolist())

                det = Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox,
                )
                result.detections.append(det)

                if class_name in self.trigger_classes:
                    result.triggered = True

        return result
