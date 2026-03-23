"""
VideoStream — OpenCV-based camera / RTSP reader.
Runs capture in a background thread so the main loop never blocks on I/O.
"""

import cv2
import sys
import threading
import logging
from typing import Optional
import numpy as np

log = logging.getLogger("sentinel.stream")


class VideoStream:
    """Threaded video capture from webcam index or RTSP URL."""

    def __init__(self, source: str | int = 0, width: int = 1280, height: int = 720):
        self.source = source
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> "VideoStream":
        """Open the capture device and begin reading frames."""
        # Use DirectShow on Windows for webcam indices (fixes FFMPEG backend errors)
        if isinstance(self.source, int) and sys.platform == "win32":
            log.info("Using DirectShow backend for webcam %d", self.source)
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        log.info("VideoStream started — source=%s", self.source)
        return self

    def stop(self):
        """Signal the reader thread to stop and release the capture."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._cap is not None:
            self._cap.release()
        log.info("VideoStream stopped.")

    # ── public API ───────────────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """Return the most recent frame (or None if nothing captured yet)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    # ── internals ────────────────────────────────────────────────────

    def _reader(self):
        """Background loop that continuously grabs the latest frame."""
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                log.warning("Frame grab failed — retrying...")
                continue
            with self._lock:
                self._frame = frame
