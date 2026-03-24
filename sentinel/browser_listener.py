"""
BrowserAudioListener - browser-native speech recognition bridge.

This backend does not launch a hidden browser. Instead, it asks the live
dashboard page to capture speech with the Web Speech API and returns the
transcript back to the Python runtime.
"""

import logging
from typing import Optional

log = logging.getLogger("sentinel.browser_listener")


class BrowserAudioListener:
    def __init__(self, language: str = "en-US", listen_duration: float = 2.0):
        self.language = language
        self.listen_duration = listen_duration
        self._closed = False

    @property
    def available(self) -> bool:
        return not self._closed

    def listen(self, duration: Optional[float] = None) -> str:
        if self._closed:
            return ""

        from dashboard.server import request_browser_live_reply

        transcript = request_browser_live_reply(duration or self.listen_duration)
        transcript = str(transcript or "").strip()
        if transcript:
            log.info('Browser STT heard: "%s"', transcript[:100])
        return transcript

    def close(self):
        self._closed = True
