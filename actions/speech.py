"""
SpeechTTS - one browser-focused speech backend for Vanguard.

Primary provider: StreamElements
Fallback provider: Edge TTS
"""

import asyncio
import logging
import os
import re
import tempfile
import threading
from typing import Optional, Tuple

log = logging.getLogger("actions.speech")


class SpeechTTS:
    """Generate speech audio and play it with pygame."""

    def __init__(
        self,
        voice: str = "Brian",
        fallback_voice: str = "en-US-GuyNeural",
    ):
        self.voice = voice
        self.fallback_voice = fallback_voice
        self._lock = threading.Lock()
        self._requests = None
        self._pygame = None
        self._edge_tts = None

        try:
            import pygame

            self._pygame = pygame
            if not self._pygame.mixer.get_init():
                self._pygame.mixer.init()
        except Exception as exc:
            log.error("Speech playback init failed: %s", exc)

        try:
            import requests

            self._requests = requests
        except Exception as exc:
            log.warning("requests unavailable for StreamElements speech: %s", exc)

        try:
            import edge_tts

            self._edge_tts = edge_tts
        except Exception as exc:
            log.warning("edge-tts unavailable for speech fallback: %s", exc)

        if self.available:
            log.info(
                "Speech backend ready (stream=%s, edge_fallback=%s)",
                bool(self._requests),
                bool(self._edge_tts),
            )
        else:
            log.error("Speech backend unavailable")

    @property
    def available(self) -> bool:
        return self._pygame is not None and (self._requests is not None or self._edge_tts is not None)

    def _sanitize_text(self, text: str, max_chars: int) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        cleaned = cleaned.replace("{", "").replace("}", "")
        return cleaned[:max_chars].rstrip()

    def _new_temp_mp3(self) -> str:
        fd, path = tempfile.mkstemp(prefix="vanguard_speech_", suffix=".mp3")
        os.close(fd)
        return path

    def _cleanup_temp_file(self, path: Optional[str]):
        if not path:
            return
        try:
            os.remove(path)
        except OSError:
            pass

    def _generate_streamelements_audio(self, text: str) -> Optional[str]:
        if self._requests is None:
            return None

        safe_text = self._sanitize_text(text, 240)
        if not safe_text:
            return None

        path = self._new_temp_mp3()
        try:
            response = self._requests.get(
                "https://api.streamelements.com/kappa/v2/speech",
                params={"voice": self.voice, "text": safe_text},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    )
                },
                timeout=30,
            )
            response.raise_for_status()
            with open(path, "wb") as handle:
                handle.write(response.content)
            return path
        except Exception as exc:
            self._cleanup_temp_file(path)
            log.warning("StreamElements speech failed, trying Edge fallback: %s", exc)
            return None

    async def _save_edge_audio(self, text: str, path: str):
        communicator = self._edge_tts.Communicate(text, self.fallback_voice)
        await communicator.save(path)

    def _generate_edge_audio(self, text: str) -> Optional[str]:
        if self._edge_tts is None:
            return None

        safe_text = self._sanitize_text(text, 1000)
        if not safe_text:
            return None

        path = self._new_temp_mp3()
        try:
            try:
                asyncio.run(self._save_edge_audio(safe_text, path))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(self._save_edge_audio(safe_text, path))
                finally:
                    loop.close()
            return path
        except Exception as exc:
            self._cleanup_temp_file(path)
            log.error("Edge speech fallback failed: %s", exc)
            return None

    def _generate_audio(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        stream_path = self._generate_streamelements_audio(text)
        if stream_path:
            return stream_path, "streamelements"

        edge_path = self._generate_edge_audio(text)
        if edge_path:
            return edge_path, "edge"

        return None, None

    def speak(self, text: str) -> bool:
        if not self.available:
            return False

        text = self._sanitize_text(text, 400)
        if not text:
            return False

        with self._lock:
            audio_path, provider = self._generate_audio(text)
            if not audio_path:
                return False

            try:
                if not self._pygame.mixer.get_init():
                    self._pygame.mixer.init()

                log.info("Speaking via %s: %s", provider or "unknown", text[:120])
                self._pygame.mixer.music.load(audio_path)
                self._pygame.mixer.music.play()

                clock = self._pygame.time.Clock()
                while self._pygame.mixer.music.get_busy():
                    clock.tick(10)

                return True
            except Exception as exc:
                log.error("Speech playback failed: %s", exc)
                return False
            finally:
                try:
                    self._pygame.mixer.music.stop()
                except Exception:
                    pass
                self._cleanup_temp_file(audio_path)

    def speak_async(self, text: str):
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()
