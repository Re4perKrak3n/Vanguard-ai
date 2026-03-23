"""
SystemTTS — lightweight offline TTS using native OS voices via pyttsx3.
"""

import logging
import threading

log = logging.getLogger("actions.system_tts")


class SystemTTS:
    """Simple local TTS backend with low startup overhead."""

    def __init__(self, rate: int = 180, voice_hint: str = ""):
        self._lock = threading.Lock()
        self._engine = None

        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", int(rate))

            if voice_hint:
                voices = self._engine.getProperty("voices") or []
                hint = voice_hint.lower()
                for voice in voices:
                    if hint in (getattr(voice, "name", "") or "").lower():
                        self._engine.setProperty("voice", voice.id)
                        break

            log.info("System TTS loaded (pyttsx3)")
        except Exception as e:
            log.error("System TTS init failed: %s", e)
            self._engine = None

    @property
    def available(self) -> bool:
        return self._engine is not None

    def speak(self, text: str) -> bool:
        if not self.available:
            return False
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
                return True
            except Exception as e:
                log.error("System TTS playback failed: %s", e)
                return False

    def speak_async(self, text: str):
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()
