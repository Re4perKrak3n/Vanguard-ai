"""
Kokoro TTS Action — expressive, realistic text-to-speech.
Uses the Kokoro-82M model (tiny, runs on CPU, emotional voice).
"""

import io
import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

log = logging.getLogger("actions.tts")


class KokoroTTS:
    """Tiny, expressive TTS using Kokoro-82M."""

    def __init__(
        self,
        model_id: str = "hexgrad/Kokoro-82M",
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "a",
    ):
        self.voice_name = voice
        self.speed = speed
        self.lang = lang
        self._pipeline = None
        self._lock = threading.Lock()

        # ── Audio Device Enumeration ─────────────────────────────────
        try:
            devices = sd.query_devices()
            default_out = sd.query_devices(kind='output')
            log.info("Audio devices found:")
            log.info("  Default output: %s", default_out.get('name', 'UNKNOWN'))
            log.info("  Sample rate:    %.0f Hz", default_out.get('default_samplerate', 0))
            log.info("  Channels:       %d", default_out.get('max_output_channels', 0))
            self._output_device = sd.default.device[1]  # default output index
            if self._output_device is None or self._output_device < 0:
                # Fallback: find first valid output device
                for i, dev in enumerate(devices):
                    if dev['max_output_channels'] > 0:
                        self._output_device = i
                        log.info("  Fallback device: [%d] %s", i, dev['name'])
                        break
        except Exception as e:
            log.error("Audio device enumeration failed: %s", e)
            self._output_device = None

        # ── Load Kokoro Model ────────────────────────────────────────
        log.info("Loading Kokoro TTS model: %s", model_id)
        try:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code=lang)
            log.info("Kokoro TTS loaded (voice=%s)", voice)
        except Exception as e:
            log.error("Failed to load Kokoro TTS: %s", e)
            log.warning("TTS will be unavailable. Install with: pip install kokoro sounddevice")

        # ── Startup Self-Test ────────────────────────────────────────
        if self._pipeline is not None:
            self._self_test()

    def _self_test(self):
        """Play a short test phrase to verify audio output works."""
        try:
            log.info("TTS self-test: 'System online'")
            generator = self._pipeline(
                "System online.",
                voice=self.voice_name,
                speed=self.speed,
            )
            audio_chunks = []
            for _, _, audio_chunk in generator:
                if audio_chunk is not None:
                    audio_chunks.append(audio_chunk)
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                sd.play(full_audio, samplerate=24000, device=self._output_device)
                sd.wait()
                log.info("TTS self-test PASSED — audio hardware OK")
            else:
                log.warning("TTS self-test: no audio generated")
        except Exception as e:
            log.error("TTS self-test FAILED: %s", e)
            log.error("Check audio device connections and drivers")

    @property
    def available(self) -> bool:
        return self._pipeline is not None

    def speak(self, text: str) -> bool:
        """
        Generate speech and play it through the default audio device.
        Returns True if successful.
        """
        if not self.available:
            log.warning("TTS unavailable — skipping: %s", text[:80])
            return False

        with self._lock:
            try:
                log.info("Speaking: %s", text[:100])

                # Generate audio with Kokoro
                generator = self._pipeline(
                    text,
                    voice=self.voice_name,
                    speed=self.speed,
                )

                # Kokoro yields (graphemes, phonemes, audio) chunks
                audio_chunks = []
                for _, _, audio_chunk in generator:
                    if audio_chunk is not None:
                        audio_chunks.append(audio_chunk)

                if not audio_chunks:
                    log.warning("Kokoro produced no audio")
                    return False

                # Concatenate all chunks
                full_audio = np.concatenate(audio_chunks)

                # Play through speakers (Kokoro outputs at 24kHz)
                sd.play(full_audio, samplerate=24000, device=self._output_device)
                sd.wait()  # block until playback finishes

                log.info("Speech complete.")
                return True

            except Exception as e:
                log.error("TTS playback failed: %s", e)
                return False

    def speak_async(self, text: str):
        """Fire-and-forget speech in a background thread."""
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
