"""
AudioListener — Speech-to-Text via faster-whisper.

Captures audio from the microphone and transcribes it using Whisper.
Used to hear what people say so the Brain can respond contextually.

Features:
  - Energy-based voice activity detection (no ML overhead)
  - Non-blocking listen() with timeout
  - Thread-safe design
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger("sentinel.listener")

# Try to import faster_whisper
_whisper_available = False
try:
    from faster_whisper import WhisperModel
    _whisper_available = True
except ImportError:
    log.warning("faster-whisper not installed. STT disabled. Install with: pip install faster-whisper")

# sounddevice should already be installed (used by TTS too)
_sd_available = False
try:
    import sounddevice as sd
    _sd_available = True
except ImportError:
    log.warning("sounddevice not installed. STT disabled.")


class AudioListener:
    """Microphone capture + Whisper speech-to-text."""

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        language: str = "en",
        sample_rate: int = 16000,
        energy_threshold: float = 0.02,
        listen_duration: float = 5.0,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.listen_duration = listen_duration
        self.language = language
        self._model = None
        self._lock = threading.Lock()

        if not _whisper_available or not _sd_available:
            log.warning("STT: missing dependencies, listener disabled")
            return

        # Detect input device
        try:
            input_device = sd.query_devices(kind='input')
            log.info("STT microphone: %s", input_device.get('name', 'UNKNOWN'))
            log.info("  Sample rate: %.0f Hz", input_device.get('default_samplerate', 0))
            log.info("  Channels: %d", input_device.get('max_input_channels', 0))
        except Exception as e:
            log.warning("No microphone detected: %s", e)
            return

        # Load Whisper model
        log.info("Loading Whisper STT model: %s (device=%s)", model_size, device)
        try:
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type="int8" if device == "cpu" else "float16",
            )
            log.info("Whisper STT model loaded")
        except Exception as e:
            log.error("Failed to load Whisper model: %s", e)

    @property
    def available(self) -> bool:
        return self._model is not None and _sd_available

    def listen(self, duration: Optional[float] = None) -> str:
        """
        Record from the microphone and transcribe.

        Args:
            duration: How long to record in seconds (default: self.listen_duration)

        Returns:
            Transcribed text, or empty string if nothing meaningful was heard.
        """
        if not self.available:
            return ""

        dur = duration or self.listen_duration

        with self._lock:
            try:
                # Record audio
                log.debug("Listening for %.1fs...", dur)
                audio = sd.rec(
                    int(dur * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()  # Block until recording finishes

                # Check if there's actual speech (energy-based VAD)
                audio_flat = audio.flatten()
                energy = np.sqrt(np.mean(audio_flat ** 2))

                if energy < self.energy_threshold:
                    log.debug("No speech detected (energy=%.4f < threshold=%.4f)", energy, self.energy_threshold)
                    return ""

                log.info("Speech detected (energy=%.4f), transcribing...", energy)

                # Transcribe
                segments, info = self._model.transcribe(
                    audio_flat,
                    language=self.language,
                    beam_size=1,
                    vad_filter=True,
                )

                text = " ".join(seg.text.strip() for seg in segments).strip()

                if text:
                    log.info("Heard: \"%s\"", text[:100])
                return text

            except Exception as e:
                log.error("STT failed: %s", e)
                return ""

    def listen_async(self, callback, duration: Optional[float] = None):
        """
        Listen in a background thread and call callback(text) when done.
        Callback is only called if actual speech was detected.
        """
        def _worker():
            text = self.listen(duration)
            if text:
                callback(text)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
