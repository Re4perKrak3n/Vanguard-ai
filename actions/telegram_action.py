"""
Telegram Command Uplink — Bot API integration for Vanguard AI.
Sends alerts (frame + reasoning) to the Boss and accepts remote commands.

Commands:
    /say <text>   — Force local TTS to speak
    /status       — System status (FPS, VRAM, uptime)
    /capture      — Snap and send current camera frame
"""

import io
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import requests

log = logging.getLogger("actions.telegram")

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramUplink:
    """Telegram Bot integration for alerts + remote commands."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str = "",
        tts_callback: Optional[Callable[[str], None]] = None,
        frame_callback: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.tts_callback = tts_callback        # KokoroTTS.speak
        self.frame_callback = frame_callback    # stream.read
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._offset = 0
        self._start_time = datetime.now()
        self._detection_count = 0

        if not bot_token:
            log.warning("Telegram bot token not set — uplink disabled")
            return

        # Verify bot token
        try:
            me = self._api("getMe")
            if me.get("ok"):
                bot_name = me["result"].get("username", "unknown")
                log.info("Telegram bot connected: @%s", bot_name)
            else:
                log.error("Telegram bot token invalid: %s", me)
        except Exception as e:
            log.error("Telegram connection failed: %s", e)

    @property
    def available(self) -> bool:
        return bool(self.bot_token)

    # ── Telegram API ─────────────────────────────────────────────────

    def _api(self, method: str, **kwargs) -> dict:
        """Call a Telegram Bot API method."""
        url = TELEGRAM_API.format(token=self.bot_token, method=method)
        resp = requests.post(url, **kwargs, timeout=30)
        return resp.json()

    def _resolve_chat_id(self, update: dict) -> bool:
        """Auto-capture the Boss's chat ID from the first message."""
        msg = update.get("message", {})
        chat = msg.get("chat", {})
        cid = str(chat.get("id", ""))
        if cid and not self.chat_id:
            self.chat_id = cid
            user = msg.get("from", {})
            name = user.get("first_name", "Boss")
            log.info("Auto-captured chat_id=%s from %s", cid, name)
            log.info("To persist, set env: TELEGRAM_CHAT_ID=%s", cid)
            self.send_message(f"Vanguard AI locked on. Welcome, {name}.")
            return True
        return False

    # ── Outbound: Send Messages ──────────────────────────────────────

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a text message to the Boss."""
        if not self.chat_id:
            log.warning("No chat_id — can't send message. Send any message to the bot first.")
            return False
        try:
            self._api("sendMessage", json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            })
            return True
        except Exception as e:
            log.error("sendMessage failed: %s", e)
            return False

    def send_photo(
        self,
        frame: np.ndarray,
        caption: str = "",
    ) -> bool:
        """Send a JPEG frame to the Boss."""
        if not self.chat_id:
            return False
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            photo = io.BytesIO(buf.tobytes())
            photo.name = "frame.jpg"
            self._api("sendPhoto", data={
                "chat_id": self.chat_id,
                "caption": caption[:1024],
                "parse_mode": "Markdown",
            }, files={"photo": photo})
            return True
        except Exception as e:
            log.error("sendPhoto failed: %s", e)
            return False

    def send_alert(
        self,
        message: str,
        frame: Optional[np.ndarray] = None,
        verdict_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Push an intruder alert: frame + formatted reasoning.
        Grouped media message: JPEG + Markdown summary.
        """
        if not self.available or not self.chat_id:
            return False

        self._detection_count += 1

        # Format the reasoning as readable Markdown
        data = verdict_data or {}
        threat = data.get("threat_score", 0)
        cot = data.get("chain_of_thought", "")
        actions = data.get("actions", [])

        threat_bar = "🟢" if threat < 0.3 else "🟡" if threat < 0.6 else "🔴"
        caption_lines = [
            f"{threat_bar} *VANGUARD ALERT*",
            f"*Threat:* `{threat:.0%}`",
        ]
        if cot:
            caption_lines.append(f"*Analysis:* {cot[:300]}")
        if actions:
            action_strs = [f"  • `{a.get('function', '?')}`: {a.get('params', {}).get('message', a.get('params', {}).get('event', ''))}" for a in actions[:3]]
            caption_lines.append("*Actions:*\n" + "\n".join(action_strs))
        caption_lines.append(f"\n_{datetime.now().strftime('%H:%M:%S')}_")
        caption = "\n".join(caption_lines)

        if frame is not None:
            return self.send_photo(frame, caption=caption)
        else:
            return self.send_message(caption)

    # ── Inbound: Command Polling ─────────────────────────────────────

    def start_polling(self):
        """Start the background polling thread for commands."""
        if not self.available:
            log.warning("Telegram uplink disabled — no token")
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info("Telegram command polling started")

    def stop_polling(self):
        """Stop the polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_loop(self):
        """Long-poll for updates from Telegram."""
        while self._running:
            try:
                resp = self._api("getUpdates", json={
                    "offset": self._offset,
                    "timeout": 10,
                })
                if not resp.get("ok"):
                    time.sleep(5)
                    continue

                for update in resp.get("result", []):
                    self._offset = update["update_id"] + 1
                    self._handle_update(update)

            except requests.exceptions.Timeout:
                continue
            except Exception as e:
                log.error("Telegram poll error: %s", e)
                time.sleep(5)

    def _handle_update(self, update: dict):
        """Process a single Telegram update."""
        # Auto-capture chat_id from the first message
        if not self.chat_id:
            self._resolve_chat_id(update)
            return

        msg = update.get("message", {})
        text = msg.get("text", "").strip()
        chat_id = str(msg.get("chat", {}).get("id", ""))

        # Only respond to the Boss
        if chat_id != self.chat_id:
            return

        if not text:
            return

        # Route commands
        if text.startswith("/say "):
            self._cmd_say(text[5:].strip())
        elif text == "/status":
            self._cmd_status()
        elif text == "/capture":
            self._cmd_capture()
        elif text == "/help":
            self._cmd_help()
        elif text.startswith("/"):
            self.send_message(f"Unknown command: `{text.split()[0]}`\nTry /help")

    # ── Command Handlers ─────────────────────────────────────────────

    def _cmd_say(self, text: str):
        """Force TTS to speak the text."""
        if not text:
            self.send_message("Usage: `/say <text>`")
            return
        self.send_message(f"🔊 Speaking: _{text[:100]}_")
        if self.tts_callback:
            try:
                self.tts_callback(text)
                self.send_message("✅ Spoken.")
            except Exception as e:
                self.send_message(f"❌ TTS failed: `{e}`")
        else:
            self.send_message("⚠️ TTS not available.")

    def _cmd_status(self):
        """Return system status."""
        import torch
        uptime = datetime.now() - self._start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        lines = [
            "*VANGUARD STATUS*",
            f"⏱ Uptime: `{hours}h {minutes}m {seconds}s`",
            f"🎯 Detections: `{self._detection_count}`",
        ]

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**2
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            gpu_name = torch.cuda.get_device_name(0)
            lines.append(f"🖥 GPU: `{gpu_name}`")
            lines.append(f"💾 VRAM: `{vram_used:.0f}/{vram_total:.0f} MB`")
        else:
            lines.append("🖥 GPU: `CPU mode`")

        self.send_message("\n".join(lines))

    def _cmd_capture(self):
        """Snap current frame and send it."""
        if not self.frame_callback:
            self.send_message("⚠️ No camera connected.")
            return
        frame = self.frame_callback()
        if frame is None:
            self.send_message("⚠️ No frame available.")
            return
        self.send_photo(frame, caption="📸 *Live Capture*")

    def _cmd_help(self):
        """Send help text."""
        self.send_message(
            "*VANGUARD AI — Commands*\n\n"
            "`/say <text>` — Force TTS to speak\n"
            "`/status` — System status\n"
            "`/capture` — Snap & send frame\n"
            "`/help` — This message"
        )
