"""
ActionDispatcher — routes the Brain's JSON function calls to real handlers.

The Brain outputs:
{
    "actions": [
        {"function": "speak", "params": {"message": "..."}},
        {"function": "alert", "params": {"message": "...", "priority": "high"}},
        {"function": "log",   "params": {"event": "..."}}
    ]
}

The dispatcher maps each "function" name to a concrete handler.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from actions.alert_action import DashboardAlert

log = logging.getLogger("actions.dispatcher")


class ActionDispatcher:
    """Routes Brain JSON actions to concrete handlers."""

    def __init__(
        self,
        tts: Any,
        dashboard: DashboardAlert,
        telegram=None,   # Optional TelegramUplink
    ):
        self.tts = tts
        self.dashboard = dashboard
        self.telegram = telegram

        # Registry of available functions
        self._handlers = {
            "speak": self._handle_speak,
            "alert": self._handle_alert,
            "log": self._handle_log,
        }
        log.info(
            "ActionDispatcher ready — functions: %s",
            list(self._handlers.keys()),
        )

    def dispatch(
        self,
        actions: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
        verdict_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Execute a list of Brain-generated actions.

        Args:
            actions: List of {"function": str, "params": dict} items.
            frame: The current camera frame (attached to alerts).
            verdict_data: Full Brain verdict (passed to dashboard + Telegram).
        """
        for action in actions:
            if not isinstance(action, dict):
                log.warning("Action is not a dictionary: %s", action)
                continue

            fn_name = action.get("function", "")
            params = action.get("params", {})

            handler = self._handlers.get(fn_name)
            if handler is None:
                log.warning("Unknown function: '%s' — skipping", fn_name)
                continue

            try:
                handler(params, frame=frame, verdict_data=verdict_data)
            except Exception as e:
                log.error("Action '%s' failed: %s", fn_name, e)

    # ── handlers ─────────────────────────────────────────────────────

    def _handle_speak(self, params: dict, **kwargs):
        """Play a spoken message through speakers."""
        message = params.get("message", "")
        if not message:
            log.warning("speak() called with empty message")
            return
        log.info("SPEAK: %s", message)
        self.tts.speak_async(message)

    def _handle_alert(self, params: dict, frame=None, verdict_data=None, **kwargs):
        """Push an alert to the PWA dashboard + Telegram."""
        message = params.get("message", "")
        priority = params.get("priority", "medium")
        if not message:
            log.warning("alert() called with empty message")
            return
        log.info("ALERT [%s]: %s", priority, message)
        self.dashboard.alert(
            message=message,
            priority=priority,
            frame=frame,
            verdict_data=verdict_data,
        )
        # Also push to Telegram if available
        if self.telegram and self.telegram.available:
            self.telegram.send_alert(
                message=message,
                frame=frame,
                verdict_data=verdict_data,
            )

    def _handle_log(self, params: dict, **kwargs):
        """Log an event (no external action)."""
        event = params.get("event", "unknown event")
        log.info("LOG: %s", event)
