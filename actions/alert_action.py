"""
Dashboard Alert Action — sends alerts via WebSocket to the PWA dashboard.
Replaces Telegram with local real-time push.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from dashboard.server import alert_clients

log = logging.getLogger("actions.alert")


class DashboardAlert:
    """Push alerts to connected PWA clients via WebSocket."""

    def __init__(self):
        log.info("Dashboard alert system initialized")

    def alert(
        self,
        message: str,
        priority: str = "medium",
        frame: Optional[np.ndarray] = None,
        verdict_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Push an alert to all connected PWA dashboard clients.

        Args:
            message: The alert message body.
            priority: low / medium / high / critical.
            frame: Optional BGR frame to attach.
            verdict_data: Full Brain verdict (for chain_of_thought, etc.)

        Returns:
            True if broadcast was attempted.
        """
        try:
            data = verdict_data or {}
            # Ensure the alert message is in the data
            if "actions" not in data:
                data["actions"] = [
                    {"function": "alert", "params": {"message": message, "priority": priority}}
                ]

            alert_clients(data, frame=frame)
            log.info("📱 Dashboard alert pushed [%s]: %s", priority, message[:80])
            return True

        except Exception as e:
            log.error("Dashboard alert failed: %s", e)
            return False
