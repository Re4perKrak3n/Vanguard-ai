"""Dashboard alert action for the browser dashboard."""

import logging
from typing import Any, Dict, Optional

import numpy as np

from dashboard.server import alert_clients

log = logging.getLogger("actions.alert")


class DashboardAlert:
    """Push alerts to connected browser clients via WebSocket."""

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
        Push an alert to all connected browser dashboard clients.

        Args:
            message: The alert message body.
            priority: low / medium / high / critical.
            frame: Optional BGR frame to attach.
            verdict_data: Full Brain verdict for dashboard rendering.

        Returns:
            True if broadcast was attempted.
        """
        try:
            data = verdict_data or {}
            if "actions" not in data:
                data["actions"] = [
                    {"function": "alert", "params": {"message": message, "priority": priority}}
                ]

            alert_clients(data, frame=frame)
            log.info("Dashboard alert pushed [%s]: %s", priority, message[:80])
            return True
        except Exception as exc:
            log.error("Dashboard alert failed: %s", exc)
            return False
