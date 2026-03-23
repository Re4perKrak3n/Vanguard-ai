"""
Vanguard AI Desktop Command Center — Native GUI
================================================
CustomTkinter desktop application showing:
  • Live camera feed (what YOLO sees)
  • Model status cards (YOLO, Brain, TTS, GPU)
  • Brain reasoning log terminal
  • System health (FPS, detections, uptime)
  • Ngrok public URL for remote access
"""

import threading
import time
from datetime import datetime
from typing import Optional

import customtkinter as ctk
from PIL import Image, ImageTk

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── Theme ───────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg": "#0a0a0f",
    "panel": "#12121a",
    "border": "#1e1e2e",
    "accent": "#6366f1",
    "pink": "#ec4899",
    "green": "#22c55e",
    "yellow": "#f59e0b",
    "red": "#ef4444",
    "text": "#f0f0f5",
    "muted": "#55556a",
}


class VanguardGUI(ctk.CTk):
    """Main desktop window for Vanguard AI engine monitoring."""

    def __init__(self):
        super().__init__()

        self.title("Vanguard AI — Command Center")
        self.geometry("1280x720")
        self.minsize(960, 540)
        self.configure(fg_color=COLORS["bg"])

        # State
        self._latest_frame: Optional[np.ndarray] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._running = True
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._start_time = time.time()
        self._detection_count = 0
        self._ngrok_url: Optional[str] = None

        self._build_ui()
        self._update_clock()
        self._update_video()

    # ── UI Construction ─────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        self._build_header()

        # Main content
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # Left: Video
        self._build_video_panel(content)

        # Right: Sidebar
        self._build_sidebar(content)

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=COLORS["panel"], height=50, corner_radius=0)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        # Title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", padx=16)

        ctk.CTkLabel(
            title_frame, text="⦿ Vanguard",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text"]
        ).pack(side="left")
        ctk.CTkLabel(
            title_frame, text="AI",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["accent"]
        ).pack(side="left")

        # Right side: status + clock
        right_frame = ctk.CTkFrame(header, fg_color="transparent")
        right_frame.pack(side="right", padx=16)

        self.ngrok_label = ctk.CTkLabel(
            right_frame, text="",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color=COLORS["green"]
        )
        self.ngrok_label.pack(side="left", padx=(0, 16))

        self.status_label = ctk.CTkLabel(
            right_frame, text="● Initializing",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=COLORS["yellow"]
        )
        self.status_label.pack(side="left", padx=(0, 16))

        self.clock_label = ctk.CTkLabel(
            right_frame, text="--:--:--",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=COLORS["muted"]
        )
        self.clock_label.pack(side="left")

    def _build_video_panel(self, parent):
        video_frame = ctk.CTkFrame(parent, fg_color=COLORS["panel"], corner_radius=12)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        video_frame.grid_rowconfigure(1, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        # Panel header
        panel_hdr = ctk.CTkFrame(video_frame, fg_color="#16161f", height=36, corner_radius=0)
        panel_hdr.grid(row=0, column=0, sticky="ew")
        panel_hdr.pack_propagate(False)
        ctk.CTkLabel(
            panel_hdr, text="LIVE CAMERA FEED",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color=COLORS["muted"]
        ).pack(side="left", padx=12)

        self.fps_label = ctk.CTkLabel(
            panel_hdr, text="FPS: --",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color=COLORS["green"]
        )
        self.fps_label.pack(side="right", padx=12)

        # Video canvas
        self.video_label = ctk.CTkLabel(
            video_frame, text="Waiting for camera...",
            font=ctk.CTkFont(size=14),
            text_color=COLORS["muted"]
        )
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # Bottom: Model status row
        model_row = ctk.CTkFrame(video_frame, fg_color="#0e0e16", height=60, corner_radius=0)
        model_row.grid(row=2, column=0, sticky="ew")
        model_row.pack_propagate(False)

        self.model_labels = {}
        for name in ["YOLO", "Brain", "Voice", "GPU"]:
            badge = ctk.CTkFrame(model_row, fg_color=COLORS["border"], corner_radius=6)
            badge.pack(side="left", fill="y", expand=True, padx=4, pady=6)

            ctk.CTkLabel(
                badge, text=name,
                font=ctk.CTkFont(family="Consolas", size=9),
                text_color=COLORS["muted"]
            ).pack(padx=8, pady=(4, 0))

            val = ctk.CTkLabel(
                badge, text="...",
                font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
                text_color=COLORS["text"]
            )
            val.pack(padx=8, pady=(0, 4))
            self.model_labels[name] = val

    def _build_sidebar(self, parent):
        sidebar = ctk.CTkFrame(parent, fg_color="transparent")
        sidebar.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        sidebar.grid_rowconfigure(0, weight=3)
        sidebar.grid_rowconfigure(1, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        # Logs panel
        log_panel = ctk.CTkFrame(sidebar, fg_color=COLORS["panel"], corner_radius=12)
        log_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        log_panel.grid_rowconfigure(1, weight=1)
        log_panel.grid_columnconfigure(0, weight=1)

        log_hdr = ctk.CTkFrame(log_panel, fg_color="#16161f", height=36, corner_radius=0)
        log_hdr.grid(row=0, column=0, sticky="ew")
        log_hdr.pack_propagate(False)
        ctk.CTkLabel(
            log_hdr, text="BRAIN ACTIVITY LOG",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color=COLORS["muted"]
        ).pack(side="left", padx=12)

        self.log_textbox = ctk.CTkTextbox(
            log_panel, font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=COLORS["bg"], text_color=COLORS["text"],
            border_width=0, corner_radius=0, wrap="word",
            state="disabled"
        )
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # Stats panel
        stats_panel = ctk.CTkFrame(sidebar, fg_color=COLORS["panel"], corner_radius=12)
        stats_panel.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        stats_hdr = ctk.CTkFrame(stats_panel, fg_color="#16161f", height=36, corner_radius=0)
        stats_hdr.pack(fill="x")
        stats_hdr.pack_propagate(False)
        ctk.CTkLabel(
            stats_hdr, text="SYSTEM HEALTH",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color=COLORS["muted"]
        ).pack(side="left", padx=12)

        stats_body = ctk.CTkFrame(stats_panel, fg_color="transparent")
        stats_body.pack(fill="both", expand=True, padx=12, pady=8)

        self.stat_labels = {}
        for label in ["Detections", "Uptime", "Alerts", "Public URL"]:
            row = ctk.CTkFrame(stats_body, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(
                row, text=label,
                font=ctk.CTkFont(family="Consolas", size=10),
                text_color=COLORS["muted"]
            ).pack(side="left")
            val = ctk.CTkLabel(
                row, text="--",
                font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
                text_color=COLORS["text"]
            )
            val.pack(side="right")
            self.stat_labels[label] = val

    # ── Public API (called by main.py) ──────────────────────────────

    def update_frame(self, frame: np.ndarray):
        """Update the live video feed from the pipeline."""
        self._latest_frame = frame
        self._frame_count += 1

    def push_log(self, message: str, level: str = "info"):
        """Add a log entry to the Brain Activity Log terminal."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "info": COLORS["text"],
            "warning": COLORS["yellow"],
            "error": COLORS["red"],
            "success": COLORS["green"],
        }
        color = color_map.get(level, COLORS["text"])
        tag_name = f"color_{level}"

        self.log_textbox.configure(state="normal")
        line = f"[{timestamp}] {message}\n"
        self.log_textbox.insert("end", line, tag_name)
        self.log_textbox.tag_config(tag_name, foreground=color)
        self.log_textbox.configure(state="disabled")
        self.log_textbox.see("end")

    def set_model_status(self, yolo: str = "", brain: str = "", voice: str = "", gpu: str = ""):
        """Update the model status badges."""
        if yolo:
            self.model_labels["YOLO"].configure(text=yolo)
        if brain:
            self.model_labels["Brain"].configure(text=brain)
        if voice:
            self.model_labels["Voice"].configure(text=voice)
        if gpu:
            self.model_labels["GPU"].configure(text=gpu)

    def set_status(self, text: str, color: str = "green"):
        """Update the top-right status label."""
        c = COLORS.get(color, COLORS["green"])
        self.status_label.configure(text=f"● {text}", text_color=c)

    def set_ngrok_url(self, url: str):
        """Show the ngrok public URL in the header."""
        self._ngrok_url = url
        self.ngrok_label.configure(text=f"🌐 {url}")
        self.stat_labels["Public URL"].configure(text=url, text_color=COLORS["accent"])

    def increment_detections(self, count: int = 1):
        self._detection_count += count

    # ── Internal Update Loops ───────────────────────────────────────

    def _update_clock(self):
        if not self._running:
            return
        now = datetime.now().strftime("%H:%M:%S")
        self.clock_label.configure(text=now)

        # Uptime
        elapsed = int(time.time() - self._start_time)
        mins, secs = divmod(elapsed, 60)
        hours, mins = divmod(mins, 60)
        self.stat_labels["Uptime"].configure(text=f"{hours}h {mins}m {secs}s")
        self.stat_labels["Detections"].configure(text=str(self._detection_count))

        self.after(1000, self._update_clock)

    def _update_video(self):
        if not self._running:
            return

        if self._latest_frame is not None and HAS_CV2:
            frame = self._latest_frame
            # Convert BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            # Scale to fit the label
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            if label_w > 10 and label_h > 10:
                img.thumbnail((label_w, label_h), Image.LANCZOS)

            self._photo_image = ImageTk.PhotoImage(img)
            self.video_label.configure(image=self._photo_image, text="")

            # FPS calculation
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self._fps = self._frame_count / (now - self._last_fps_time)
                self._frame_count = 0
                self._last_fps_time = now
                self.fps_label.configure(text=f"FPS: {self._fps:.1f}")

        self.after(33, self._update_video)  # ~30fps refresh

    def on_closing(self):
        self._running = False
        self.destroy()


def launch_gui() -> VanguardGUI:
    """Create and return the GUI instance (call mainloop from main thread)."""
    app = VanguardGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    return app
