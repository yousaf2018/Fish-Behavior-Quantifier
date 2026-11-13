# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Fish Behavior Quantification GUI.

Load a trained model session, select raw videos, draw ROIs per video,
and run head/tail/center keypoint extraction + behavior analysis.
"""

from __future__ import annotations

import os
import sys
import json
import time
import threading
import queue
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.path_utils import resolve_config_paths, to_relative_path
from tools.project_meta import PROJECT_DISPLAY_NAME as PROJECT_NAME
from tools.settings_manager import (
    create_default_session_settings,
    find_session_dir_from_videos,
    get_session_settings_path,
    load_session_settings,
    merge_settings,
    save_session_settings,
)

# Settings persistence file (user config directory) - legacy fallback
SETTINGS_DIR = os.path.expanduser(os.path.join("~", ".config", "fbq"))
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "behavior_quantification_settings.json")


def _start_maximized(window: tk.Tk) -> None:
    """Make window full-screen on all platforms."""
    def maximize() -> None:
        if sys.platform == "darwin":
            try:
                window.attributes("-fullscreen", True)
                window.update_idletasks()
                return
            except Exception:
                pass

        try:
            window.state("zoomed")
            window.update_idletasks()
            if str(window.state()) == "zoomed":
                return
        except Exception:
            pass

        try:
            window.attributes("-zoomed", True)
            window.update_idletasks()
            return
        except Exception:
            pass

        try:
            screen_w = window.winfo_screenwidth()
            screen_h = window.winfo_screenheight()
            window.geometry(f"{screen_w}x{screen_h}+0+0")
        except Exception:
            pass

    maximize()
    for delay_ms in (50, 250, 1000):
        window.after(delay_ms, maximize)


@dataclass
class ROIEntry:
    points: List[tuple] = field(default_factory=list)
    shape: str = "rectangle"  # rectangle, circle, grid
    label: str = ""
    enabled: bool = True
    fish_count: int = 1


@dataclass
class VideoEntry:
    path: str
    session_dir: str = ""
    rois: List[ROIEntry] = field(default_factory=list)
    enabled: bool = True


class BehaviorQuantificationGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"{PROJECT_NAME} — Fish Behavior Quantification")
        # Full screen
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.geometry(f"{screen_w}x{screen_h}+0+0")
        self.minsize(1000, 700)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._video_entries: List[VideoEntry] = []
        self._model_session: str = ""
        self._weights_path: str = ""
        self._output_dir: str = ""
        self._queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        # Video playback state
        self._video_cap = None
        self._video_fps = 30
        self._video_total_frames = 0
        self._video_playing = False
        self._canvas_photo = None

        # Tracking results for endpoints extraction
        self._tracking_results: Dict[str, str] = {}  # video_path -> tracking_csv
        self._session_path = ""
        
        # Endpoints extraction settings (shared with endpoints dialog)
        self._endpoints_metric_selections: Dict[str, bool] = {}
        self._endpoints_global_config: Dict[str, Any] = {}
        self._endpoints_arena_configs: Dict[str, Any] = {}
        self._endpoints_calibration_ratio: Optional[float] = None
        
        self._build_ui()

        # Load persisted settings after UI is built
        self._load_settings()
        
        # Ensure full-screen
        _start_maximized(self)

    def _build_ui(self):
        # Top bar
        top = ctk.CTkFrame(self, corner_radius=0, height=50)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        ctk.CTkLabel(
            top,
            text="Fish Behavior Quantification",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left", padx=20, pady=10)

        ctk.CTkButton(
            top,
            text="← Back to Home",
            width=120,
            command=self._go_home,
        ).pack(side="right", padx=20, pady=10)

        # Main content: two columns
        content = ctk.CTkFrame(self, corner_radius=0)
        content.pack(fill="both", expand=True, padx=10, pady=10)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        # Left panel: Model & Videos
        left = ctk.CTkScrollableFrame(content, corner_radius=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left.grid_columnconfigure(0, weight=1)

        # Model selection
        model_frame = ctk.CTkFrame(left, corner_radius=6)
        model_frame.pack(fill="x", pady=(0, 10))
        model_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(model_frame, text="Trained Model Session", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 5)
        )

        self._model_entry = ctk.CTkEntry(model_frame, placeholder_text="Select session folder with trained weights...")
        self._model_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(model_frame, text="Browse", width=80, command=self._browse_model).grid(
            row=1, column=2, padx=10, pady=5
        )

        ctk.CTkLabel(model_frame, text="Model Weights:", font=ctk.CTkFont(weight="bold")).grid(
            row=2, column=0, sticky="w", padx=10, pady=(10, 5)
        )
        self._weights_var = ctk.StringVar(value="")
        self._weights_dropdown = ctk.CTkOptionMenu(
            model_frame, variable=self._weights_var, values=["Select session first..."],
            command=self._on_weights_selected
        )
        self._weights_dropdown.grid(row=2, column=1, columnspan=2, sticky="ew", padx=10, pady=5)

        self._weights_label = ctk.CTkLabel(model_frame, text="Weights: not selected", text_color="gray")
        self._weights_label.grid(row=3, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 10))

        # Video list
        video_frame = ctk.CTkFrame(left, corner_radius=6)
        video_frame.pack(fill="x", pady=(0, 10))
        video_frame.grid_columnconfigure(0, weight=1)

        header_row = ctk.CTkFrame(video_frame, corner_radius=0)
        header_row.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(header_row, text="Raw Videos", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(
            header_row, text="+ Add Video", width=100, command=self._add_video
        ).pack(side="right", padx=(5, 0))
        ctk.CTkButton(
            header_row, text="📁 Load Folder", width=100, command=self._load_videos_from_folder
        ).pack(side="right")

        self._video_list_frame = ctk.CTkFrame(video_frame, corner_radius=0)
        self._video_list_frame.pack(fill="x", padx=10, pady=(0, 10))
        self._video_list_frame.grid_columnconfigure(0, weight=1)

        # Keypoint options
        kp_frame = ctk.CTkFrame(left, corner_radius=6)
        kp_frame.pack(fill="x", pady=(0, 10))
        kp_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(kp_frame, text="Keypoint Extraction", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5)
        )

        self._extract_head = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(kp_frame, text="Extract Head (snout)", variable=self._extract_head).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=10, pady=2
        )

        self._extract_center = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(kp_frame, text="Extract Center (body centroid)", variable=self._extract_center).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=10, pady=2
        )

        self._extract_tail = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(kp_frame, text="Extract Tail (caudal fin)", variable=self._extract_tail).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=10, pady=2
        )

        self._behavior_analysis = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(kp_frame, text="Run Behavior Analysis", variable=self._behavior_analysis).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=10, pady=(2, 10)
        )

        # Display options
        disp_frame = ctk.CTkFrame(left, corner_radius=6)
        disp_frame.pack(fill="x", pady=(0, 10))
        disp_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(disp_frame, text="Display Settings", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 5)
        )

        ctk.CTkLabel(disp_frame, text="Arrow Size:").grid(row=1, column=0, sticky="w", padx=10, pady=2)
        self._arrow_size = ctk.IntVar(value=45)
        ctk.CTkEntry(disp_frame, textvariable=self._arrow_size, width=50).grid(
            row=1, column=1, sticky="w", padx=10, pady=2
        )

        ctk.CTkLabel(disp_frame, text="ID Font Size:").grid(row=2, column=0, sticky="w", padx=10, pady=2)
        self._id_font_size = ctk.DoubleVar(value=0.6)
        ctk.CTkEntry(disp_frame, textvariable=self._id_font_size, width=50).grid(
            row=2, column=1, sticky="w", padx=10, pady=2
        )

        ctk.CTkLabel(disp_frame, text="Box Thickness:").grid(row=3, column=0, sticky="w", padx=10, pady=2)
        self._box_thickness = ctk.IntVar(value=2)
        ctk.CTkEntry(disp_frame, textvariable=self._box_thickness, width=50).grid(
            row=3, column=1, sticky="w", padx=10, pady=2
        )

        # Output directory
        out_frame = ctk.CTkFrame(left, corner_radius=6)
        out_frame.pack(fill="x", pady=(0, 10))
        out_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(out_frame, text="Output Directory", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 5)
        )

        self._output_entry = ctk.CTkEntry(out_frame, placeholder_text="Select output directory...")
        self._output_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(out_frame, text="Browse", width=80, command=self._browse_output).grid(
            row=1, column=2, padx=10, pady=5
        )

        # Endpoints output directory
        ep_out_frame = ctk.CTkFrame(out_frame, corner_radius=0)
        ep_out_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(5, 10))
        ep_out_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(ep_out_frame, text="Endpoints Output:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self._endpoints_output_entry = ctk.CTkEntry(ep_out_frame, placeholder_text="Same as output directory (optional)")
        self._endpoints_output_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(ep_out_frame, text="Browse", width=80, command=self._browse_endpoints_output).grid(
            row=0, column=2, padx=5, pady=5
        )

        # Run button
        self._run_btn = ctk.CTkButton(
            left,
            text="Run Quantification",
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._run_quantification,
        )
        self._run_btn.pack(fill="x", pady=10)

        # Endpoints extraction button (enabled after tracking completes)
        self._endpoints_btn = ctk.CTkButton(
            left,
            text="📊 Extract Behavioral Endpoints",
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#5b2c91",
            hover_color="#4a2373",
            command=self._run_endpoints_extraction,
            state="disabled",
        )
        self._endpoints_btn.pack(fill="x", pady=(0, 10))

        # Progress
        self._progress = ctk.CTkProgressBar(left)
        self._progress.pack(fill="x", pady=(0, 5))
        self._progress.set(0)

        self._status_label = ctk.CTkLabel(left, text="Ready", text_color="gray")
        self._status_label.pack(fill="x", pady=(0, 10))

        # Progress stats (videos, fps, ETA)
        self._stats_label = ctk.CTkLabel(left, text="", text_color="gray", font=ctk.CTkFont(size=11))
        self._stats_label.pack(fill="x", pady=(0, 10))

        # Right panel: ROI Canvas + Output Video Preview
        right = ctk.CTkFrame(content, corner_radius=6)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right.grid_rowconfigure(1, weight=0)
        right.grid_rowconfigure(6, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right, text="ROI Editor (select video from list)", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10, pady=10)

        self._roi_photo = None
        self._roi_image = None
        self._roi_scale = 1.0
        self._roi_drag_start = None
        self._current_roi_rect = None
        self._current_video_idx = -1
        self._current_roi_idx = -1
        self._zoom_scale = 1.0
        self._base_fit_scale = 1.0

        # ROI list panel (scrollable, fixed height)
        roi_list_frame = ctk.CTkScrollableFrame(right, corner_radius=6, height=220)
        roi_list_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
        roi_list_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(roi_list_frame, text="ROIs (Tanks/Arenas)", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 5)
        )

        self._roi_list_container = ctk.CTkFrame(roi_list_frame, corner_radius=0)
        self._roi_list_container.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self._roi_list_container.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(roi_list_frame, text="+ Add ROI", width=100, command=self._add_roi).grid(
            row=0, column=1, padx=10, pady=5, sticky="e"
        )

        # ROI controls
        roi_ctrl = ctk.CTkFrame(right, corner_radius=0)
        roi_ctrl.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        roi_ctrl.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(roi_ctrl, text="ROI Shape:").grid(row=0, column=0, padx=5, pady=5)
        self._roi_shape_var = ctk.StringVar(value="rectangle")
        ctk.CTkOptionMenu(
            roi_ctrl, variable=self._roi_shape_var,
            values=["rectangle", "circle", "grid"]
        ).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(roi_ctrl, text="Label:").grid(row=0, column=2, padx=5, pady=5)
        self._roi_label_entry = ctk.CTkEntry(roi_ctrl, placeholder_text="Tank 1", width=100)
        self._roi_label_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(roi_ctrl, text="Zoom:").grid(row=0, column=4, padx=5, pady=5)
        self._zoom_label = ctk.CTkLabel(roi_ctrl, text="100%")
        self._zoom_label.grid(row=0, column=5, padx=5, pady=5)

        ctk.CTkButton(roi_ctrl, text="Reset View", width=80, command=self._reset_roi_view).grid(
            row=0, column=6, padx=5, pady=5
        )
        ctk.CTkButton(roi_ctrl, text="Clear ROI", width=80, command=self._clear_roi).grid(
            row=0, column=7, padx=5, pady=5
        )

        # Settings persistence row
        settings_ctrl = ctk.CTkFrame(right, corner_radius=0)
        settings_ctrl.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        settings_ctrl.grid_columnconfigure(4, weight=1)

        ctk.CTkButton(settings_ctrl, text="Apply ROIs to All Videos", width=150,
                      command=self._apply_rois_to_all_videos).grid(
            row=0, column=0, padx=5, pady=5
        )
        ctk.CTkButton(settings_ctrl, text="Save Settings", width=110,
                      command=self._save_settings_profile).grid(
            row=0, column=1, padx=5, pady=5
        )
        ctk.CTkButton(settings_ctrl, text="Load Settings", width=110,
                      command=self._load_settings_profile).grid(
            row=0, column=2, padx=5, pady=5
        )

        # View toggle: ROI editor vs Results video
        view_ctrl = ctk.CTkFrame(right, corner_radius=0)
        view_ctrl.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 10))
        view_ctrl.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(view_ctrl, text="View:").grid(row=0, column=0, padx=5, pady=5)
        self._view_var = ctk.StringVar(value="roi")
        self._view_toggle = ctk.CTkSegmentedButton(
            view_ctrl, variable=self._view_var,
            values=["ROI View", "Results"],
            command=self._on_view_toggle,
            width=180,
        )
        self._view_toggle.grid(row=0, column=1, padx=5, pady=5)

        # Shared canvas: ROI view by default; shows result video after run
        self._canvas_frame = ctk.CTkFrame(right, corner_radius=6)
        self._canvas_frame.grid(row=6, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._canvas_frame.grid_columnconfigure(0, weight=1)
        self._canvas_frame.grid_rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(self._canvas_frame, bg="#1a1a1a", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._canvas.bind("<Button-1>", self._on_canvas_press)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Button-4>", self._on_mousewheel)
        self._canvas.bind("<Button-5>", self._on_mousewheel)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas_photo = None

        # Video playback controls (shown only in Results view)
        self._video_ctrl_frame = ctk.CTkFrame(self._canvas_frame, corner_radius=0)
        self._video_ctrl_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self._video_ctrl_frame.grid_columnconfigure(1, weight=1)
        self._video_ctrl_frame.grid_remove()

        # Row 1: navigation + slider
        self._nav_row = ctk.CTkFrame(self._video_ctrl_frame, corner_radius=0)
        self._nav_row.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=(5, 2))
        self._nav_row.grid_columnconfigure(1, weight=1)

        self._step_back_btn = ctk.CTkButton(
            self._nav_row, text="◀◀", width=40,
            command=lambda: self._step_video(-1), state="disabled",
        )
        self._step_back_btn.grid(row=0, column=0, padx=2, pady=2)

        self._frame_slider = ctk.CTkSlider(
            self._nav_row, from_=0, to=100, number_of_steps=100,
            command=self._on_frame_slider,
        )
        self._frame_slider.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        self._step_fwd_btn = ctk.CTkButton(
            self._nav_row, text="▶▶", width=40,
            command=lambda: self._step_video(1), state="disabled",
        )
        self._step_fwd_btn.grid(row=0, column=2, padx=2, pady=2)

        self._frame_label = ctk.CTkLabel(self._nav_row, text="Frame 0 / 0", width=100)
        self._frame_label.grid(row=0, column=3, padx=5, pady=2)

        # Row 2: play + time + folder
        self._ctrl_row = ctk.CTkFrame(self._video_ctrl_frame, corner_radius=0)
        self._ctrl_row.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=(2, 5))

        self._play_btn = ctk.CTkButton(
            self._ctrl_row, text="▶ Play", width=80,
            command=self._toggle_video_playback, state="disabled",
        )
        self._play_btn.grid(row=0, column=0, padx=5, pady=5)

        self._video_time_label = ctk.CTkLabel(self._ctrl_row, text="00:00 / 00:00")
        self._video_time_label.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkButton(self._ctrl_row, text="Open Folder", width=100,
                      command=self._open_output_folder).grid(
            row=0, column=2, padx=5, pady=5
        )

        # Poll queue
        self.after(100, self._poll_queue)

    def _go_home(self):
        self.destroy()
        import subprocess
        subprocess.Popen([sys.executable, "-m", "gui.gui_home"])

    def _browse_model(self):
        path = filedialog.askdirectory(title="Select Trained Model Session")
        if path:
            self._model_session = path
            self._model_entry.delete(0, tk.END)
            self._model_entry.insert(0, path)
            self._find_weights(path)
            self._save_settings()

    def _find_weights(self, session_path: str):
        # Look for weights in typical locations
        weight_dirs = [
            os.path.join(session_path, "main"),
            os.path.join(session_path, "weights"),
            session_path,
        ]
        weight_files = []
        for d in weight_dirs:
            if os.path.isdir(d):
                for root, dirs, files in os.walk(d):
                    for f in files:
                        if f.endswith(".pt"):
                            weight_files.append(os.path.join(root, f))

        if weight_files:
            # Populate dropdown
            display_names = [os.path.relpath(w, session_path) for w in weight_files]
            self._weights_dropdown.configure(values=display_names)
            self._weights_var.set(display_names[0])
            self._weights_path = weight_files[0]
            self._weights_label.configure(text=f"Found {len(weight_files)} model(s)", text_color="green")
        else:
            self._weights_dropdown.configure(values=["No weights found"])
            self._weights_var.set("No weights found")
            self._weights_path = ""
            self._weights_label.configure(text="Weights: not found in session", text_color="orange")

    def _on_weights_selected(self, choice: str):
        # Map display name back to full path
        session_path = self._model_session
        if not session_path:
            return
        weight_files = []
        weight_dirs = [
            os.path.join(session_path, "main"),
            os.path.join(session_path, "weights"),
            session_path,
        ]
        for d in weight_dirs:
            if os.path.isdir(d):
                for root, dirs, files in os.walk(d):
                    for f in files:
                        if f.endswith(".pt"):
                            weight_files.append(os.path.join(root, f))

        for i, w in enumerate(weight_files):
            if os.path.relpath(w, session_path) == choice:
                self._weights_path = w
                self._weights_label.configure(text=f"Weights: {choice}", text_color="white")
                break

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self._output_dir = path
            self._output_entry.delete(0, tk.END)
            self._output_entry.insert(0, path)
            self._save_settings()

    def _browse_endpoints_output(self):
        path = filedialog.askdirectory(title="Select Endpoints Output Directory")
        if path:
            self._endpoints_output_entry.delete(0, tk.END)
            self._endpoints_output_entry.insert(0, path)
            self._save_settings()

    def _toggle_video_playback(self):
        if not hasattr(self, '_video_cap') or self._video_cap is None:
            # Open the output video
            output_video = os.path.join(getattr(self, '_output_dir', ''), 'behavior_analysis.mp4')
            if not os.path.exists(output_video):
                messagebox.showinfo("Info", "No output video found. Run quantification first.")
                return
            self._video_cap = cv2.VideoCapture(output_video)
            self._video_fps = self._video_cap.get(cv2.CAP_PROP_FPS) or 30
            self._video_total_frames = int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._video_playing = True
            self._play_btn.configure(text="⏸ Pause")
            self._play_video_frame()
        elif self._video_playing:
            self._video_playing = False
            self._play_btn.configure(text="▶ Play")
        else:
            self._video_playing = True
            self._play_btn.configure(text="⏸ Pause")
            self._play_video_frame()

    def _play_video_frame(self):
        if not self._video_playing or not hasattr(self, '_video_cap') or self._video_cap is None:
            return
        ret, frame = self._video_cap.read()
        if not ret:
            # Loop video
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._video_cap.read()
            if not ret:
                self._video_playing = False
                self._play_btn.configure(text="▶ Play")
                return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        cw = max(self._canvas.winfo_width(), 400)
        ch = max(self._canvas.winfo_height(), 200)
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        self._canvas_photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self._canvas.create_image(x, y, anchor="nw", image=self._canvas_photo)

        # Update time label
        current_frame = int(self._video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame / max(self._video_fps, 1)
        total_time = self._video_total_frames / max(self._video_fps, 1)
        self._video_time_label.configure(text=f"{self._format_time(current_time)} / {self._format_time(total_time)}")

        # Schedule next frame
        delay = int(1000 / max(self._video_fps, 1))
        self.after(delay, self._play_video_frame)
        self._update_playback_ui()

    def _update_playback_ui(self):
        """Refresh slider, frame label, and time label from capture position."""
        if not hasattr(self, '_video_cap') or self._video_cap is None:
            return
        try:
            current_frame = int(self._video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        except Exception:
            return
        total = max(self._video_total_frames, 1)
        # Clamp to valid range
        current_frame = max(0, min(current_frame, total - 1))
        # Update slider without triggering its callback
        self._frame_slider.set(current_frame / max(total - 1, 1) * 100)
        self._frame_label.configure(text=f"Frame {current_frame + 1} / {total}")
        current_time = current_frame / max(self._video_fps, 1)
        total_time = self._video_total_frames / max(self._video_fps, 1)
        self._video_time_label.configure(
            text=f"{self._format_time(current_time)} / {self._format_time(total_time)}"
        )

    def _step_video(self, delta: int):
        """Step forward/backward by delta frames and refresh the display."""
        if not hasattr(self, '_video_cap') or self._video_cap is None:
            return
        was_playing = self._video_playing
        if was_playing:
            self._video_playing = False
            self._play_btn.configure(text="▶ Play")
        try:
            current = int(self._video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            target = max(0, min(current + delta - 1, self._video_total_frames - 1))
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        except Exception:
            return
        ret, frame = self._video_cap.read()
        if ret:
            self._display_frame(frame)
        self._update_playback_ui()

    def _on_frame_slider(self, value: float):
        """Seek to the frame indicated by the slider."""
        if not hasattr(self, '_video_cap') or self._video_cap is None:
            return
        was_playing = self._video_playing
        if was_playing:
            self._video_playing = False
            self._play_btn.configure(text="▶ Play")
        try:
            total = max(self._video_total_frames, 1)
            target = int(float(value) / 100.0 * (total - 1))
            target = max(0, min(target, total - 1))
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        except Exception:
            return
        ret, frame = self._video_cap.read()
        if ret:
            self._display_frame(frame)
        self._update_playback_ui()

    def _display_frame(self, frame):
        """Render a BGR frame onto the canvas."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        cw = max(self._canvas.winfo_width(), 400)
        ch = max(self._canvas.winfo_height(), 200)
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        self._canvas_photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self._canvas.create_image(x, y, anchor="nw", image=self._canvas_photo)

    def _format_time(self, seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _open_output_folder(self):
        output_dir = getattr(self, '_output_dir', '')
        if output_dir and os.path.exists(output_dir):
            import subprocess
            subprocess.Popen(['xdg-open', output_dir])
        else:
            messagebox.showinfo("Info", "Output directory not set or doesn't exist.")

    def _add_video(self):
        paths = filedialog.askopenfilenames(
            title="Select Raw Video Files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        for p in paths:
            entry = VideoEntry(path=p)
            self._video_entries.append(entry)
            self._add_video_row(entry)
        self._save_settings()

    def _load_videos_from_folder(self):
        """Load all videos from a selected folder and auto-load session settings."""
        folder = filedialog.askdirectory(title="Select Folder Containing Raw Videos")
        if not folder:
            return
        
        # Find video files
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
        video_paths = []
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith(video_extensions):
                    video_paths.append(os.path.join(root, f))
        
        if not video_paths:
            messagebox.showinfo("No Videos", f"No video files found in:\n{folder}")
            return
        
        # Clear existing videos
        self._video_entries.clear()
        if hasattr(self, "_video_list_frame"):
            for widget in self._video_list_frame.winfo_children():
                widget.destroy()
        
        # Add all videos
        for p in sorted(video_paths):
            entry = VideoEntry(path=p)
            self._video_entries.append(entry)
            self._add_video_row(entry)
        
        # Load session settings from this folder
        self._load_session_settings_from_folder(folder)
        
        self._save_settings()
        self._log(f"📁 Loaded {len(video_paths)} videos from {folder}")

    def _load_session_settings_from_folder(self, folder: str):
        """Load session settings from the specified folder."""
        data = load_session_settings(folder)
        if data:
            self._apply_settings_dict(data)
            print(f"[BEHAVIOR] Loaded settings from folder: {folder}")
            self._log(f"⚙️ Loaded session settings from folder")
        else:
            print(f"[BEHAVIOR] No session settings found in folder: {folder}")

    def _add_video_row(self, entry: VideoEntry):
        row = ctk.CTkFrame(self._video_list_frame, corner_radius=4)
        row.pack(fill="x", pady=2)
        row.grid_columnconfigure(1, weight=1)

        enabled_var = ctk.BooleanVar(value=entry.enabled)
        ctk.CTkCheckBox(row, text="", variable=enabled_var, width=24,
                        command=lambda e=entry, v=enabled_var: setattr(e, "enabled", v.get())).grid(
            row=0, column=0, padx=5, pady=5
        )

        name = os.path.basename(entry.path)
        ctk.CTkLabel(row, text=name, anchor="w").grid(row=0, column=1, sticky="ew", padx=5)

        ctk.CTkButton(row, text="ROI", width=50,
                      command=lambda e=entry: self._select_video_for_roi(e)).grid(
            row=0, column=2, padx=5
        )
        ctk.CTkButton(row, text="×", width=30, fg_color="red", hover_color="#aa0000",
                      command=lambda e=entry, r=row: self._remove_video(e, r)).grid(
            row=0, column=3, padx=5
        )

    def _remove_video(self, entry: VideoEntry, row_frame):
        self._video_entries.remove(entry)
        row_frame.destroy()
        self._save_settings()

    def _select_video_for_roi(self, entry: VideoEntry):
        idx = self._video_entries.index(entry)
        self._current_video_idx = idx
        self._current_roi_idx = -1
        self._load_video_frame(entry.path)
        self._refresh_roi_list()
        self._save_settings()

    def _add_roi(self):
        if self._current_video_idx < 0:
            messagebox.showinfo("Info", "Select a video first")
            return
        entry = self._video_entries[self._current_video_idx]
        new_roi = ROIEntry(shape=self._roi_shape_var.get(), label=self._roi_label_entry.get() or f"Tank {len(entry.rois)+1}")
        entry.rois.append(new_roi)
        self._current_roi_idx = len(entry.rois) - 1
        self._refresh_roi_list()
        self._fit_image_to_canvas()
        self._save_settings()

    def _refresh_roi_list(self):
        # Clear existing ROI rows
        for widget in self._roi_list_container.winfo_children():
            widget.destroy()

        if self._current_video_idx < 0:
            return

        entry = self._video_entries[self._current_video_idx]
        for i, roi in enumerate(entry.rois):
            row = ctk.CTkFrame(self._roi_list_container, corner_radius=4)
            row.pack(fill="x", pady=2)
            row.grid_columnconfigure(1, weight=1)

            enabled_var = ctk.BooleanVar(value=roi.enabled)
            ctk.CTkCheckBox(row, text="", variable=enabled_var, width=24,
                            command=lambda r=roi, v=enabled_var: setattr(r, "enabled", v.get())).grid(
                row=0, column=0, padx=5, pady=5
            )

            label = roi.label or f"Tank {i+1} ({roi.shape})"
            ctk.CTkLabel(row, text=f"{label}  [{roi.fish_count} fish]", anchor="w").grid(
                row=0, column=1, sticky="ew", padx=5
            )

            # Centroid position display
            if roi.points:
                cx = sum(p[0] for p in roi.points) / len(roi.points)
                cy = sum(p[1] for p in roi.points) / len(roi.points)
                centroid_text = f"Centroid: ({cx:.1f}, {cy:.1f})"
            else:
                centroid_text = "Centroid: N/A"
            ctk.CTkLabel(row, text=centroid_text, text_color="#88ccff", font=ctk.CTkFont(size=10)).grid(
                row=1, column=1, sticky="w", padx=5
            )

            # Fish count entry
            fish_count_var = ctk.StringVar(value=str(roi.fish_count))
            ctk.CTkLabel(row, text="Fish:").grid(row=0, column=2, padx=(10, 2), pady=5)
            fish_spinbox = ctk.CTkEntry(row, textvariable=fish_count_var, width=40)
            fish_spinbox.grid(row=0, column=3, padx=2, pady=5)

            def _update_fish_count(r=roi, v=fish_count_var, *, refresh: bool = False):
                try:
                    val = int(v.get())
                    if val >= 1:
                        r.fish_count = val
                        if refresh:
                            self._refresh_roi_list()
                            self._save_settings()
                except ValueError:
                    pass
            # Save immediately as the user types (no FocusOut/Return required).
            fish_count_var.trace_add("write", lambda *args: _update_fish_count())
            # Refresh the UI label once editing is done.
            fish_spinbox.bind("<FocusOut>", lambda e: _update_fish_count(refresh=True))
            fish_spinbox.bind("<Return>", lambda e: _update_fish_count(refresh=True))

            ctk.CTkButton(row, text="Edit", width=50,
                          command=lambda idx=i: self._select_roi(idx)).grid(
                row=0, column=4, padx=5
            )
            ctk.CTkButton(row, text="×", width=30, fg_color="red", hover_color="#aa0000",
                          command=lambda idx=i: self._remove_roi(idx)).grid(
                row=0, column=5, padx=5
            )

    def _select_roi(self, roi_idx: int):
        self._current_roi_idx = roi_idx
        self._fit_image_to_canvas()
        self._save_settings()

    def _remove_roi(self, roi_idx: int):
        if self._current_video_idx < 0:
            return
        entry = self._video_entries[self._current_video_idx]
        if 0 <= roi_idx < len(entry.rois):
            entry.rois.pop(roi_idx)
            if self._current_roi_idx >= roi_idx:
                self._current_roi_idx = max(-1, self._current_roi_idx - 1)
            self._refresh_roi_list()
            self._fit_image_to_canvas()
            self._save_settings()

    def _load_video_frame(self, video_path: str):
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video: {video_path}")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", f"Cannot read first frame: {video_path}")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._roi_image = Image.fromarray(frame_rgb)
        self._fit_image_to_canvas()

    def _fit_image_to_canvas(self):
        if self._roi_image is None:
            return
        # Ensure canvas is laid out before getting dimensions
        self._canvas.update_idletasks()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        # Fallback if canvas not laid out yet
        if cw <= 1 or ch <= 1:
            cw, ch = 800, 600
        iw, ih = self._roi_image.size
        scale = min(cw / iw, ch / ih) * self._zoom_scale
        self._base_fit_scale = min(cw / iw, ch / ih)
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        img = self._roi_image.resize((new_w, new_h), Image.LANCZOS)
        self._roi_photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self._canvas.create_image(x, y, anchor="nw", image=self._roi_photo)
        # Draw all ROIs for this video
        if self._current_video_idx >= 0:
            entry = self._video_entries[self._current_video_idx]
            self._draw_all_rois_on_canvas(entry)

    def _canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> tuple:
        """Convert canvas coordinates to image coordinates."""
        if self._roi_image is None:
            return (0, 0)
        # Ensure canvas is laid out before getting dimensions
        self._canvas.update_idletasks()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        # Fallback if canvas not laid out yet
        if cw <= 1 or ch <= 1:
            cw, ch = 800, 600
        iw, ih = self._roi_image.size
        scale = min(cw / iw, ch / ih) * self._zoom_scale
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        x_offset = (cw - new_w) // 2
        y_offset = (ch - new_h) // 2
        img_x = (canvas_x - x_offset) / scale
        img_y = (canvas_y - y_offset) / scale
        return (img_x, img_y)

    def _image_to_canvas_coords(self, img_x: float, img_y: float) -> tuple:
        """Convert image coordinates to canvas coordinates."""
        if self._roi_image is None:
            return (0, 0)
        # Ensure canvas is laid out before getting dimensions
        self._canvas.update_idletasks()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        # Fallback if canvas not laid out yet
        if cw <= 1 or ch <= 1:
            cw, ch = 800, 600
        iw, ih = self._roi_image.size
        scale = min(cw / iw, ch / ih) * self._zoom_scale
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        x_offset = (cw - new_w) // 2
        y_offset = (ch - new_h) // 2
        canvas_x = x_offset + img_x * scale
        canvas_y = y_offset + img_y * scale
        return (canvas_x, canvas_y)

    def _draw_all_rois_on_canvas(self, entry: VideoEntry):
        """Draw all ROIs for a video entry."""
        for i, roi in enumerate(entry.rois):
            if not roi.enabled or not roi.points:
                continue
            is_selected = (i == self._current_roi_idx)
            color = "#38bdf8" if is_selected else "#888888"
            width = 3 if is_selected else 1

            # Convert points to canvas coordinates
            canvas_points = []
            for pt in roi.points:
                cx, cy = self._image_to_canvas_coords(pt[0], pt[1])
                canvas_points.append((cx, cy))

            # Draw ROI polygon using all points
            if len(canvas_points) >= 3:
                flat = [coord for pt in canvas_points for coord in pt]
                self._canvas.create_polygon(*flat, fill="", outline=color, width=width, tags=f"roi_{i}")
            elif roi.shape == "circle" and len(canvas_points) >= 2:
                x0, y0 = canvas_points[0]
                x1, y1 = canvas_points[1]
                self._canvas.create_oval(x0, y0, x1, y1, outline=color, width=width, tags=f"roi_{i}")
            elif roi.shape in ("rectangle", "grid") and len(canvas_points) >= 2:
                x0, y0 = canvas_points[0]
                x1, y1 = canvas_points[1]
                dash = (4, 4) if roi.shape == "grid" else ()
                self._canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=width, dash=dash, tags=f"roi_{i}")
                if roi.shape == "grid":
                    rows = 3
                    cols = 3
                    for r in range(1, rows):
                        ry = y0 + (y1 - y0) * r / rows
                        self._canvas.create_line(x0, ry, x1, ry, fill=color, width=width, dash=(4, 4), tags=f"roi_{i}")
                    for c in range(1, cols):
                        cx = x0 + (x1 - x0) * c / cols
                        self._canvas.create_line(cx, y0, cx, y1, fill=color, width=width, dash=(4, 4), tags=f"roi_{i}")

            # Draw ROI number/label on the canvas
            if canvas_points:
                # Use centroid of first 3 points for label position
                if len(canvas_points) >= 3:
                    cx = sum(p[0] for p in canvas_points[:3]) / 3
                    cy = sum(p[1] for p in canvas_points[:3]) / 3
                else:
                    cx, cy = canvas_points[0]

                # ROI number (1-based) and fish count
                label_text = f"Tank {i+1} ({roi.fish_count})"
                if roi.label:
                    label_text = f"{roi.label} ({roi.fish_count})"

                self._canvas.create_text(
                    cx, cy - 15, text=label_text, fill=color,
                    font=("Arial", 10, "bold"), tags=f"roi_{i}_label"
                )

    def _draw_roi_on_canvas(self, entry: VideoEntry):
        # Legacy method - now draws all ROIs
        self._draw_all_rois_on_canvas(entry)

    def _on_canvas_press(self, event):
        if self._current_video_idx < 0 or self._current_roi_idx < 0:
            return
        entry = self._video_entries[self._current_video_idx]
        roi = entry.rois[self._current_roi_idx]
        self._roi_drag_start = (event.x, event.y)
        # Initialize points if empty
        if not roi.points:
            roi.points = []
            roi.shape = self._roi_shape_var.get()
            roi.label = self._roi_label_entry.get() or roi.label
        if self._current_roi_rect:
            self._canvas.delete(self._current_roi_rect)

    def _on_canvas_drag(self, event):
        if self._roi_drag_start is None or self._current_video_idx < 0 or self._current_roi_idx < 0:
            return
        x0, y0 = self._roi_drag_start
        if self._current_roi_rect:
            self._canvas.delete(self._current_roi_rect)
        shape = self._roi_shape_var.get()
        if shape == "rectangle":
            self._current_roi_rect = self._canvas.create_rectangle(
                x0, y0, event.x, event.y, outline="#38bdf8", width=2
            )
        elif shape == "circle":
            self._current_roi_rect = self._canvas.create_oval(
                x0, y0, event.x, event.y, outline="#38bdf8", width=2
            )
        elif shape == "grid":
            self._current_roi_rect = self._canvas.create_rectangle(
                x0, y0, event.x, event.y, outline="#38bdf8", width=2, dash=(4, 4)
            )

    def _on_canvas_release(self, event):
        if self._roi_drag_start is None or self._current_video_idx < 0 or self._current_roi_idx < 0:
            self._roi_drag_start = None
            return
        x0, y0 = self._roi_drag_start
        x1, y1 = event.x, event.y
        entry = self._video_entries[self._current_video_idx]
        roi = entry.rois[self._current_roi_idx]
        # Convert to image coordinates
        img_pt1 = self._canvas_to_image_coords(x0, y0)
        img_pt2 = self._canvas_to_image_coords(x1, y1)

        shape = self._roi_shape_var.get()
        if shape == "rectangle":
            # Convert 2 points to 4-point rectangle polygon
            x_min = min(img_pt1[0], img_pt2[0])
            x_max = max(img_pt1[0], img_pt2[0])
            y_min = min(img_pt1[1], img_pt2[1])
            y_max = max(img_pt1[1], img_pt2[1])
            roi.points = [
                [x_min, y_min],  # top-left
                [x_max, y_min],  # top-right
                [x_max, y_max],  # bottom-right
                [x_min, y_max],  # bottom-left
            ]
        elif shape == "circle":
            # Convert 2 points to 8-point circle approximation
            cx = (img_pt1[0] + img_pt2[0]) / 2
            cy = (img_pt1[1] + img_pt2[1]) / 2
            rx = abs(img_pt1[0] - img_pt2[0]) / 2
            ry = abs(img_pt1[1] - img_pt2[1]) / 2
            import math
            roi.points = []
            for i in range(8):
                angle = i * math.pi / 4
                x = cx + rx * math.cos(angle)
                y = cy + ry * math.sin(angle)
                roi.points.append([x, y])
        elif shape == "grid":
            # Grid uses same as rectangle
            x_min = min(img_pt1[0], img_pt2[0])
            x_max = max(img_pt1[0], img_pt2[0])
            y_min = min(img_pt1[1], img_pt2[1])
            y_max = max(img_pt1[1], img_pt2[1])
            roi.points = [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ]
        else:
            roi.points = [img_pt1, img_pt2]

        roi.shape = shape
        roi.label = self._roi_label_entry.get() or roi.label
        self._roi_drag_start = None
        if self._current_roi_rect:
            self._canvas.delete(self._current_roi_rect)
            self._current_roi_rect = None
        self._refresh_roi_list()
        self._fit_image_to_canvas()

    def _on_mousewheel(self, event):
        if event.delta > 0 or event.num == 4:
            self._zoom_scale = min(3.0, self._zoom_scale * 1.1)
        else:
            self._zoom_scale = max(0.2, self._zoom_scale / 1.1)
        self._zoom_label.configure(text=f"{int(self._zoom_scale * 100)}%")
        self._fit_image_to_canvas()

    def _reset_roi_view(self):
        self._zoom_scale = 1.0
        self._zoom_label.configure(text="100%")
        self._fit_image_to_canvas()

    def _clear_roi(self):
        if self._current_video_idx >= 0 and self._current_roi_idx >= 0:
            entry = self._video_entries[self._current_video_idx]
            roi = entry.rois[self._current_roi_idx]
            roi.points = []
            roi.shape = self._roi_shape_var.get()
            roi.label = self._roi_label_entry.get() or roi.label
            self._fit_image_to_canvas()

    def _on_canvas_configure(self, event):
        """Handle canvas resize - re-fit image to new canvas size."""
        if self._roi_image is not None:
            self._fit_image_to_canvas()

    def _run_quantification(self):
        if not self._model_session:
            messagebox.showerror("Error", "Select a trained model session first")
            return
        if not self._weights_path:
            messagebox.showerror("Error", "No trained weights found in session")
            return
        if not self._video_entries:
            messagebox.showerror("Error", "Add at least one raw video")
            return
        if not any(v.enabled for v in self._video_entries):
            messagebox.showerror("Error", "Enable at least one video")
            return

        self._save_settings()

        self._run_btn.configure(state="disabled", text="Running...")
        self._progress.set(0)
        self._status_label.configure(text="Starting...")
        self._stats_label.configure(text="")

        self._start_time = time.monotonic()
        # Count videos that have at least one valid ROI (>= 3 points)
        vids_with_rois = 0
        for entry in self._video_entries:
            valid_rois = [r for r in entry.rois if r.enabled and len(r.points) >= 3]
            if valid_rois:
                vids_with_rois += 1
        self._total_videos = max(vids_with_rois, 1)
        self._videos_done = 0
        self._frames_done = 0
        self._frames_total = 0

        self._worker = threading.Thread(target=self._worker_run, daemon=True)
        self._worker.start()

    def _worker_run(self):
            try:
                import subprocess
                import tempfile
                import yaml
                import cv2

                self._queue.put(("progress", 0.05, "Preparing configuration...", self._progress_stats(0, self._total_videos, 0.05)))

                # Create a temporary config for the pipeline
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_path = os.path.join(tmpdir, "behavior_config.yaml")

                    # Build config for the full pipeline
                    session_path = self._model_session
                    self._session_path = session_path  # Store for endpoints extraction
                    if not os.path.isdir(session_path):
                        self._queue.put(("error", "Invalid model session directory"))
                        return

                    # Load existing session config and update it
                    session_config = os.path.join(session_path, "config.yaml")
                    if not os.path.exists(session_config):
                        self._queue.put(("error", "Session config.yaml not found"))
                        return

                    with open(session_config, "r") as f:
                        cfg = yaml.safe_load(f) or {}

                    # Update with our settings
                    # Use 'best' checkpoint selector for inference with pre-trained weights
                    cfg["WEIGHT"] = "best"

                    # Behavior settings
                    if "BEHAVIOR" not in cfg:
                        cfg["BEHAVIOR"] = {}
                    cfg["BEHAVIOR"]["ENABLE"] = True
                    cfg["BEHAVIOR"]["EXTRACT_HEAD"] = self._extract_head.get()
                    cfg["BEHAVIOR"]["EXTRACT_CENTER"] = self._extract_center.get()
                    cfg["BEHAVIOR"]["EXTRACT_TAIL"] = self._extract_tail.get()
                    cfg["BEHAVIOR"]["ANALYSIS"] = self._behavior_analysis.get()

                    # Display settings (arrow size, id font size, box thickness)
                    cfg["DISPLAY"] = {
                        "ARROW_SIZE": int(self._arrow_size.get()),
                        "ID_FONT_SIZE": float(self._id_font_size.get()),
                        "ID_FONT_THICKNESS": int(self._box_thickness.get()),
                        "BOX_THICKNESS": int(self._box_thickness.get()),
                    }

                    videos_to_process = []
                    for entry in self._video_entries:
                        valid_rois = [r for r in entry.rois if r.enabled and len(r.points) >= 3]
                        if valid_rois:
                            videos_to_process.append(entry)

                    if not videos_to_process:
                        self._queue.put(("error", "No videos with valid ROIs (need at least 3 points per ROI)"))
                        return

                    all_output_videos = []
                    total_videos = len(videos_to_process)

                    self._queue.put(("progress", 0.1, "Running behavioral quantification...", self._progress_stats(0, self._total_videos, 0.1, 0, self._frames_total)))

                    # Pre-compute total frames across all videos for realtime FPS
                    self._frames_total = 0
                    video_frame_counts = []
                    for entry in videos_to_process:
                        video_path = entry.path
                        if not video_path or not os.path.exists(video_path):
                            video_frame_counts.append(0)
                            continue
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            video_frame_counts.append(0)
                            continue
                        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        if tf <= 0:
                            tf = 0
                        self._frames_total += tf
                        video_frame_counts.append(tf)

                    cumulative_frames_before = 0

                    for vid_idx, entry in enumerate(videos_to_process):
                        video_path = entry.path
                        if not video_path or not os.path.exists(video_path):
                            self._queue.put(("log", f"Skipping {os.path.basename(video_path)}: file not found"))
                            cumulative_frames_before += video_frame_counts[vid_idx]
                            continue

                        # Get this video's dimensions for ROI normalization
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            self._queue.put(("error", f"Cannot open video: {video_path}"))
                            return
                        v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        if v_width <= 0 or v_height <= 0:
                            cumulative_frames_before += video_frame_counts[vid_idx]
                            continue

                        # Build ROI_SETS for THIS video only
                        roi_sets = []
                        tank_entries = []
                        roi_index = 0
                        for roi in entry.rois:
                            if not roi.enabled or len(roi.points) < 2:
                                continue
                            pts_norm = []
                            for pt in roi.points:
                                x_norm = pt[0] / v_width
                                y_norm = pt[1] / v_height
                                pts_norm.append([float(x_norm), float(y_norm)])

                            if len(pts_norm) >= 3:
                                roi_sets.append({
                                    "polygon": pts_norm,
                                    "fish_count": roi.fish_count,
                                    "population": "fixed",
                                })
                                tank_id = roi.label.strip().replace(" ", "_") if roi.label else f"tank_{roi_index + 1}"
                                tank_entries.append({
                                    "id": tank_id,
                                    "name": roi.label or f"Tank {roi_index + 1}",
                                    "polygon": pts_norm,
                                    "fish_count": roi.fish_count,
                                    "population": "fixed",
                                    "view": "top",
                                })
                                roi_index += 1

                        if not roi_sets:
                            self._queue.put(("log", f"Skipping {os.path.basename(video_path)}: no valid ROIs"))
                            cumulative_frames_before += video_frame_counts[vid_idx]
                            continue

                        cfg["ROI_SETS"] = roi_sets
                        cfg["TANKS"] = tank_entries
                        cfg["TRACKING_VIDEO_PATH"] = video_path
                        cfg["TRACKING_VIDEO_PATH_IS_DIR"] = False

                        with open(config_path, "w") as f:
                            yaml.dump(cfg, f)

                        # Each video occupies an equal slice of the 0.1..0.9 progress range.
                        # Detection pct (0..1) maps into that slice.
                        prog_start = 0.1 + 0.8 * vid_idx / max(total_videos, 1)
                        prog_end = 0.1 + 0.8 * (vid_idx + 1) / max(total_videos, 1)
                        frames_done = cumulative_frames_before
                        self._queue.put(("progress", prog_start, f"Processing {os.path.basename(video_path)} ({vid_idx + 1}/{total_videos})", self._progress_stats(vid_idx, total_videos, prog_start, frames_done, self._frames_total)))

                        # Run the standalone behavior quantification module (inference only)
                        cmd = [
                            sys.executable, "-u",
                            os.path.join(PROJECT_ROOT, "main", "behavior_quantification.py"),
                            config_path
                        ]

                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )

                        # Monitor progress - parse PROGRESS: lines for real-time frame updates
                        v_total_frames = video_frame_counts[vid_idx]
                        for line in proc.stdout:
                            line = line.strip()
                            if line.startswith("PROGRESS:"):
                                try:
                                    pct = float(line.split(":", 1)[1])
                                    pct = max(0.0, min(1.0, pct))
                                    prog = prog_start + (prog_end - prog_start) * pct
                                    frames_done = cumulative_frames_before + int(pct * v_total_frames)
                                    self._frames_done = frames_done
                                    self._queue.put(("progress", prog, f"Processing {os.path.basename(video_path)} ({vid_idx + 1}/{total_videos}) - {pct*100:.1f}% frames", self._progress_stats(vid_idx, total_videos, prog, frames_done, self._frames_total)))
                                except (ValueError, IndexError):
                                    pass
                            elif line:
                                self._queue.put(("log", line))

                        proc.wait()

                        if proc.returncode != 0:
                            self._queue.put(("error", f"Behavioral quantification failed for {os.path.basename(video_path)} with exit code {proc.returncode}"))
                            return

                        # Find the output video for this video
                        output_dir = os.path.join(session_path, "behavior_results")
                        video_name = os.path.splitext(os.path.basename(video_path))[0]
                        expected_video = os.path.join(output_dir, f"{video_name}_behavior.mp4")
                        expected_csv = os.path.join(output_dir, f"{video_name}_behavior.csv")
                        if os.path.exists(expected_video):
                            all_output_videos.append(expected_video)
                            self._queue.put(("log", f"Output: {expected_video}"))
                        # Track CSV for endpoints extraction
                        if os.path.exists(expected_csv):
                            self._queue.put(("tracking_result", video_path, expected_csv))

                        cumulative_frames_before += v_total_frames
                        self._frames_done = cumulative_frames_before
                        self._videos_done = vid_idx + 1
                        prog_done = min(prog_end, 0.9)
                        self._queue.put(("progress", prog_done, f"Completed {os.path.basename(video_path)} ({vid_idx + 1}/{total_videos})", self._progress_stats(self._videos_done, total_videos, prog_done, self._frames_done, self._frames_total)))

                    if not all_output_videos:
                        self._queue.put(("error", "No output videos were produced"))
                        return

                    # Show the last output video in preview
                    self._queue.put(("video_ready", all_output_videos[-1]))

                    self._queue.put(("progress", 1.0, f"Behavioral quantification complete! ({len(all_output_videos)} video(s))", self._progress_stats(len(all_output_videos), total_videos, 1.0, self._frames_done, self._frames_total)))
                    self._queue.put(("done", f"Behavioral quantification complete! ({len(all_output_videos)} video(s))"))

            except Exception as e:
                import traceback
                self._queue.put(("error", f"{e}\n{traceback.format_exc()}"))

    def _settings_dict(self) -> Dict[str, Any]:
        """Serialize current GUI state to a settings dict."""
        return {
            "model_session": self._model_session,
            "weights_path": self._weights_path,
            "output_dir": self._output_dir,
            "endpoints_output_dir": self._endpoints_output_entry.get() if hasattr(self, "_endpoints_output_entry") else "",
            "display": {
                "arrow_size": int(self._arrow_size.get()) if hasattr(self, "_arrow_size") else 45,
                "id_font_size": float(self._id_font_size.get()) if hasattr(self, "_id_font_size") else 0.6,
                "box_thickness": int(self._box_thickness.get()) if hasattr(self, "_box_thickness") else 2,
                "keypoint_radius": 4,
                "keypoint_font_size": 0.4,
                "trail_length": 30,
            },
            "behavior": {
                "extract_head": bool(self._extract_head.get()) if hasattr(self, "_extract_head") else True,
                "extract_center": bool(self._extract_center.get()) if hasattr(self, "_extract_center") else True,
                "extract_tail": bool(self._extract_tail.get()) if hasattr(self, "_extract_tail") else True,
                "analysis": bool(self._behavior_analysis.get()) if hasattr(self, "_behavior_analysis") else True,
            },
            "videos": [
                {
                    "path": entry.path,
                    "enabled": entry.enabled,
                    "rois": [
                        {
                            "points": roi.points,
                            "shape": roi.shape,
                            "label": roi.label,
                            "enabled": roi.enabled,
                            "fish_count": roi.fish_count,
                        }
                        for roi in entry.rois
                    ],
                }
                for entry in self._video_entries
            ],
        }

    def _apply_settings_dict(self, data: Dict[str, Any]):
        """Apply a settings dict to the GUI state."""
        self._model_session = data.get("model_session", "")
        self._weights_path = data.get("weights_path", "")
        self._output_dir = data.get("output_dir", "")

        # Update UI fields if they exist
        if hasattr(self, "_model_entry") and self._model_session:
            self._model_entry.delete(0, tk.END)
            self._model_entry.insert(0, self._model_session)
        if hasattr(self, "_weights_path") and self._weights_path:
            if hasattr(self, "_weights_label"):
                self._weights_label.configure(text=f"Weights: {os.path.basename(self._weights_path)}", text_color="white")
        if hasattr(self, "_output_entry") and self._output_dir:
            self._output_entry.delete(0, tk.END)
            self._output_entry.insert(0, self._output_dir)

        disp = data.get("display", {})
        if hasattr(self, "_arrow_size"):
            self._arrow_size.set(int(disp.get("arrow_size", 45)))
        if hasattr(self, "_id_font_size"):
            self._id_font_size.set(float(disp.get("id_font_size", 0.6)))
        if hasattr(self, "_box_thickness"):
            self._box_thickness.set(int(disp.get("box_thickness", 2)))

        beh = data.get("behavior", {})
        if hasattr(self, "_extract_head"):
            self._extract_head.set(bool(beh.get("extract_head", True)))
        if hasattr(self, "_extract_center"):
            self._extract_center.set(bool(beh.get("extract_center", True)))
        if hasattr(self, "_extract_tail"):
            self._extract_tail.set(bool(beh.get("extract_tail", True)))
        if hasattr(self, "_behavior_analysis"):
            self._behavior_analysis.set(bool(beh.get("analysis", True)))

        # Restore endpoints output directory
        ep_out_dir = data.get("endpoints_output_dir", "")
        if hasattr(self, "_endpoints_output_entry") and ep_out_dir:
            self._endpoints_output_entry.delete(0, tk.END)
            self._endpoints_output_entry.insert(0, ep_out_dir)

        # Restore endpoints settings
        if hasattr(self, "_endpoints_metric_selections"):
            self._endpoints_metric_selections = data.get("endpoints_metric_selections", {})
        if hasattr(self, "_endpoints_global_config"):
            self._endpoints_global_config = data.get("endpoints_global_config", {})
        if hasattr(self, "_endpoints_arena_configs"):
            self._endpoints_arena_configs = data.get("endpoints_arena_configs", {})
        if hasattr(self, "_endpoints_calibration_ratio"):
            self._endpoints_calibration_ratio = data.get("endpoints_calibration_ratio")

        # Restore video entries with ROIs
        self._video_entries = []
        for vd in data.get("videos", []):
            entry = VideoEntry(path=vd.get("path", ""), enabled=vd.get("enabled", True))
            for rd in vd.get("rois", []):
                roi = ROIEntry(
                    points=rd.get("points", []),
                    shape=rd.get("shape", "rectangle"),
                    label=rd.get("label", ""),
                    enabled=rd.get("enabled", True),
                    fish_count=rd.get("fish_count", 1),
                )
                entry.rois.append(roi)
            self._video_entries.append(entry)

        # Refresh UI lists
        if hasattr(self, "_video_list_frame"):
            for widget in self._video_list_frame.winfo_children():
                widget.destroy()
            for entry in self._video_entries:
                self._add_video_row(entry)

        if hasattr(self, "_current_video_idx"):
            self._current_video_idx = -1
        if hasattr(self, "_current_roi_idx"):
            self._current_roi_idx = -1

    def _save_settings(self):
        """Persist current settings to disk (session folder first, then legacy)."""
        # Try session folder first (where videos are loaded from)
        video_paths = [entry.path for entry in self._video_entries]
        session_dir = find_session_dir_from_videos(video_paths)
        
        settings_data = self._settings_dict()
        
        # Add endpoints settings if available
        if hasattr(self, "_endpoints_metric_selections"):
            settings_data["endpoints_metric_selections"] = self._endpoints_metric_selections
        if hasattr(self, "_endpoints_global_config"):
            settings_data["endpoints_global_config"] = self._endpoints_global_config
        if hasattr(self, "_endpoints_arena_configs"):
            settings_data["endpoints_arena_configs"] = self._endpoints_arena_configs
        if hasattr(self, "_endpoints_calibration_ratio"):
            settings_data["endpoints_calibration_ratio"] = self._endpoints_calibration_ratio
        
        # Save to session folder
        if session_dir:
            if save_session_settings(session_dir, settings_data):
                print(f"[BEHAVIOR] Saved settings to session folder: {session_dir}")
                return
        
        # Fallback to legacy user config directory
        try:
            os.makedirs(SETTINGS_DIR, exist_ok=True)
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings_data, f, indent=2)
            print(f"[BEHAVIOR] Saved settings to legacy location: {SETTINGS_FILE}")
        except Exception as e:
            print(f"[BEHAVIOR] Failed to save settings: {e}")

    def _load_settings(self):
        """Load persisted settings from disk (session folder first, then legacy)."""
        # Try session folder first
        video_paths = [entry.path for entry in self._video_entries]
        session_dir = find_session_dir_from_videos(video_paths)
        
        loaded = False
        if session_dir:
            data = load_session_settings(session_dir)
            if data:
                self._apply_settings_dict(data)
                print(f"[BEHAVIOR] Loaded settings from session folder: {session_dir}")
                loaded = True
        
        # Fallback to legacy user config directory
        if not loaded:
            try:
                if os.path.exists(SETTINGS_FILE):
                    with open(SETTINGS_FILE, "r") as f:
                        data = json.load(f)
                    self._apply_settings_dict(data)
                    print(f"[BEHAVIOR] Loaded settings from legacy location: {SETTINGS_FILE}")
                    loaded = True
            except Exception as e:
                print(f"[BEHAVIOR] Failed to load legacy settings: {e}")
        
        # If still not loaded, try to find session folder from model session
        if not loaded and self._model_session:
            data = load_session_settings(self._model_session)
            if data:
                self._apply_settings_dict(data)
                print(f"[BEHAVIOR] Loaded settings from model session: {self._model_session}")
                loaded = True
        
        # Scan for existing tracking results in model session
        self._scan_existing_tracking_results()
        
    def _scan_existing_tracking_results(self):
        """Scan model session for existing behavior_results tracking CSVs."""
        if not self._model_session or not os.path.isdir(self._model_session):
            return
            
        behavior_results_dir = os.path.join(self._model_session, "behavior_results")
        if not os.path.isdir(behavior_results_dir):
            return
            
        # Look for *_behavior.csv files
        for entry in self._video_entries:
            video_name = os.path.splitext(os.path.basename(entry.path))[0]
            csv_path = os.path.join(behavior_results_dir, f"{video_name}_behavior.csv")
            if os.path.exists(csv_path):
                self._tracking_results[entry.path] = csv_path
                self._log(f"📝 Found existing tracking: {os.path.basename(csv_path)}")
        
        # Enable endpoints button if we found any
        if self._tracking_results:
            self._endpoints_btn.configure(state="normal")
            self._log(f"✅ {len(self._tracking_results)} pre-tracked video(s) ready for endpoints extraction.")

    def _save_settings_profile(self):
        """Save current settings to a user-chosen file."""
        path = filedialog.asksaveasfilename(
            title="Save Settings Profile",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=SETTINGS_DIR,
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._settings_dict(), f, indent=2)
            messagebox.showinfo("Saved", f"Settings saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def _load_settings_profile(self):
        """Load settings from a user-chosen file."""
        path = filedialog.askopenfilename(
            title="Load Settings Profile",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=SETTINGS_DIR,
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._apply_settings_dict(data)
            self._save_settings()
            messagebox.showinfo("Loaded", f"Settings loaded from:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {e}")

    def _on_view_toggle(self, value=None):
        """Switch between ROI editor view and Results video view."""
        view = self._view_var.get()
        if view == "Results":
            self._canvas_frame.grid(row=6, column=0, sticky="nsew", padx=10, pady=(0, 10))
            self._video_ctrl_frame.grid()
            self._canvas.delete("all")
            if hasattr(self, '_video_cap') and self._video_cap is not None:
                ret, frame = self._video_cap.read()
                if ret:
                    self._display_frame(frame)
                self._update_playback_ui()
        else:
            self._video_ctrl_frame.grid_remove()
            self._canvas_frame.grid(row=6, column=0, sticky="nsew", padx=10, pady=(0, 10))
            self._fit_image_to_canvas()

    def _progress_stats(self, videos_done: int, total_videos: int, progress: float,
                        frames_done: int = 0, frames_total: int = 0) -> Dict[str, Any]:
        """Compute a stats dict for the progress display."""
        elapsed = time.monotonic() - getattr(self, "_start_time", time.monotonic())
        # Use frames per second for realtime FPS during active processing.
        fps = (frames_done / elapsed) if elapsed > 0 and frames_done > 0 else 0.0
        eta = ((frames_total - frames_done) / fps) if fps > 0 else 0.0
        pct = progress * 100.0
        return {
            "videos_done": videos_done,
            "total_videos": total_videos,
            "progress": progress,
            "pct": pct,
            "elapsed": elapsed,
            "fps": fps,
            "eta": eta,
            "frames_done": frames_done,
            "frames_total": frames_total,
        }

    def _apply_rois_to_all_videos(self):
        """Copy the currently selected video's ROIs to all other videos,
        normalizing coordinates per-video by resolution."""
        if self._current_video_idx < 0:
            messagebox.showinfo("Info", "Select a video first")
            return
        src_entry = self._video_entries[self._current_video_idx]
        if not src_entry.rois:
            messagebox.showinfo("Info", "No ROIs defined for the selected video")
            return

        # Get source video dimensions
        try:
            cap = cv2.VideoCapture(src_entry.path)
            if not cap.isOpened():
                messagebox.showerror("Error", f"Cannot open source video: {src_entry.path}")
                return
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read source video: {e}")
            return

        if src_w <= 0 or src_h <= 0:
            messagebox.showerror("Error", "Invalid source video dimensions")
            return

        # Normalize source ROIs to [0,1]
        normalized_rois = []
        for roi in src_entry.rois:
            norm_points = []
            for pt in roi.points:
                norm_points.append([pt[0] / src_w, pt[1] / src_h])
            normalized_rois.append({
                "points": norm_points,
                "shape": roi.shape,
                "label": roi.label,
                "enabled": roi.enabled,
                "fish_count": roi.fish_count,
            })

        # Apply to all other videos
        applied_count = 0
        for entry in self._video_entries:
            if entry is src_entry:
                continue
            try:
                cap = cv2.VideoCapture(entry.path)
                if not cap.isOpened():
                    continue
                t_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                t_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if t_w <= 0 or t_h <= 0:
                    continue
            except Exception:
                continue

            entry.rois = []
            for nr in normalized_rois:
                pts = []
                for pt in nr["points"]:
                    pts.append((pt[0] * t_w, pt[1] * t_h))
                roi = ROIEntry(
                    points=pts,
                    shape=nr["shape"],
                    label=nr["label"],
                    enabled=nr["enabled"],
                    fish_count=nr["fish_count"],
                )
                entry.rois.append(roi)
            applied_count += 1

        self._save_settings()
        messagebox.showinfo(
            "Applied",
            f"ROIs applied to {applied_count} other video(s).\n"
            "Coordinates are normalized per-video resolution.",
        )

    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg[0] == "progress":
                    self._progress.set(msg[1])
                    self._status_label.configure(text=msg[2])
                    # Display detailed stats if provided (4-tuple)
                    if len(msg) >= 4 and isinstance(msg[3], dict):
                        st = msg[3]
                        elapsed_s = st.get("elapsed", 0)
                        eta_s = st.get("eta", 0)
                        fps = st.get("fps", 0)
                        vd = st.get("videos_done", 0)
                        tv = st.get("total_videos", 0)
                        pct = st.get("progress", 0) * 100
                        stats_text = (
                            f"Videos: {vd}/{tv}  |  Progress: {pct:.1f}%  |  "
                            f"FPS: {fps:.2f}  |  Elapsed: {self._format_time(elapsed_s)}  |  "
                            f"ETA: {self._format_time(eta_s) if eta_s > 0 else '--:--'}"
                        )
                        self._stats_label.configure(text=stats_text)
                elif msg[0] == "log":
                    print(f"[BEHAVIOR] {msg[1]}")
                elif msg[0] == "tracking_result":
                    video_path, csv_path = msg[1], msg[2]
                    self._tracking_results[video_path] = csv_path
                    self._log(f"📝 Tracking CSV ready: {os.path.basename(csv_path)}")
                elif msg[0] == "video_ready":
                    self._load_output_video(msg[1])
                elif msg[0] == "done":
                    self._progress.set(1)
                    self._status_label.configure(text=msg[1], text_color="green")
                    self._run_btn.configure(state="normal", text="Run Quantification")
                    # Enable endpoints button if we have tracking results
                    if self._tracking_results:
                        self._endpoints_btn.configure(state="normal")
                        self._log(f"✅ Tracking complete! {len(self._tracking_results)} video(s) ready for endpoints extraction.")
                    messagebox.showinfo("Complete", msg[1])
                elif msg[0] == "error":
                    self._status_label.configure(text=f"Error: {msg[1]}", text_color="red")
                    self._stats_label.configure(text="")
                    self._run_btn.configure(state="normal", text="Run Quantification")
                    messagebox.showerror("Error", msg[1])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _log(self, msg: str):
        """Log message to console (stdout for now, could be extended to UI)."""
        print(f"[BEHAVIOR] {msg}")

    def _run_endpoints_extraction(self):
        """Launch the endpoints extraction dialog."""
        if not self._tracking_results:
            messagebox.showwarning("No Data", "No tracking results available. Run quantification first.")
            return
        if not self._model_session:
            messagebox.showwarning("No Session", "Model session not set.")
            return
            
        from gui.endpoints_dialog import show_endpoints_extraction_dialog
        
        # Default all metrics enabled (selection happens in endpoints dialog)
        all_default_metrics = {
            "Average Speed (cm/s)": True,
            "Freezing Time Ratio (%)": True,
            "Swimming Time Ratio (%)": True,
            "Rapid Movement Time Ratio (%)": True,
            "Time in Top Percentage (%)": True,
            "Time in Bottom Percentage (%)": True,
            "Average Thigmotaxis (cm)": True,
            "Fractal Dimension": True,
            "Entropy": True,
            "Average Speed (body-length/s)": True,
            "Freezing Time Ratio (body-length/s) (%)": True,
            "Swimming Time Ratio (body-length/s) (%)": True,
            "Rapid Movement Time Ratio (body-length/s) (%)": True,
        }
        
        # Get endpoints output directory
        ep_output_dir = self._endpoints_output_entry.get().strip()
        
        def on_complete(result):
            self._log(f"Endpoints extraction complete: {result.get('output_dir', 'unknown')}")
            
        show_endpoints_extraction_dialog(
            self,
            self._video_entries,
            self._tracking_results,
            self._model_session,
            all_default_metrics,
            ep_output_dir,
            on_complete,
        )

    def _load_output_video(self, video_path: str):
        """Load the output video for preview."""
        try:
            self._video_cap = cv2.VideoCapture(video_path)
            self._video_fps = self._video_cap.get(cv2.CAP_PROP_FPS) or 30
            self._video_total_frames = int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._play_btn.configure(state="normal")
            self._step_back_btn.configure(state="normal")
            self._step_fwd_btn.configure(state="normal")
            self._frame_slider.configure(from_=0, to=100, number_of_steps=max(self._video_total_frames - 1, 1))
            # Switch to Results view and show controls
            self._view_var.set("Results")
            self._on_view_toggle()
            # Show first frame
            ret, frame = self._video_cap.read()
            if ret:
                self._display_frame(frame)
            self._update_playback_ui()
        except Exception as e:
            print(f"Failed to load output video: {e}")


def main() -> int:
    app = BehaviorQuantificationGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())