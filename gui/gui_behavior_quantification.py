# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Fish Behavior Quantification GUI.

Load a trained model session, select raw videos, draw ROIs per video,
and run head/tail/center keypoint extraction + behavior analysis.
"""

from __future__ import annotations

import os
import sys
import threading
import queue
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main.path_utils import resolve_config_paths, to_relative_path
from tools.project_meta import PROJECT_DISPLAY_NAME as PROJECT_NAME


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
        self.geometry("1000x700")
        self.minsize(900, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._video_entries: List[VideoEntry] = []
        self._model_session: str = ""
        self._weights_path: str = ""
        self._output_dir: str = ""
        self._queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        self._build_ui()

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

        # Run button
        self._run_btn = ctk.CTkButton(
            left,
            text="Run Quantification",
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._run_quantification,
        )
        self._run_btn.pack(fill="x", pady=10)

        # Progress
        self._progress = ctk.CTkProgressBar(left)
        self._progress.pack(fill="x", pady=(0, 5))
        self._progress.set(0)

        self._status_label = ctk.CTkLabel(left, text="Ready", text_color="gray")
        self._status_label.pack(fill="x", pady=(0, 10))

        # Output video preview
        self._video_preview_frame = ctk.CTkFrame(left, corner_radius=6)
        self._video_preview_frame.pack(fill="x", pady=(0, 10))
        self._video_preview_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self._video_preview_frame, text="Output Video Preview", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 5)
        )

        self._video_canvas = tk.Canvas(self._video_preview_frame, bg="#1a1a1a", height=200, highlightthickness=0)
        self._video_canvas.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self._video_photo = None

        # Video controls
        vid_ctrl = ctk.CTkFrame(self._video_preview_frame, corner_radius=0)
        vid_ctrl.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        vid_ctrl.grid_columnconfigure(1, weight=1)

        self._play_btn = ctk.CTkButton(vid_ctrl, text="▶ Play", width=80, command=self._toggle_video_playback, state="disabled")
        self._play_btn.grid(row=0, column=0, padx=5, pady=5)

        self._video_time_label = ctk.CTkLabel(vid_ctrl, text="00:00 / 00:00")
        self._video_time_label.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkButton(vid_ctrl, text="Open Folder", width=100, command=self._open_output_folder).grid(
            row=0, column=2, padx=5, pady=5
        )

        # Right panel: ROI Canvas
        right = ctk.CTkFrame(content, corner_radius=6)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right, text="ROI Editor (select video from list)", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10, pady=10)

        self._roi_canvas = tk.Canvas(right, bg="#1a1a1a", highlightthickness=0)
        self._roi_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._roi_canvas.bind("<Button-1>", self._on_canvas_press)
        self._roi_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._roi_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._roi_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._roi_canvas.bind("<Button-4>", self._on_mousewheel)
        self._roi_canvas.bind("<Button-5>", self._on_mousewheel)
        self._roi_canvas.bind("<Configure>", self._on_canvas_configure)

        self._roi_photo = None
        self._roi_image = None
        self._roi_scale = 1.0
        self._roi_drag_start = None
        self._current_roi_rect = None
        self._current_video_idx = -1
        self._current_roi_idx = -1
        self._zoom_scale = 1.0
        self._base_fit_scale = 1.0

        # ROI list panel
        roi_list_frame = ctk.CTkFrame(right, corner_radius=6)
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
        cw = max(self._video_canvas.winfo_width(), 400)
        ch = max(self._video_canvas.winfo_height(), 200)
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        self._video_photo = ImageTk.PhotoImage(img)
        self._video_canvas.delete("all")
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self._video_canvas.create_image(x, y, anchor="nw", image=self._video_photo)

        # Update time label
        current_frame = int(self._video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame / max(self._video_fps, 1)
        total_time = self._video_total_frames / max(self._video_fps, 1)
        self._video_time_label.configure(text=f"{self._format_time(current_time)} / {self._format_time(total_time)}")

        # Schedule next frame
        delay = int(1000 / max(self._video_fps, 1))
        self.after(delay, self._play_video_frame)

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

    def _select_video_for_roi(self, entry: VideoEntry):
        idx = self._video_entries.index(entry)
        self._current_video_idx = idx
        self._current_roi_idx = -1
        self._load_video_frame(entry.path)
        self._refresh_roi_list()

    def _add_roi(self):
        if self._current_video_idx < 0:
            messagebox.showinfo("Info", "Select a video first")
            return
        entry = self._video_entries[self._current_video_idx]
        new_roi = ROIEntry(shape=self._roi_shape_var.get(), label=self._roi_label_entry.get() or f"ROI {len(entry.rois)+1}")
        entry.rois.append(new_roi)
        self._current_roi_idx = len(entry.rois) - 1
        self._refresh_roi_list()
        self._fit_image_to_canvas()

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

            label = roi.label or f"ROI {i+1} ({roi.shape})"
            ctk.CTkLabel(row, text=label, anchor="w").grid(row=0, column=1, sticky="ew", padx=5)

            # Fish count entry
            fish_count_var = ctk.StringVar(value=str(roi.fish_count))
            ctk.CTkLabel(row, text="Fish:").grid(row=0, column=2, padx=(10, 2), pady=5)
            fish_spinbox = ctk.CTkEntry(row, textvariable=fish_count_var, width=40)
            fish_spinbox.grid(row=0, column=3, padx=2, pady=5)
            def _update_fish_count(r=roi, v=fish_count_var):
                try:
                    val = int(v.get())
                    if val >= 1:
                        r.fish_count = val
                except ValueError:
                    pass
            fish_spinbox.bind("<FocusOut>", lambda e: _update_fish_count())
            fish_spinbox.bind("<Return>", lambda e: _update_fish_count())

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
        self._roi_canvas.update_idletasks()
        cw = self._roi_canvas.winfo_width()
        ch = self._roi_canvas.winfo_height()
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
        self._roi_canvas.delete("all")
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self._roi_canvas.create_image(x, y, anchor="nw", image=self._roi_photo)
        # Draw all ROIs for this video
        if self._current_video_idx >= 0:
            entry = self._video_entries[self._current_video_idx]
            self._draw_all_rois_on_canvas(entry)

    def _canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> tuple:
        """Convert canvas coordinates to image coordinates."""
        if self._roi_image is None:
            return (0, 0)
        # Ensure canvas is laid out before getting dimensions
        self._roi_canvas.update_idletasks()
        cw = self._roi_canvas.winfo_width()
        ch = self._roi_canvas.winfo_height()
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
        self._roi_canvas.update_idletasks()
        cw = self._roi_canvas.winfo_width()
        ch = self._roi_canvas.winfo_height()
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
                self._roi_canvas.create_polygon(*flat, fill="", outline=color, width=width, tags=f"roi_{i}")
            elif roi.shape == "circle" and len(canvas_points) >= 2:
                x0, y0 = canvas_points[0]
                x1, y1 = canvas_points[1]
                self._roi_canvas.create_oval(x0, y0, x1, y1, outline=color, width=width, tags=f"roi_{i}")
            elif roi.shape in ("rectangle", "grid") and len(canvas_points) >= 2:
                x0, y0 = canvas_points[0]
                x1, y1 = canvas_points[1]
                dash = (4, 4) if roi.shape == "grid" else ()
                self._roi_canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=width, dash=dash, tags=f"roi_{i}")
                if roi.shape == "grid":
                    rows = 3
                    cols = 3
                    for r in range(1, rows):
                        ry = y0 + (y1 - y0) * r / rows
                        self._roi_canvas.create_line(x0, ry, x1, ry, fill=color, width=width, dash=(4, 4), tags=f"roi_{i}")
                    for c in range(1, cols):
                        cx = x0 + (x1 - x0) * c / cols
                        self._roi_canvas.create_line(cx, y0, cx, y1, fill=color, width=width, dash=(4, 4), tags=f"roi_{i}")

            # Draw ROI number/label on the canvas
            if canvas_points:
                # Use centroid of first 3 points for label position
                if len(canvas_points) >= 3:
                    cx = sum(p[0] for p in canvas_points[:3]) / 3
                    cy = sum(p[1] for p in canvas_points[:3]) / 3
                else:
                    cx, cy = canvas_points[0]

                # ROI number (1-based) and fish count
                label_text = f"ROI {i+1} ({roi.fish_count})"
                if roi.label:
                    label_text = f"{roi.label} ({roi.fish_count})"

                self._roi_canvas.create_text(
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
            self._roi_canvas.delete(self._current_roi_rect)

    def _on_canvas_drag(self, event):
        if self._roi_drag_start is None or self._current_video_idx < 0 or self._current_roi_idx < 0:
            return
        x0, y0 = self._roi_drag_start
        if self._current_roi_rect:
            self._roi_canvas.delete(self._current_roi_rect)
        shape = self._roi_shape_var.get()
        if shape == "rectangle":
            self._current_roi_rect = self._roi_canvas.create_rectangle(
                x0, y0, event.x, event.y, outline="#38bdf8", width=2
            )
        elif shape == "circle":
            self._current_roi_rect = self._roi_canvas.create_oval(
                x0, y0, event.x, event.y, outline="#38bdf8", width=2
            )
        elif shape == "grid":
            self._current_roi_rect = self._roi_canvas.create_rectangle(
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
            self._roi_canvas.delete(self._current_roi_rect)
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

        self._run_btn.configure(state="disabled", text="Running...")
        self._progress.set(0)
        self._status_label.configure(text="Starting...")

        self._worker = threading.Thread(target=self._worker_run, daemon=True)
        self._worker.start()

    def _worker_run(self):
            try:
                import subprocess
                import tempfile
                import yaml
                import cv2

                self._queue.put(("progress", 0.05, "Preparing configuration..."))

                # Create a temporary config for the pipeline
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_path = os.path.join(tmpdir, "behavior_config.yaml")

                    # Get video dimensions for ROI normalization
                    video_path = self._video_entries[0].path if self._video_entries else ""
                    if not video_path or not os.path.exists(video_path):
                        self._queue.put(("error", "No valid video path"))
                        return

                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        self._queue.put(("error", "Cannot open video"))
                        return
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    self._queue.put(("progress", 0.07, f"Video: {width}x{height}"))

                    # Build config for the full pipeline
                    session_path = self._model_session
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
                    cfg["TRACKING_VIDEO_PATH"] = video_path
                    cfg["TRACKING_VIDEO_PATH_IS_DIR"] = False

                    # Behavior settings
                    if "BEHAVIOR" not in cfg:
                        cfg["BEHAVIOR"] = {}
                    cfg["BEHAVIOR"]["ENABLE"] = True
                    cfg["BEHAVIOR"]["EXTRACT_HEAD"] = self._extract_head.get()
                    cfg["BEHAVIOR"]["EXTRACT_CENTER"] = self._extract_center.get()
                    cfg["BEHAVIOR"]["EXTRACT_TAIL"] = self._extract_tail.get()
                    cfg["BEHAVIOR"]["ANALYSIS"] = self._behavior_analysis.get()

                    # Build ROI_SETS from GUI ROIs (normalized coordinates)
                    roi_sets = []
                    tank_entries = []
                    roi_index = 0
                    for entry in self._video_entries:
                        if not entry.enabled:
                            continue
                        for roi in entry.rois:
                            if not roi.enabled or len(roi.points) < 2:
                                continue
                            # Convert pixel coordinates to normalized [0,1]
                            pts_norm = []
                            for pt in roi.points:
                                x_norm = pt[0] / width
                                y_norm = pt[1] / height
                                pts_norm.append([float(x_norm), float(y_norm)])

                            if len(pts_norm) >= 3:
                                roi_sets.append({
                                    "polygon": pts_norm,
                                    "fish_count": roi.fish_count,
                                    "population": "fixed",
                                })
                                # TANKS needs an id field
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
                        self._queue.put(("error", "No valid ROIs with at least 3 points"))
                        return

                    cfg["ROI_SETS"] = roi_sets

                    # Also set TANKS for per-ROI analysis (with required id field)
                    cfg["TANKS"] = tank_entries

                    with open(config_path, "w") as f:
                        yaml.dump(cfg, f)

                    self._queue.put(("progress", 0.1, "Running behavioral quantification..."))

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

                    # Monitor progress
                    for line in proc.stdout:
                        self._queue.put(("log", line.strip()))

                    proc.wait()

                    if proc.returncode != 0:
                        self._queue.put(("error", f"Behavioral quantification failed with exit code {proc.returncode}"))
                        return

                    self._queue.put(("progress", 0.9, "Finding output files..."))

                    # Find the output video and CSV
                    output_dir = os.path.join(session_path, "behavior_results")
                    video_found = False
                    if os.path.exists(output_dir):
                        for f in os.listdir(output_dir):
                            if f.endswith(".mp4") and "behavior" in f.lower():
                                output_video = os.path.join(output_dir, f)
                                self._queue.put(("video_ready", output_video))
                                video_found = True

                    if not video_found:
                        self._queue.put(("error", "Output video not found"))
                        return

                    self._queue.put(("progress", 1.0, "Behavioral quantification complete!"))
                    self._queue.put(("done", "Behavioral quantification complete!"))

            except Exception as e:
                import traceback
                self._queue.put(("error", f"{e}\n{traceback.format_exc()}"))

    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg[0] == "progress":
                    self._progress.set(msg[1])
                    self._status_label.configure(text=msg[2])
                elif msg[0] == "log":
                    print(f"[BEHAVIOR] {msg[1]}")
                elif msg[0] == "video_ready":
                    self._load_output_video(msg[1])
                elif msg[0] == "done":
                    self._progress.set(1)
                    self._status_label.configure(text=msg[1], text_color="green")
                    self._run_btn.configure(state="normal", text="Run Quantification")
                    messagebox.showinfo("Complete", msg[1])
                elif msg[0] == "error":
                    self._status_label.configure(text=f"Error: {msg[1]}", text_color="red")
                    self._run_btn.configure(state="normal", text="Run Quantification")
                    messagebox.showerror("Error", msg[1])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _load_output_video(self, video_path: str):
        """Load the output video for preview."""
        try:
            self._video_cap = cv2.VideoCapture(video_path)
            self._video_fps = self._video_cap.get(cv2.CAP_PROP_FPS) or 30
            self._video_total_frames = int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._play_btn.configure(state="normal")
            # Show first frame
            ret, frame = self._video_cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                cw = max(self._video_canvas.winfo_width(), 400)
                ch = max(self._video_canvas.winfo_height(), 200)
                iw, ih = img.size
                scale = min(cw / iw, ch / ih)
                new_w = int(iw * scale)
                new_h = int(ih * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                self._video_photo = ImageTk.PhotoImage(img)
                self._video_canvas.delete("all")
                x = (cw - new_w) // 2
                y = (ch - new_h) // 2
                self._video_canvas.create_image(x, y, anchor="nw", image=self._video_photo)
                self._video_time_label.configure(text=f"00:00 / {self._format_time(self._video_total_frames / max(self._video_fps, 1))}")
        except Exception as e:
            print(f"Failed to load output video: {e}")


def main() -> int:
    app = BehaviorQuantificationGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())