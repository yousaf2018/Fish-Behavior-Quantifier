# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Endpoints Extraction Dialog for Fish Behavior Quantification.

Provides a comprehensive UI for configuring and running behavioral endpoint
extraction after tracking is complete.
"""

from __future__ import annotations

import os
import json
import math
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field, asdict

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np

from main.endpoints_engine import EndpointsEngine, calculate_pixel_cm_ratio

# Add project root to path for settings_manager
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.settings_manager import (
    create_default_session_settings,
    find_session_dir_from_videos,
    get_session_settings_path,
    load_session_settings,
    merge_settings,
    save_session_settings,
)


@dataclass
class ArenaConfig:
    """Per-arena configuration for endpoint extraction."""
    cx_pct: float = 50.0
    cy_pct: float = 50.0
    view_type: str = "Side / Lateral View"
    division_rule: str = "Horizontal Split (Y < Center is Top)"


class PixelCmCalibrationDialog(ctk.CTkToplevel):
    """Dialog for calibrating pixel/cm ratio by drawing a scale on a known distance."""
    
    def __init__(self, parent, video_path: str, callback: Callable[[float], None]):
        super().__init__(parent)
        self.title("Pixel/cm Calibration - Draw Scale on Known Distance")
        self.geometry("900x700")
        self.minsize(800, 600)
        
        self._video_path = video_path
        self._callback = callback
        self._scale_points: List[tuple] = []
        self._known_distance_cm = 1.0
        self._video_frame = None
        self._photo = None
        self._zoom = 1.0
        self._drag_start = None
        
        self._setup_ui()
        self._load_first_frame()
        
    def _setup_ui(self):
        # Instructions
        instr_frame = ctk.CTkFrame(self, corner_radius=6)
        instr_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            instr_frame,
            text="📏 Pixel/cm Calibration",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        ctk.CTkLabel(
            instr_frame,
            text="1. Enter the known distance (e.g., 10 cm for a 10 cm ruler in the video)\n"
                 "2. Click two points on the image representing that known distance\n"
                 "3. The pixel/cm ratio will be calculated automatically\n"
                 "4. Click 'Apply' to use this calibration",
            justify="left"
        ).pack(padx=15, pady=(0, 10))
        
        # Known distance input
        dist_frame = ctk.CTkFrame(instr_frame, corner_radius=0)
        dist_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(dist_frame, text="Known Distance (cm):").pack(side="left", padx=5)
        self._dist_entry = ctk.CTkEntry(dist_frame, width=100)
        self._dist_entry.insert(0, "1.0")
        self._dist_entry.pack(side="left", padx=5)
        
        self._calc_label = ctk.CTkLabel(dist_frame, text="Pixel/cm: --", text_color="#39FF14", font=ctk.CTkFont(weight="bold"))
        self._calc_label.pack(side="right", padx=10)
        
        # Canvas
        canvas_frame = ctk.CTkFrame(self, corner_radius=6)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self._canvas = tk.Canvas(canvas_frame, bg="#1a1a1a", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Button-4>", self._on_mousewheel)
        self._canvas.bind("<Button-5>", self._on_mousewheel)
        self._canvas.bind("<Configure>", self._on_configure)
        
        # Buttons
        btn_frame = ctk.CTkFrame(self, corner_radius=0)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkButton(btn_frame, text="Clear Points", command=self._clear_points, 
                      fg_color="#c42b1c", hover_color="#a02016").pack(side="left", padx=5, pady=10)
        ctk.CTkButton(btn_frame, text="Reset View", command=self._reset_view).pack(side="left", padx=5, pady=10)
        ctk.CTkButton(btn_frame, text="Cancel", command=self.destroy, 
                      fg_color="gray", hover_color="#555").pack(side="right", padx=5, pady=10)
        ctk.CTkButton(btn_frame, text="Apply Calibration", command=self._apply, 
                      fg_color="#28a745", hover_color="#218838").pack(side="right", padx=5, pady=10)
        
    def _load_first_frame(self):
        cap = cv2.VideoCapture(self._video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self._video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()
        self._fit_to_canvas()
        
    def _fit_to_canvas(self):
        if self._video_frame is None:
            return
        self.update_idletasks()
        cw = max(self._canvas.winfo_width(), 400)
        ch = max(self._canvas.winfo_height(), 300)
        ih, iw = self._video_frame.shape[:2]
        self._zoom = min(cw / iw, ch / ih) * 0.95
        new_w, new_h = int(iw * self._zoom), int(ih * self._zoom)
        img = Image.fromarray(self._video_frame).resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        x, y = (cw - new_w) // 2, (ch - new_h) // 2
        self._canvas.create_image(x, y, anchor="nw", image=self._photo)
        self._canvas.config(scrollregion=(0, 0, new_w, new_h))
        self._redraw_scale()
        
    def _canvas_to_image(self, cx: int, cy: int) -> tuple:
        if self._video_frame is None:
            return 0, 0
        self.update_idletasks()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        ih, iw = self._video_frame.shape[:2]
        new_w, new_h = int(iw * self._zoom), int(ih * self._zoom)
        x_off, y_off = (cw - new_w) // 2, (ch - new_h) // 2
        return (cx - x_off) / self._zoom, (cy - y_off) / self._zoom
        
    def _on_click(self, event):
        if self._video_frame is None:
            return
        img_pt = self._canvas_to_image(event.x, event.y)
        self._scale_points.append(img_pt)
        if len(self._scale_points) > 2:
            self._scale_points = self._scale_points[-2:]
        self._redraw_scale()
        self._update_calc_label()
        
    def _redraw_scale(self):
        self._canvas.delete("scale")
        if len(self._scale_points) >= 1:
            for i, pt in enumerate(self._scale_points):
                cx, cy = self._image_to_canvas(pt[0], pt[1])
                self._canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill="#39FF14", outline="white", width=2, tags="scale")
                self._canvas.create_text(cx, cy-15, text=str(i+1), fill="white", font=("Arial", 10, "bold"), tags="scale")
        if len(self._scale_points) == 2:
            x1, y1 = self._image_to_canvas(self._scale_points[0][0], self._scale_points[0][1])
            x2, y2 = self._image_to_canvas(self._scale_points[1][0], self._scale_points[1][1])
            self._canvas.create_line(x1, y1, x2, y2, fill="#39FF14", width=3, tags="scale")
            
    def _image_to_canvas(self, ix: float, iy: float) -> tuple:
        if self._video_frame is None:
            return 0, 0
        self.update_idletasks()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        ih, iw = self._video_frame.shape[:2]
        new_w, new_h = int(iw * self._zoom), int(ih * self._zoom)
        x_off, y_off = (cw - new_w) // 2, (ch - new_h) // 2
        return x_off + ix * self._zoom, y_off + iy * self._zoom
        
    def _update_calc_label(self):
        if len(self._scale_points) == 2:
            try:
                known_cm = float(self._dist_entry.get())
                if known_cm > 0:
                    ratio = calculate_pixel_cm_ratio(self._video_path, known_cm, self._scale_points)
                    self._calc_label.configure(text=f"Pixel/cm: {ratio:.3f}")
            except ValueError:
                pass
                
    def _on_mousewheel(self, event):
        if event.delta > 0 or event.num == 4:
            self._zoom = min(3.0, self._zoom * 1.1)
        else:
            self._zoom = max(0.2, self._zoom / 1.1)
        self._fit_to_canvas()
        
    def _on_configure(self, event):
        self._fit_to_canvas()
        
    def _clear_points(self):
        self._scale_points.clear()
        self._redraw_scale()
        self._calc_label.configure(text="Pixel/cm: --")
        
    def _reset_view(self):
        self._zoom = 1.0
        self._fit_to_canvas()
        
    def _apply(self):
        if len(self._scale_points) != 2:
            messagebox.showwarning("Incomplete", "Please click exactly two points on the image.")
            return
        try:
            known_cm = float(self._dist_entry.get())
            if known_cm <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid known distance in cm.")
            return
            
        ratio = calculate_pixel_cm_ratio(self._video_path, known_cm, self._scale_points)
        self._callback(ratio)
        self.destroy()


class EndpointsExtractionDialog(ctk.CTkToplevel):
    """Main dialog for configuring and running behavioral endpoint extraction."""
    
    DIVISION_RULES = [
        "Horizontal Split (Y < Center is Top)",
        "Horizontal Split (Y > Center is Top)",
        "Vertical Split (X < Center is Top)",
        "Vertical Split (X > Center is Top)",
        "Diagonal Split (Top-Left to Bottom-Right)",
        "Diagonal Split (Bottom-Left to Top-Right)",
    ]
    
    VIEW_TYPES = ["Side / Lateral View", "Top View"]
    
    def __init__(
        self,
        parent,
        video_entries: List[Any],
        tracking_results: Dict[str, str],  # video_path -> tracking_csv
        session_path: str,
        selected_metrics: Dict[str, bool],  # metric_name -> enabled
        endpoints_output_dir: str = "",
        on_complete: Callable[[Dict[str, Any]], None] = None,
    ):
        super().__init__(parent)
        self.title("Behavioral Endpoints Extraction")
        self.geometry("1100x800")
        self.minsize(1000, 700)
        
        self._video_entries = video_entries
        self._tracking_results = tracking_results
        self._session_path = session_path
        self._selected_metrics = selected_metrics  # Dict[metric_name, bool]
        self._endpoints_output_dir = endpoints_output_dir
        self._on_complete = on_complete
        
        # State
        self._arena_configs: Dict[int, Dict[int, ArenaConfig]] = {}
        self._current_video_idx = 0
        self._current_arena_idx = 0
        self._video_frame = None
        self._photo = None
        self._zoom = 1.0
        self._drag_start = None
        self._calibration_ratio = None
        self._worker_thread = None
        self._queue = queue.Queue()
        
        # Global config
        self._global_config = {
            'conversion': 18.74,
            'fps': 24.99,
            'freeze_thresh': 0.1,
            'rapid_thresh': 3.0,
            'body_length': 3.0,
            'freeze_thresh_bl': 0.05,
            'rapid_thresh_bl': 1.0,
            'export_kinetics': True,
            'global_view_type': 'Side / Lateral View',
            'global_division_rule': 'Horizontal Split (Y < Center is Top)',
            'global_cx_pct': 50.0,
            'global_cy_pct': 50.0,
        }
        
        self._setup_ui()
        self._load_first_video()
        self.after(100, self._poll_queue)
        
    def _setup_ui(self):
        # Main layout: left panel (controls), right panel (canvas)
        main_frame = ctk.CTkFrame(self, corner_radius=0)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # LEFT PANEL - Controls
        left = ctk.CTkScrollableFrame(main_frame, corner_radius=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left.grid_columnconfigure(0, weight=1)
        
        # 1. Global Parameters
        g1 = ctk.CTkFrame(left, corner_radius=6)
        g1.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(g1, text="1. Global Parameters", font=ctk.CTkFont(weight="bold", size=14)).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Pixel/cm with calibration button
        conv_frame = ctk.CTkFrame(g1, corner_radius=0)
        conv_frame.pack(fill="x", padx=10, pady=2)
        conv_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(conv_frame, text="Pixel/cm Ratio:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self._conv_var = ctk.DoubleVar(value=self._global_config['conversion'])
        ctk.CTkEntry(conv_frame, textvariable=self._conv_var, width=80).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ctk.CTkButton(conv_frame, text="📏 Calibrate", width=100, command=self._open_calibration).grid(row=0, column=2, padx=5, pady=5)
        
        # FPS
        fps_frame = ctk.CTkFrame(g1, corner_radius=0)
        fps_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(fps_frame, text="FPS:").pack(side="left", padx=5)
        self._fps_var = ctk.DoubleVar(value=self._global_config['fps'])
        ctk.CTkEntry(fps_frame, textvariable=self._fps_var, width=80).pack(side="left", padx=5)
        
        # Thresholds
        ctk.CTkLabel(g1, text="Speed Thresholds (cm/s):", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 2))
        
        frz_frame = ctk.CTkFrame(g1, corner_radius=0)
        frz_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(frz_frame, text="Freezing (< cm/s):").pack(side="left", padx=5)
        self._frz_var = ctk.DoubleVar(value=self._global_config['freeze_thresh'])
        ctk.CTkEntry(frz_frame, textvariable=self._frz_var, width=80).pack(side="left", padx=5)
        
        rpd_frame = ctk.CTkFrame(g1, corner_radius=0)
        rpd_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(rpd_frame, text="Rapid (> cm/s):").pack(side="left", padx=5)
        self._rpd_var = ctk.DoubleVar(value=self._global_config['rapid_thresh'])
        ctk.CTkEntry(rpd_frame, textvariable=self._rpd_var, width=80).pack(side="left", padx=5)
        
        # Body-length scaling (collapsible)
        self._bl_frame = ctk.CTkFrame(g1, corner_radius=6)
        self._bl_frame.pack(fill="x", padx=10, pady=(10, 10))
        ctk.CTkLabel(self._bl_frame, text="Body-Length Scaling (optional)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5, 2))
        
        self._bl_enabled = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self._bl_frame, text="Enable Body-Length Scaling", variable=self._bl_enabled, 
                        command=self._toggle_body_length).pack(anchor="w", padx=10, pady=2)
        
        self._bl_container = ctk.CTkFrame(self._bl_frame, corner_radius=0)
        self._bl_container.pack(fill="x", padx=10, pady=(0, 10))
        
        bl_len_frame = ctk.CTkFrame(self._bl_container, corner_radius=0)
        bl_len_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(bl_len_frame, text="Body Length (cm):").pack(side="left", padx=5)
        self._bl_len_var = ctk.DoubleVar(value=self._global_config['body_length'])
        ctk.CTkEntry(bl_len_frame, textvariable=self._bl_len_var, width=80).pack(side="left", padx=5)
        
        bl_frz_frame = ctk.CTkFrame(self._bl_container, corner_radius=0)
        bl_frz_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(bl_frz_frame, text="Freezing (< BL/s):").pack(side="left", padx=5)
        self._bl_frz_var = ctk.DoubleVar(value=self._global_config['freeze_thresh_bl'])
        ctk.CTkEntry(bl_frz_frame, textvariable=self._bl_frz_var, width=80).pack(side="left", padx=5)
        
        bl_rpd_frame = ctk.CTkFrame(self._bl_container, corner_radius=0)
        bl_rpd_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(bl_rpd_frame, text="Rapid (> BL/s):").pack(side="left", padx=5)
        self._bl_rpd_var = ctk.DoubleVar(value=self._global_config['rapid_thresh_bl'])
        ctk.CTkEntry(bl_rpd_frame, textvariable=self._bl_rpd_var, width=80).pack(side="left", padx=5)
        
        self._bl_container.pack_forget()  # Initially hidden
        
        # Export kinetics
        self._export_kinetics = ctk.BooleanVar(value=self._global_config['export_kinetics'])
        ctk.CTkCheckBox(g1, text="Export Frame-Wise Kinetics (Slower)", variable=self._export_kinetics).pack(anchor="w", padx=10, pady=(5, 10))
        
        # Endpoints Output Directory
        ep_out_frame = ctk.CTkFrame(g1, corner_radius=0)
        ep_out_frame.pack(fill="x", padx=10, pady=(0, 10))
        ep_out_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(ep_out_frame, text="Endpoints Output:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self._ep_output_var = ctk.StringVar(value=self._endpoints_output_dir)
        ctk.CTkEntry(ep_out_frame, textvariable=self._ep_output_var, placeholder_text="Same as session (optional)").grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ctk.CTkButton(ep_out_frame, text="Browse", width=80, command=self._browse_ep_output).grid(row=0, column=2, padx=5, pady=5)

        # 2. Endpoints Metrics Selection
        g_metrics = ctk.CTkFrame(left, corner_radius=6)
        g_metrics.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(g_metrics, text="2. Endpoints Metrics to Extract", font=ctk.CTkFont(weight="bold", size=14)).pack(anchor="w", padx=10, pady=(10, 5))
        
        self._metric_vars = {}
        # Use the selected metrics from parent GUI
        all_metrics = [
            "Average Speed (cm/s)",
            "Freezing Time Ratio (%)",
            "Swimming Time Ratio (%)",
            "Rapid Movement Time Ratio (%)",
            "Time in Top Percentage (%)",
            "Time in Bottom Percentage (%)",
            "Average Thigmotaxis (cm)",
            "Fractal Dimension",
            "Entropy",
"Average Speed (body-length/s)",
            "Freezing Time Ratio (body-length/s) (%)",
            "Swimming Time Ratio (body-length/s) (%)",
            "Rapid Movement Time Ratio (body-length/s) (%)",
        ]
        
        # Metrics that are only for lateral view
        self._lateral_only_metrics = {"Time in Top Percentage (%)", "Time in Bottom Percentage (%)"}
        
        # Create scrollable frame for metrics (both vertical and horizontal)
        self._metrics_scroll = ctk.CTkScrollableFrame(g_metrics, corner_radius=0, height=200)
        self._metrics_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._metrics_scroll.grid_columnconfigure(0, weight=1)
        self._metrics_scroll.grid_columnconfigure(1, weight=1)
        self._metrics_scroll.grid_columnconfigure(2, weight=1)
        
        self._metric_vars = {}
        self._metric_checkboxes = {}  # Store checkbox widgets for show/hide
        
        for i, name in enumerate(all_metrics):
            row, col = divmod(i, 3)
            enabled = self._selected_metrics.get(name, True)
            var = ctk.BooleanVar(value=enabled)
            self._metric_vars[name] = var
            cb = ctk.CTkCheckBox(self._metrics_scroll, text=name, variable=var)
            cb.grid(row=row, column=col, sticky="w", padx=10, pady=2)
            self._metric_checkboxes[name] = cb
        
        # Select All / Deselect All buttons
        btn_frame = ctk.CTkFrame(g_metrics, corner_radius=0)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkButton(btn_frame, text="Select All", width=100, 
                      command=self._select_all_metrics).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(btn_frame, text="Deselect All", width=100,
                      command=self._deselect_all_metrics).pack(side="left", padx=5, pady=5)
        
        # Kinematics export option
        self._export_kinetics = ctk.BooleanVar(value=self._global_config.get('export_kinetics', True))
        ctk.CTkCheckBox(g_metrics, text="Export Frame-by-Frame Kinematics (detailed CSV per point)", 
                        variable=self._export_kinetics, font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=10, pady=(0, 10)
        )
        
        # 3. View & Division Settings
        g2 = ctk.CTkFrame(left, corner_radius=6)
        g2.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(g2, text="2. View & Division Settings", font=ctk.CTkFont(weight="bold", size=14)).pack(anchor="w", padx=10, pady=(10, 5))
        
        # View type
        view_frame = ctk.CTkFrame(g2, corner_radius=0)
        view_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(view_frame, text="View Type:").pack(side="left", padx=5)
        self._view_var = ctk.StringVar(value=self._global_config['global_view_type'])
        view_combo = ctk.CTkOptionMenu(view_frame, variable=self._view_var, values=self.VIEW_TYPES, command=self._on_view_changed)
        view_combo.pack(side="left", padx=5, fill="x", expand=True)
        
        # Initially update visibility based on view type (now that _view_var exists)
        self._update_metric_visibility()
        
        # Division rule
        self._div_frame = ctk.CTkFrame(g2, corner_radius=0)
        self._div_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(self._div_frame, text="Division Rule:").pack(side="left", padx=5)
        self._div_var = ctk.StringVar(value=self._global_config['global_division_rule'])
        self._div_combo = ctk.CTkOptionMenu(self._div_frame, variable=self._div_var, values=self.DIVISION_RULES)
        self._div_combo.pack(side="left", padx=5, fill="x", expand=True)
        
        # Center sliders
        cx_frame = ctk.CTkFrame(g2, corner_radius=0)
        cx_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(cx_frame, text="Center X (%):").pack(side="left", padx=5)
        self._cx_var = ctk.DoubleVar(value=self._global_config['global_cx_pct'])
        cx_slider = ctk.CTkSlider(cx_frame, from_=0, to=100, variable=self._cx_var, command=self._on_center_changed)
        cx_slider.pack(side="left", padx=5, fill="x", expand=True)
        self._cx_label = ctk.CTkLabel(cx_frame, text="50%", width=40)
        self._cx_label.pack(side="left", padx=5)
        
        cy_frame = ctk.CTkFrame(g2, corner_radius=0)
        cy_frame.pack(fill="x", padx=10, pady=(2, 10))
        ctk.CTkLabel(cy_frame, text="Center Y (%):").pack(side="left", padx=5)
        self._cy_var = ctk.DoubleVar(value=self._global_config['global_cy_pct'])
        cy_slider = ctk.CTkSlider(cy_frame, from_=0, to=100, variable=self._cy_var, command=self._on_center_changed)
        cy_slider.pack(side="left", padx=5, fill="x", expand=True)
        self._cy_label = ctk.CTkLabel(cy_frame, text="50%", width=40)
        self._cy_label.pack(side="left", padx=5)
        
        # Auto-sync checkbox
        self._auto_sync = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(g2, text="🔗 Auto-sync across all arenas", variable=self._auto_sync, 
                        font=ctk.CTkFont(weight="bold"), text_color="#39FF14").pack(anchor="w", padx=10, pady=(5, 10))
        
        # 3. Video & Arena Selection
        g3 = ctk.CTkFrame(left, corner_radius=6)
        g3.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(g3, text="3. Video & Arena Selection", font=ctk.CTkFont(weight="bold", size=14)).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Video selector
        vid_frame = ctk.CTkFrame(g3, corner_radius=0)
        vid_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(vid_frame, text="Video:").pack(side="left", padx=5)
        self._video_var = ctk.StringVar()
        self._video_combo = ctk.CTkOptionMenu(vid_frame, variable=self._video_var, values=[], command=self._on_video_changed)
        self._video_combo.pack(side="left", padx=5, fill="x", expand=True)
        
        # Arena selector
        arena_frame = ctk.CTkFrame(g3, corner_radius=0)
        arena_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(arena_frame, text="Arena:").pack(side="left", padx=5)
        self._arena_var = ctk.StringVar()
        self._arena_combo = ctk.CTkOptionMenu(arena_frame, variable=self._arena_var, values=[], command=self._on_arena_changed)
        self._arena_combo.pack(side="left", padx=5, fill="x", expand=True)
        
        # Apply to all button
        ctk.CTkButton(g3, text="⚡ Apply Current Settings to ALL Arenas", 
                      command=self._apply_to_all, fg_color="#5b2c91", hover_color="#4a2373").pack(fill="x", padx=10, pady=(5, 10))
        
        # 4. Run Button
        self._run_btn = ctk.CTkButton(
            left, text="🚀 RUN ENDPOINTS EXTRACTION", height=50,
            font=ctk.CTkFont(size=16, weight="bold"), fg_color="#28a745", hover_color="#218838",
            command=self._run_extraction
        )
        self._run_btn.pack(fill="x", pady=10)
        
        # Progress
        self._progress = ctk.CTkProgressBar(left)
        self._progress.pack(fill="x", pady=(0, 5))
        self._progress.set(0)
        self._status_label = ctk.CTkLabel(left, text="Ready", text_color="gray")
        self._status_label.pack(fill="x", pady=(0, 10))
        
        # Console
        self._console = ctk.CTkTextbox(left, height=120, font=ctk.CTkFont(family="Consolas", size=11))
        self._console.pack(fill="x", pady=(0, 10))
        self._console.configure(state="disabled")
        
        # RIGHT PANEL - Canvas
        right = ctk.CTkFrame(main_frame, corner_radius=6)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(right, text="ROI View & Division Preview", font=ctk.CTkFont(weight="bold", size=14)).grid(
            row=0, column=0, sticky="w", padx=10, pady=10)
        
        canvas_frame = ctk.CTkFrame(right, corner_radius=6)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)
        
        self._canvas = tk.Canvas(canvas_frame, bg="#1a1a1a", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Button-4>", self._on_mousewheel)
        self._canvas.bind("<Button-5>", self._on_mousewheel)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Load settings if available
        self._load_settings()
        
    def _toggle_body_length(self):
        if self._bl_enabled.get():
            self._bl_container.pack(fill="x", padx=10, pady=(0, 10))
        else:
            self._bl_container.pack_forget()
            
    def _on_view_changed(self, value=None):
        is_lateral = (self._view_var.get() == "Side / Lateral View")
        if is_lateral:
            self._div_frame.pack(fill="x", padx=10, pady=2)
        else:
            self._div_frame.pack_forget()
        self._sync_global_to_arena()
        self._update_canvas()
        self._update_metric_visibility()
        
    def _update_metric_visibility(self):
        """Show/hide lateral-only metrics based on view type."""
        is_lateral = (self._view_var.get() == "Side / Lateral View")
        for metric_name in self._lateral_only_metrics:
            cb = self._metric_checkboxes.get(metric_name)
            if cb:
                if is_lateral:
                    cb.grid()  # Show
                else:
                    cb.grid_remove()  # Hide
        
    def _select_all_metrics(self):
        """Select all visible metrics."""
        is_lateral = (self._view_var.get() == "Side / Lateral View")
        for name, var in self._metric_vars.items():
            if name in self._lateral_only_metrics and not is_lateral:
                continue  # Skip hidden metrics
            var.set(True)
        
    def _deselect_all_metrics(self):
        """Deselect all visible metrics."""
        is_lateral = (self._view_var.get() == "Side / Lateral View")
        for name, var in self._metric_vars.items():
            if name in self._lateral_only_metrics and not is_lateral:
                continue  # Skip hidden metrics
            var.set(False)
        
    def _on_center_changed(self, value=None):
        self._cx_label.configure(text=f"{self._cx_var.get():.0f}%")
        self._cy_label.configure(text=f"{self._cy_var.get():.0f}%")
        self._sync_global_to_arena()
        self._update_canvas()
        
    def _sync_global_to_arena(self):
        """Sync global settings to current arena, and optionally to all arenas."""
        vid_idx = self._current_video_idx
        arena_idx = self._current_arena_idx
        
        if vid_idx not in self._arena_configs:
            self._arena_configs[vid_idx] = {}
        if arena_idx not in self._arena_configs[vid_idx]:
            self._arena_configs[vid_idx][arena_idx] = ArenaConfig()
            
        cfg = self._arena_configs[vid_idx][arena_idx]
        cfg.cx_pct = self._cx_var.get()
        cfg.cy_pct = self._cy_var.get()
        cfg.view_type = self._view_var.get()
        cfg.division_rule = self._div_var.get()
        
        if self._auto_sync.get():
            # Apply to all arenas in current video
            for a_idx in self._arena_configs.get(vid_idx, {}):
                self._arena_configs[vid_idx][a_idx].cx_pct = cfg.cx_pct
                self._arena_configs[vid_idx][a_idx].cy_pct = cfg.cy_pct
                self._arena_configs[vid_idx][a_idx].view_type = cfg.view_type
                self._arena_configs[vid_idx][a_idx].division_rule = cfg.division_rule
            
            # Also apply to other videos' arenas
            for v_idx in self._arena_configs:
                for a_idx in self._arena_configs[v_idx]:
                    self._arena_configs[v_idx][a_idx].view_type = cfg.view_type
                    self._arena_configs[v_idx][a_idx].division_rule = cfg.division_rule
        
        self._update_arena_combo()
        
    def _load_first_video(self):
        if self._video_entries:
            self._video_combo.configure(values=[os.path.basename(v.path) for v in self._video_entries])
            self._video_var.set(os.path.basename(self._video_entries[0].path))
            self._on_video_changed()
            
    def _on_video_changed(self, value=None):
        vid_name = self._video_var.get()
        self._current_video_idx = next(i for i, v in enumerate(self._video_entries) if os.path.basename(v.path) == vid_name)
        entry = self._video_entries[self._current_video_idx]
        
        # Load first frame
        cap = cv2.VideoCapture(entry.path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self._video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()
            
        # Update arena combo
        self._update_arena_combo()
        if entry.rois:
            self._arena_var.set(f"Arena 1")
            self._on_arena_changed()
        else:
            self._arena_combo.configure(values=[])
            self._arena_var.set("")
            
        self._update_canvas()
        
    def _update_arena_combo(self):
        entry = self._video_entries[self._current_video_idx]
        arena_names = [f"Arena {i+1}" for i in range(len(entry.rois))]
        self._arena_combo.configure(values=arena_names)
        if arena_names and self._arena_var.get() not in arena_names:
            self._arena_var.set(arena_names[0])
            self._on_arena_changed()
            
    def _on_arena_changed(self, value=None):
        try:
            self._current_arena_idx = int(self._arena_var.get().split()[-1]) - 1
        except (ValueError, IndexError):
            self._current_arena_idx = 0
            
        # Load arena config into UI
        cfg = self._arena_configs.get(self._current_video_idx, {}).get(self._current_arena_idx)
        if cfg:
            self._cx_var.set(cfg.cx_pct)
            self._cy_var.set(cfg.cy_pct)
            self._view_var.set(cfg.view_type)
            self._div_var.set(cfg.division_rule)
        else:
            # Initialize with global defaults
            self._cx_var.set(self._global_config['global_cx_pct'])
            self._cy_var.set(self._global_config['global_cy_pct'])
            self._view_var.set(self._global_config['global_view_type'])
            self._div_var.set(self._global_config['global_division_rule'])
            
        self._cx_label.configure(text=f"{self._cx_var.get():.0f}%")
        self._cy_label.configure(text=f"{self._cy_var.get():.0f}%")
        self._on_view_changed()
        self._update_canvas()
        
    def _apply_to_all(self):
        """Force apply current settings to all arenas in all videos."""
        cfg = ArenaConfig(
            cx_pct=self._cx_var.get(),
            cy_pct=self._cy_var.get(),
            view_type=self._view_var.get(),
            division_rule=self._div_var.get(),
        )
        for vid_idx, entry in enumerate(self._video_entries):
            if vid_idx not in self._arena_configs:
                self._arena_configs[vid_idx] = {}
            for arena_idx in range(len(entry.rois)):
                self._arena_configs[vid_idx][arena_idx] = ArenaConfig(
                    cx_pct=cfg.cx_pct, cy_pct=cfg.cy_pct,
                    view_type=cfg.view_type, division_rule=cfg.division_rule
                )
        self._log(f"✅ Applied settings to ALL arenas across ALL videos")
        self._update_canvas()
        
    def _open_calibration(self):
        entry = self._video_entries[self._current_video_idx]
        PixelCmCalibrationDialog(self, entry.path, self._on_calibration_done)
        
    def _on_calibration_done(self, ratio: float):
        self._conv_var.set(round(ratio, 3))
        self._calibration_ratio = ratio
        self._log(f"📏 Calibration complete: {ratio:.3f} pixel/cm")
        
    def _browse_ep_output(self):
        path = filedialog.askdirectory(title="Select Endpoints Output Directory")
        if path:
            self._ep_output_var.set(path)
        
    def _update_canvas(self):
        if self._video_frame is None:
            return
            
        entry = self._video_entries[self._current_video_idx]
        canvas_img = self._video_frame.copy()
        h, w = canvas_img.shape[:2]
        
        # Draw all ROIs for this video
        for i, roi in enumerate(entry.rois):
            color = (0, 255, 255) if i == self._current_arena_idx else (0, 200, 0)
            thickness = 3 if i == self._current_arena_idx else 2
            
            pts = roi.points
            if len(pts) >= 3:
                cv2_pts = np.array([[int(p[0]), int(p[1])] for p in pts], dtype=np.int32)
                cv2.polylines(canvas_img, [cv2_pts], True, color, thickness, cv2.LINE_AA)
            elif len(pts) == 2:
                cv2.rectangle(canvas_img, 
                    (int(pts[0][0]), int(pts[0][1])), 
                    (int(pts[1][0]), int(pts[1][1])), 
                    color, thickness, cv2.LINE_AA)
                    
            # Draw center point and division line for current arena
            if i == self._current_arena_idx:
                cfg = self._arena_configs.get(self._current_video_idx, {}).get(i)
                if cfg:
                    # Calculate center in pixel coordinates
                    roi_x = min(p[0] for p in pts)
                    roi_y = min(p[1] for p in pts)
                    roi_w = max(p[0] for p in pts) - roi_x
                    roi_h = max(p[1] for p in pts) - roi_y
                    
                    # Convert to int for OpenCV
                    roi_x_i = int(roi_x)
                    roi_y_i = int(roi_y)
                    roi_w_i = int(roi_w)
                    roi_h_i = int(roi_h)
                    
                    cx = int(roi_x + roi_w * (cfg.cx_pct / 100.0))
                    cy = int(roi_y + roi_h * (cfg.cy_pct / 100.0))
                    
                    # Draw center
                    cv2.circle(canvas_img, (cx, cy), 10, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas_img, (cx, cy), 12, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Draw division line for lateral view
                    if cfg.view_type == "Side / Lateral View":
                        line_color = (0, 165, 255)
                        if "Vertical Split (X <" in cfg.division_rule:
                            cv2.line(canvas_img, (cx, roi_y_i), (cx, roi_y_i + roi_h_i), line_color, 2, cv2.LINE_AA)
                        elif "Vertical Split (X >" in cfg.division_rule:
                            cv2.line(canvas_img, (cx, roi_y_i), (cx, roi_y_i + roi_h_i), line_color, 2, cv2.LINE_AA)
                        elif "Horizontal Split (Y <" in cfg.division_rule:
                            cv2.line(canvas_img, (roi_x_i, cy), (roi_x_i + roi_w_i, cy), line_color, 2, cv2.LINE_AA)
                        elif "Horizontal Split (Y >" in cfg.division_rule:
                            cv2.line(canvas_img, (roi_x_i, cy), (roi_x_i + roi_w_i, cy), line_color, 2, cv2.LINE_AA)
                        elif "Top-Left to Bottom-Right" in cfg.division_rule:
                            k = roi_h / roi_w if roi_w > 0 else 1
                            y1 = int(cy + k * (roi_x - cx))
                            y2 = int(cy + k * (roi_x + roi_w - cx))
                            cv2.line(canvas_img, (roi_x_i, y1), (roi_x_i + roi_w_i, y2), line_color, 2, cv2.LINE_AA)
                        elif "Bottom-Left to Top-Right" in cfg.division_rule:
                            k = roi_h / roi_w if roi_w > 0 else 1
                            y1 = int(cy - k * (roi_x - cx))
                            y2 = int(cy - k * (roi_x + roi_w - cx))
                            cv2.line(canvas_img, (roi_x_i, y1), (roi_x_i + roi_w_i, y2), line_color, 2, cv2.LINE_AA)
                    
                    # Label
                    cv2.putText(canvas_img, f"Arena {i+1}", 
                        (roi_x_i + 10, roi_y_i + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
        # Convert to PhotoImage
        self.update_idletasks()
        cw = max(self._canvas.winfo_width(), 400)
        ch = max(self._canvas.winfo_height(), 300)
        scale = min(cw / w, ch / h) * 0.95
        new_w, new_h = int(w * scale), int(h * scale)
        img = Image.fromarray(canvas_img).resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        x, y = (cw - new_w) // 2, (ch - new_h) // 2
        self._canvas.create_image(x, y, anchor="nw", image=self._photo)
        
    def _canvas_to_image(self, cx: int, cy: int) -> tuple:
        if self._video_frame is None:
            return 0, 0
        self.update_idletasks()
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        h, w = self._video_frame.shape[:2]
        scale = min(cw / w, ch / h) * 0.95
        new_w, new_h = int(w * scale), int(h * scale)
        x_off, y_off = (cw - new_w) // 2, (ch - new_h) // 2
        return (cx - x_off) / scale, (cy - y_off) / scale
        
    def _on_canvas_click(self, event):
        if self._current_arena_idx < 0:
            return
        entry = self._video_entries[self._current_video_idx]
        roi = entry.rois[self._current_arena_idx]
        if not roi.points:
            return
            
        self._drag_start = self._canvas_to_image(event.x, event.y)
        
    def _on_canvas_drag(self, event):
        if self._drag_start is None:
            return
        # Real-time preview of center movement
        pass
        
    def _on_canvas_release(self, event):
        if self._drag_start is None:
            return
        img_pt = self._canvas_to_image(event.x, event.y)
        entry = self._video_entries[self._current_video_idx]
        roi = entry.rois[self._current_arena_idx]
        pts = roi.points
        
        roi_x = min(p[0] for p in pts)
        roi_y = min(p[1] for p in pts)
        roi_w = max(p[0] for p in pts) - roi_x
        roi_h = max(p[1] for p in pts) - roi_y
        
        if roi_w > 0 and roi_h > 0:
            cx_pct = ((img_pt[0] - roi_x) / roi_w) * 100.0
            cy_pct = ((img_pt[1] - roi_y) / roi_h) * 100.0
            cx_pct = max(0, min(100, cx_pct))
            cy_pct = max(0, min(100, cy_pct))
            
            self._cx_var.set(cx_pct)
            self._cy_var.set(cy_pct)
            self._cx_label.configure(text=f"{cx_pct:.0f}%")
            self._cy_label.configure(text=f"{cy_pct:.0f}%")
            self._on_center_changed()
            
        self._drag_start = None
        
    def _on_mousewheel(self, event):
        if event.delta > 0 or event.num == 4:
            self._zoom = min(3.0, self._zoom * 1.1)
        else:
            self._zoom = max(0.2, self._zoom / 1.1)
        self._update_canvas()
        
    def _on_canvas_configure(self, event):
        self._update_canvas()
        
    def _log(self, msg: str):
        self._console.configure(state="normal")
        self._console.insert("end", msg + "\n")
        self._console.see("end")
        self._console.configure(state="disabled")
        
    def _run_extraction(self):
        # Validate
        if not self._tracking_results:
            messagebox.showerror("Error", "No tracking results available. Run tracking first.")
            return
            
        # Collect selected metrics
        selected_metrics = [name for name, var in self._metric_vars.items() if var.get()]
        if not selected_metrics:
            messagebox.showwarning("No Metrics", "Please select at least one endpoint metric.")
            return
        
        # Collect global config
        self._global_config.update({
            'conversion': self._conv_var.get(),
            'fps': self._fps_var.get(),
            'freeze_thresh': self._frz_var.get(),
            'rapid_thresh': self._rpd_var.get(),
            'body_length': self._bl_len_var.get() if self._bl_enabled.get() else 3.0,
            'freeze_thresh_bl': self._bl_frz_var.get() if self._bl_enabled.get() else 0.05,
            'rapid_thresh_bl': self._bl_rpd_var.get() if self._bl_enabled.get() else 1.0,
            'export_kinetics': self._export_kinetics.get(),
            'global_view_type': self._view_var.get(),
            'global_division_rule': self._div_var.get(),
            'global_cx_pct': self._cx_var.get(),
            'global_cy_pct': self._cy_var.get(),
            'selected_metrics': selected_metrics,
        })
        
        # Determine output directory
        ep_output_dir = self._ep_output_var.get().strip()
        if not ep_output_dir:
            ep_output_dir = os.path.join(self._session_path, "endpoints_results")
        
        # Prepare arena settings for engine
        arena_settings = {}
        for vid_idx, entry in enumerate(self._video_entries):
            for arena_idx in range(len(entry.rois)):
                cfg = self._arena_configs.get(vid_idx, {}).get(arena_idx, ArenaConfig())
                # Use global index for engine
                global_idx = sum(len(self._video_entries[i].rois) for i in range(vid_idx)) + arena_idx
                arena_settings[global_idx] = {
                    'cx_pct': cfg.cx_pct,
                    'cy_pct': cfg.cy_pct,
                    'view_type': cfg.view_type,
                    'division_rule': cfg.division_rule,
                }
        
        # Prepare ROI list for engine (global)
        roi_list = []
        for entry in self._video_entries:
            for roi in entry.rois:
                pts = roi.points
                if len(pts) >= 2:
                    roi_x = min(p[0] for p in pts)
                    roi_y = min(p[1] for p in pts)
                    roi_w = max(p[0] for p in pts) - roi_x
                    roi_h = max(p[1] for p in pts) - roi_y
                    roi_list.append({
                        'x': roi_x, 'y': roi_y, 'w': roi_w, 'h': roi_h,
                        'type': roi.shape
                    })
        
        self._run_btn.configure(state="disabled", text="Running...")
        self._progress.set(0)
        self._status_label.configure(text="Starting endpoints extraction...")
        self._log("🚀 Starting behavioral endpoints extraction...")
        
        # Run in background thread
        self._worker_thread = threading.Thread(
            target=self._worker_run,
            args=(roi_list, arena_settings, ep_output_dir),
            daemon=True
        )
        self._worker_thread.start()
    def _worker_run(self, roi_list: List[Dict], arena_settings: Dict[int, Dict], output_dir: str):
        try:
            all_summaries = []
            all_kinetics = []
            selected_metrics = self._global_config.get('selected_metrics', [])
            
            total_videos = len(self._tracking_results)
            for vid_idx, (video_path, tracking_csv) in enumerate(self._tracking_results.items()):
                self._queue.put(("progress", 0.1 + 0.8 * vid_idx / total_videos, 
                                 f"Processing {os.path.basename(video_path)}..."))
                
                if not os.path.exists(tracking_csv):
                    self._queue.put(("log", f"⚠️ Tracking CSV not found: {tracking_csv}"))
                    continue
                    
                summaries, kinetics = EndpointsEngine.calculate_endpoints(
                    tracking_csv, self._global_config, roi_list, arena_settings, self._queue.put
                )
                
                # Filter summaries to only include selected metrics
                if summaries:
                    for s in summaries:
                        s['Video'] = os.path.basename(video_path)
                        # Keep only selected metrics + Arena_ID + Video
                        filtered = {'Arena_ID': s['Arena_ID'], 'Video': s['Video']}
                        for metric in selected_metrics:
                            if metric in s:
                                filtered[metric] = s[metric]
                        all_summaries.append(filtered)
                
                if not kinetics.empty:
                    kinetics['Video'] = os.path.basename(video_path)
                    all_kinetics.append(kinetics)
                    
                self._queue.put(("log", f"✅ Completed {os.path.basename(video_path)}: {len(summaries)} arenas"))
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            
            if all_summaries:
                summary_df = pd.DataFrame(all_summaries)
                summary_path = os.path.join(output_dir, "endpoints_summary.csv")
                summary_df.to_csv(summary_path, index=False)
                self._queue.put(("log", f"📊 Summary saved: {summary_path}"))
                
                # Also save Excel with multiple sheets
                excel_path = os.path.join(output_dir, "endpoints_summary.xlsx")
                with pd.ExcelWriter(excel_path) as writer:
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
                    # Per-video sheets
                    for vid_name in summary_df['Video'].unique():
                        vid_df = summary_df[summary_df['Video'] == vid_name]
                        sheet_name = vid_name[:31]  # Excel limit
                        vid_df.to_excel(writer, sheet_name=sheet_name, index=False)
                self._queue.put(("log", f"📊 Excel saved: {excel_path}"))
            
            if all_kinetics:
                kinetics_df = pd.concat(all_kinetics, ignore_index=True)
                kinetics_path = os.path.join(output_dir, "endpoints_kinetics.csv")
                kinetics_df.to_csv(kinetics_path, index=False)
                self._queue.put(("log", f"📈 Kinetics saved: {kinetics_path}"))
            
            self._queue.put(("progress", 1.0, "Endpoints extraction complete!"))
            self._queue.put(("done", {
                'summary_path': os.path.join(output_dir, "endpoints_summary.csv"),
                'excel_path': os.path.join(output_dir, "endpoints_summary.xlsx"),
                'kinetics_path': os.path.join(output_dir, "endpoints_kinetics.csv") if all_kinetics else None,
                'output_dir': output_dir,
            }))
        
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
                    self._log(msg[1])
                elif msg[0] == "done":
                    self._progress.set(1)
                    self._status_label.configure(text="Complete!", text_color="green")
                    self._run_btn.configure(state="normal", text="🚀 RUN ENDPOINTS EXTRACTION")
                    self._log("✅ Endpoints extraction complete!")
                    self._save_settings()
                    if self._on_complete:
                        self._on_complete(msg[1])
                    messagebox.showinfo("Complete", f"Endpoints extraction complete!\nResults saved to:\n{msg[1]['output_dir']}")
                elif msg[0] == "error":
                    self._status_label.configure(text="Error", text_color="red")
                    self._run_btn.configure(state="normal", text="🚀 RUN ENDPOINTS EXTRACTION")
                    self._log(f"❌ ERROR: {msg[1]}")
                    messagebox.showerror("Error", msg[1])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _save_settings(self):
        """Save settings to session folder and sync with parent behavior window."""
        # Save current metric selections
        metric_selections = {name: var.get() for name, var in self._metric_vars.items()}
        
        data = {
            'global_config': self._global_config,
            'metric_selections': metric_selections,
            'export_kinetics': self._export_kinetics.get() if hasattr(self, '_export_kinetics') else True,
            'arena_configs': {
                str(v): {str(a): asdict(c) for a, c in arenas.items()}
                for v, arenas in self._arena_configs.items()
            },
            'calibration_ratio': self._calibration_ratio,
        }
        
        # Try session folder first
        video_paths = [entry.path for entry in self._video_entries]
        session_dir = find_session_dir_from_videos(video_paths)
        
        # Also try parent's model session
        if not session_dir and self._session_path:
            session_dir = self._session_path
        
        saved = False
        if session_dir:
            if save_session_settings(session_dir, data):
                print(f"[ENDPOINTS] Saved settings to session folder: {session_dir}")
                saved = True
        
        # Fallback to legacy user config directory
        if not saved:
            try:
                settings_dir = os.path.expanduser(os.path.join("~", ".config", "fbq"))
                os.makedirs(settings_dir, exist_ok=True)
                settings_file = os.path.join(settings_dir, "endpoints_extraction_settings.json")
                with open(settings_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"[ENDPOINTS] Saved settings to legacy location: {settings_file}")
            except Exception as e:
                print(f"[ENDPOINTS] Failed to save settings: {e}")
        
        # Sync with parent behavior window if available
        if hasattr(self.master, '_endpoints_metric_selections'):
            self.master._endpoints_metric_selections = metric_selections
        if hasattr(self.master, '_endpoints_global_config'):
            self.master._endpoints_global_config = self._global_config
        if hasattr(self.master, '_endpoints_arena_configs'):
            self.master._endpoints_arena_configs = data['arena_configs']
        if hasattr(self.master, '_endpoints_calibration_ratio'):
            self.master._endpoints_calibration_ratio = self._calibration_ratio
        
        # Trigger parent to save as well
        if hasattr(self.master, '_save_settings'):
            self.master._save_settings()

    def _load_settings(self):
        """Load settings from session folder, sync with parent if available."""
        # Try to get settings from parent first (if already loaded there)
        parent_data = {}
        if hasattr(self.master, '_endpoints_metric_selections'):
            parent_data['metric_selections'] = getattr(self.master, '_endpoints_metric_selections', {})
        if hasattr(self.master, '_endpoints_global_config'):
            parent_data['global_config'] = getattr(self.master, '_endpoints_global_config', {})
        if hasattr(self.master, '_endpoints_arena_configs'):
            parent_data['arena_configs'] = getattr(self.master, '_endpoints_arena_configs', {})
        if hasattr(self.master, '_endpoints_calibration_ratio'):
            parent_data['calibration_ratio'] = getattr(self.master, '_endpoints_calibration_ratio', None)
        
        # Try session folder
        video_paths = [entry.path for entry in self._video_entries]
        session_dir = find_session_dir_from_videos(video_paths)
        
        loaded = False
        if session_dir:
            data = load_session_settings(session_dir)
            if data:
                parent_data = merge_settings(parent_data, data)
                loaded = True
        
        # Fallback to legacy user config directory
        if not loaded:
            try:
                settings_file = os.path.expanduser(os.path.join("~", ".config", "fbq", "endpoints_extraction_settings.json"))
                if os.path.exists(settings_file):
                    with open(settings_file, 'r') as f:
                        data = json.load(f)
                    parent_data = merge_settings(parent_data, data)
                    loaded = True
            except Exception as e:
                print(f"[ENDPOINTS] Failed to load legacy settings: {e}")
        
        # Apply loaded settings
        if parent_data:
            self._global_config.update(parent_data.get('global_config', {}))
            
            # Restore metric selections
            saved_metrics = parent_data.get('metric_selections', {})
            if hasattr(self, '_metric_vars'):
                for name, var in self._metric_vars.items():
                    var.set(saved_metrics.get(name, True))
            
            # Restore export_kinetics
            if hasattr(self, '_export_kinetics'):
                self._export_kinetics.set(parent_data.get('export_kinetics', True))
            
            # Restore arena configs
            for v_str, arenas in parent_data.get('arena_configs', {}).items():
                v = int(v_str)
                self._arena_configs[v] = {}
                for a_str, cfg_dict in arenas.items():
                    a = int(a_str)
                    self._arena_configs[v][a] = ArenaConfig(**cfg_dict)
                    
            self._calibration_ratio = parent_data.get('calibration_ratio')
            if self._calibration_ratio:
                self._conv_var.set(self._calibration_ratio)
            
            print(f"[ENDPOINTS] Loaded settings from {'session folder' if loaded else 'parent/legacy'}")


def show_endpoints_extraction_dialog(
    parent,
    video_entries: List[Any],
    tracking_results: Dict[str, str],
    session_path: str,
    selected_metrics: Dict[str, bool],
    endpoints_output_dir: str = "",
    on_complete: Callable[[Dict[str, Any]], None] = None,
) -> EndpointsExtractionDialog:
    """Show the endpoints extraction dialog."""
    dialog = EndpointsExtractionDialog(
        parent, video_entries, tracking_results, session_path, 
        selected_metrics, endpoints_output_dir, on_complete
    )
    return dialog