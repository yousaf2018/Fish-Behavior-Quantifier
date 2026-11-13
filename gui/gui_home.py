# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""
FBQ Home Window — Modern Full-Screen Design

Main entry point with two primary workflows:
1. Train AI Model
2. Inference & Behavior Quantification

Optional tools accessible via bottom bar.
"""

from __future__ import annotations

import math
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import uuid
import webbrowser
from pathlib import Path
from tkinter import messagebox
from typing import Callable
from PIL import Image, ImageTk

import customtkinter as ctk

if __package__:
    from .addon_gui import AddonController
    from .app_uninstall import UNINSTALL_EXIT_CODE, start_uninstall
    from .app_update import probe_update_status, start_update_check
    from .project_paths import PROJECT_ROOT, gui_asset, gui_script
    from .splash_ipc import (
        TOKEN_ENV_VAR,
        external_closed,
        external_dismissed,
        external_started_at,
        signal_stop,
    )
    from .splash_reveal import MAX_DISPLAY_MS, reveal_from_left, reveal_total_ms
    from .splash_geometry import SPLASH_LOGO_WIDTH_RATIO, configure_splash_window
    from .window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon, tk_font_spec
    from .ui_theme import apply_theme
else:
    from addon_gui import AddonController
    from app_uninstall import UNINSTALL_EXIT_CODE, start_uninstall
    from app_update import probe_update_status, start_update_check
    from project_paths import PROJECT_ROOT, gui_asset, gui_script
    from splash_ipc import (
        TOKEN_ENV_VAR,
        external_closed,
        external_dismissed,
        external_started_at,
        signal_stop,
    )
    from splash_reveal import MAX_DISPLAY_MS, reveal_from_left, reveal_total_ms
    from splash_geometry import SPLASH_LOGO_WIDTH_RATIO, configure_splash_window
    from window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon, tk_font_spec
    from ui_theme import apply_theme

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.project_meta import MANUAL_URL  # noqa: E402

apply_theme()

APP_TITLE = "FBQ"

# Modern color palette - Blue theme
BG_DARK = "#080f1a"
BG_CARD = "#0e1a2e"
BG_CARD_HOVER = "#142540"
BORDER_COLOR = "#1a2d4a"
BORDER_HOVER = "#1e3a5f"
TEXT_PRIMARY = "#eef4fb"
TEXT_SECONDARY = "#9ab0cc"
TEXT_MUTED = "#5a7a9a"
ACCENT_BLUE = "#1e90ff"
ACCENT_BLUE_HOVER = "#3aa0ff"
ACCENT_BLUE_DIM = "#0d6bd4"
ACCENT_BLUE_LIGHT = "#4db8ff"
ACCENT_CYAN = "#00d4ff"
ACCENT_CYAN_HOVER = "#33ddff"
ACCENT_ORANGE = "#ff8c00"
ACCENT_ORANGE_HOVER = "#ff9f33"
ACCENT_GREEN = "#00cc66"
ACCENT_RED = "#ff4444"
WHITE = "#ffffff"


def home_window_title() -> str:
    """Return the Home window title with the installed FBQ version."""
    try:
        version = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        version = ""
    if not version:
        return APP_TITLE
    version_label = version if version.lower().startswith("v") else f"v{version}"
    return f"{APP_TITLE} {version_label}"


ONLINE_MANUAL_BASE_URL = MANUAL_URL


def python_executable() -> str:
    if os.name == "nt":
        candidate = Path(sys.prefix) / "Scripts" / "python.exe"
    else:
        candidate = Path(sys.prefix) / "bin" / "python"
    return str(candidate if candidate.is_file() else Path(sys.executable))


def ensure_external_splash() -> tuple[bool, subprocess.Popen | None]:
    """Use a separate splash process so reveal stays smooth during GUI setup."""
    if os.environ.get(TOKEN_ENV_VAR):
        return True, None

    token = uuid.uuid4().hex
    environment = os.environ.copy()
    environment[TOKEN_ENV_VAR] = token
    try:
        kwargs = {
            "cwd": str(PROJECT_ROOT),
            "env": environment,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(
            [
                python_executable(),
                "-u",
                str(gui_script("splash_standalone.py")),
                token,
                str(MAX_DISPLAY_MS / 1000),
            ],
            **kwargs,
        )
    except OSError as exc:
        print(f"[FBQ] Could not launch external splash: {exc}", flush=True)
        return False, None

    os.environ[TOKEN_ENV_VAR] = token
    return True, process


def show_splash(root: ctk.CTk, reveal: bool = False) -> tk.Toplevel:
    """Show a fullscreen window with the FBQ logo centered."""
    splash = tk.Toplevel(root)
    splash.configure(bg=BG_DARK)
    screen_w, screen_h = configure_splash_window(splash)
    splash.attributes("-topmost", True)

    canvas = tk.Canvas(splash, width=screen_w, height=screen_h, bg=BG_DARK, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    splash.FBQ_SPLASH_CANVAS = canvas

    logo_path = gui_asset("logo_fbq_splash.png")
    if logo_path.is_file():
        with Image.open(logo_path) as source:
            source = source.convert("RGBA")
            target_w = max(1, round(screen_w * SPLASH_LOGO_WIDTH_RATIO))
            target_h = max(1, round(source.height * (target_w / source.width)))
            logo = source.resize((target_w, target_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(logo, master=splash)
        canvas.FBQ_SPLASH_PHOTO = photo
        cx, cy = screen_w / 2, screen_h / 2
        canvas.create_image(cx, cy, image=photo)
        if reveal:
            w, h = photo.width(), photo.height()
            reveal_from_left(canvas, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, BG_DARK)
    else:
        canvas.create_text(screen_w / 2, screen_h / 2, text=APP_TITLE, font=tk_font_spec(splash, "Arial", 32, "bold"), fill=ACCENT_BLUE)

    splash.lift()
    return splash


def launch_gui_script(
    name: str, *script_args: str, capture_errors: bool = False,
) -> subprocess.Popen:
    kwargs = {"cwd": str(PROJECT_ROOT)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    if capture_errors:
        kwargs.update(stderr=subprocess.PIPE, text=True, errors="replace")
    return subprocess.Popen(
        [sys.executable, "-u", str(gui_script(name)), *script_args], **kwargs
    )


# Module launchers
def launch_train_model() -> None:
    """Launch the Train AI Model workflow."""
    launch_gui_script("gui_easy_tracking.py")


def launch_inference_behavior() -> None:
    """Launch the Inference & Behavior Quantification workflow."""
    launch_gui_script("gui_behavior_quantification.py")


def launch_advanced_tracking() -> None:
    launch_gui_script("gui_advanced_tracking.py")


def launch_refinement() -> None:
    launch_gui_script("gui_refinement.py")


def launch_segmentation() -> None:
    launch_gui_script("gui_segmentation.py")


def launch_multi_config_batch() -> None:
    launch_gui_script("gui_multi_config_batch.py")


def launch_preprocess() -> None:
    launch_gui_script("gui_preprocess.py")


def launch_colab() -> None:
    launch_gui_script("gui_colab.py")


def manual_url(language: str) -> str:
    return ONLINE_MANUAL_BASE_URL


def open_manual(language: str) -> None:
    url = manual_url(language)
    if not webbrowser.open(url):
        messagebox.showinfo("Open manual", f"Open this page in your browser:\n{url}")


class WorkflowCard(tk.Canvas):
    """Large workflow card with smooth hover animations."""
    
    def __init__(
        self,
        parent: tk.Misc,
        root: ctk.CTk,
        title: str,
        subtitle: str,
        description: str,
        icon: str,
        command: Callable[[], None],
        accent_color: str,
        accent_hover: str,
        width: int,
        height: int,
    ):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=BG_CARD,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            takefocus=1,
        )
        
        self.root = root
        self.command = command
        self.accent_color = accent_color
        self.accent_hover = accent_hover
        self.width = width
        self.height = height
        self.hover_progress = 0.0
        self.target_hover = 0.0
        self.anim_job = None
        self.title = title
        self.subtitle = subtitle
        self.description = description
        self.icon_text = icon
        
        # Fonts
        self.icon_font = tk_font_spec(root, "Segoe UI", 72, "bold")
        self.title_font = tk_font_spec(root, "Segoe UI", 28, "bold")
        self.subtitle_font = tk_font_spec(root, "Segoe UI", 16, "normal")
        self.desc_font = tk_font_spec(root, "Segoe UI", 13, "normal")
        self.arrow_font = tk_font_spec(root, "Segoe UI", 24, "bold")
        
        # Card elements
        self.bg_rect_id = None
        self.border_id = None
        self.accent_bar_id = None
        self.icon_id = None
        self.title_id = None
        self.subtitle_id = None
        self.desc_id = None
        self.arrow_id = None
        self.glow_id = None
        
        self._bind_events()
        self._start_animation()
    
    def _bind_events(self):
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<Button-1>", lambda e: self._on_click())
        self.bind("<Return>", lambda e: self._on_click())
        self.bind("<space>", lambda e: self._on_click())
        self.bind("<Configure>", lambda e: self._redraw())
    
    def _set_hover(self, hover: bool):
        self.target_hover = 1.0 if hover else 0.0
    
    def _on_click(self):
        self.command()
    
    def _start_animation(self):
        self._animate()
    
    def _animate(self):
        diff = self.target_hover - self.hover_progress
        if abs(diff) > 0.005:
            self.hover_progress += diff * 0.12
        else:
            self.hover_progress = self.target_hover
        
        self._redraw()
        if self.winfo_exists():
            self.anim_job = self.root.after(16, self._animate)
    
    def _redraw(self):
        if not self.winfo_exists():
            return
            
        w = max(1, self.winfo_width() or self.width)
        h = max(1, self.winfo_height() or self.height)
        r = 20
        
        # Delete old elements
        self.delete("all")
        
        # Interpolate colors
        bg_color = self._lerp_color(BG_CARD, BG_CARD_HOVER, self.hover_progress)
        border_color = self._lerp_color(BORDER_COLOR, BORDER_HOVER, self.hover_progress)
        accent_color = self._lerp_color(self.accent_color, self.accent_hover, self.hover_progress)
        text_color = self._lerp_color(TEXT_PRIMARY, WHITE, self.hover_progress * 0.3)
        desc_color = self._lerp_color(TEXT_SECONDARY, TEXT_PRIMARY, self.hover_progress * 0.4)
        
        # Glow effect on hover (outer shadow)
        if self.hover_progress > 0.1:
            glow_alpha = self.hover_progress
            glow_offset = int(8 * self.hover_progress)
            # Create glow by drawing multiple expanding rects with decreasing opacity simulation
            for i in range(3):
                offset = glow_offset + i * 4
                alpha = 1.0 - (i * 0.3)
                glow_color = self._lerp_color(BG_DARK, accent_color, glow_alpha * alpha * 0.15)
                self.create_rounded_rect(
                    -offset, -offset, w + offset, h + offset, 
                    r + offset, fill="", outline=glow_color, width=2, tags="glow"
                )
        
        # Background rounded rect
        self.bg_rect_id = self.create_rounded_rect(0, 0, w, h, r, fill=bg_color, outline="")
        
        # Border
        border_width = self._lerp_int(1, 3, self.hover_progress)
        self.border_id = self.create_rounded_rect(
            border_width//2, border_width//2, 
            w - border_width//2, h - border_width//2, 
            r - border_width//2, fill="", outline=border_color, width=border_width
        )
        
        # Accent bar (left side) - animated width
        bar_w = self._lerp_int(6, 14, self.hover_progress)
        self.accent_bar_id = self.create_rounded_rect(
            0, 0, bar_w, h, r, fill=accent_color, outline=""
        )
        
        # Top accent line
        line_w = self._lerp_int(w * 0.15, w * 0.35, self.hover_progress)
        self.create_rounded_rect(0, 0, line_w, 3, 0, fill=accent_color, outline="")
        
        # Bottom accent line
        self.create_rounded_rect(0, h - 3, line_w, h, 0, fill=accent_color, outline="")
        
        # Icon
        icon_y = h * 0.28
        icon_scale = 1.0 + self.hover_progress * 0.08
        self.icon_id = self.create_text(
            w/2, icon_y, text=self.icon_text, 
            font=self.icon_font, fill=accent_color, anchor="center"
        )
        
        # Title
        title_y = h * 0.48
        self.title_id = self.create_text(
            w/2, title_y, text=self.title, 
            font=self.title_font, fill=text_color, anchor="center", width=w-60
        )
        
        # Subtitle
        subtitle_y = h * 0.60
        self.subtitle_id = self.create_text(
            w/2, subtitle_y, text=self.subtitle, 
            font=self.subtitle_font, fill=accent_color, anchor="center", width=w-60
        )
        
        # Description
        desc_y = h * 0.75
        self.desc_id = self.create_text(
            w/2, desc_y, text=self.description, 
            font=self.desc_font, fill=desc_color, anchor="center", width=w-80, justify="center"
        )
        
        # Arrow indicator (right side)
        arrow_x = w - 50 + self.hover_progress * 20
        arrow_y = h - 50
        self.arrow_id = self.create_text(
            arrow_x, arrow_y, text="▶", 
            font=self.arrow_font, fill=accent_color, anchor="center"
        )
        
        self.tag_lower(self.bg_rect_id)
        if self.glow_id:
            self.tag_lower(self.glow_id)
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        """Create a rounded rectangle."""
        if radius <= 0:
            return self.create_rectangle(x1, y1, x2, y2, **kwargs)
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, splinesteps=32, **kwargs)
    
    def _lerp_color(self, color1: str, color2: str, t: float) -> str:
        t = max(0.0, min(1.0, t))
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _lerp_int(self, a: int, b: int, t: float) -> int:
        t = max(0.0, min(1.0, t))
        return int(a + (b - a) * t)


class ToolButton(ctk.CTkButton):
    """Compact tool button for bottom bar."""
    
    def __init__(self, parent, text: str, command: Callable, description: str = "", **kwargs):
        super().__init__(
            parent,
            text=text,
            command=command,
            width=170,
            height=48,
            corner_radius=10,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="normal"),
            fg_color=BG_CARD,
            hover_color=BG_CARD_HOVER,
            border_color=BORDER_COLOR,
            border_width=1,
            text_color=TEXT_PRIMARY,
            **kwargs
        )
        self.description = description
        self.bind("<Enter>", lambda e: self._show_tooltip())
        self.bind("<Leave>", lambda e: self._hide_tooltip())
        self.tooltip = None
    
    def _show_tooltip(self):
        if self.description and not self.tooltip:
            self.tooltip = ctk.CTkToplevel(self)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.attributes("-topmost", True)
            x = self.winfo_rootx() + self.winfo_width() // 2
            y = self.winfo_rooty() - 50
            self.tooltip.geometry(f"+{x}+{y}")
            label = ctk.CTkLabel(
                self.tooltip,
                text=self.description,
                font=ctk.CTkFont(family="Segoe UI", size=11),
                fg_color=BG_CARD,
                text_color=TEXT_SECONDARY,
                corner_radius=8,
                padx=14,
                pady=8
            )
            label.pack()
    
    def _hide_tooltip(self):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


def build_home(root: ctk.CTk) -> None:
    """Build the modern full-screen home window."""
    # Get screen dimensions
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    
    # Main container - full screen
    main_frame = ctk.CTkFrame(root, fg_color=BG_DARK, corner_radius=0)
    main_frame.pack(fill="both", expand=True)
    main_frame.grid_rowconfigure(0, weight=0)  # Header
    main_frame.grid_rowconfigure(1, weight=1)  # Workflows (expand)
    main_frame.grid_rowconfigure(2, weight=0)  # Tools
    main_frame.grid_rowconfigure(3, weight=0)  # Footer
    main_frame.grid_columnconfigure(0, weight=1)
    
    # === HEADER SECTION ===
    header = ctk.CTkFrame(main_frame, fg_color="transparent")
    header.grid(row=0, column=0, sticky="ew", padx=60, pady=(30, 10))
    header.grid_columnconfigure(0, weight=1)
    
    # Logo + Title in a horizontal layout for better visibility
    header_top = ctk.CTkFrame(header, fg_color="transparent")
    header_top.grid(row=0, column=0, sticky="ew")
    header_top.grid_columnconfigure(0, weight=0)  # Logo
    header_top.grid_columnconfigure(1, weight=1)  # Title
    
    # Logo
    logo_path = gui_asset("logo_fbq_splash.png")
    if logo_path.is_file():
        with Image.open(logo_path) as img:
            img = img.convert("RGBA")
            logo_img = ctk.CTkImage(light_image=img, dark_image=img, size=(80, 80))
        logo_label = ctk.CTkLabel(header_top, image=logo_img, text="")
        logo_label.grid(row=0, column=0, rowspan=3, padx=(0, 20), sticky="n")
    
    # Title - larger and more prominent
    title_label = ctk.CTkLabel(
        header_top,
        text="Fish Behavior Quantifier",
        font=ctk.CTkFont(family="Segoe UI", size=36, weight="bold"),
        text_color=TEXT_PRIMARY,
        anchor="w"
    )
    title_label.grid(row=0, column=1, sticky="ew", pady=(5, 0))
    
    # Subtitle
    subtitle_label = ctk.CTkLabel(
        header_top,
        text="Annotation-free multi-fish tracking & behavior analysis",
        font=ctk.CTkFont(family="Segoe UI", size=16),
        text_color=TEXT_SECONDARY,
        anchor="w"
    )
    subtitle_label.grid(row=1, column=1, sticky="ew", pady=(0, 8))
    
    # Version badge
    try:
        version = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
        if version and not version.lower().startswith("v"):
            version = f"v{version}"
    except OSError:
        version = "v1.2.4"
    
    version_label = ctk.CTkLabel(
        header_top,
        text=f"{version}  •  AGPL-3.0",
        font=ctk.CTkFont(family="Segoe UI", size=12),
        text_color=TEXT_MUTED,
        anchor="w"
    )
    version_label.grid(row=2, column=1, sticky="ew", pady=(0, 0))
    
    # === PRIMARY WORKFLOWS SECTION ===
    workflows_section = ctk.CTkFrame(main_frame, fg_color="transparent")
    workflows_section.grid(row=1, column=0, sticky="nsew", padx=60, pady=15)
    workflows_section.grid_columnconfigure(0, weight=1)
    workflows_section.grid_columnconfigure(1, weight=1)
    workflows_section.grid_rowconfigure(1, weight=1)
    
    # Section label
    workflows_label = ctk.CTkLabel(
        workflows_section,
        text="Primary Workflows",
        font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
        text_color=TEXT_PRIMARY,
        anchor="w"
    )
    workflows_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
    
    # Calculate card dimensions based on available space
    # Reserve space for header (~130), tools (~110), footer (~80), padding
    available_height = sh - 350
    card_width = min(560, max(400, int((sw - 180) / 2)))
    card_height = min(460, max(340, available_height))
    
    # Left card - Train AI Model
    train_frame = ctk.CTkFrame(workflows_section, fg_color="transparent")
    train_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 15))
    train_frame.grid_rowconfigure(0, weight=1)
    train_frame.grid_columnconfigure(0, weight=1)
    
    train_card = WorkflowCard(
        train_frame,
        root,
        title="Train AI Model",
        subtitle="Self-Supervised Learning",
        description="Build a custom fish detector from your own videos.\nNo manual annotation needed — the model learns\nfrom synthetic data generated from your footage.",
        icon="🎯",
        command=launch_train_model,
        accent_color=ACCENT_BLUE,
        accent_hover=ACCENT_BLUE_HOVER,
        width=card_width,
        height=card_height,
    )
    train_card.grid(row=0, column=0, sticky="nsew")
    train_card.bind("<Configure>", lambda e: train_card._redraw())
    
    # Right card - Inference & Behavior Quantification
    infer_frame = ctk.CTkFrame(workflows_section, fg_color="transparent")
    infer_frame.grid(row=1, column=1, sticky="nsew", padx=(15, 0))
    infer_frame.grid_rowconfigure(0, weight=1)
    infer_frame.grid_columnconfigure(0, weight=1)
    
    infer_card = WorkflowCard(
        infer_frame,
        root,
        title="Inference & Behavior\nQuantification",
        subtitle="Keypoints • Tracking • Metrics",
        description="Run trained models on new videos to extract\nhead/center/tail keypoints, maintain identities,\nand quantify 13 behavioral metrics automatically.",
        icon="📊",
        command=launch_inference_behavior,
        accent_color=ACCENT_CYAN,
        accent_hover=ACCENT_CYAN_HOVER,
        width=card_width,
        height=card_height,
    )
    infer_card.grid(row=0, column=0, sticky="nsew")
    infer_card.bind("<Configure>", lambda e: infer_card._redraw())
    
    # === OPTIONAL TOOLS BAR ===
    tools_section = ctk.CTkFrame(main_frame, fg_color="transparent")
    tools_section.grid(row=2, column=0, sticky="ew", padx=60, pady=(5, 20))
    tools_section.grid_columnconfigure(0, weight=1)
    
    # Tools label
    tools_label = ctk.CTkLabel(
        tools_section,
        text="Optional Tools",
        font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
        text_color=TEXT_PRIMARY,
        anchor="w"
    )
    tools_label.grid(row=0, column=0, sticky="ew", pady=(0, 12))
    
    # Tool buttons container - use dynamic column count
    buttons_frame = ctk.CTkFrame(tools_section, fg_color="transparent")
    buttons_frame.grid(row=1, column=0, sticky="ew")
    
    optional_tools = [
        ("Segmentation", launch_segmentation, "Configure foreground/background separation"),
        ("Cropping & Trimming", launch_preprocess, "Crop and trim raw videos"),
        ("Advanced Tracking", launch_advanced_tracking, "Full pipeline parameter control"),
        ("Refinement", launch_refinement, "Manual correction of tracking results"),
        ("Colab (beta)", launch_colab, "Google Colab GPU runtime package") if sys.platform.startswith("linux") else ("Colab (beta)", launch_colab, "Google Colab GPU runtime package"),
    ]
    
    # Configure columns dynamically based on number of tools
    num_tools = len(optional_tools)
    for i in range(num_tools):
        buttons_frame.grid_columnconfigure(i, weight=1)
    
    for col, (text, command, desc) in enumerate(optional_tools):
        btn = ToolButton(
            buttons_frame,
            text=text,
            command=command,
            description=desc,
        )
        btn.grid(row=0, column=col, padx=10, pady=8, sticky="ew")
    
    # === FOOTER ACTIONS ===
    footer = ctk.CTkFrame(main_frame, fg_color="transparent")
    footer.grid(row=3, column=0, sticky="ew", padx=60, pady=(0, 20))
    footer.grid_columnconfigure(0, weight=1)  # Left spacer
    footer.grid_columnconfigure(1, weight=0)  # Center group
    footer.grid_columnconfigure(2, weight=1)  # Right spacer
    
    # Center action buttons group
    actions_frame = ctk.CTkFrame(footer, fg_color="transparent")
    actions_frame.grid(row=0, column=1)
    
    doc_btn = ctk.CTkButton(
        actions_frame,
        text="📖  Documentation",
        command=lambda: open_manual("EN"),
        width=170,
        height=44,
        corner_radius=10,
        font=ctk.CTkFont(family="Segoe UI", size=12, weight="normal"),
        fg_color="transparent",
        hover_color=BG_CARD,
        border_color=BORDER_COLOR,
        border_width=1,
        text_color=TEXT_SECONDARY,
    )
    doc_btn.pack(side="left", padx=8)
    
    update_btn = ctk.CTkButton(
        actions_frame,
        text="🔄  Check Updates",
        command=lambda: check_updates(root, update_btn),
        width=170,
        height=44,
        corner_radius=10,
        font=ctk.CTkFont(family="Segoe UI", size=12, weight="normal"),
        fg_color="transparent",
        hover_color=BG_CARD,
        border_color=BORDER_COLOR,
        border_width=1,
        text_color=TEXT_SECONDARY,
    )
    update_btn.pack(side="left", padx=8)
    
    uninstall_btn = ctk.CTkButton(
        actions_frame,
        text="🗑️  Uninstall",
        command=lambda: uninstall_app(root),
        width=140,
        height=44,
        corner_radius=10,
        font=ctk.CTkFont(family="Segoe UI", size=12, weight="normal"),
        fg_color="transparent",
        hover_color="#2a1010",
        border_color="#4a1a1a",
        border_width=1,
        text_color="#ff8888",
    )
    uninstall_btn.pack(side="left", padx=8)
    
    # Keyboard shortcuts
    root.bind("<F1>", lambda e: open_manual("EN"))
    root.bind("<Control-t>", lambda e: launch_train_model())
    root.bind("<Control-i>", lambda e: launch_inference_behavior())
    root.bind("<Escape>", lambda e: root.destroy())
    
    # Start update probe in background
    root.after(500, lambda: start_update_probe(root, update_btn))


def start_update_probe(root: ctk.CTk, update_btn: ctk.CTkButton) -> None:
    """Start background update check."""
    def probe():
        try:
            _current, latest, available = probe_update_status()
        except Exception:
            return
        if available and latest is not None:
            root.after(0, lambda: mark_update_available(root, update_btn, latest))
    
    threading.Thread(target=probe, daemon=True, name="FBQ Home update probe").start()


def mark_update_available(root: ctk.CTk, update_btn: ctk.CTkButton, latest_version: str) -> None:
    """Mark update as available with visual indicator."""
    update_btn.configure(
        text=f"🔄  Update to {latest_version}",
        fg_color=ACCENT_GREEN,
        hover_color="#00e673",
        text_color=WHITE,
        border_color=ACCENT_GREEN,
    )
    root.title(f"{home_window_title()}  ·  update {latest_version} available")


def check_updates(root: ctk.CTk, update_btn: ctk.CTkButton) -> None:
    """Manual update check triggered by user."""
    update_btn.configure(text="🔄  Checking...", state="disabled")
    
    def do_check():
        try:
            _current, latest, available = probe_update_status()
            if available and latest is not None:
                root.after(0, lambda: mark_update_available(root, update_btn, latest))
                root.after(0, lambda: start_update_check(
                    root,
                    on_status=lambda msg: root.after(0, lambda: update_btn.configure(text=msg)),
                    on_update_start=lambda: root.after(0, lambda: setattr(root, '_fbq_update_requested', True) or root.destroy()),
                ))
            else:
                root.after(0, lambda: update_btn.configure(text="✅  Up to date", fg_color="transparent", text_color=ACCENT_GREEN, border_color=ACCENT_GREEN))
                root.after(3000, lambda: update_btn.configure(text="🔄  Check Updates", state="normal", fg_color="transparent", text_color=TEXT_SECONDARY, border_color=BORDER_COLOR))
        except Exception:
            root.after(0, lambda: update_btn.configure(text="❌  Check failed", fg_color="transparent", text_color=ACCENT_RED, border_color=ACCENT_RED))
            root.after(3000, lambda: update_btn.configure(text="🔄  Check Updates", state="normal", fg_color="transparent", text_color=TEXT_SECONDARY, border_color=BORDER_COLOR))
    
    threading.Thread(target=do_check, daemon=True).start()


def uninstall_app(root: ctk.CTk) -> None:
    """Handle uninstall request."""
    def close_for_uninstall():
        root._fbq_uninstall_requested = True
        root.destroy()
    start_uninstall(root, on_uninstall_start=close_for_uninstall)


def main() -> int:
    """Entry point for the Home window."""
    # External splash FIRST - before creating main window
    has_splash, splash_proc = ensure_external_splash()
    
    # Main window - hidden initially
    root = ctk.CTk()
    root.title(home_window_title())
    install_window_icon(root)
    configure_dpi_scaling(root)
    configure_taskbar_identity()
    
    # Full screen dimensions
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{sw}x{sh}+0+0")
    root.minsize(1000, 700)
    
    # Build UI BEFORE showing window
    build_home(root)
    
    # Hide window initially
    root.withdraw()
    
    # Show splash reveal
    if has_splash:
        splash = show_splash(root, reveal=True)
        
        def show_main_window():
            splash.destroy()
            root.deiconify()
            root.lift()
            root.focus_force()
        
        root.after(MAX_DISPLAY_MS, show_main_window)
    else:
        root.deiconify()
        root.lift()
        root.focus_force()
    
    root.mainloop()
    
    # Cleanup splash process
    if splash_proc and splash_proc.poll() is None:
        splash_proc.terminate()
    
    # Return exit code for uninstall/update
    if getattr(root, "_fbq_uninstall_requested", False):
        return UNINSTALL_EXIT_CODE
    if getattr(root, "_fbq_update_requested", False):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())