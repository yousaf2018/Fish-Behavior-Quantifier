# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Unified settings management for FBQ behavior tracking and endpoints extraction.

Settings are stored per-session-folder (where videos are loaded from) to enable
auto-loading when the user opens the same video folder again.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


SETTINGS_FILENAME = "fbq_session_settings.json"


def get_session_settings_path(session_dir: str) -> str:
    """Get the settings file path for a session directory."""
    return os.path.join(session_dir, SETTINGS_FILENAME)


def save_session_settings(session_dir: str, settings: Dict[str, Any]) -> bool:
    """Save settings to the session directory."""
    try:
        path = get_session_settings_path(session_dir)
        os.makedirs(session_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"[SETTINGS] Failed to save session settings: {e}")
        return False


def load_session_settings(session_dir: str) -> Optional[Dict[str, Any]]:
    """Load settings from the session directory."""
    try:
        path = get_session_settings_path(session_dir)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"[SETTINGS] Failed to load session settings: {e}")
    return None


def find_session_dir_from_videos(video_paths: List[str]) -> Optional[str]:
    """Find the common session directory from a list of video paths.
    
    Returns the common parent directory of all videos, or the directory
    of the first video if they're in different directories.
    """
    if not video_paths:
        return None
    
    dirs = [os.path.dirname(os.path.abspath(p)) for p in video_paths]
    
    # Find common parent
    common = os.path.commonpath(dirs)
    if common:
        return common
    
    return dirs[0]


@dataclass
class DisplaySettings:
    """Video display settings."""
    arrow_size: int = 45
    id_font_size: float = 0.6
    box_thickness: int = 2
    keypoint_radius: int = 4
    keypoint_font_size: float = 0.4
    trail_length: int = 30


@dataclass
class BehaviorSettings:
    """Behavior analysis settings."""
    extract_head: bool = True
    extract_center: bool = True
    extract_tail: bool = True
    analysis: bool = True


@dataclass
class EndpointsGlobalConfig:
    """Global configuration for endpoints extraction."""
    conversion: float = 18.74
    fps: float = 24.99
    freeze_thresh: float = 0.1
    rapid_thresh: float = 3.0
    body_length: float = 3.0
    freeze_thresh_bl: float = 0.05
    rapid_thresh_bl: float = 1.0
    export_kinetics: bool = True
    global_view_type: str = "Side / Lateral View"
    global_division_rule: str = "Horizontal Split (Y < Center is Top)"
    global_cx_pct: float = 50.0
    global_cy_pct: float = 50.0


@dataclass
class ArenaConfig:
    """Per-arena configuration for endpoints extraction."""
    cx_pct: float = 50.0
    cy_pct: float = 50.0
    view_type: str = "Side / Lateral View"
    division_rule: str = "Horizontal Split (Y < Center is Top)"


def create_default_session_settings() -> Dict[str, Any]:
    """Create default session settings structure."""
    return {
        "version": 1,
        "model_session": "",
        "weights_path": "",
        "output_dir": "",
        "endpoints_output_dir": "",
        "display": asdict(DisplaySettings()),
        "behavior": asdict(BehaviorSettings()),
        "endpoints_global_config": asdict(EndpointsGlobalConfig()),
        "endpoints_metric_selections": {},
        "endpoints_arena_configs": {},
        "endpoints_calibration_ratio": None,
        "videos": [],
    }


def merge_settings(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override settings into base settings."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_settings(result[key], value)
        else:
            result[key] = value
    return result