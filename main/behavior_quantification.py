# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Standalone Behavioral Quantification Module for Fish Behavior Analysis.

This module provides a complete inference pipeline for fish behavior quantification
using pre-trained weights. It does not require training and handles:
- OBB detection using pre-trained YOLO model
- Multi-object tracking with identity preservation
- Keypoint extraction (head, center, tail) from detections
- Behavioral analysis (speed, acceleration, angles, distances)
- Output video with annotations and CSV results
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

try:
    from tracking_artifacts import artifact_path as tracking_artifact_path
except ImportError:
    def tracking_artifact_path(out_dir: str, base_name: str, suffix: str = "") -> str:
        return os.path.join(out_dir, "buffer", base_name)


KEYPOINT_NAMES = ["head", "center", "tail"]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_device(device_str: str = "auto") -> str:
    if device_str == "auto":
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_str


def read_video_frame_info(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "usable_frame_count": frame_count,
    }


def polygon_to_mask(polygon: List[List[float]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def obb_to_keypoints(obb: List[float], contour: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, float]]:
    """Extract head, center, tail keypoints from OBB.
    
    OBB format: [cx, cy, w, h, angle_rad]
    Head = midpoint of the shorter edge facing forward
    Center = centroid
    Tail = opposite edge midpoint
    """
    cx, cy, w, h, angle = obb[:5]
    
    # Direction vector (pointing to "head" side)
    dx = math.cos(angle)
    dy = math.sin(angle)
    
    # Perpendicular vector
    px = -dy
    py = dx
    
    # Half dimensions
    hw = w / 2
    hh = h / 2
    
    # Four corners
    corners = [
        (cx + hw * dx + hh * px, cy + hw * dy + hh * py),  # head-right
        (cx + hw * dx - hh * px, cy + hw * dy - hh * py),  # head-left
        (cx - hw * dx - hh * px, cy - hw * dy - hh * py),  # tail-left
        (cx - hw * dx + hh * px, cy - hw * dy + hh * py),  # tail-right
    ]
    
    # Head = midpoint of head edge (first two corners)
    head_x = (corners[0][0] + corners[1][0]) / 2
    head_y = (corners[0][1] + corners[1][1]) / 2
    
    # Tail = midpoint of tail edge (last two corners)
    tail_x = (corners[2][0] + corners[3][0]) / 2
    tail_y = (corners[2][1] + corners[3][1]) / 2
    
    # Center = centroid
    center_x = cx
    center_y = cy
    
    return {
        "head": (head_x, head_y),
        "center": (center_x, center_y),
        "tail": (tail_x, tail_y),
    }


def heading_from_axis(dx: float, dy: float) -> float:
    """Return heading in degrees from the upward axis toward +x (clockwise).

    Matches :func:`main.direction_class_assignment.head_angle_deg_from_axis`:
    ``0`` points up, angle grows clockwise on screen.
    """
    return (math.degrees(math.atan2(float(dx), -float(dy))) + 360.0) % 360.0


def direction_class(dx: float, dy: float) -> int:
    """Quantize a direction vector into one of 8 direction classes.

    Matches :func:`main.direction_class_assignment.angle_to_class`.
    Classes: upper, upper_right, right, lower_right, lower, lower_left,
    left, upper_left.
    """
    angle = heading_from_axis(dx, dy)
    return int(((angle + 22.5) % 360.0) // 45.0)


DIRECTION_CLASS_NAMES = [
    "upper", "upper_right", "right", "lower_right",
    "lower", "lower_left", "left", "upper_left"
]


def track_label(track_id: Any) -> str:
    """Return a display label like ``Tank1-ID1`` for a track id.

    Parses the ``roi{idx}_{num}`` string ids used when tracking multiple ROIs,
    converting to 1-based ``Tank{idx+1}-ID{num}`` labels. Integer ids fall back
    to ``ID{n}``.
    """
    if isinstance(track_id, str) and track_id.startswith("roi"):
        try:
            parts = track_id.split("_")
            roi_idx = int(parts[0].replace("roi", ""))
            track_num = int(parts[-1])
            return f"Tank{roi_idx + 1}-ID{track_num}"
        except (ValueError, IndexError):
            return str(track_id)
    return f"ID{track_id}"


def track_color(track_id: Any) -> Tuple[int, int, int]:
    """Return a distinct BGR color for a track id.

    Accepts integer ids or the ``roi{idx}_{id}`` string ids used when
    tracking multiple ROIs. Falls back to a stable hash for unknown formats.
    """
    palette = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
        (255, 255, 0), (255, 0, 255), (0, 128, 255), (0, 255, 128),
        (128, 0, 255), (255, 128, 0), (128, 255, 0), (255, 0, 128),
        (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128),
    ]
    if isinstance(track_id, str) and track_id.startswith("roi"):
        # Extract the numeric track id after the last underscore.
        try:
            numeric = int(track_id.rsplit("_", 1)[-1])
        except ValueError:
            numeric = 0
    elif isinstance(track_id, (int, float)):
        numeric = int(track_id)
    else:
        # Stable hash for any other format.
        numeric = sum(ord(c) for c in str(track_id))
    return palette[numeric % len(palette)]


def display_config(cfg: dict) -> dict:
    """Return display settings with sensible defaults."""
    disp = cfg.get("DISPLAY", {}) or {}
    if not isinstance(disp, dict):
        disp = {}
    return {
        "arrow_size": int(disp.get("ARROW_SIZE", 45)),
        "id_font_size": float(disp.get("ID_FONT_SIZE", 0.6)),
        "id_font_thickness": int(disp.get("ID_FONT_THICKNESS", 2)),
        "box_thickness": int(disp.get("BOX_THICKNESS", 2)),
        "keypoint_radius": int(disp.get("KEYPOINT_RADIUS", 4)),
        "keypoint_font_size": float(disp.get("KEYPOINT_FONT_SIZE", 0.4)),
        "trail_length": int(disp.get("TRAIL_LENGTH", 30)),
    }


def compute_heading(keypoints: Dict[str, Tuple[float, float]]) -> float:
    """Compute heading angle from center to head (direction the fish faces).

    Uses the AMADEUS convention: measured from the upward axis toward +x,
    clockwise, matching the direction estimation classes.
    """
    hx, hy = keypoints["head"]
    cx, cy = keypoints["center"]
    return heading_from_axis(hx - cx, hy - cy)


def compute_speed(prev_center: Tuple[float, float], curr_center: Tuple[float, float], fps: float) -> float:
    """Compute speed in pixels per second."""
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    return math.hypot(dx, dy) * fps


def run_obb_detection(
    model: YOLO,
    video_path: str,
    roi_mask: Optional[np.ndarray] = None,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
    imgsz: int = 1920,
    batch_size: int = 8,
    device: str = "cuda:0",
    max_det: int = 300,
    first_frame: int = 0,
    last_frame: int = -1,
) -> List[Dict[str, Any]]:
    """Run OBB detection on video frames.

    When ``roi_mask`` is None the full frame is processed (no masking),
    so detection runs exactly once per video and detections are later
    assigned to ROIs by checking whether their centre lies inside each
    ROI mask. This avoids re-running the model once per ROI.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if last_frame < 0 or last_frame >= total_frames:
        last_frame = total_frames - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    
    all_detections = []
    frame_buffer = []
    frame_indices = []
    
    n_frames = last_frame - first_frame + 1
    
    for frame_idx in range(first_frame, last_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply ROI mask only when one is provided; otherwise use full frame.
        if roi_mask is not None:
            frame_in = cv2.bitwise_and(frame, frame, mask=roi_mask)
        else:
            frame_in = frame
        
        frame_buffer.append(frame_in)
        frame_indices.append(frame_idx)
        
        # Report frame-level progress for GUI responsiveness
        if (frame_idx - first_frame + 1) % 30 == 0 or frame_idx == last_frame:
            pct = (frame_idx - first_frame + 1) / max(n_frames, 1)
            print(f"PROGRESS:{pct:.3f}")
        
        if len(frame_buffer) >= batch_size or frame_idx == last_frame:
            results = model.predict(
                frame_buffer,
                imgsz=imgsz,
                conf=conf_thresh,
                iou=iou_thresh,
                max_det=max_det,
                device=device,
                verbose=False,
                stream=False,
            )
            
            for i, result in enumerate(results):
                fi = frame_indices[i]
                if result.obb is not None:
                    for obb in result.obb.data.cpu().numpy():
                        # OBB format: [cx, cy, w, h, angle, conf, cls]
                        cx, cy, w, h, angle, conf, cls = obb[:7]
                        
                        # When a mask is given, keep detections whose centre is
                        # inside the ROI; otherwise keep everything.
                        if roi_mask is not None:
                            if roi_mask[int(cy), int(cx)] <= 0:
                                continue
                        
                        all_detections.append({
                            "frame": fi,
                            "cx": float(cx),
                            "cy": float(cy),
                            "w": float(w),
                            "h": float(h),
                            "angle": float(angle),
                            "conf": float(conf),
                            "class": int(cls),
                            "class_name": DIRECTION_CLASS_NAMES[int(cls)] if int(cls) < len(DIRECTION_CLASS_NAMES) else str(int(cls)),
                        })
            
            frame_buffer.clear()
            frame_indices.clear()
    
    cap.release()
    return all_detections


def simple_tracking(
    detections: List[Dict[str, Any]],
    num_objects: int,
    max_age: int = 30,
    iou_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Simple multi-object tracking using IoU matching."""
    # Group detections by frame
    by_frame: Dict[int, List[Dict]] = {}
    for det in detections:
        by_frame.setdefault(det["frame"], []).append(det)
    
    # Track objects
    tracks = {}  # track_id -> {last_frame, last_bbox, age, positions}
    next_track_id = 1
    results = []
    
    for frame_idx in sorted(by_frame.keys()):
        frame_dets = by_frame[frame_idx]
        
        # Match existing tracks
        matched_dets = set()
        for track_id, track in list(tracks.items()):
            if frame_idx - track["last_frame"] > max_age:
                continue
            
            best_iou = 0
            best_det_idx = -1
            last_bbox = track["last_bbox"]
            
            for i, det in enumerate(frame_dets):
                if i in matched_dets:
                    continue
                
                # Compute IoU between OBBs (approximate with axis-aligned boxes)
                iou = compute_obb_iou(last_bbox, det)
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx >= 0:
                det = frame_dets[best_det_idx]
                tracks[track_id] = {
                    "last_frame": frame_idx,
                    "last_bbox": det,
                    "age": track["age"] + 1,
                    "positions": track["positions"] + [(frame_idx, det["cx"], det["cy"], det["angle"])],
                }
                matched_dets.add(best_det_idx)
                det["track_id"] = track_id
                results.append(det)
            else:
                # Track lost
                if frame_idx - track["last_frame"] > max_age:
                    del tracks[track_id]
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(frame_dets):
            if i not in matched_dets and len(tracks) < num_objects:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = {
                    "last_frame": frame_idx,
                    "last_bbox": det,
                    "age": 1,
                    "positions": [(frame_idx, det["cx"], det["cy"], det["angle"])],
                }
                det["track_id"] = track_id
                results.append(det)
    
    return results


def compute_obb_iou(obb1: Dict, obb2: Dict) -> float:
    """Approximate IoU between two OBBs using axis-aligned bounding boxes."""
    # Convert OBB to AABB
    def obb_to_aabb(obb):
        cx, cy, w, h, angle = obb["cx"], obb["cy"], obb["w"], obb["h"], obb["angle"]
        hw, hh = w / 2, h / 2
        ca, sa = math.cos(angle), math.sin(angle)
        corners = [
            (cx + hw * ca - hh * sa, cy + hw * sa + hh * ca),
            (cx + hw * ca + hh * sa, cy + hw * sa - hh * ca),
            (cx - hw * ca + hh * sa, cy - hw * sa - hh * ca),
            (cx - hw * ca - hh * sa, cy - hw * sa + hh * ca),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)
    
    x1_min, y1_min, x1_max, y1_max = obb_to_aabb(obb1)
    x2_min, y2_min, x2_max, y2_max = obb_to_aabb(obb2)
    
    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0
    
    inter = (xi_max - xi_min) * (yi_max - yi_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def extract_keypoints_for_tracks(
    tracked_detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Add keypoints (head, center, tail) to each tracked detection.

    Follows the same direction-estimation logic as the AMADEUS pipeline
    (``main/direction_class_assignment.py``):

    * The OBB long axis is taken from the bounding box angle (if ``w >= h``
      the axis runs along ``angle``, otherwise along ``angle + 90°``), matching
      AMADEUS's ``_obb_geometry_from_contour``.
    * Consecutive OBB long axes are aligned by dot-product sign so the body
      axis does not flip 180° frame to frame.
    * The global axis sign is chosen by majority vote against the motion
      vectors, exactly like the direction stage.
    * Heading and direction class use the AMADEUS convention: measured from
      the upward axis toward +x, clockwise, quantised to eight 45° classes
      (upper, upper_right, right, lower_right, lower, lower_left, left,
      upper_left).

    Head is placed at the leading end of movement, tail at the trailing end,
    center at the OBB centre. When motion is too small to decide, the OBB
    geometry is used as a fallback.
    """
    # Group by track and sort by frame.
    by_track: Dict[int, List[Dict]] = {}
    for det in tracked_detections:
        by_track.setdefault(det["track_id"], []).append(det)

    for track_id in by_track:
        by_track[track_id].sort(key=lambda x: x["frame"])

    for track_id, frames in by_track.items():
        n = len(frames)
        if n == 0:
            continue

        # Motion vectors between consecutive frame centres (previous -> current
        # points toward the head).
        motions: List[Tuple[float, float]] = [(0.0, 0.0)] * n
        for i in range(1, n):
            dx = float(frames[i]["cx"]) - float(frames[i - 1]["cx"])
            dy = float(frames[i]["cy"]) - float(frames[i - 1]["cy"])
            motions[i] = (dx, dy)

        # First frame borrows the next motion vector if available.
        if n > 1:
            motions[0] = motions[1]

        # OBB long-axis unit vector per frame (from the OBB angle).
        # If width >= height the long axis runs along the OBB angle;
        # otherwise it runs along angle + 90° (same as AMADEUS).
        axes: List[Tuple[float, float]] = []
        for det in frames:
            angle = float(det["angle"])
            w = float(det["w"])
            h = float(det["h"])
            axis_angle = angle if w >= h else angle + math.pi / 2.0
            axes.append((math.cos(axis_angle), math.sin(axis_angle)))

        # Align consecutive axes by dot-product sign (avoid 180° flips).
        aligned: List[Tuple[float, float]] = [axes[0]]
        for i in range(1, n):
            ax, ay = axes[i]
            px, py = aligned[i - 1]
            if ax * px + ay * py < 0.0:
                ax, ay = -ax, -ay
            aligned.append((ax, ay))

        # Global sign chosen by majority vote against motion (AMADEUS pattern).
        score_pos = 0.0
        score_neg = 0.0
        for ax, ay in aligned:
            for dx, dy in motions:
                mag = math.hypot(dx, dy)
                if mag <= 1e-6:
                    continue
                d = ax * (dx / mag) + ay * (dy / mag)
                score_pos += mag * d
                score_neg += mag * (-d)
        global_flip = -1.0 if score_neg > score_pos else 1.0

        for i, det in enumerate(frames):
            ax, ay = aligned[i][0] * global_flip, aligned[i][1] * global_flip
            mag = math.hypot(ax, ay)
            if mag <= 1e-6:
                # Degenerate axis: fall back to OBB geometry.
                kpts = obb_to_keypoints([
                    det["cx"], det["cy"], det["w"], det["h"], det["angle"],
                ])
                head_pt = kpts["head"]
                tail_pt = kpts["tail"]
            else:
                ux, uy = ax / mag, ay / mag
                # Head is at the positive end of the aligned axis (direction of
                # motion), tail at the negative end. Distance from centre is half
                # the *long* side of the OBB (the axis may run along w or h).
                cx, cy = float(det["cx"]), float(det["cy"])
                half_len = max(float(det["w"]), float(det["h"])) / 2.0
                head_pt = (cx + ux * half_len, cy + uy * half_len)
                tail_pt = (cx - ux * half_len, cy - uy * half_len)

            det["head_x"], det["head_y"] = head_pt
            det["center_x"], det["center_y"] = float(det["cx"]), float(det["cy"])
            det["tail_x"], det["tail_y"] = tail_pt
            det["heading"] = compute_heading({
                "head": head_pt,
                "center": (float(det["cx"]), float(det["cy"])),
                "tail": tail_pt,
            })
            det["direction_class"] = direction_class(ax, ay)
            det["direction_class_name"] = DIRECTION_CLASS_NAMES[
                det["direction_class"]
            ]

    return tracked_detections


def compute_behavior_metrics(
    tracked_detections: List[Dict[str, Any]],
    fps: float,
) -> List[Dict[str, Any]]:
    """Compute behavioral metrics per track per frame."""
    # Group by track
    by_track: Dict[int, List[Dict]] = {}
    for det in tracked_detections:
        by_track.setdefault(det["track_id"], []).append(det)
    
    # Sort each track by frame
    for track_id in by_track:
        by_track[track_id].sort(key=lambda x: x["frame"])
    
    # Compute metrics
    results = []
    for track_id, frames in by_track.items():
        prev_center = None
        prev_heading = None
        for i, det in enumerate(frames):
            center = (det["center_x"], det["center_y"])
            heading = det["heading"]
            
            speed = 0.0
            angular_vel = 0.0
            acceleration = 0.0
            
            if prev_center is not None:
                speed = compute_speed(prev_center, center, fps)
                if prev_heading is not None:
                    # Smallest signed angular difference, handling 360° wraparound.
                    d = (heading - prev_heading + 180.0) % 360.0 - 180.0
                    angular_vel = d * fps
            
            # Simple acceleration (difference in speed)
            if i > 0:
                prev_speed = frames[i-1].get("speed", 0)
                acceleration = (speed - prev_speed) * fps
            
            det["speed"] = speed
            det["angular_velocity"] = angular_vel
            det["acceleration"] = acceleration
            
            prev_center = center
            prev_heading = heading
            
            results.append(det)
    
    return results


def write_results_csv(
    detections: List[Dict[str, Any]],
    output_path: str,
):
    """Write tracking and behavior results to CSV."""
    fieldnames = [
        "frame", "track_id", "class", "class_name", "conf",
        "cx", "cy", "w", "h", "angle",
        "head_x", "head_y", "center_x", "center_y", "tail_x", "tail_y",
        "heading", "speed", "angular_velocity", "acceleration",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for det in detections:
            row = {k: det.get(k, "") for k in fieldnames}
            writer.writerow(row)


def create_output_video(
    video_path: str,
    detections: List[Dict[str, Any]],
    output_path: str,
    fps: float,
    arrow_size: int = 45,
    id_font_size: float = 0.6,
    id_font_thickness: int = 2,
    box_thickness: int = 2,
    keypoint_radius: int = 4,
    keypoint_font_size: float = 0.4,
    trail_length: int = 30,
):
    """Create annotated output video."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Group detections by frame
    by_frame: Dict[int, List[Dict]] = {}
    for det in detections:
        by_frame.setdefault(det["frame"], []).append(det)
    
    # Track positions for trails
    track_trails: Dict[str, List[Tuple[int, int]]] = {}
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # ROIs are intentionally not drawn on the annotated video.
        
        if frame_idx in by_frame:
            for det in by_frame[frame_idx]:
                track_id = det.get("track_id", 0)
                cx, cy = int(det["cx"]), int(det["cy"])
                w, h = int(det["w"]), int(det["h"])
                angle = det["angle"]
                
                color = track_color(track_id)
                
                # Draw OBB
                box = cv2.boxPoints(((cx, cy), (w, h), math.degrees(angle)))
                box = np.intp(box)
                cv2.drawContours(frame, [box], 0, color, box_thickness)
                
                # Draw track ID as Tank{roi}-ID{num}
                cv2.putText(frame, track_label(track_id), (cx - 20, cy - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, id_font_size, color, id_font_thickness)
                
                # Draw keypoints
                for kpt_name, kpt_color in [("head", (0, 0, 255)), ("center", (255, 0, 0)), ("tail", (0, 255, 255))]:
                    kx = int(det.get(f"{kpt_name}_x", cx))
                    ky = int(det.get(f"{kpt_name}_y", cy))
                    cv2.circle(frame, (kx, ky), keypoint_radius, kpt_color, -1)
                    cv2.putText(frame, kpt_name[0].upper(), (kx + 5, ky - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, keypoint_font_size, kpt_color, 1)
                
                # Draw heading arrow from tail to head (matches AMADEUS direction estimation).
                # Arrow length scales with the bounding box size so it stays
                # proportional across fish of different sizes.
                if "heading" in det:
                    heading_rad = math.radians(float(det["heading"]))
                    # Arrow length is a fraction of the box's long side.
                    box_len = max(float(w), float(h))
                    dyn_arrow = arrow_size * box_len / 100.0
                    # Arrow starts at tail, ends at head.
                    tx, ty = int(det.get("tail_x", cx)), int(det.get("tail_y", cy))
                    hx = int(tx + dyn_arrow * math.sin(heading_rad))
                    hy = int(ty - dyn_arrow * math.cos(heading_rad))
                    # Clamp to frame bounds
                    hx = max(0, min(width - 1, hx))
                    hy = max(0, min(height - 1, hy))
                    cv2.arrowedLine(frame, (tx, ty), (hx, hy), (255, 255, 0), 2)
                
                # Update and draw trail
                if track_id not in track_trails:
                    track_trails[track_id] = []
                track_trails[track_id].append((cx, cy))
                if len(track_trails[track_id]) > trail_length:
                    track_trails[track_id].pop(0)
                
                for j in range(1, len(track_trails[track_id])):
                    pt1 = track_trails[track_id][j-1]
                    pt2 = track_trails[track_id][j]
                    alpha = j / len(track_trails[track_id])
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, pt1, pt2, trail_color, 2)
                
                # Draw speed
                if "speed" in det:
                    cv2.putText(frame, f"v:{det['speed']:.1f}px/s", (cx - 40, cy + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(description="Fish Behavior Quantification Inference")
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    session_path = cfg.get("SESSION_PATH", "")
    video_path = cfg.get("TRACKING_VIDEO_PATH", "")
    # Resolve relative video path
    if video_path and not os.path.isabs(video_path):
        video_path = os.path.join(session_path, video_path)
    weight_selector = cfg.get("WEIGHT", "best")
    roi_sets = cfg.get("ROI_SETS", [])
    tanks = cfg.get("TANKS", roi_sets)
    num_objects = cfg.get("NUM_OBJECTS", 1)
    behavior_cfg = cfg.get("BEHAVIOR", {})
    
    # Behavior settings
    extract_head = behavior_cfg.get("EXTRACT_HEAD", True)
    extract_center = behavior_cfg.get("EXTRACT_CENTER", True)
    extract_tail = behavior_cfg.get("EXTRACT_TAIL", True)
    run_analysis = behavior_cfg.get("ANALYSIS", True)
    
    # Detection settings
    analysis = cfg.get("analysis", {}) or {}
    conf_thresh = float(analysis.get("CONF", 0.25))
    iou_thresh = float(analysis.get("NMS_IOU", 0.5))
    device = resolve_device(analysis.get("DEVICE", "auto"))
    imgsz = int(analysis.get("IMGSZ", 1920))
    batch_size_str = analysis.get("BATCH_SIZE", 8)
    batch_size = int(batch_size_str) if str(batch_size_str).isdigit() else 8
    first_frame = int(analysis.get("FIRST_FRAME", 0))
    last_frame = int(analysis.get("LAST_FRAME", -1))
    
    # Load model
    # Use WEIGHT from analysis if available, otherwise top-level
    analysis_weight = analysis.get("WEIGHT", weight_selector)
    print(f"Using weight selector: {analysis_weight}")
    
    if analysis_weight in ("best", "last"):
        # Find weight in session directory
        main_dir = os.path.join(session_path, "main")
        if os.path.exists(main_dir):
            # Find the training directory
            for root, dirs, files in os.walk(main_dir):
                for f in files:
                    if f == f"{analysis_weight}.pt":
                        model_path = os.path.join(root, f)
                        break
                else:
                    continue
                break
            else:
                # Try standard location
                model_path = os.path.join(session_path, "weights", f"{analysis_weight}.pt")
        else:
            model_path = os.path.join(session_path, "weights", f"{analysis_weight}.pt")
    else:
        model_path = analysis_weight
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Read video info
    video_info = read_video_frame_info(video_path)
    width, height = video_info["width"], video_info["height"]
    video_fps = video_info["fps"]
    
    # Build list of ROIs to track. Each ROI gets its own mask and independent
    # tracking. Track ids are prefixed with "roi{roi_idx}_" so ids are unique
    # across ROIs.
    roi_list = []
    if tanks:
        for i, tank in enumerate(tanks):
            polygon = tank.get("polygon", tank.get("points_normalized", []))
            fish_count = int(tank.get("fish_count", tank.get("n_animals", num_objects)))
            roi_list.append((i, polygon, fish_count))
    elif roi_sets:
        for i, roi in enumerate(roi_sets):
            polygon = roi.get("polygon", roi.get("points_normalized", []))
            fish_count = int(roi.get("fish_count", roi.get("n_animals", num_objects)))
            roi_list.append((i, polygon, fish_count))
    else:
        # Full frame
        roi_list.append((0, [[0, 0], [1, 0], [1, 1], [0, 1]], int(num_objects)))
    
    print(f"ROI count: {len(roi_list)}")
    
    # Output directory
    output_dir = os.path.join(session_path, "behavior_results")
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video = os.path.join(output_dir, f"{video_name}_behavior.mp4")
    output_csv = os.path.join(output_dir, f"{video_name}_behavior.csv")
    
    # Build ROI masks (pixel coordinates) - used only for assignment, not detection.
    roi_masks = []
    for roi_idx, polygon, fish_count in roi_list:
        pixel_polygon = [[int(x * width), int(y * height)] for x, y in polygon]
        roi_mask = polygon_to_mask(pixel_polygon, width, height)
        roi_masks.append((roi_idx, roi_mask, fish_count))
    
    # Output directory
    output_dir = os.path.join(session_path, "behavior_results")
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video = os.path.join(output_dir, f"{video_name}_behavior.mp4")
    output_csv = os.path.join(output_dir, f"{video_name}_behavior.csv")
    
    # STEP 1: Run detection ONCE on the full frame (no ROI mask).
    # Detections are later assigned to ROIs by checking whether their centre
    # lies inside each ROI mask. This avoids re-running the model once per ROI.
    print("Running OBB detection on full frame (once per video)...")
    all_detections = run_obb_detection(
        model=model,
        video_path=video_path,
        roi_mask=None,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        imgsz=imgsz,
        batch_size=batch_size,
        device=device,
        max_det=max(300, num_objects * 2 * len(roi_list)),
        first_frame=first_frame,
        last_frame=last_frame,
    )
    print(f"Found {len(all_detections)} total detections")
    
    # Group detections by frame for ROI assignment.
    det_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for det in all_detections:
        det_by_frame.setdefault(det["frame"], []).append(det)
    
    # STEP 2: For each ROI, filter detections by mask centre, track, collect.
    all_tracked: List[Dict[str, Any]] = []
    
    for roi_idx, roi_mask, fish_count in roi_masks:
        print(f"ROI {roi_idx}: assigning detections and tracking...")
        
        # Filter detections to this ROI (centre inside mask).
        roi_detections = []
        for frame_idx, frame_dets in det_by_frame.items():
            for det in frame_dets:
                cx, cy = int(det["cx"]), int(det["cy"])
                if 0 <= cx < width and 0 <= cy < height and roi_mask[cy, cx] > 0:
                    roi_detections.append(det)
        
        print(f"  {len(roi_detections)} detections inside ROI {roi_idx}")
        
        # Track within this ROI.
        tracked = simple_tracking(roi_detections, num_objects=fish_count)
        print(f"  Tracked {len(tracked)} detections in ROI {roi_idx}")
        
        # Prefix track ids so they are unique across ROIs.
        for det in tracked:
            det["track_id"] = f"roi{roi_idx}_{det.get('track_id', 0)}"
        
        all_tracked.extend(tracked)
    
    print(f"Total tracked detections across all ROIs: {len(all_tracked)}")
    
    print("Extracting keypoints...")
    all_tracked = extract_keypoints_for_tracks(all_tracked)
    
    if run_analysis:
        print("Computing behavior metrics...")
        all_tracked = compute_behavior_metrics(all_tracked, video_fps)
    
    print(f"Writing results to {output_csv}...")
    write_results_csv(all_tracked, output_csv)
    
    print(f"Creating output video {output_video}...")
    disp = display_config(cfg)
    create_output_video(
        video_path=video_path,
        detections=all_tracked,
        output_path=output_video,
        fps=video_fps,
        arrow_size=disp["arrow_size"],
        id_font_size=disp["id_font_size"],
        id_font_thickness=disp["id_font_thickness"],
        box_thickness=disp["box_thickness"],
        keypoint_radius=disp["keypoint_radius"],
        keypoint_font_size=disp["keypoint_font_size"],
        trail_length=disp["trail_length"],
    )
    
    print("Done!")


if __name__ == "__main__":
    main()