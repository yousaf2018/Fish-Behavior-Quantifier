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


DIRECTION_CLASS_NAMES = [
    "upper", "upper_right", "right", "lower_right",
    "lower", "lower_left", "left", "upper_left"
]

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


def compute_heading(keypoints: Dict[str, Tuple[float, float]]) -> float:
    """Compute heading angle from head to tail."""
    hx, hy = keypoints["head"]
    tx, ty = keypoints["tail"]
    return math.atan2(ty - hy, tx - hx)


def compute_speed(prev_center: Tuple[float, float], curr_center: Tuple[float, float], fps: float) -> float:
    """Compute speed in pixels per second."""
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    return math.hypot(dx, dy) * fps


def run_obb_detection(
    model: YOLO,
    video_path: str,
    roi_mask: np.ndarray,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
    imgsz: int = 1920,
    batch_size: int = 8,
    device: str = "cuda:0",
    max_det: int = 300,
    first_frame: int = 0,
    last_frame: int = -1,
) -> List[Dict[str, Any]]:
    """Run OBB detection on video frames within ROI."""
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
    
    for frame_idx in range(first_frame, last_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply ROI mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
        
        frame_buffer.append(masked_frame)
        frame_indices.append(frame_idx)
        
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
                        
                        # Check if center is inside ROI
                        if roi_mask[int(cy), int(cx)] > 0:
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
    """Add keypoints (head, center, tail) to each tracked detection."""
    for det in tracked_detections:
        obb = [det["cx"], det["cy"], det["w"], det["h"], det["angle"]]
        kpts = obb_to_keypoints(obb)
        det["head_x"], det["head_y"] = kpts["head"]
        det["center_x"], det["center_y"] = kpts["center"]
        det["tail_x"], det["tail_y"] = kpts["tail"]
        det["heading"] = compute_heading(kpts)
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
                    angular_vel = (heading - prev_heading) * fps
            
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
    roi_mask: np.ndarray,
    fps: float,
    draw_keypoints: bool = True,
    draw_trails: bool = True,
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
    track_trails: Dict[int, List[Tuple[int, int]]] = {}
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw ROI outline
        roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, roi_contours, -1, (0, 255, 255), 2)
        
        if frame_idx in by_frame:
            for det in by_frame[frame_idx]:
                track_id = det.get("track_id", 0)
                cx, cy = int(det["cx"]), int(det["cy"])
                w, h = int(det["w"]), int(det["h"])
                angle = det["angle"]
                
                # Draw OBB
                box = cv2.boxPoints(((cx, cy), (w, h), math.degrees(angle)))
                box = np.intp(box)
                color = (0, 255, 0) if track_id > 0 else (128, 128, 128)
                cv2.drawContours(frame, [box], 0, color, 2)
                
                # Draw track ID
                cv2.putText(frame, f"ID:{track_id}", (cx - 20, cy - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw keypoints
                if draw_keypoints:
                    for kpt_name, color in [("head", (0, 0, 255)), ("center", (255, 0, 0)), ("tail", (0, 255, 255))]:
                        kx = int(det.get(f"{kpt_name}_x", cx))
                        ky = int(det.get(f"{kpt_name}_y", cy))
                        cv2.circle(frame, (kx, ky), 4, color, -1)
                        cv2.putText(frame, kpt_name[0].upper(), (kx + 5, ky - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw heading arrow
                if "heading" in det:
                    hx = int(det["center_x"] + 30 * math.cos(det["heading"]))
                    hy = int(det["center_y"] + 30 * math.sin(det["heading"]))
                    cv2.arrowedLine(frame, (cx, cy), (hx, hy), (255, 255, 0), 2)
                
                # Update and draw trail
                if draw_trails:
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
    
    # Create ROI mask from first tank/ROI
    # Handle both TANKS and ROI_SETS formats
    polygon = []
    if tanks:
        tank = tanks[0]
        polygon = tank.get("polygon", tank.get("points_normalized", []))
        num_objects = tank.get("fish_count", tank.get("n_animals", num_objects))
    elif roi_sets:
        roi = roi_sets[0]
        polygon = roi.get("polygon", roi.get("points_normalized", []))
        num_objects = roi.get("fish_count", roi.get("n_animals", num_objects))
    else:
        # Full frame
        polygon = [[0, 0], [1, 0], [1, 1], [0, 1]]
    
    print(f"Polygon points: {len(polygon)}")
    print(f"Num objects: {num_objects}")
    
    # Convert normalized polygon to pixel coordinates
    pixel_polygon = [[int(x * width), int(y * height)] for x, y in polygon]
    print(f"Pixel polygon: {pixel_polygon}")
    roi_mask = polygon_to_mask(pixel_polygon, width, height)
    
    # Output directory
    output_dir = os.path.join(session_path, "behavior_results")
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video = os.path.join(output_dir, f"{video_name}_behavior.mp4")
    output_csv = os.path.join(output_dir, f"{video_name}_behavior.csv")
    
    print(f"Running OBB detection on {video_path}...")
    detections = run_obb_detection(
        model=model,
        video_path=video_path,
        roi_mask=roi_mask,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        imgsz=imgsz,
        batch_size=batch_size,
        device=device,
        max_det=max(300, num_objects * 2),
        first_frame=first_frame,
        last_frame=last_frame,
    )
    print(f"Found {len(detections)} detections")
    
    print("Running tracking...")
    tracked = simple_tracking(detections, num_objects=num_objects)
    print(f"Tracked {len(tracked)} detections")
    
    print("Extracting keypoints...")
    tracked = extract_keypoints_for_tracks(tracked)
    
    if run_analysis:
        print("Computing behavior metrics...")
        tracked = compute_behavior_metrics(tracked, video_fps)
    
    print(f"Writing results to {output_csv}...")
    write_results_csv(tracked, output_csv)
    
    print(f"Creating output video {output_video}...")
    create_output_video(
        video_path=video_path,
        detections=tracked,
        output_path=output_video,
        roi_mask=roi_mask,
        fps=video_fps,
    )
    
    print("Done!")


if __name__ == "__main__":
    main()