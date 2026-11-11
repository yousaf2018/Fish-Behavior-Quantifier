# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Behaviour analysis of the tracked fish.

The tracking stages answer *where* every fish is and *which way* it points.
This stage answers the questions a behaviour experiment asks about that
trajectory:

* how fast does the fish move, and how much of the recording is it moving?
* how sharply does it turn?
* does the body stretch while it swims, and how much?
* does it stay close to the wall of its tank?

The input is the landmark side-car written by the direction stage
(``single_animal_images/keypoints.csv``), which holds the ``head``,
``center`` and ``tail`` of every accepted fish on every analysed frame. The
identity of a fish is the trajectory id that the initial tracking stage
assigned to it, so the analysis does not depend on the detector and can run
once per configuration instead of once per learning-rate pair.

Two files are written into ``<session>/results/behavior``:

``behavior_tracks.csv``
    One row per fish per frame with the derived kinematics.

``behavior_summary.csv``
    One row per fish with the summary of the whole recording.

The stage is additive: it reads artifacts and writes new files, and it never
touches the tracking results.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from behavior_config import behavior_settings
from path_utils import resolve_config_paths
from roi import (
    POPULATION_CROWDED,
    roi_expected_animals,
    roi_population_mode,
)
from tank_roster import polygon_pixels
from arena_labels import arena_name, identity_label

#: Directory of the session that holds the per-tank behaviour results.
BEHAVIOR_DIR_NAME = "behavior"

#: Name of the landmark side-car written by the direction stage.
KEYPOINTS_CSV_NAME = "keypoints.csv"


def ensure_dirs(paths: Sequence[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping.")
    return resolve_config_paths(cfg)


def wrap_signed_deg(angle: float) -> float:
    """Return the angle in ``(-180, 180]``."""
    value = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if value == -180.0 else value


def read_keypoints(path: str) -> List[dict]:
    """Read the landmark side-car and return the accepted rows only.

    A missing file is not an error: the direction stage may have been
    skipped, and then there is nothing to analyse.
    """
    if not os.path.exists(path):
        print(f"[behavior_analysis] no landmark side-car at {path}; nothing to analyse.")
        return []
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                accepted = int(raw.get("accepted", "1") or "1")
            except ValueError:
                accepted = 1
            if not accepted:
                continue
            traj_raw = str(raw.get("traj_id", "") or "").strip()
            if not traj_raw:
                continue
            try:
                track_id = int(traj_raw)
            except ValueError:
                continue
            try:
                row = {
                    "frame": int(raw["frame"]),
                    "track_id": track_id,
                    "head_x": float(raw["head_x"]),
                    "head_y": float(raw["head_y"]),
                    "center_x": float(raw["center_x"]),
                    "center_y": float(raw["center_y"]),
                    "tail_x": float(raw["tail_x"]),
                    "tail_y": float(raw["tail_y"]),
                    "heading_deg": float(raw["heading_deg"]),
                    "body_length": float(raw["body_length"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def _tank_contour(cfg: dict, image_shape: Optional[Sequence[int]]) -> Optional[np.ndarray]:
    """Return the tank polygon in pixels, or None when the run has no tank."""
    roi_sets = cfg.get("ROI_SETS") or []
    for roi in roi_sets:
        if not roi.get("enabled", False):
            continue
        points = roi.get("points") or []
        if len(points) >= 3:
            return np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        normalized = roi.get("points_normalized") or []
        if len(normalized) >= 3 and image_shape is not None:
            pixels = polygon_pixels(normalized, image_shape)
            return np.asarray(pixels, dtype=np.int32).reshape(-1, 1, 2)
    return None


def _roi_contours(cfg: dict, image_shape: Optional[Sequence[int]]) -> List[dict]:
    """Return the polygon of every enabled ROI, in pixels."""
    contours: List[dict] = []
    for index, roi in enumerate(cfg.get("ROI_SETS") or []):
        if not roi.get("enabled", False):
            continue
        points = roi.get("points") or []
        contour = None
        if len(points) >= 3:
            contour = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        else:
            normalized = roi.get("points_normalized") or []
            if len(normalized) >= 3 and image_shape is not None:
                pixels = polygon_pixels(normalized, image_shape)
                contour = np.asarray(pixels, dtype=np.int32).reshape(-1, 1, 2)
        if contour is None:
            continue
        contours.append({
            "index": index + 1,
            "label": str(roi.get("name", "") or "").strip() or f"ROI {index + 1}",
            "contour": contour,
            "mode": roi_population_mode(roi),
            "expected": roi_expected_animals(roi),
        })
    return contours


def _track_center(track: Sequence[dict]) -> Optional[Tuple[float, float]]:
    """Return the middle position of a track, in pixels."""
    xs = [float(row["center_x"]) for row in track if "center_x" in row]
    ys = [float(row["center_y"]) for row in track if "center_y" in row]
    if not xs or not ys:
        return None
    return float(np.median(xs)), float(np.median(ys))


def roi_population_rows(
    cfg: dict, tracks: Dict[int, List[dict]], image_shape: Optional[Sequence[int]],
) -> List[dict]:
    """Compare the animals every ROI declares with the fish found in it.

    A fish that lies in two overlapping ROIs counts for both, and a fish
    outside every ROI is reported on its own row, so no fish is lost.
    """
    contours = _roi_contours(cfg, image_shape)
    if not contours:
        return []
    rows: List[dict] = []
    counts = {entry["index"]: 0 for entry in contours}
    unassigned: List[int] = []
    for track_id, track in sorted(tracks.items()):
        center = _track_center(track)
        if center is None:
            continue
        inside = False
        for entry in contours:
            if cv2.pointPolygonTest(entry["contour"], center, True) >= 0:
                counts[entry["index"]] += 1
                inside = True
        if not inside:
            unassigned.append(track_id)
    for entry in contours:
        found = counts[entry["index"]]
        expected = entry["expected"]
        rows.append({
            "roi": entry["label"],
            "mode": entry["mode"],
            "expected_animals": expected,
            "detected_animals": found,
            "difference": (found - expected) if expected > 0 else "",
            "note": _population_note(entry["mode"], expected, found),
        })
    if unassigned:
        rows.append({
            "roi": "outside every ROI",
            "mode": "",
            "expected_animals": "",
            "detected_animals": len(unassigned),
            "difference": "",
            "note": "fish that were tracked outside every declared ROI",
        })
    return rows


def _population_note(mode: str, expected: int, found: int) -> str:
    """Explain how the detected number of animals relates to the declaration."""
    if mode == POPULATION_CROWDED:
        return "crowded ROI: the number of animals is not known exactly"
    if expected <= 0:
        return "no number of animals declared"
    if found == expected:
        return "the declared number of animals was found"
    if found < expected:
        return f"{expected - found} animal(s) fewer than declared"
    return f"{found - expected} animal(s) more than declared"


def distance_to_wall(contour: Optional[np.ndarray], point: Sequence[float]) -> Optional[float]:
    """Return the distance from a point to the tank wall in pixels."""
    if contour is None:
        return None
    distance = float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))
    return abs(distance)


def group_tracks(rows: Sequence[dict]) -> Dict[int, List[dict]]:
    """Group the landmark rows by fish identity, ordered by frame."""
    tracks: Dict[int, List[dict]] = {}
    for row in rows:
        tracks.setdefault(int(row["track_id"]), []).append(row)
    for track in tracks.values():
        track.sort(key=lambda item: int(item["frame"]))
    return tracks


def analyze_track(
    track: Sequence[dict], fps: float, settings: dict, contour: Optional[np.ndarray],
) -> Tuple[List[dict], dict]:
    """Return the per-frame kinematics and the summary of one fish."""
    if not track:
        return [], {}

    frame_interval = 1
    if len(track) > 1:
        gaps = [
            int(track[index + 1]["frame"]) - int(track[index]["frame"])
            for index in range(len(track) - 1)
        ]
        gaps = [gap for gap in gaps if gap > 0]
        if gaps:
            frame_interval = int(round(sum(gaps) / len(gaps)))
    seconds_per_frame = (frame_interval / fps) if fps > 1e-9 else float(frame_interval)

    lengths = [float(row["body_length"]) for row in track]
    reference_length = float(np.median(lengths)) if lengths else 0.0

    min_speed = float(settings["MIN_SPEED_PX_S"])
    turn_threshold = float(settings["TURN_THRESHOLD_DEG"])
    scale = settings.get("SCALE_PX_PER_MM")

    rows: List[dict] = []
    path_length = 0.0
    turning_sum = 0.0
    turning_count = 0
    turning_abs_sum = 0.0
    moving_frames = 0
    speeds: List[float] = []
    elongations: List[float] = []
    wall_distances: List[float] = []
    heading_sectors = [0] * 8

    for index, row in enumerate(track):
        center = (float(row["center_x"]), float(row["center_y"]))
        heading = float(row["heading_deg"])
        length = float(row["body_length"])

        speed_px_s = 0.0
        turning_rate = None
        if index > 0:
            previous = track[index - 1]
            dx = center[0] - float(previous["center_x"])
            dy = center[1] - float(previous["center_y"])
            step = math.hypot(dx, dy)
            path_length += step
            speed_px_s = step / seconds_per_frame
            delta = wrap_signed_deg(heading - float(previous["heading_deg"]))
            turning_rate = delta / seconds_per_frame
            turning_sum += abs(turning_rate)
            turning_abs_sum += abs(turning_rate)
            turning_count += 1

        elongation = (length / reference_length) if reference_length > 1e-6 else None
        state = "moving" if speed_px_s >= min_speed else "resting"
        if state == "moving":
            moving_frames += 1

        speeds.append(speed_px_s)
        if elongation is not None:
            elongations.append(elongation)

        wall = distance_to_wall(contour, center)
        if wall is not None:
            wall_distances.append(wall)

        sector = int(((heading + 22.5) % 360.0) // 45.0) % 8
        heading_sectors[sector] += 1

        rows.append({
            "frame": int(row["frame"]),
            "track_id": int(row["track_id"]),
            "head_x": float(row["head_x"]),
            "head_y": float(row["head_y"]),
            "center_x": center[0],
            "center_y": center[1],
            "tail_x": float(row["tail_x"]),
            "tail_y": float(row["tail_y"]),
            "heading_deg": heading,
            "body_length": length,
            "elongation": elongation,
            "speed_px_s": speed_px_s,
            "speed_mm_s": (speed_px_s * scale) if scale else None,
            "body_lengths_s": (speed_px_s / reference_length) if reference_length > 1e-6 else None,
            "turning_rate_deg_s": turning_rate,
            "distance_to_wall_px": wall,
            "state": state,
        })

    first = track[0]
    last = track[-1]
    net = math.hypot(
        float(last["center_x"]) - float(first["center_x"]),
        float(last["center_y"]) - float(first["center_y"]),
    )
    duration = (len(track) - 1) * seconds_per_frame
    straightness = (net / path_length) if path_length > 1e-6 else None

    summary = {
        "track_id": int(track[0]["track_id"]),
        "frames": len(track),
        "duration_s": duration,
        "path_length_px": path_length,
        "path_length_mm": (path_length * scale) if scale else None,
        "net_displacement_px": net,
        "straightness": straightness,
        "mean_speed_px_s": (sum(speeds) / len(speeds)) if speeds else 0.0,
        "max_speed_px_s": max(speeds) if speeds else 0.0,
        "mean_turning_rate_deg_s": (turning_sum / turning_count) if turning_count else None,
        "mean_abs_turning_rate_deg_s": (turning_abs_sum / turning_count) if turning_count else None,
        "turn_events": sum(
            1 for row in rows
            if row["turning_rate_deg_s"] is not None
            and abs(row["turning_rate_deg_s"]) >= turn_threshold
        ),
        "mean_body_length_px": reference_length,
        "max_elongation": max(elongations) if elongations else None,
        "time_moving_s": moving_frames * seconds_per_frame,
        "time_resting_s": (len(track) - moving_frames) * seconds_per_frame,
        "mean_distance_to_wall_px": (sum(wall_distances) / len(wall_distances)) if wall_distances else None,
        "min_distance_to_wall_px": min(wall_distances) if wall_distances else None,
    }
    for sector in range(8):
        summary[f"heading_sector_{sector}"] = heading_sectors[sector]

    return rows, summary


def _write_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _video_info(cfg: dict) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Return the frame rate and the frame size of the analysed video."""
    candidates = (
        str(cfg.get("TRAINING_VIDEO_PATH", "") or "").strip(),
        str(cfg.get("TRACKING_VIDEO_PATH", "") or "").strip(),
    )
    for video_path in candidates:
        if not video_path or not os.path.exists(video_path):
            continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            ok, frame = cap.read()
            shape = (int(frame.shape[0]), int(frame.shape[1])) if ok and frame is not None else None
        finally:
            cap.release()
        return (fps if fps > 0.0 else 0.0), shape
    return 0.0, None


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: behavior_analysis.py CONFIG_PATH")
    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)

    session_path = str(cfg["SESSION_PATH"])
    settings = behavior_settings(cfg)

    keypoints_path = os.path.join(session_path, "single_animal_images", KEYPOINTS_CSV_NAME)
    out_dir = os.path.join(session_path, "results", BEHAVIOR_DIR_NAME)
    ensure_dirs([out_dir])

    rows = read_keypoints(keypoints_path)
    if not rows:
        print("[behavior_analysis] no accepted fish landmarks were found; nothing to analyse.")
        return

    fps, image_shape = _video_info(cfg)
    contour = _tank_contour(cfg, image_shape)

    track_rows: List[dict] = []
    summaries: List[dict] = []
    for track_id, track in sorted(group_tracks(rows).items()):
        per_frame, summary = analyze_track(track, fps, settings, contour)
        if not summary:
            continue
        track_rows.extend(per_frame)
        summaries.append(summary)

    tracks_path = os.path.join(out_dir, "behavior_tracks.csv")
    summary_path = os.path.join(out_dir, "behavior_summary.csv")
    roi_path = os.path.join(out_dir, "behavior_rois.csv")

    # Every fish is named after the arena it belongs to, so a table of one
    # arena can be read on its own.
    arena = arena_name(cfg)
    for row in track_rows:
        row["id_label"] = identity_label(arena, int(row.get("track_id", 0) or 0) + 1)
    for summary in summaries:
        summary["id_label"] = identity_label(arena, int(summary.get("track_id", 0) or 0) + 1)

    track_fields = [
        "frame", "track_id", "id_label",
        "head_x", "head_y", "center_x", "center_y", "tail_x", "tail_y",
        "heading_deg", "body_length", "elongation",
        "speed_px_s", "speed_mm_s", "body_lengths_s", "turning_rate_deg_s",
        "distance_to_wall_px", "state",
    ]
    _write_csv(tracks_path, track_rows, track_fields)

    summary_fields = [
        "track_id", "id_label", "frames", "duration_s",
        "path_length_px", "path_length_mm", "net_displacement_px", "straightness",
        "mean_speed_px_s", "max_speed_px_s",
        "mean_turning_rate_deg_s", "mean_abs_turning_rate_deg_s", "turn_events",
        "mean_body_length_px", "max_elongation",
        "time_moving_s", "time_resting_s",
        "mean_distance_to_wall_px", "min_distance_to_wall_px",
    ] + [f"heading_sector_{sector}" for sector in range(8)]
    _write_csv(summary_path, summaries, summary_fields)

    roi_rows = roi_population_rows(cfg, group_tracks(rows), image_shape)
    if roi_rows:
        roi_fields = [
            "roi", "mode", "expected_animals", "detected_animals", "difference", "note",
        ]
        _write_csv(roi_path, roi_rows, roi_fields)
        print(f"[behavior_analysis] ROI populations: {roi_path}")

    print(f"[behavior_analysis] analysed {len(summaries)} fish from {len(track_rows)} frames")
    print(f"[behavior_analysis] per-frame kinematics: {tracks_path}")
    print(f"[behavior_analysis] summary: {summary_path}")


if __name__ == "__main__":
    main()
