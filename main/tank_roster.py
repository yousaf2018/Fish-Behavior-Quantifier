# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Tank roster: the multi-ROI / multi-tank configuration model.

Labs record fish in several tanks, either with one camera per tank or with a
single camera that sees several tanks at once. FBQ handles both by giving
every tank its own *tank entry* in the configuration and running the existing
pipeline once per tank.

Configuration shape::

    TANKS:
      - id: tank_a
        name: "Tank A"
        view: top                 # top | side
        fish_count: 12
        polygon: [[0.02, 0.05], [0.48, 0.05], [0.48, 0.95], [0.02, 0.95]]
        scale_px_per_mm: 42.5
      - id: tank_b
        ...

    VIDEO_TANK_MAP:
      "rig_cam1.mp4": ["tank_a", "tank_b"]
      "rig_cam2.mp4": ["tank_c"]

Polygons are stored in **normalized** coordinates so the same tank geometry
survives a change of resolution and can be reused across recordings of the
same rig.

The roster is deliberately additive: a configuration without ``TANKS`` keeps
the single-tank behaviour and the single session layout exactly as before.
"""

from __future__ import annotations

import copy
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from roi import (
    POPULATION_CROWDED,
    POPULATION_ESTIMATED,
    POPULATION_FIXED,
    normalize_population_mode,
)


#: Sub-directory of the session that holds one sub-session per tank.
TANKS_DIR_NAME = "tanks"

#: The two camera geometries the behaviour models distinguish.
SUPPORTED_VIEWS = ("top", "side")

#: Configuration key that restricts a run to the blobs inside one polygon.
ROI_SETS_KEY = "ROI_SETS"

#: Configuration key that decides whether the tank runs happen at the same time.
TANKS_PARALLEL_KEY = "TANKS_PARALLEL"

#: Configuration key that limits how many tank runs happen at the same time.
TANKS_MAX_WORKERS_KEY = "TANKS_MAX_WORKERS"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")


class TankConfigurationError(ValueError):
    """Raised when the tank roster in the configuration is not usable."""


def _as_float(value: object, field: str, tank_id: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TankConfigurationError(
            f"Tank {tank_id!r}: {field} must be a number."
        ) from exc
    if not math.isfinite(result):
        raise TankConfigurationError(f"Tank {tank_id!r}: {field} must be finite.")
    return result


def normalize_polygon(raw: object, tank_id: str) -> List[List[float]]:
    """Validate and return the tank polygon in normalized coordinates."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        raise TankConfigurationError(
            f"Tank {tank_id!r}: polygon must list at least three vertices."
        )
    polygon: List[List[float]] = []
    for vertex in raw:
        if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
            raise TankConfigurationError(
                f"Tank {tank_id!r}: every polygon vertex needs an x and a y."
            )
        x = _as_float(vertex[0], "polygon x", tank_id)
        y = _as_float(vertex[1], "polygon y", tank_id)
        if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
            raise TankConfigurationError(
                f"Tank {tank_id!r}: polygon vertices must be normalized to 0..1."
            )
        polygon.append([x, y])
    return polygon


def normalize_tank(raw: dict, index: int) -> dict:
    """Validate one tank entry and return its normalized form."""
    if not isinstance(raw, dict):
        raise TankConfigurationError(f"Tank entry #{index + 1} must be a mapping.")

    tank_id = str(raw.get("id", "") or "").strip()
    if not tank_id:
        raise TankConfigurationError(f"Tank entry #{index + 1} needs an id.")
    if not _ID_PATTERN.match(tank_id):
        raise TankConfigurationError(
            f"Tank id {tank_id!r} may only contain letters, digits, '-' and '_'."
        )

    view = str(raw.get("view", "top") or "top").strip().lower()
    if view not in SUPPORTED_VIEWS:
        raise TankConfigurationError(
            f"Tank {tank_id!r}: view must be one of {SUPPORTED_VIEWS}, got {view!r}."
        )

    polygon = normalize_polygon(raw.get("polygon"), tank_id)

    fish_count = raw.get("fish_count", None)
    if fish_count is not None:
        fish_count = int(fish_count)
        if fish_count < 0:
            raise TankConfigurationError(f"Tank {tank_id!r}: fish_count must not be negative.")

    population = normalize_population_mode(raw.get("population", ""), fish_count or 0)

    scale = raw.get("scale_px_per_mm", None)
    if scale is not None:
        scale = _as_float(scale, "scale_px_per_mm", tank_id)
        if scale <= 0.0:
            raise TankConfigurationError(f"Tank {tank_id!r}: scale_px_per_mm must be positive.")

    return {
        "id": tank_id,
        "name": str(raw.get("name", "") or tank_id).strip() or tank_id,
        "view": view,
        "fish_count": fish_count,
        "population": population,
        "polygon": polygon,
        "scale_px_per_mm": scale,
    }


def normalize_tanks(raw: object) -> List[dict]:
    """Validate the ``TANKS`` section and return the normalized roster."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise TankConfigurationError("TANKS must be a list of tank mappings.")
    tanks = [normalize_tank(entry, index) for index, entry in enumerate(raw)]
    seen: set[str] = set()
    for tank in tanks:
        if tank["id"] in seen:
            raise TankConfigurationError(f"Duplicate tank id {tank['id']!r}.")
        seen.add(tank["id"])
    return tanks


def normalize_video_tank_map(raw: object, tanks: Sequence[dict]) -> Dict[str, List[str]]:
    """Validate the ``VIDEO_TANK_MAP`` section against the roster."""
    known = {tank["id"] for tank in tanks}
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TankConfigurationError("VIDEO_TANK_MAP must be a mapping of video name to tank ids.")
    result: Dict[str, List[str]] = {}
    for video, tank_ids in raw.items():
        if not isinstance(tank_ids, (list, tuple)):
            raise TankConfigurationError(
                f"VIDEO_TANK_MAP[{video!r}] must list the tank ids that video covers."
            )
        ids = [str(item).strip() for item in tank_ids if str(item).strip()]
        unknown = [item for item in ids if item not in known]
        if unknown:
            raise TankConfigurationError(
                f"VIDEO_TANK_MAP[{video!r}] references unknown tank ids: {', '.join(unknown)}."
            )
        result[str(video)] = ids
    return result


def polygon_pixels(polygon: Sequence[Sequence[float]], image_shape: Sequence[int]) -> np.ndarray:
    """Convert a normalized polygon into pixel coordinates for a frame."""
    height, width = int(image_shape[0]), int(image_shape[1])
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    pixels = points.copy()
    pixels[:, 0] *= max(width - 1, 0)
    pixels[:, 1] *= max(height - 1, 0)
    return pixels


def point_in_polygon(point: Sequence[float], polygon_px: np.ndarray) -> bool:
    """Return True when the pixel lies inside the polygon.

    Uses the even-odd ray casting rule so no OpenCV import is needed at
    configuration time.
    """
    x = float(point[0])
    y = float(point[1])
    points = np.asarray(polygon_px, dtype=np.float64).reshape(-1, 2)
    inside = False
    count = points.shape[0]
    if count < 3:
        return False
    j = count - 1
    for i in range(count):
        xi, yi = points[i]
        xj, yj = points[j]
        if (yi > y) != (yj > y):
            crossing = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing:
                inside = not inside
        j = i
    return inside


def roi_sets_for_tank(tank: dict, image_shape: Optional[Sequence[int]] = None) -> List[dict]:
    """Return the ROI set that selects the blobs of one tank.

    The polygon is stored in **normalized** coordinates so that the same
    tank geometry survives a change of resolution, and so that every stage
    converts it with the frame size of the video it actually reads. The
    pixel form is added when the frame size is known. The declared number of
    animals travels with the ROI so that every stage knows how many fish
    the tank holds.
    """
    normalized = [[float(x), float(y)] for x, y in tank["polygon"]]
    roi = {
        "enabled": True,
        "points_normalized": normalized,
        "frame_start": 0,
        "frame_end": -1,
        "n_animals": int(tank.get("fish_count") or 0),
        "population": tank.get("population", POPULATION_FIXED),
    }
    if image_shape is not None:
        pixels = polygon_pixels(tank["polygon"], image_shape)
        roi["points"] = [[int(round(x)), int(round(y))] for x, y in pixels]
    return [roi]


def roi_polygon_pixels(roi: dict, image_shape: Sequence[int]) -> Optional[np.ndarray]:
    """Return the pixel polygon of one ROI, or None when it has none."""
    points = roi.get("points") or []
    if len(points) >= 3:
        return np.asarray(points, dtype=np.float64).reshape(-1, 2)
    normalized = roi.get("points_normalized") or []
    if len(normalized) >= 3:
        return polygon_pixels(normalized, image_shape)
    return None


def blob_in_rois(
    center: Sequence[float], roi_sets: Sequence[dict], image_shape: Sequence[int],
) -> bool:
    """Return True when a blob center falls inside one of the active ROIs.

    An empty ROI list means "no restriction", which keeps the single-tank
    behaviour unchanged.
    """
    if not roi_sets:
        return True
    for roi in roi_sets:
        if not roi.get("enabled", False):
            continue
        polygon = roi_polygon_pixels(roi, image_shape)
        if polygon is None:
            continue
        if point_in_polygon(center, polygon):
            return True
    return False


def filter_records_by_rois(records: dict, roi_sets: Sequence[dict], image_shape: Sequence[int]) -> dict:
    """Drop every blob whose center lies outside the active ROIs.

    The segmentation pass is shared by all tanks, so the ROI restriction is
    applied to the blob records here. Without ROI sets nothing is dropped.
    """
    if not roi_sets:
        return records
    filtered: Dict[int, List[object]] = {}
    dropped = 0
    for frame, blobs in records.items():
        kept = [
            blob for blob in blobs
            if blob_in_rois(getattr(blob, "center", (0.0, 0.0)), roi_sets, image_shape)
        ]
        dropped += len(blobs) - len(kept)
        filtered[frame] = kept
    if dropped:
        print(f"[tank_roster] dropped {dropped} blobs outside the active tank ROI(s)")
    return filtered


def tank_session_path(session_path: str, tank_id: str) -> str:
    """Return the sub-session directory that holds one tank's results."""
    return os.path.join(str(session_path), TANKS_DIR_NAME, str(tank_id))


def tank_results_path(session_path: str, tank_id: str) -> str:
    """Return the results directory of one tank."""
    return os.path.join(tank_session_path(session_path, tank_id), "results")


def videos_for_tank(tank: dict, video_tank_map: Dict[str, List[str]], videos: Sequence[str]) -> List[str]:
    """Return the videos that cover one tank.

    A tank that no video claims belongs to every listed video, which is the
    single-tank-per-video case the labs use most.
    """
    claimed = [name for name, ids in video_tank_map.items() if tank["id"] in ids]
    if not claimed:
        return list(videos)
    matched = [video for video in videos if os.path.basename(video) in claimed]
    return matched or list(videos)


def build_tank_config(
    base_cfg: dict, tank: dict, session_path: str, video: Optional[str],
) -> dict:
    """Derive the configuration that runs the pipeline for one tank.

    The derived configuration keeps every analysis parameter and only
    redirects the session, the fish count and the ROI restriction. The
    roster itself is removed, so a derived configuration describes exactly
    one tank and never starts a second round of tank runs.
    """
    config = copy.deepcopy(base_cfg)
    config["SESSION_PATH"] = tank_session_path(session_path, tank["id"])
    config["TANK_ID"] = tank["id"]
    config["TANK_NAME"] = tank["name"]
    config["TANK_VIEW"] = tank["view"]
    if tank.get("scale_px_per_mm"):
        config["TANK_SCALE_PX_PER_MM"] = float(tank["scale_px_per_mm"])
    if tank.get("fish_count") is not None and int(tank["fish_count"]) > 0:
        config["NUM_OBJECTS"] = int(tank["fish_count"])
    if tank.get("population", POPULATION_FIXED) != POPULATION_FIXED:
        # A crowded tank has no exact population, so the run must not force
        # a fixed number of identities.
        config["VARIABLE_NUM_OBJECTS"] = True
    if video:
        config["TRACKING_VIDEO_PATH"] = video
        config["TRACKING_VIDEO_FILES"] = [video]
    # The polygon stays normalized so that every stage converts it with the
    # frame size of the video it actually reads.
    config[ROI_SETS_KEY] = roi_sets_for_tank(tank)
    config.pop("TANKS", None)
    config.pop("VIDEO_TANK_MAP", None)
    return config


def collect_video_tank_pairs(
    cfg: dict, videos: Sequence[str],
) -> List[Tuple[dict, List[str]]]:
    """Return every ``(tank, videos)`` pair the roster describes."""
    tanks = normalize_tanks(cfg.get("TANKS"))
    if not tanks:
        return []
    video_tank_map = normalize_video_tank_map(cfg.get("VIDEO_TANK_MAP"), tanks)
    return [(tank, videos_for_tank(tank, video_tank_map, videos)) for tank in tanks]


def tank_parallelism(cfg: dict, run_count: int) -> Tuple[bool, int]:
    """Return whether the tank runs run at the same time and how many at once.

    ``TANKS_PARALLEL`` switches the simultaneous mode on (the default) and
    ``TANKS_MAX_WORKERS`` limits how many tanks are processed at once; zero
    means one worker per tank, so every ROI is processed at the same time.
    """
    if run_count <= 1:
        return False, 1
    parallel = cfg.get(TANKS_PARALLEL_KEY, True)
    if isinstance(parallel, str):
        parallel = parallel.strip().lower() in {"1", "true", "yes", "on"}
    parallel = bool(parallel)
    try:
        max_workers = int(cfg.get(TANKS_MAX_WORKERS_KEY, 0) or 0)
    except (TypeError, ValueError):
        max_workers = 0
    if max_workers <= 0:
        max_workers = run_count
    return parallel, max(1, min(max_workers, run_count))


def iter_tank_runs(
    cfg: dict, videos: Sequence[str],
) -> Iterable[Tuple[dict, dict, Optional[str]]]:
    """Iterate the tank runs of a configuration, one per tank and video."""
    for tank, tank_videos in collect_video_tank_pairs(cfg, videos):
        for video in (tank_videos or [None]):
            derived = build_tank_config(
                cfg, tank, str(cfg.get("SESSION_PATH", "")), video,
            )
            yield tank, derived, video
