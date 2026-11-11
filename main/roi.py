# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Polygon ROI settings and masks shared by preview and processing.

Image libraries are imported only when a mask is requested. Importing this
module during GUI startup does not load OpenCV, NumPy, or the segmentation core.
"""

from __future__ import annotations

import math

#: The shape a ROI was drawn with. The geometry itself always lives in
#: ``points``, so every shape can be turned into a mask the same way.
ROI_SHAPE_RECTANGLE = "rectangle"
ROI_SHAPE_CIRCLE = "circle"
ROI_SHAPE_POLYGON = "polygon"
ROI_SHAPE_GRID = "grid"

#: Every shape a ROI can be drawn with in the user interface.
ROI_SHAPES = (ROI_SHAPE_RECTANGLE, ROI_SHAPE_CIRCLE, ROI_SHAPE_POLYGON, ROI_SHAPE_GRID)

#: The number of vertices a circle is approximated with.
CIRCLE_VERTEX_COUNT = 48


def default_roi_set() -> dict:
    return {
        "enabled": False,
        "name": "",
        "shape": ROI_SHAPE_RECTANGLE,
        "points": [],
        "frame_start": 0,
        "frame_end": -1,
        "n_animals": 0,
        "population": POPULATION_CROWDED,
    }


def initial_roi_points(image_shape: tuple) -> list[list[int]]:
    """A centered rectangle spanning half the image width and height."""
    height, width = image_shape[:2]
    x0, x1 = round((width - 1) * 0.25), round((width - 1) * 0.75)
    y0, y1 = round((height - 1) * 0.25), round((height - 1) * 0.75)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def rectangle_points(x0: float, y0: float, x1: float, y1: float) -> list[list[int]]:
    """Return the four corners of a rectangle, in any order of the corners."""
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    return [
        [round(left), round(top)],
        [round(right), round(top)],
        [round(right), round(bottom)],
        [round(left), round(bottom)],
    ]


def circle_points(center_x: float, center_y: float, radius: float) -> list[list[int]]:
    """Return a circle as a polygon of evenly spaced vertices."""
    radius = max(0.0, float(radius))
    points: list[list[int]] = []
    for index in range(CIRCLE_VERTEX_COUNT):
        angle = 2.0 * math.pi * index / CIRCLE_VERTEX_COUNT
        points.append([
            round(float(center_x) + radius * math.cos(angle)),
            round(float(center_y) + radius * math.sin(angle)),
        ])
    return points


def grid_cell_points(
    box: list[list[int]], columns: int, rows: int,
) -> list[list[list[int]]]:
    """Split a rectangle into a grid of cells and return their corners.

    The cells are returned row by row, from the top left cell to the bottom
    right one, so a grid of arenas always numbers in a predictable order.
    """
    columns = max(1, int(columns))
    rows = max(1, int(rows))
    x0, y0 = min(point[0] for point in box), min(point[1] for point in box)
    x1, y1 = max(point[0] for point in box), max(point[1] for point in box)
    cell_width = (x1 - x0) / columns
    cell_height = (y1 - y0) / rows
    cells: list[list[list[int]]] = []
    for row in range(rows):
        for column in range(columns):
            left = x0 + column * cell_width
            top = y0 + row * cell_height
            cells.append(rectangle_points(left, top, left + cell_width, top + cell_height))
    return cells


def normalize_roi_shape(value: object) -> str:
    """Return a valid ROI shape name."""
    shape = str(value or "").strip().lower()
    if shape not in ROI_SHAPES:
        shape = ROI_SHAPE_RECTANGLE
    return shape


def normalize_roi_name(value: object, fallback: str = "") -> str:
    """Return the display name of a ROI."""
    name = str(value or "").strip()
    if not name:
        return fallback
    return name[:64]


def normalize_roi_set(roi: dict, image_shape: tuple | None = None) -> dict:
    points = roi.get("points", [])
    if not isinstance(points, (list, tuple)) or (len(points) != 0 and len(points) < 3):
        raise ValueError("Each ROI must have at least three polygon vertices, or no vertices yet.")
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each ROI vertex must contain x and y.")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("ROI vertex coordinates must be finite.")
        x, y = round(x), round(y)
        if image_shape is not None:
            height, width = image_shape[:2]
            x, y = max(0, min(width - 1, x)), max(0, min(height - 1, y))
        normalized.append([x, y])
    n_animals = _animal_count(roi.get("n_animals", 0))
    return {
        "enabled": bool(roi.get("enabled", False)),
        "name": normalize_roi_name(roi.get("name", "")),
        "shape": normalize_roi_shape(roi.get("shape", ROI_SHAPE_RECTANGLE)),
        "points": normalized,
        "frame_start": int(roi.get("frame_start", 0)),
        "frame_end": int(roi.get("frame_end", -1)),
        "n_animals": n_animals,
        "population": normalize_population_mode(roi.get("population", ""), n_animals),
    }


def _animal_count(value: object) -> int:
    """Return the number of animals declared for one ROI.

    Zero means "not declared"; the count is a hint, not a hard limit.
    """
    try:
        count = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, count)


def roi_animal_counts(roi_sets: list) -> list[int]:
    """Return the declared animal count of every enabled ROI, in ROI order."""
    return [
        _animal_count(roi.get("n_animals", 0))
        for roi in roi_sets
        if roi.get("enabled", False)
    ]


#: The ROI holds exactly the declared number of animals.
POPULATION_FIXED = "fixed"

#: The ROI holds about the declared number of animals; more or fewer are possible.
POPULATION_ESTIMATED = "estimated"

#: The ROI is crowded; the number of animals is unknown and may be large.
POPULATION_CROWDED = "crowded"

#: Every population mode a ROI can declare.
POPULATION_MODES = (POPULATION_FIXED, POPULATION_ESTIMATED, POPULATION_CROWDED)


def normalize_population_mode(value: object, n_animals: int = 0) -> str:
    """Return a valid population mode for a ROI.

    A ROI that declares no count cannot have a fixed population, and a ROI
    that declares a count without a mode is treated as an estimate.
    """
    mode = str(value or "").strip().lower()
    if mode not in POPULATION_MODES:
        mode = POPULATION_FIXED if n_animals > 0 else POPULATION_CROWDED
    if mode == POPULATION_FIXED and n_animals <= 0:
        # A fixed population without a number is a contradiction: the number
        # of animals is unknown, so the ROI is treated as crowded.
        return POPULATION_CROWDED
    return mode


def roi_population_mode(roi: dict) -> str:
    """Return the population mode of one ROI."""
    return normalize_population_mode(
        roi.get("population", ""),
        _animal_count(roi.get("n_animals", 0)),
    )


def roi_expected_animals(roi: dict) -> int:
    """Return the number of animals a ROI declares, zero when it is unknown."""
    return _animal_count(roi.get("n_animals", 0))


def roi_display_name(roi: dict, index: int = 0) -> str:
    """Return the name a ROI is shown with, falling back to its number."""
    name = str(roi.get("name", "") or "").strip()
    return name or f"ROI {index + 1}"


def roi_population_model(roi_sets: list) -> dict:
    """Describe the population that the enabled ROIs declare.

    The model tells the pipeline how many animals to expect and whether that
    number is exact: a crowded or estimated ROI means the population is not
    known, so the run must not force an exact number of identities.
    """
    rois: list[dict] = []
    total_expected = 0
    fixed_total = 0
    variable = False
    for index, roi in enumerate(roi_sets or []):
        if not roi.get("enabled", False):
            continue
        expected = roi_expected_animals(roi)
        mode = roi_population_mode(roi)
        if mode != POPULATION_FIXED:
            variable = True
        total_expected += expected
        if mode == POPULATION_FIXED:
            fixed_total += expected
        rois.append({
            "index": index + 1,
            "name": roi_display_name(roi, index),
            "mode": mode,
            "expected": expected,
            "label": roi_display_name(roi, index),
        })
    return {
        "rois": rois,
        "total_expected": total_expected,
        "fixed_total": fixed_total,
        "variable": variable,
        "count": len(rois),
    }


def total_expected_animals(roi_sets: list) -> int:
    """Return the total number of animals declared across the enabled ROIs.

    The sum is the expected population of a run whose ROIs partition the
    tank, which is what the tracker is told to expect.
    """
    return sum(roi_animal_counts(roi_sets))


def describe_roi_population(roi_sets: list) -> str:
    """Return a one-line description of the population the ROIs declare."""
    model = roi_population_model(roi_sets)
    if not model["rois"]:
        return "no ROI declares a population"
    parts = [
        f"{roi['label']} {roi['expected']} ({roi['mode']})"
        for roi in model["rois"]
    ]
    return ", ".join(parts) + f"; total {model['total_expected']}"


def load_roi_settings(settings: dict, image_shape: tuple | None = None) -> tuple[list, bool]:
    """Read the current polygon ROI settings."""
    raw_sets = settings.get("roi_sets", [])
    if not isinstance(raw_sets, list) or any(not isinstance(roi, dict) for roi in raw_sets):
        raise ValueError("settings.roi_sets must be a list of ROI objects.")
    roi_sets = [normalize_roi_set(roi, image_shape) for roi in raw_sets]
    return roi_sets or [default_roi_set()], bool(settings.get("roi_reverse", False))


def roi_is_active(roi: dict, frame_idx: int | None) -> bool:
    if not roi.get("enabled", False):
        return False
    if frame_idx is None:
        return True
    start, end = int(roi.get("frame_start", 0)), int(roi.get("frame_end", -1))
    return frame_idx >= start and (end < 0 or frame_idx <= end)


def roi_signature(roi_sets: list, reverse: bool) -> tuple:
    return (
        bool(reverse),
        tuple((bool(roi["enabled"]), tuple(tuple(point) for point in roi["points"]),
               int(roi["frame_start"]), int(roi["frame_end"]),
               str(roi.get("name", "")), int(roi.get("n_animals", 0)),
               str(roi.get("population", ""))) for roi in roi_sets),
    )


def build_single_roi_mask(image_shape: tuple, roi: dict):
    import cv2
    import numpy as np

    points = roi.get("points", [])
    if not points:
        return None
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    vertices = np.asarray(points, dtype=np.int32)
    cv2.fillPoly(mask, [vertices], 255)
    return mask


def build_roi_mask_for_frame(roi_sets: list, image_shape: tuple, frame_idx: int | None, reverse: bool = False):
    import cv2

    combined = None
    for roi in roi_sets:
        if not roi_is_active(roi, frame_idx):
            continue
        partial = build_single_roi_mask(image_shape, roi)
        if partial is not None:
            combined = partial if combined is None else cv2.bitwise_or(combined, partial)
    if combined is not None and reverse:
        combined = cv2.bitwise_not(combined)
    return combined


def contour_mask_region(contour, image_shape: tuple):
    """Filled contour pixels in their bounding box, clipped to the image."""
    import cv2
    import numpy as np

    x, y, w, h = cv2.boundingRect(contour)
    H, W = image_shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x+w), min(H, y+h)
    if x0 >= x1 or y0 >= y1:
        return 0, 0, np.zeros((0, 0), dtype=np.uint8)
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED, offset=(-x0, -y0))
    return x0, y0, mask


def convex_polygon_mask_region(poly, image_shape: tuple):
    """Rasterize a globally rounded polygon before cropping it to any blob.

    Drawing directly into a smaller blob window changes OpenCV's edge clipping
    and can change the intersection by a few pixels.
    """
    import cv2
    import numpy as np

    H, W = image_shape[:2]
    x0 = max(0, math.floor(float(poly[:, 0].min())))
    y0 = max(0, math.floor(float(poly[:, 1].min())))
    x1 = min(W, math.ceil(float(poly[:, 0].max())) + 1)
    y1 = min(H, math.ceil(float(poly[:, 1].max())) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0, 0, np.zeros((0, 0), dtype=np.uint8)
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    vertices = np.round(poly).astype(np.int32) - np.array((x0, y0), dtype=np.int32)
    cv2.fillConvexPoly(mask, vertices, 255)
    return x0, y0, mask


def mask_region_overlap(first, second) -> int:
    """Count shared pixels by intersecting two already rasterized regions."""
    import cv2

    ax, ay, a = first
    bx, by, b = second
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax+a.shape[1], bx+b.shape[1]), min(ay+a.shape[0], by+b.shape[0])
    if x0 >= x1 or y0 >= y1:
        return 0
    return cv2.countNonZero(a[y0-ay:y1-ay, x0-ax:x1-ax] & b[y0-by:y1-by, x0-bx:x1-bx])
