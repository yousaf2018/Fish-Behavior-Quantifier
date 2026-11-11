# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Head / center / tail keypoint geometry for fish bodies.

The OBB pipeline represents a fish as one oriented rectangle whose long axis
has no intrinsic direction; deciding which end is the head is a downstream
heuristic. This module produces the three ordered body landmarks FBQ needs:

    head    - the snout tip
    center  - the body centroid (not the midpoint of head and tail)
    tail    - the caudal fin tip

All functions are pure geometry on a contour or a filled mask. They carry no
state and no I/O so they can be reused by the direction stage, by the
self-supervised synthesis stage and by the behaviour analysis stage.

Angles use the same convention as the rest of the pipeline: the heading is
measured in degrees from the upwards image direction (``-y``) towards ``+x``,
so ``0`` points up and the angle grows clockwise on screen. That makes
``heading_deg`` directly comparable with the ``heading`` column of the
result CSV and with the eight direction classes of the OBB path.

The head/tail order is decided by two independent cues:

1. **Width profile.** Fish bodies are widest near the head and taper towards
   the tail, so the inscribed radius measured along the body axis is
   normally larger at the head end.
2. **Motion.** When the animal is moving, the end that leads the trajectory
   is the head. This is the same motion vote the OBB pipeline already uses to
   orient its axis, applied to the point order instead of the axis sign.

Neither cue is trusted on its own. The caller decides which one to use; see
:func:`order_keypoints_by_motion`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from obb_fitting import fit_obb, normalize_obb_fit_mode


#: Number of samples taken along the body axis when building the width
#: profile. The profile only needs to distinguish a blunt end from a tapered
#: one, so a coarse sampling is enough and keeps the cost negligible.
PROFILE_SAMPLES = 24

#: Minimum number of contour points required to place body landmarks.
MIN_CONTOUR_POINTS = 5


@dataclass(frozen=True)
class BodyKeypoints:
    """Three ordered body landmarks plus the quantities derived from them."""

    head: Tuple[float, float]
    center: Tuple[float, float]
    tail: Tuple[float, float]
    #: Continuous head direction in degrees, measured from the upwards axis.
    heading_deg: float
    length: float
    #: Signed distance along the body axis, positive towards the head.
    head_asymmetry: float
    #: Relative confidence in [0, 1]; low when the width profile is flat.
    confidence: float
    #: Unit vector of the body axis the landmarks were fitted against.
    axis: Tuple[float, float] = (1.0, 0.0)
    #: ``+1`` when the head sits at the ``+axis`` end, ``-1`` otherwise.
    head_sign: float = 1.0

    def as_dict(self) -> dict:
        return {
            "head_x": float(self.head[0]),
            "head_y": float(self.head[1]),
            "center_x": float(self.center[0]),
            "center_y": float(self.center[1]),
            "tail_x": float(self.tail[0]),
            "tail_y": float(self.tail[1]),
            "heading_deg": float(self.heading_deg),
            "body_length": float(self.length),
            "head_asymmetry": float(self.head_asymmetry),
            "confidence": float(self.confidence),
        }

    def with_head_sign(self, sign: float) -> "BodyKeypoints":
        """Return the landmarks with head and tail exchanged when ``sign`` < 0."""
        sign = 1.0 if float(sign) >= 0.0 else -1.0
        if sign > 0.0:
            return self
        head = self.tail
        tail = self.head
        heading = heading_from_head_center(head, self.center)
        if heading is None:
            heading = self.heading_deg
        return BodyKeypoints(
            head=head,
            center=self.center,
            tail=tail,
            heading_deg=float(heading % 360.0),
            length=self.length,
            head_asymmetry=-self.head_asymmetry,
            confidence=self.confidence,
            axis=self.axis,
            head_sign=-self.head_sign,
        )


def unit_vector(x: float, y: float) -> Optional[Tuple[float, float]]:
    """Return the normalized ``(x, y)`` vector, or None when it is degenerate."""
    norm = math.hypot(float(x), float(y))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    return (float(x) / norm, float(y) / norm)


def heading_from_head_center(
    head: Tuple[float, float], center: Tuple[float, float],
) -> Optional[float]:
    """Return the head direction in degrees, measured from the upwards axis.

    The angle is continuous and lives in ``[0, 360)``: ``0`` points up and
    the angle grows clockwise on screen, exactly like the direction classes
    of the OBB path. ``None`` is returned when the head and the center
    coincide, which carries no direction.
    """
    dx = float(head[0]) - float(center[0])
    dy = float(head[1]) - float(center[1])
    if not math.isfinite(dx) or not math.isfinite(dy):
        return None
    if math.hypot(dx, dy) <= 1e-9:
        return None
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def heading_to_unit_vector(heading_deg: float) -> Optional[Tuple[float, float]]:
    """Return the image-space unit vector of a heading in degrees.

    This is the inverse of :func:`heading_from_head_center`, so a heading
    read from a result CSV can be turned back into the direction vector the
    association stage works with.
    """
    try:
        angle = float(heading_deg)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(angle):
        return None
    radians = math.radians(angle % 360.0)
    return (math.sin(radians), -math.cos(radians))


def _filled_mask(contour: np.ndarray, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Rasterize a contour into a filled binary mask."""
    pts = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    if shape is None:
        x, y, w, h = cv2.boundingRect(pts)
        shape = (h + 2, w + 2)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [pts], -1, 255, cv2.FILLED, offset=(1, 1))
    return mask


def _body_axis(contour: np.ndarray, obb_fit_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return the ``(center, unit_axis)`` of the fitted body axis."""
    fit_mode = normalize_obb_fit_mode(obb_fit_mode)
    rect, fitted_axis = fit_obb(np.asarray(contour, dtype=np.float32).reshape(-1, 2), fit_mode)
    (cx, cy), (w, h), angle = rect
    if fit_mode == "pca":
        axis = np.asarray(fitted_axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            axis = np.asarray([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        else:
            axis = axis / norm
    else:
        axis_angle = float(angle) if float(w) >= float(h) else float(angle) + 90.0
        axis = np.asarray([math.cos(math.radians(axis_angle)), math.sin(math.radians(axis_angle))])
    return np.asarray([float(cx), float(cy)], dtype=np.float64), axis


def _width_profile(
    mask: np.ndarray, origin: np.ndarray, axis: np.ndarray,
    samples: int = PROFILE_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Measure the inscribed radius along the body axis.

    Returns ``(positions, radii)`` where ``positions`` are the signed
    distances from ``origin`` along ``axis`` and ``radii`` the distance
    transform value at each sample. The profile is what distinguishes a
    blunt head end from a tapered tail end.
    """
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(distance > 0)
    if xs.size == 0:
        return np.zeros(0), np.zeros(0)
    points = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    # The mask origin is offset by one pixel relative to the contour.
    offsets = points - (origin + 1.0)
    projections = offsets @ axis
    if projections.size == 0:
        return np.zeros(0), np.zeros(0)
    low = float(projections.min())
    high = float(projections.max())
    if not math.isfinite(low) or not math.isfinite(high) or high - low <= 1e-6:
        return np.zeros(0), np.zeros(0)
    positions = np.linspace(low, high, samples)
    radii = np.zeros(samples, dtype=np.float64)
    for index, position in enumerate(positions):
        target = origin + axis * float(position) + 1.0
        column = int(round(target[0]))
        row = int(round(target[1]))
        if 0 <= row < distance.shape[0] and 0 <= column < distance.shape[1]:
            radii[index] = float(distance[row, column])
    return positions, radii


def _extreme_points(
    contour: np.ndarray, origin: np.ndarray, axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the contour points furthest along ``+axis`` and ``-axis``."""
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    projections = (pts - origin) @ axis
    forward = pts[int(np.argmax(projections))]
    backward = pts[int(np.argmin(projections))]
    return forward, backward


def estimate_body_keypoints(
    contour: np.ndarray,
    *,
    obb_fit_mode: str = "min_area",
    mask: Optional[np.ndarray] = None,
) -> Optional[BodyKeypoints]:
    """Estimate the ``head`` / ``center`` / ``tail`` landmarks of one body.

    The head is placed at the end of the body axis whose width profile is
    wider, which is the blunt end for nearly every fish. The centre is the
    centroid of the filled body, which stays on the animal even when the body
    is curved. The tail is the opposite end.

    Returns ``None`` when the contour is too small or degenerate.
    """
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < MIN_CONTOUR_POINTS:
        return None

    origin, axis = _body_axis(contour, obb_fit_mode)
    forward, backward = _extreme_points(pts, origin, axis)

    body_mask = mask if mask is not None else _filled_mask(contour)
    positions, radii = _width_profile(body_mask, origin, axis)

    if radii.size >= 4:
        half = radii.size // 2
        forward_radius = float(radii[half:].mean())
        backward_radius = float(radii[:half].mean())
        total = forward_radius + backward_radius
        # A flat profile means neither end is clearly blunter.
        confidence = 0.0 if total <= 1e-6 else abs(forward_radius - backward_radius) / total
        head_is_forward = forward_radius >= backward_radius
    else:
        forward_radius = backward_radius = 0.0
        confidence = 0.0
        head_is_forward = True

    head = forward if head_is_forward else backward
    tail = backward if head_is_forward else forward

    # The centre is the centroid of the filled body, not the midpoint of the
    # two extremes: a curved body has its midpoint off the animal.
    moments = cv2.moments(body_mask)
    if abs(moments["m00"]) > 1e-6:
        center = (
            float(moments["m10"] / moments["m00"]) - 1.0,
            float(moments["m01"] / moments["m00"]) - 1.0,
        )
    else:
        center = ((float(head[0]) + float(tail[0])) / 2.0, (float(head[1]) + float(tail[1])) / 2.0)

    length = float(math.hypot(float(head[0]) - float(tail[0]), float(head[1]) - float(tail[1])))
    heading = heading_from_head_center((float(head[0]), float(head[1])), center)
    if heading is None:
        return None

    signed = (np.asarray(head, dtype=np.float64) - np.asarray(center, dtype=np.float64)) @ axis
    head_asymmetry = float(signed) / max(length, 1e-6)

    return BodyKeypoints(
        head=(float(head[0]), float(head[1])),
        center=(float(center[0]), float(center[1])),
        tail=(float(tail[0]), float(tail[1])),
        heading_deg=float(heading),
        length=length,
        head_asymmetry=head_asymmetry,
        confidence=float(min(max(confidence, 0.0), 1.0)),
        axis=(float(axis[0]), float(axis[1])),
        head_sign=1.0 if head_is_forward else -1.0,
    )


def order_keypoints_by_motion(
    keypoints: BodyKeypoints, motion: Tuple[float, float],
) -> BodyKeypoints:
    """Return the same landmarks with head and tail ordered by the motion cue.

    The leading end of the trajectory is the head. This is the same
    motion vote the OBB path uses to orient its axis, applied to the point
    order. When the motion is too small to decide, the landmarks are
    returned unchanged.
    """
    dx = float(motion[0])
    dy = float(motion[1])
    if not math.isfinite(dx) or not math.isfinite(dy):
        return keypoints
    if math.hypot(dx, dy) <= 1e-6:
        return keypoints

    to_head = (keypoints.head[0] - keypoints.center[0]) * dx + (keypoints.head[1] - keypoints.center[1]) * dy
    if to_head >= 0.0:
        return keypoints

    # The head is behind the motion: swap the two ends.
    swapped = keypoints.with_head_sign(-1.0)
    heading = heading_from_head_center(swapped.head, swapped.center)
    if heading is None:
        return keypoints
    return BodyKeypoints(
        head=swapped.head,
        center=swapped.center,
        tail=swapped.tail,
        heading_deg=float(heading),
        length=swapped.length,
        head_asymmetry=swapped.head_asymmetry,
        confidence=swapped.confidence,
        axis=swapped.axis,
        head_sign=swapped.head_sign,
    )


def rotate_keypoints(
    keypoints: BodyKeypoints, angle_deg: float, pivot: Tuple[float, float],
) -> BodyKeypoints:
    """Rotate the landmarks around ``pivot`` by ``angle_deg``."""
    radians = math.radians(float(angle_deg))
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    px, py = float(pivot[0]), float(pivot[1])

    def rotate(point: Tuple[float, float]) -> Tuple[float, float]:
        dx = float(point[0]) - px
        dy = float(point[1]) - py
        return (px + dx * cos_a - dy * sin_a, py + dx * sin_a + dy * cos_a)

    head = rotate(keypoints.head)
    tail = rotate(keypoints.tail)
    center = rotate(keypoints.center)
    heading = heading_from_head_center(head, center)
    if heading is None:
        heading = keypoints.heading_deg
    axis = (keypoints.axis[0] * cos_a - keypoints.axis[1] * sin_a,
            keypoints.axis[0] * sin_a + keypoints.axis[1] * cos_a)
    return BodyKeypoints(
        head=head,
        center=center,
        tail=tail,
        heading_deg=float(heading % 360.0),
        length=keypoints.length,
        head_asymmetry=keypoints.head_asymmetry,
        confidence=keypoints.confidence,
        axis=axis,
        head_sign=keypoints.head_sign,
    )


def scale_keypoints(
    keypoints: BodyKeypoints, scale: float, pivot: Tuple[float, float],
) -> BodyKeypoints:
    """Scale the landmarks around ``pivot`` by ``scale``."""
    px, py = float(pivot[0]), float(pivot[1])

    def scale_point(point: Tuple[float, float]) -> Tuple[float, float]:
        return (px + (float(point[0]) - px) * float(scale), py + (float(point[1]) - py) * float(scale))

    head = scale_point(keypoints.head)
    tail = scale_point(keypoints.tail)
    center = scale_point(keypoints.center)
    heading = heading_from_head_center(head, center)
    if heading is None:
        heading = keypoints.heading_deg
    return BodyKeypoints(
        head=head,
        center=center,
        tail=tail,
        heading_deg=float(heading % 360.0),
        length=keypoints.length * float(scale),
        head_asymmetry=keypoints.head_asymmetry,
        confidence=keypoints.confidence,
        axis=keypoints.axis,
        head_sign=keypoints.head_sign,
    )


def body_axis_unit_vector(keypoints: BodyKeypoints) -> Optional[Tuple[float, float]]:
    """Return the unit vector that points from the tail towards the head."""
    return unit_vector(
        keypoints.head[0] - keypoints.tail[0],
        keypoints.head[1] - keypoints.tail[1],
    )


def elongation(
    keypoints: BodyKeypoints, reference_length: Optional[float],
) -> Optional[float]:
    """Return the body length relative to a reference length.

    A value above one means the body is stretched relative to its resting
    length, which is the signature of carangiform swimming.
    """
    if reference_length is None or reference_length <= 1e-6:
        return None
    return float(keypoints.length) / float(reference_length)
