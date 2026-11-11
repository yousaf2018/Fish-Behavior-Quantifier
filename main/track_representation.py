# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Switch between the OBB and the keypoint body representation.

The OBB representation describes a fish as one oriented rectangle. The
keypoint representation describes the same fish as three ordered landmarks
(``head``, ``center``, ``tail``) and a continuous heading.

The switch is additive. ``TRACK_REPRESENTATION: obb`` keeps the existing
behaviour of every stage untouched and stays the default, so the OBB path
keeps working until the keypoint path has been validated on the same video.
"""

from __future__ import annotations

#: The oriented bounding box representation.
OBB = "obb"

#: The head / center / tail keypoint representation.
KEYPOINTS = "keypoints"

#: Every representation the pipeline understands.
REPRESENTATIONS = (OBB, KEYPOINTS)

#: The configuration key that selects the representation.
CONFIG_KEY = "TRACK_REPRESENTATION"


class TrackRepresentationError(ValueError):
    """Raised when the configured representation is not known."""


def normalize_track_representation(value: object, default: str = OBB) -> str:
    """Return the normalized representation name.

    An empty value falls back to ``default`` so a configuration written
    before the switch existed keeps working unchanged.
    """
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return default
    if text not in REPRESENTATIONS:
        raise TrackRepresentationError(
            f'{CONFIG_KEY} must be one of {", ".join(REPRESENTATIONS)}; got {value!r}.'
        )
    return text


def track_representation(cfg: dict) -> str:
    """Return the normalized representation of a configuration."""
    return normalize_track_representation(cfg.get(CONFIG_KEY), OBB)


def uses_keypoints(cfg: dict) -> bool:
    """Return True when the configuration selects the keypoint path."""
    return track_representation(cfg) == KEYPOINTS
