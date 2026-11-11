# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Settings of the behaviour analysis stage.

The settings can be written in two ways:

* in the ``BEHAVIOR`` section of the configuration, which is the canonical
  form used by hand written configurations, and
* as flat ``BEHAVIOR_*`` keys, which is the form the advanced GUI writes.

The flat keys are a convenience for the GUI; the section always wins when
both are present.
"""

from __future__ import annotations

#: Default movement threshold in pixels per second.
DEFAULT_MIN_SPEED_PX_S = 1.0

#: Default turn threshold in degrees per second.
DEFAULT_TURN_THRESHOLD_DEG = 30.0


def _section(cfg: dict) -> dict:
    section = cfg.get("BEHAVIOR", {}) or {}
    if not isinstance(section, dict):
        raise ValueError("BEHAVIOR must be a mapping.")
    return section


def _number(cfg: dict, section: dict, section_key: str, flat_key: str, default: float) -> float:
    value = section.get(section_key, None)
    if value is None:
        value = cfg.get(flat_key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{section_key} must be a number.") from exc


def behavior_enabled(cfg: dict) -> bool:
    """Return True when the behaviour analysis stage should run."""
    section = _section(cfg)
    if "ENABLE" in section:
        return bool(section["ENABLE"])
    return bool(cfg.get("BEHAVIOR_ENABLE", True))


def behavior_settings(cfg: dict) -> dict:
    """Return the resolved settings of the behaviour analysis stage."""
    section = _section(cfg)

    scale = section.get("SCALE_PX_PER_MM", None)
    if scale is None:
        scale = cfg.get("BEHAVIOR_SCALE_PX_PER_MM", None)
    if scale is None:
        scale = cfg.get("TANK_SCALE_PX_PER_MM", None)
    if scale is not None and str(scale).strip() != "":
        try:
            scale = float(scale)
        except (TypeError, ValueError) as exc:
            raise ValueError("SCALE_PX_PER_MM must be a number.") from exc
        if scale <= 0.0:
            raise ValueError("SCALE_PX_PER_MM must be positive.")
    else:
        scale = None

    return {
        "ENABLE": behavior_enabled(cfg),
        "MIN_SPEED_PX_S": _number(
            cfg, section, "MIN_SPEED_PX_S", "BEHAVIOR_MIN_SPEED_PX_S", DEFAULT_MIN_SPEED_PX_S,
        ),
        "TURN_THRESHOLD_DEG": _number(
            cfg, section, "TURN_THRESHOLD_DEG", "BEHAVIOR_TURN_THRESHOLD_DEG", DEFAULT_TURN_THRESHOLD_DEG,
        ),
        "SCALE_PX_PER_MM": scale,
    }
