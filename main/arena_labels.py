# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Identity labels of the fish of one arena.

A run that covers several arenas gives every fish an identity of its own, so
the number alone does not tell the user which arena a fish belongs to. The
label of a fish is the arena name and the number of the fish in that arena,
for example ``Arena1-ID3``.

The label is written to the result tables, drawn on the result video and
shown in the user interface, so every part of the application names a fish
the same way.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

#: The configuration keys that name the arena of a run, in the order they win.
ARENA_NAME_KEYS = ("TANK_NAME", "ARENA_NAME")

#: The configuration keys that identify the arena of a run.
ARENA_ID_KEYS = ("TANK_ID", "ARENA_ID")

#: The prefix of the identity number inside a label.
IDENTITY_PREFIX = "ID"


def arena_name(cfg: Optional[dict], fallback: str = "") -> str:
    """Return the name of the arena a run belongs to.

    A run without an arena name keeps the plain number, so a single-tank run
    behaves exactly as it did before the arenas existed.
    """
    if not cfg:
        return fallback
    for key in ARENA_NAME_KEYS:
        value = str(cfg.get(key, "") or "").strip()
        if value:
            return value
    return fallback


def arena_id(cfg: Optional[dict], fallback: str = "") -> str:
    """Return the identifier of the arena a run belongs to."""
    if not cfg:
        return fallback
    for key in ARENA_ID_KEYS:
        value = str(cfg.get(key, "") or "").strip()
        if value:
            return value
    return fallback


def identity_label(arena: str, fish_id: object) -> str:
    """Return the label of one fish, for example ``Arena1-ID1``.

    The number of a fish is the number the user sees, so the first fish of
    an arena is ``ID1`` even though the tables of the application number the
    columns of a fish from zero.
    """
    number = str(fish_id).strip()
    if not number:
        number = "0"
    name = str(arena or "").strip()
    if not name:
        return f"{IDENTITY_PREFIX}{number}"
    return f"{name}-{IDENTITY_PREFIX}{number}"


def identity_label_for_index(arena: str, index: int) -> str:
    """Return the label of the fish that the tables number ``index``."""
    return identity_label(arena, int(index) + 1)


def identity_labels(cfg: Optional[dict], fish_ids: Iterable[object]) -> List[str]:
    """Return the labels of several fish of one run."""
    arena = arena_name(cfg)
    return [identity_label(arena, fish_id) for fish_id in fish_ids]


def identity_labels_for_indices(arena: str, indices: Iterable[object]) -> List[str]:
    """Return the labels of the fish that the tables number by ``indices``."""
    return [identity_label_for_index(arena, index) for index in indices]
