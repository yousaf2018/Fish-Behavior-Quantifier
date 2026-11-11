# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""The FBQ interface design system.

Every screen of FBQ shares one palette, one set of radii and one type scale.
The tokens live here so a colour is written down once, and the widgets that
are built from them look the same in the Home window, in the guided workflow
and in the parameter panels.

The CustomTkinter side of the design system lives in ``assets/fbq_dark.json``;
this module mirrors those colours for the parts that are drawn on a canvas
and offers ready-made widgets for the parts that are not.
"""

from __future__ import annotations

import json
import pathlib
import tkinter as tk
from typing import Callable, Optional

import customtkinter as ctk

try:
    from .project_paths import gui_asset
except ImportError:  # Preserve direct execution with: python gui/<script>.py
    from project_paths import gui_asset


#: Name of the CustomTkinter theme that carries the widget colours.
THEME_FILE = "fbq_dark.json"

# --- Surfaces --------------------------------------------------------------

#: The application background.
BG = "#0D1216"

#: A card or panel that sits on the background.
SURFACE = "#151C21"

#: A card that sits on another card.
SURFACE_ELEVATED = "#1C242A"

#: A surface in its hover state.
SURFACE_HOVER = "#232E35"

#: A hairline separator.
BORDER = "#27333A"

#: A separator that needs to be visible against a card.
BORDER_STRONG = "#38454E"

# --- Text ------------------------------------------------------------------

#: Primary text.
TEXT = "#E6EDF3"

#: Supporting text.
TEXT_MUTED = "#93A1AC"

#: Text that is deliberately quiet.
TEXT_FAINT = "#68787F"

#: Text drawn on the accent colour.
TEXT_ON_ACCENT = "#FFFFFF"

# --- Accents ---------------------------------------------------------------

#: The primary action colour.
ACCENT = "#3B82F6"

#: The primary action colour under the pointer.
ACCENT_HOVER = "#60A5FA"

#: The primary action colour while it is pressed.
ACCENT_PRESSED = "#2563EB"

#: A destructive action.
DANGER = "#F0616D"

#: A caution colour.
WARNING = "#F2B544"

# --- Shape and spacing -----------------------------------------------------

#: Radius of a card.
RADIUS_CARD = 14

#: Radius of a control.
RADIUS_CONTROL = 9

#: Gap between two cards.
GAP = 16

#: Padding inside a card.
PADDING = 22

#: Style of the plain Tk spinboxes and entries that the panels use.
SPIN_STYLE = {
    "bg": "#0F171C",
    "fg": TEXT,
    "disabledbackground": "#0F171C",
    "disabledforeground": TEXT_FAINT,
    "insertbackground": TEXT,
    "buttonbackground": BORDER_STRONG,
    "relief": "flat",
    "highlightthickness": 2,
    "highlightbackground": BORDER_STRONG,
    "highlightcolor": ACCENT,
    "selectbackground": ACCENT,
    "selectforeground": "white",
}


# --- Type scale ------------------------------------------------------------

def title_font(size: int = 24) -> ctk.CTkFont:
    """Return the font of a screen title."""
    return ctk.CTkFont(size=size, weight="bold")


def heading_font(size: int = 17) -> ctk.CTkFont:
    """Return the font of a section heading."""
    return ctk.CTkFont(size=size, weight="bold")


def body_font(size: int = 13) -> ctk.CTkFont:
    """Return the font of running text."""
    return ctk.CTkFont(size=size)


def caption_font(size: int = 12) -> ctk.CTkFont:
    """Return the font of a quiet caption."""
    return ctk.CTkFont(size=size)


def mono_font(size: int = 12) -> ctk.CTkFont:
    """Return the font of a code or path."""
    return ctk.CTkFont(family="DejaVu Sans Mono", size=size)


# --- Theme -----------------------------------------------------------------

def _installed_theme_defaults() -> dict:
    """Return the theme that ships with the installed CustomTkinter."""
    try:
        import customtkinter

        assets = pathlib.Path(customtkinter.__file__).resolve().parent / "assets" / "themes"
        for name in ("dark-blue.json", "blue.json"):
            candidate = assets / name
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def apply_theme(appearance: str = "dark") -> None:
    """Apply the FBQ appearance and colour theme to CustomTkinter.

    Every GUI module calls this before it builds any widget, so the whole
    application shares one look.  Keys that the FBQ theme does not define are
    taken from the theme that ships with CustomTkinter, so a new widget
    release cannot break the interface with a missing-key error.
    """
    ctk.set_appearance_mode(appearance)
    ctk.set_default_color_theme(str(gui_asset(THEME_FILE)))

    defaults = _installed_theme_defaults()
    if not defaults:
        return
    try:
        from customtkinter.windows.widgets.theme.theme_manager import ThemeManager
    except Exception:
        return
    theme = ThemeManager.theme
    for section, values in defaults.items():
        if not isinstance(values, dict):
            continue
        current = theme.setdefault(section, {})
        if not isinstance(current, dict):
            continue
        for key, value in values.items():
            current.setdefault(key, value)


# --- Widgets ---------------------------------------------------------------

def primary_button(
    master: tk.Misc,
    text: str,
    command: Optional[Callable[[], None]] = None,
    *,
    width: int = 200,
    height: int = 42,
    font: Optional[ctk.CTkFont] = None,
) -> ctk.CTkButton:
    """Return the primary action button."""
    return ctk.CTkButton(
        master,
        text=text,
        command=command,
        width=width,
        height=height,
        corner_radius=RADIUS_CONTROL,
        fg_color=ACCENT,
        hover_color=ACCENT_HOVER,
        text_color=TEXT_ON_ACCENT,
        font=font or ctk.CTkFont(size=14, weight="bold"),
    )


def secondary_button(
    master: tk.Misc,
    text: str,
    command: Optional[Callable[[], None]] = None,
    *,
    width: int = 120,
    height: int = 36,
    font: Optional[ctk.CTkFont] = None,
) -> ctk.CTkButton:
    """Return a secondary action button."""
    return ctk.CTkButton(
        master,
        text=text,
        command=command,
        width=width,
        height=height,
        corner_radius=RADIUS_CONTROL,
        fg_color="transparent",
        hover_color=SURFACE_HOVER,
        border_width=1,
        border_color=BORDER_STRONG,
        text_color=TEXT,
        font=font or ctk.CTkFont(size=13),
    )


def ghost_button(
    master: tk.Misc,
    text: str,
    command: Optional[Callable[[], None]] = None,
    *,
    width: int = 110,
    height: int = 34,
    font: Optional[ctk.CTkFont] = None,
) -> ctk.CTkButton:
    """Return a quiet button for the header bar."""
    return ctk.CTkButton(
        master,
        text=text,
        command=command,
        width=width,
        height=height,
        corner_radius=RADIUS_CONTROL,
        fg_color="transparent",
        hover_color=SURFACE_HOVER,
        text_color=TEXT_MUTED,
        font=font or ctk.CTkFont(size=13),
    )


def card(
    master: tk.Misc,
    *,
    fg_color: str = SURFACE,
    border_color: str = BORDER,
    corner_radius: int = RADIUS_CARD,
) -> ctk.CTkFrame:
    """Return a card surface."""
    return ctk.CTkFrame(
        master,
        fg_color=fg_color,
        border_color=border_color,
        border_width=1,
        corner_radius=corner_radius,
    )


def divider(master: tk.Misc) -> ctk.CTkFrame:
    """Return a hairline separator."""
    return ctk.CTkFrame(master, height=1, fg_color=BORDER, corner_radius=0)
