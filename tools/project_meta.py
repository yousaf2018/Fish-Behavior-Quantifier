# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Single source of truth for the FBQ project identity and remote endpoints.

Everything that needs to know the project's public name, the on-disk launcher
names, or the remote release/manual locations reads them from here so a future
re-brand or a move to a different hosting account is a one-file change.

Import from anywhere in the project with::

    from tools.project_meta import RELEASES_URL

The module intentionally has no third-party imports so it can be loaded from
``gui/``, ``main/`` and ``tools/`` contexts alike.
"""

from __future__ import annotations

# --- Identity -------------------------------------------------------------

PROJECT_DISPLAY_NAME = "Fish Behavior Quantifier"
PROJECT_SHORT_NAME = "FBQ"
PROJECT_SLUG = "fbq"
PROJECT_DISTRIBUTION = "fbq"
PROJECT_COMMAND = "fbq"
PROJECT_COMMAND_ALIAS = "fq"

#: Environment variable prefix used by every FBQ-era switch, now FBQ.
ENV_PREFIX = "FBQ"

# --- Launcher / asset file names -----------------------------------------

LAUNCHER_SH = "FBQ.sh"
LAUNCHER_BAT = "FBQ.bat"
LAUNCHER_COMMAND = "FBQ.command"
LOGO_MINI = "logo_fbq_mini.png"
LOGO_SPLASH = "logo_fbq_splash.png"
COLAB_NOTEBOOK = "FBQ_Colab.ipynb"
COLAB_NOTEBOOK_TEMPLATE = "FBQ_Colab.template.ipynb"

# --- Remote endpoints -----------------------------------------------------
#
# These point at the repository that hosts this working copy. Point them at a
# different owner/name if the project is published elsewhere; the update
# checker validates that the release redirect stays on the same host and
# path prefix before trusting a version number.

REPO_OWNER = "yousaf2018"
REPO_NAME = "FBQ"
REPO_SLUG = f"{REPO_OWNER}/{REPO_NAME}"

RELEASES_URL = f"https://github.com/{REPO_SLUG}/releases/latest"
RELEASE_TAG_PATH = f"/{REPO_SLUG}/releases/tag/"
ARCHIVE_URL_TEMPLATE = f"https://github.com/{REPO_SLUG}/archive/refs/tags/{{tag}}.zip"

#: In-app manual. The upstream project published a website; FBQ ships its
#: manual inside the repository so the app never depends on a domain.
MANUAL_URL = f"https://github.com/{REPO_SLUG}/blob/main/docs/MANUAL.md"
PROJECT_URL = f"https://github.com/{REPO_SLUG}"

#: Add-on guide and template, kept next to the project they belong to.
ADDON_GUIDE_URL = f"https://github.com/{REPO_SLUG}/blob/main/tools/ADDONS.md"
ADDON_TEMPLATE_URL = f"https://github.com/{REPO_SLUG}-Addon-Template"

#: Google Colab entry point used by the beta cloud workflow.
COLAB_URL = "https://colab.research.google.com/"


def user_agent(component: str) -> str:
    """Return the HTTP User-Agent used for this project's network calls."""
    return f"{PROJECT_SHORT_NAME}-{component}"


__all__ = [
    "ADDON_GUIDE_URL",
    "ADDON_TEMPLATE_URL",
    "ARCHIVE_URL_TEMPLATE",
    "COLAB_NOTEBOOK",
    "COLAB_NOTEBOOK_TEMPLATE",
    "COLAB_URL",
    "ENV_PREFIX",
    "LAUNCHER_BAT",
    "LAUNCHER_COMMAND",
    "LAUNCHER_SH",
    "LOGO_MINI",
    "LOGO_SPLASH",
    "MANUAL_URL",
    "PROJECT_COMMAND",
    "PROJECT_COMMAND_ALIAS",
    "PROJECT_DISPLAY_NAME",
    "PROJECT_DISTRIBUTION",
    "PROJECT_ROOT_URL",
    "PROJECT_SHORT_NAME",
    "PROJECT_SLUG",
    "PROJECT_URL",
    "RELEASES_URL",
    "RELEASE_TAG_PATH",
    "REPO_NAME",
    "REPO_OWNER",
    "REPO_SLUG",
    "user_agent",
]

PROJECT_ROOT_URL = PROJECT_URL
