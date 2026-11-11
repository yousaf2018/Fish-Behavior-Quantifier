#!/usr/bin/env bash
# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
# Open the shared launcher from Finder or Terminal.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +e
bash "$SCRIPT_DIR/FBQ.sh" "$@"
status=$?
set -e
if (( status == 42 )); then
  echo "[FBQ] Update accepted. The updater will restart FBQ automatically."
  exit 0
fi
if (( status == 43 )); then
  echo "[FBQ] Uninstall accepted. A separate terminal will remove FBQ."
  exit 0
fi
if (( status == 130 )); then
  exit 130
fi
if (( status != 0 )); then
  echo "FBQ failed with exit code ${status}. See the run's log.txt for the full diagnostic log."
  if [[ -t 0 && -t 1 ]]; then
    read -r -p "Press Enter to close..."
  fi
fi
exit "$status"
