# AGENTS.md — working on FBQ

**Fish Behavior Quantifier (FBQ)** — annotation-free multi-fish tracking for
laboratory tank recordings.

This file is the orientation guide for anyone (human or agent) changing this
repository. Read it before editing.

---

## 1. What this project is

A GUI application that learns a fish detector **from the user's own videos**
(no manual annotation) and then produces per-frame fish identities, oriented
bounding boxes and head direction.

It is a derivative work of AMADEUS (Notomi, Matsumura & Dobata, 2026),
reworked for fish behavior work. See `README.md` and
`THIRD_PARTY_NOTICES.md` for the attribution.

**The next generation replaces the OBB with `head` / `center` / `tail`
keypoints and adds multi-tank ROIs and a behavior-analysis stage.** The
design is in `docs/REDESIGN.md` — read it before touching the pipeline. The
OBB path must keep working until the keypoint path is validated.

---

## 2. Architecture in one page

```
FBQ.sh / FBQ.bat / FBQ.command      launcher: uv → Python → .venv → deps → GUI
  └─ gui/runtime.py:main            console script `fbq`
       └─ gui/gui_home.py:main      Home window; spawns every tool as a subprocess
            └─ gui/gui_easy_tracking.py     authors config.yaml
                 └─ main/batch.py config.yaml     THE PIPELINE (spawns each stage)
                      └─ main/<stage>.py config.yaml   one process per stage
```

**The GUI never runs analysis.** It writes a configuration and shells out.
Every stage is a standalone `python main/<stage>.py config.yaml` process that
reads the config and writes artifacts. This is what makes the pipeline
resumable via the `skip_*` flags — preserve that property in every change.

### The twelve stages (`main/batch.py:377-462`)

1. `initial_tracking.py` · 2. `direction_class_assignment.py` ·
3. `direction_class_filtering.py` · 4. `interaction_image_synthesis_clustered.py` ·
5. `interaction_image_synthesis.py` · 6. `crop_images.py` ·
7. `create_dataset.py` · 8. `obb_detector_training.py` ·
9. `obb_detection.py` · 10. `multi_staged_association.py` ·
11. `refinement.py` · 12. `create_video.py`

Stages 1–7 run once per configuration. Stages 8–12 run once per
learning-rate pair.

`main/behavior_analysis.py` is an **optional** extra stage. It runs once per
configuration, after `create_dataset.py`, and is switched off with
`BEHAVIOR.ENABLE: false`.

---

## 3. Rules that must not be broken

1. **Licence headers.** Every source file keeps
   `SPDX-License-Identifier: AGPL-3.0-only` and the original copyright line.
   Never remove them, and never change the licence of the project.
2. **Attribution.** The upstream AMADEUS citation stays in `README.md` and
   `THIRD_PARTY_NOTICES.md`. AGPL-3.0 §5 requires conveying the source of
   modified versions to network users.
3. **One source of truth for identity.** The project name, the launcher file
   names, the asset names and every remote URL live in
   `tools/project_meta.py`. Change them there, never inline.
4. **Config-driven.** No stage may read global state, environment paths or
   hard-coded locations. Everything comes from `config.yaml`.
5. **Relative paths.** All config paths except `SESSION_PATH` are stored
   relative to `SESSION_PATH` (`main/path_utils.py`). Use
   `to_relative_path` / `resolve_config_paths` rather than raw `os.path`.
6. **Determinism.** The master `RANDOM_SEED` is propagated to every child
   process (`main/batch.py:265-274`). Any new randomness must derive from it.
7. **Fail fast.** A failed stage aborts the run; a failed LR pair aborts the
   sweep; a failed configuration aborts the batch queue. Do not add
   "continue on error" paths.
8. **No telemetry.** Nothing may be uploaded. The only network calls during a
   tracking run are the 2-second GitHub release check and the package-index
   access during setup.

---

## 4. Environment

```bash
bash FBQ.sh              # Linux / WSL2
```

The launcher creates `.venv/` with a pinned `uv` and Python 3.10–3.12, then
verifies the PyTorch profile, the numeric stack, the GUI stack and FFmpeg
before starting the GUI. A ready-marker file (`.venv/.fbq-ready`) lets the
next launch skip the expensive work.

Useful overrides:

| Variable | Effect |
| --- | --- |
| `FBQ_TORCH_PROFILE` | force `cpu` / `macos` / `cu126` / `cu132` / `rocm72` / … |
| `FBQ_VENV` | use a different virtual-environment directory |
| `FBQ_PYTHON` | use a different interpreter to build the environment |

**There is no test suite.** The repository has no `pytest`/`unittest` files.
When you change numerical code, validate it on a real video and compare
against the previous behaviour — that is currently the only regression net.

---

## 5. Where things live

| Concern | Module |
| --- | --- |
| Project identity, URLs, file names | `tools/project_meta.py` |
| Environment bootstrap | `tools/setup_environment.py`, `tools/runtime_profiles.py` |
| GUI shell and navigation | `gui/gui_home.py` |
| Guided workflow | `gui/gui_easy_tracking.py` |
| Full parameter control | `gui/gui_advanced_tracking.py`, `gui/advanced_tracking_layout.py` |
| Segmentation | `gui/gui_segmentation.py`, `main/segmentation_core.py` |
| ROI polygons | `main/roi.py` |
| Stage driver | `main/batch.py` |
| Direction birth (head end) | `main/direction_class_assignment.py` |
| Body landmarks (head/center/tail) | `main/keypoint_geometry.py` |
| Body representation switch | `main/track_representation.py` |
| Tank roster and per-tank runs | `main/tank_roster.py` |
| Behaviour analysis | `main/behavior_analysis.py`, `main/behavior_config.py` |
| OBB fitting | `main/obb_fitting.py` |
| Self-supervised synthesis | `main/interaction_image_synthesis*.py` |
| Dataset assembly | `main/create_dataset.py` |
| Detector training | `main/obb_detector_training.py` |
| Association / identity | `main/multi_staged_association.py` |
| Stage and correction codes | `main/assign_types.py` |
| Correction (C1–C5) | `main/refinement.py` |
| Embedding re-identification | `main/identity_correction.py` |
| Rendering | `main/create_video.py` |
| Batch / multi-config | `gui/gui_multi_config_batch.py` |
| Colab packaging | `main/colab_utils.py`, `tools/colab_runner.py`, `colab/FBQ_Colab.ipynb` |
| Update / uninstall | `gui/app_update.py`, `gui/app_uninstall.py`, `tools/update_worker.py` |

---

## 6. Conventions

- Python 3.10–3.12. `from __future__ import annotations` at the top of new
  modules.
- Type hints on public functions. Keep the existing style: short
  `dict[str, object]` payloads rather than dataclasses, unless the module
  already uses dataclasses.
- Comments explain **why**, not what. Do not add comments that restate the
  code.
- Every module starts with the copyright + SPDX header.
- GUI code keeps all Tk work on the main thread. Long jobs run in a
  subprocess or a worker thread and push updates through a queue that the Tk
  thread drains on a timer.
- User-facing strings are plain English, sentence case, no trailing
  punctuation inside message boxes.

---

## 7. Working on the redesign

`docs/REDESIGN.md` is the contract. The order it proposes is deliberate:

1. Add helpers and a config switch — do not change default behaviour.
2. Build the keypoint path alongside the OBB path.
3. Validate both paths against each other on the same video.
4. Only then switch the default.

Two invariants the redesign depends on:

- **`heading` must be continuous in time.** A single-frame 180° flip is always
  a model error, never a real behaviour. The Viterbi direction post-filter
  (`ATYPE_VITERBI`) relies on that.
- **The head/tail decision belongs to the model, not to a heuristic.** The
  geometric and motion heuristics only label synthetic training data.

Multi-tank work adds a `TANKS:` section to the configuration and runs the
existing pipeline once per tank into `<session>/tanks/<tank_id>/`. Do not
refactor the stages to be ROI-native until the per-tank path is proven.

The switch itself lives in `main/track_representation.py`
(`TRACK_REPRESENTATION: obb | keypoints`, default `obb`). The head/tail
heuristics of `main/keypoint_geometry.py` only label synthetic training data
and the segmentation contours; they must never become the authority for the
final heading of a tracked fish.

---

## 8. Definition of done

- The app starts: `bash FBQ.sh` reaches the Home window.
- A full Easy Tracking run completes on a real video and produces a result
  CSV and a result video.
- `python -m compileall main gui tools` is clean.
- No file in the repository contains the string `AMADEUS` outside the
  upstream attribution in `README.md`, `THIRD_PARTY_NOTICES.md`,
  `AGENTS.md`, `docs/MANUAL.md`, `docs/REDESIGN.md` and
  `docs/legal/BETA_TEST_LICENSE.txt`.
- `tools/project_meta.py` still holds every name and URL.
