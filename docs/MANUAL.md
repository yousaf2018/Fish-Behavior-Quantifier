# FBQ Manual

Fish Behavior Quantifier (FBQ) — annotation-free multi-fish tracking and
behavior quantification for laboratory tank recordings.

This manual covers installation, the tracking workflow, every parameter, the
output format, and the limitations of each mode. For the architecture of the
upcoming keypoint / multi-tank / behavior work see
[REDESIGN.md](REDESIGN.md).

---

## 1. Install and launch

### 1.1 Installer

Download the installer for your operating system and run it. The first setup
needs an internet connection and provisions:

- a pinned `uv` (the exact version is declared in `pyproject.toml` under
  `[tool.uv].required-version`),
- a Python 3.10–3.12 environment in `.venv/`,
- the PyTorch/torchvision pair matched to your GPU,
- a checksum-pinned FFmpeg build.

The environment is verified before the GUI starts: the PyTorch profile must
pass a real GPU computation and a `torchvision.ops.nms` call, NumPy and SciPy
must be ABI-compatible, and CustomTkinter, Tk, OpenCV and PyAV must import.

### 1.2 Launchers

| Operating system | Launcher |
| --- | --- |
| Windows 10 / 11 | Double-click `FBQ.bat` |
| macOS on Apple Silicon | Double-click `FBQ.command` |
| Linux / WSL2 | `bash FBQ.sh` |

After the first successful setup, `fbq` (and the alias `fq`) are installed
per-user and open the GUI from any terminal.

### 1.3 Environment variables

All switches use the `FBQ_` prefix.

| Variable | Purpose |
| --- | --- |
| `FBQ_VENV` | Override the virtual-environment location (macOS/Linux only). |
| `FBQ_PYTHON` | Override the Python interpreter used to build the environment. |
| `FBQ_TORCH_PROFILE` | Force a PyTorch profile: `cpu`, `macos`, `cu126`, `cu132`, `rocm72`, `rocm100`, `rocmwin100`. `auto` (default) selects by detected hardware and falls back. |
| `FBQ_GUI_PROGRESS_PROTOCOL` | Set by the GUI so the batch runner forwards raw progress to the window. |

### 1.4 Updating and uninstalling

The Home window has **Update** and **Uninstall** buttons. The update checker
compares the installed `VERSION` against the latest GitHub release of this
repository and applies the change after the GUI exits, with automatic
rollback. Uninstall removes the installation folder, the installed commands and
any add-ons.

---

## 2. The Home window

| Panel | Purpose |
| --- | --- |
| **Easy Tracking** | the guided workflow — the normal way to run FBQ |
| **Cropping & Trimming** | cut, rotate and re-encode source video before analysis |
| **Segmentation** | configure foreground segmentation and review blobs |
| **Advanced Tracking** | every stage and parameter exposed individually |
| **Multi Config Batch** | run a queue of saved configurations sequentially |
| **Refinement** | correct identities and directions in a finished result |
| **Colab (beta)** | prepare a Google Colab run (Linux only) |

---

## 3. Segmentation

Segmentation turns frames into foreground blobs. It is classical
image processing, not a neural network, so it is fast, deterministic and
inspectable.

**Modes**

| Mode | Behaviour |
| --- | --- |
| `dark_region` | animals darker than the threshold |
| `bright_region` | animals brighter than the threshold |
| `background_diff` | pixels that differ from the static background |
| hybrid modes | intensity threshold **and** background difference |

**Parameters**: threshold (and separate dark/bright/difference thresholds for
the hybrid modes), blur kernel, morphological open/close iterations, hole
filling, mask expansion (merge-only corridors), inversion, and the static ROI
mask.

**Outliers.** A blob can be marked as an *outlier* when it is not a usable
single-animal observation — for example two fish that overlap and merge into
one blob. Outliers are excluded from normal rasterization, are prioritised as
crossing candidates during synthesis, and are colour-coded in the export.

**Single-fish smoothing (beta)** removes thin protrusions with a
thickness-relative morphological opening, and refuses to split an animal,
create a hole, or remove more than a third of the animal's pixels.

**Additional Outlier ROIs** exclude fixed regions (tank walls, dividers,
reflections) from every frame.

**Arenas (ROIs) drawn on the frame.** The segmentation window has an
**ROI** section with one row per arena. Every arena carries

| Field | Meaning |
| --- | --- |
| **Arena name** | the name of the arena, for example `Arena1`. It names the fish of the arena everywhere in the application |
| **Shape** | `Rectangle`, `Circle` or `Grid` |
| **Animals in this ROI** | how many animals the arena holds |
| **Population** | how exact that number is |

With **Edit ROI** switched on, the arena is drawn on the frame: a drag makes
a rectangle, a drag from the centre makes a circle, and a drag with the
**Grid** shape makes a box that is split into cells. **Split into grid** turns
the current arena into a grid of `columns × rows` arenas, each with its own
name (`Rig-1`, `Rig-2`, …) and its own number of animals. The selector above
the section lists the arenas by name, and the summary line shows what every
enabled arena declares.

**ROIs with a population.** Every ROI set can declare how many animals it
holds (**Animals in this ROI**) and how exact that number is
(**Population**):

| Population | Meaning |
| --- | --- |
| `fixed` | the ROI holds exactly that many animals |
| `estimated` | the ROI holds about that many animals |
| `crowded` | the number of animals is unknown and may be large |

Typing a number switches the mode to `fixed`; choosing `crowded` or
`estimated` by hand keeps that choice, so a crowded ROI can still carry an
estimate. The panel shows the declaration of every enabled ROI and the exact
total of the run.

**The workflow buttons.** The four steps of the segmentation window —
**1. Confirm threshold**, **2. Analyze frames**, **3. Confirm outliers** and
**4. Start processing** — are always on screen at the bottom of the window.
A step that cannot be used yet is disabled instead of hidden, and the step the
workflow is on is marked with a moving border. The settings above them scroll,
so no button is ever pushed out of the window.

**Outputs**, written to `<session>/segmentation/`:

- `background.png` — the static background used by `background_diff`,
- `segmentation_gui_config.json` — the full configuration, re-loadable,
- `<video>_list_of_blobs_gui.pickle` — the extracted blobs.

---

## 4. The tracking pipeline

Twelve stages run in order. Each is an independent process that reads the
configuration and writes its artifacts, so a run can be stopped and resumed at
any stage boundary.

| # | Stage | Skip flag | What it does |
| --- | --- | --- | --- |
| 1 | Initial Tracking | `skip_initial_tracking` | provisional blob-to-track association |
| 2 | Direction Estimation | `skip_trajectory_direction_filtering` | fits an OBB per blob, votes with motion to find the head end, bins the heading into 8 classes, writes single-fish images |
| 3 | Refine Blobs | `skip_refine_blobs_through_tracking` | trains a small classifier on the clean images and deletes blobs whose heading label disagrees with the model |
| 4 | Clustered Paste | `skip_paste_blobs_clustered` | synthesises clustered fish-pasting training images |
| 5 | Mixed Paste | `skip_paste_blobs_with_crossing` | synthesises crossing / overlap training images |
| 6 | Crop Images | `skip_cropping` | writes per-fish crops |
| 7 | Create Dataset | `skip_creating_direction_dataset` | assembles the YOLO-OBB dataset and its train/val split |
| 8 | Training | `skip_training` | trains the YOLO-OBB detector |
| 9 | Detection | `skip_detection` | runs the detector over the analysis video |
| 10 | ID Tracking | `skip_id_tracking` | associates detections into persistent identities (S0–S6) |
| 11 | Refinement | `skip_refinement` | post-correction C1–C5 |
| 12 | Create Video | `skip_creating_video` | renders the annotated result video |

Stages 1–7 run **once** per configuration. Stages 8–12 run **once per
learning-rate pair**, so `training.LR0` and `training.LRF` may each be a list
to sweep a grid. Each pair is stored in its own experiment directory named
`dataset{N}_lr0{LR0}_lrf{LRF}_seed{SEED}`.

### 4.1 How direction is determined

For each track, FBQ:

1. fits an OBB to the blob contour (`min_area` or `pca`),
2. orients the long axis consistently across consecutive frames,
3. compares a motion-weighted vote for both axis directions,
4. flips the axis if the motion contradicts it,
5. bins the surviving axis into one of eight 45° classes.

Blobs whose track does not move far enough, or whose contour is degenerate,
are dropped and the reason is recorded.

The stage only trusts a heading when the track travelled at least
`DIR_MIN_DISP` body lengths (1.0 by default). A short clip can leave every
fish below that value even though the fish clearly moved. When no track
reaches the configured value and `AUTO_PARAMS` is on, the stage measures the
motion instead and relaxes the gate to half of what the longest track
travelled, never below 0.25 body lengths. The relaxation is printed in the
log. Set `DIR_MIN_DISP` explicitly, or turn `AUTO_PARAMS` off, to keep the
strict gate.

### 4.2 How identity is maintained

Stage 10 runs the association cascade:

| Code | Meaning |
| --- | --- |
| S0 | start-frame initialization |
| S1 | strong IoU + heading gate |
| S2 | strong IoU + axis cost |
| S3 | loose IoU (IoU > 0) + heading gate |
| S4 | loose IoU (IoU > 0) + axis cost |
| S5 | fast-distance recovery of an active track |
| S6 | long-gap recovery of a dormant track |

Three Kalman filters per track model position, box shape and direction. The
pass runs **forward and backward** from a common seed frame and the two
results are merged, which is what gives complete coverage of the timeline.

### 4.3 How results are corrected

Stage 11 applies:

| Code | Meaning |
| --- | --- |
| C1 | Kalman-gated interpolation of a gap |
| C2 | a gap filled from an overlapping identity |
| C3 | embedding-based identity resolution over contact episodes |
| C4 | position-spike, direction-spike and segment-flip repair |
| C5 | frames still missing, flagged for manual review |

C3 trains a small embedding network on isolated fragments and only relabels a
contact episode when the embedding decision is confident (similarity ≥ 0.91).

---

## 5. Configuration file

Every run is driven by one YAML file, normally `<session>/config.yaml`.
All paths except `SESSION_PATH` are stored **relative to `SESSION_PATH`**, so
a session folder can be moved or copied.

```yaml
SESSION_PATH: /path/to/session
TRAINING_VIDEO_PATH: training.mp4
TRACKING_VIDEO_PATH: analysis.mp4
NUM_OBJECTS: 12
MIN_OVERLAP: 0.05
MAX_OVERLAP: 0.4
RANDOM_SEED: 0

training:
  EPOCHS: 50
  LR0: 0.01          # a list turns this into a sweep
  LRF: 1.0
  BATCH_SIZE: auto
  PRETRAINED_MODEL: yolo11n

analysis:
  MATCH_IOU: 0.5
  MATCH_ANGLE: 90
  MAX_AGE: 10

EMBEDDING:
  ENABLE: false
```

### 5.1 Key parameters

| Parameter | Meaning |
| --- | --- |
| `NUM_OBJECTS` | number of fish in the video |
| `MIN_OVERLAP` / `MAX_OVERLAP` | how much two fish may overlap before the model treats the pair specially |
| `REFINE_EPOCHS` / `REFINE_ITERS` | the refine-classifier budget and how many refine passes to run |
| `CLUSTER_FRAMES`, `PASTE_*`, `ALPHA_MODE` | synthesis controls |
| `VAL_RATIO` | validation split of the generated dataset |
| `NUM_WORKERS` | dataloader / correction workers |
| `FRAME_INTERVAL`, `NUM_PREVIEW_FRAMES` | sampling density |
| `RANDOM_SEED` | master seed; propagated to every child process |
| `TRACK_REPRESENTATION` | `obb` (default) or `keypoints`; see §5.3 |
| `ROI_SETS` | the regions of the frame that hold the fish; see §5.5 |
| `TANKS`, `VIDEO_TANK_MAP` | multi-tank rigs; see §5.4 |
| `BEHAVIOR` | behaviour analysis settings; see §5.6 |

### 5.2 Logs

`log.txt` (next to the configuration) holds the run header with the full
environment and every stage's output. `time.csv` holds one row per stage with
status and elapsed time.

### 5.3 Body representation

`TRACK_REPRESENTATION` selects how a fish body is described.

| Value | Meaning |
| --- | --- |
| `obb` | the default: one oriented bounding box per fish, heading from the box axis |
| `keypoints` | the heading comes from the head landmark, so it is continuous and follows the snout instead of the box axis |

Both values write the same landmark side-car
(`<session>/single_animal_images/keypoints.csv`) with the `head`, `center`
and `tail` of every fish, so the body geometry is available either way.
The eight direction classes are still written on both paths.

`keypoints` is the experimental path. It is not the default until it has been
validated against the `obb` path on the same video.

### 5.4 Multiple tanks

A rig that records several tanks at once is described by a `TANKS` section.
Every tank is one **ROI** of the video, and the pipeline then runs **once per
ROI**, each with its own session directory under
`<session>/tanks/<tank_id>/`.

```yaml
TANKS:
  - id: tank_a
    name: "Tank A (adult zebrafish)"
    view: top                 # top | side
    fish_count: 12
    polygon: [[0.02, 0.05], [0.48, 0.05], [0.48, 0.95], [0.02, 0.95]]
    scale_px_per_mm: 42.5

VIDEO_TANK_MAP:
  "rig_cam1.mp4": ["tank_a", "tank_b"]

TANKS_PARALLEL: true
TANKS_MAX_WORKERS: 0            # 0 = every ROI at once
```

* `polygon` is in **normalized** coordinates (0–1), so the same tank geometry
  survives a change of resolution.
* `fish_count` is the **number of animals in that ROI**. It sets `NUM_OBJECTS`
  for that tank, so each ROI is tracked with its own population. Leave it out
  to use the global value.
* `scale_px_per_mm` is optional; it turns pixel speeds into millimetres.
* `VIDEO_TANK_MAP` says which tanks a video shows. A tank that no video
  claims is analysed on every listed video.
* `TANKS_PARALLEL` (default `true`) processes **all ROIs at the same time**
  instead of one after the other. `TANKS_MAX_WORKERS` limits how many run
  concurrently; `0` runs every ROI at once. A tank that fails stops the
  whole run, as every other stage does.

**In the interface.** The Advanced Tracking window has a **Tanks (multi-ROI)**
panel: one row per ROI with its id, name, view, animal count and rectangle
(`x0 y0 x1 y1`, normalized), a switch for processing all ROIs at the same
time, and a field for how many run at once. The panel writes the `TANKS`
section of `config.yaml`.

**In the segmentation window.** Every ROI set can declare how many animals
it holds (**Animals in this ROI**). The counts of the enabled ROIs are
shown next to the field, and their sum is the number of animals the run
expects.

Only the blobs whose centre lies inside the polygon are used, so one
segmentation pass serves every tank. Without a `TANKS` section the run is
exactly the single-tank run it always was.

### 5.5 ROIs of the analysis

`ROI_SETS` describes the parts of the frame that hold the fish. It is used
when the analysis runs on a video with several groups of animals, for example
a camera that sees two tanks, or one tank that is divided.

```yaml
ROI_SETS:
  - enabled: true
    points_normalized: [[0.02, 0.05], [0.48, 0.05], [0.48, 0.95], [0.02, 0.95]]
    frame_start: 0
    frame_end: -1
    n_animals: 12
    population: fixed
  - enabled: true
    points_normalized: [[0.52, 0.05], [0.98, 0.05], [0.98, 0.95], [0.52, 0.95]]
    n_animals: 0
    population: crowded
```

* `points_normalized` is in **normalized** coordinates (0–1), so the same
  geometry survives a change of resolution.
* `n_animals` is the number of animals in that ROI, and `population` says how
  exact that number is: `fixed`, `estimated` or `crowded`.

**In the interface.** The Easy Tracking window has a **ROIs of the analysis**
panel in Step 2. It is available after the segmentation model is trained, so
the ROIs can be added, changed or removed while the final analysis is being
prepared. Each row holds a name, the rectangle (`x0 y0 x1 y1`, normalized),
the number of animals and the population. **From segmentation** copies the
arenas of the segmentation step into the panel, with their names, their shape
and their number of animals, so a polygon drawn there keeps its shape. The
panel writes the `ROI_SETS` section of `config.yaml`.

**One arena per ROI.** With **One arena per ROI** switched on, every ROI
becomes an arena of its own: the pipeline runs once per arena, tracks only the
fish of that arena, and numbers them on their own. The results of an arena
are written to `<session>/tanks/<arena>/`, and the fish of an arena are named
after it, for example `Arena1-ID1` and `Arena2-ID1`. The name of a fish is
written to the result tables (`label` columns), drawn on the result video and
shown in the refinement window. With the option switched off, all the ROIs
share one run and one set of numbers.

**What the declaration does.**

| Situation | What FBQ does |
| --- | --- |
| every ROI has a fixed population | the sum of the counts is the population of the run, so it becomes `NUM_OBJECTS` |
| a ROI is `estimated` or `crowded` | the number of animals is not known exactly, so the run uses the variable-population mode and the counts stay expectations |
| the counts disagree with `NUM_OBJECTS` | the run reports the disagreement and keeps the configured value |
| an ROI holds no fish | the behaviour report shows `0` against the declared number |

The behaviour analysis writes the comparison to
`<session>/results/behavior/behavior_rois.csv`: one row per ROI with the
declared number, the number of fish found in it, the difference and a note.
A fish that lies in two overlapping ROIs counts for both, and a fish outside
every ROI is reported on its own row.

### 5.6 Behaviour analysis

After the tracking data is prepared, FBQ writes a per-fish behaviour summary
to `<session>/results/behavior/`:

| File | Content |
| --- | --- |
| `behavior_tracks.csv` | one row per fish per frame: landmarks, heading, elongation, speed, turning rate, distance to the wall, movement state |
| `behavior_summary.csv` | one row per fish: path length, straightness, mean and maximum speed, turning statistics, body length, time moving and resting, wall proximity, time per heading sector |
| `behavior_rois.csv` | one row per ROI: the declared number of animals, the number of fish found in it, the difference and a note |

```yaml
BEHAVIOR:
  ENABLE: true
  MIN_SPEED_PX_S: 1.0        # movement threshold in pixels per second
  TURN_THRESHOLD_DEG: 30.0   # turn threshold in degrees per second
  SCALE_PX_PER_MM: null      # optional; falls back to the scale of the tank
```

Set `BEHAVIOR.ENABLE: false` to skip the stage.

### 5.7 Training on several videos

A run can learn its detector from more than one video. The videos are listed
in `TRAINING_VIDEO_PATHS`; the first entry of the list is also the
`TRAINING_VIDEO_PATH`, so the stages that need one reference video keep
working.

```yaml
TRAINING_VIDEO_PATH: "rig_cam1.mp4"
TRAINING_VIDEO_PATHS:
  - "rig_cam1.mp4"
  - "rig_cam2.mp4"
```

Every video is prepared in a session of its own, under
`<session>/training_videos/<video>/`, so the fish crops of one video never mix
with the crops of another. The dataset of the run is then built from the pools
of all the videos, and the training uses all of them.

**In the interface.** The Easy Tracking window has an **Add videos** button
next to the training video. The picked videos are listed under the field, and
the run learns from all of them. Every video needs its own segmentation, so
the segmentation step has to be run for each of them.

The analysis side already takes several videos: the **Analysis Video Path**
accepts a directory, and the run then analyses every video it holds.

---

## 6. Beta modes

### 6.1 Without direction estimation

Tracks OBBs and identities without assigning front or rear. Selectable in
Easy Tracking; the run is stored under
`<session>/without_direction_estimation/`.

That folder is a session of its own, so the stages whose output the later
stages read — the initial tracking and the single-fish images — run again
there even when the parent session already produced them. Every other skip
flag keeps the value the user gave it.

### 6.2 Variable population

Fish may enter or leave the frame. A candidate must be seen in enough
consecutive frames before it is given an ID, and confirmed tracks expire after
a configurable age. **A fish that leaves and re-enters may receive a new ID.**

### 6.3 Colab (beta, Linux only)

Prepares two bundles in a Google Drive folder of your choice:

- an **input bundle** — the configuration, the segmentation outputs, the
  videos and the dataset images the run actually needs,
- a **runtime package** — the FBQ source at the exact revision you are
  running, with a per-file SHA-256 manifest.

A personalized notebook is written next to them. Open it on a Colab GPU
runtime: the notebook re-verifies the manifest, builds the environment and
runs the same `main/batch.py`. Results return **through Google Drive only** —
there is no FBQ server in the loop. The local workspace is deleted on
success and kept on failure for debugging.

---

## 7. Refinement GUI

Opens a finished result and lets you correct it interactively:

- move or re-assign identities across a contact episode,
- flip a direction,
- fill or clear a frame,
- review the frames that stage 11 flagged as still missing.

Corrections are written back with their own assign-type codes, so the
corrected output stays distinguishable from the automatic one.

---

## 8. Output format

The final CSV lives in `<session>/results/` and is named
`<model>_<dataset>_<run>_<video>.csv` (plus `_id_resolved` for the corrected
variant).

```
frame, cx0, cy0, w0, h0, heading0, cx1, cy1, w1, h1, heading1, ...
```

| Column | Meaning |
| --- | --- |
| `frame` | frame index, 0-based |
| `cx{n}`, `cy{n}` | OBB center of fish `n`, in pixels |
| `w{n}`, `h{n}` | OBB long and short side, in pixels |
| `heading{n}` | head direction of fish `n`, in degrees, `[0, 360)` |
| `label{n}` | the name of fish `n`, for example `Arena1-ID1` (only when the run has an arena) |

`heading` is quantized to 45° steps in the current OBB pipeline. The
keypoint pipeline planned in [REDESIGN.md](REDESIGN.md) makes it continuous.

**Session directory layout**

```
<session>/
  config.yaml
  log.txt
  time.csv
  segmentation/            background.png, config json, blob pickle
  initial_tracking/        track_assignments.csv, tracking_stats.csv
  single_animal_images/    clean per-fish images, keypoints.csv
  paste_blobs/             synthesised crossing images
  paste_blobs_clustered/   synthesised clustered images
  yolo_dataset/            train/ and val/ with data.yaml
  model/                   downloaded pretrained weights
  main/<model>/training/<experiment>/   weights/, results.csv, args.yaml
  main/<model>/tracking/<experiment>/    obbs*.csv, directions*.csv
  results/                 final CSVs and the result video
  results/behavior/        behavior_tracks.csv, behavior_summary.csv, behavior_rois.csv
  tanks/<tank_id>/         one session directory per tank, when TANKS is used
  training_videos/<video>/ one session directory per training video, when several are used
```

**Landmark side-car**

`single_animal_images/keypoints.csv` holds the body landmarks of every
accepted fish:

```
frame, blob_index, traj_id, class_id, accepted,
head_x, head_y, center_x, center_y, tail_x, tail_y,
heading_deg, body_length, head_asymmetry, confidence, erase_reasons
```

`traj_id` is the identity the initial-tracking stage gave to the fish, and
`heading_deg` is the continuous head direction in degrees, measured from the
upwards image direction.

---

## 9. Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Setup fails on GPU detection | set `FBQ_TORCH_PROFILE` explicitly |
| "GPU unavailable: the loaded NVIDIA kernel module does not match the installed userspace driver" | the NVIDIA driver was updated without a reboot; reboot the machine so the matching kernel module is loaded, then start FBQ again |
| "GPU: none (CPU, unavailable)" although the machine has a GPU | the same driver mismatch, or no driver is installed for the running kernel |
| "the reference video ... cannot be read" | `TRAINING_VIDEO_PATH` points at a file that is not there; the run falls back to the tracking video, but the training video has to be correct for the training stages |
| "No single_animal_images segments were produced" | no fish travelled far enough for a heading; the log names the dominant removal reason. Use a longer clip, lower `DIR_MIN_DISP`, or run without direction estimation |
| Tracking merges two fish | segmentation produced one merged blob; mark it as an outlier in Segmentation |
| Head direction is 180° wrong | the fish was not moving during the direction stage; increase the amount of separated footage or record a training clip with clear motion |
| Identities swap at a crossing | raise `REFINE_EPOCHS` / enable `EMBEDDING.ENABLE` |
| Out of memory during training | lower `BATCH_SIZE`, or set it to `auto` |
| The GUI will not start | run the launcher from a terminal to read the diagnostic output |

---

## 10. Citation and licence

FBQ is a derivative work of AMADEUS (Notomi, Matsumura & Dobata, 2026,
<https://doi.org/10.64898/2026.09.19.752854>). Cite the upstream work when you
publish results.

FBQ is licensed under the GNU Affero General Public License v3.0 only
(`AGPL-3.0-only`). See [LICENSE](../LICENSE) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
