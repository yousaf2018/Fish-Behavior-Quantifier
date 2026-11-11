# FBQ Redesign Roadmap

**Fish Behavior Quantifier (FBQ)** — from oriented bounding boxes to exact
`head` / `center` / `tail` keypoints, multi-tank (multi-ROI) recording, and a
behavior-analysis layer for top-view and side-view tanks.

This document is the design contract for the next generation of FBQ. It
describes what changes, what stays, and in which order the work is done so
that the current pipeline keeps working throughout.

---

## 0. Why the change

The current pipeline represents each fish as an **oriented bounding box**
(OBB) plus one of **eight quantized heading classes**:

```
cx, cy, w, h, heading      # heading ∈ {0°, 45°, ..., 315°}
```

That is enough for "which way is it going", but it cannot answer the
questions fish-behaviour labs actually ask:

| Question | OBB answer | Keypoint answer |
|---|---|---|
| Where is the snout? | not representable | `head_x, head_y` |
| Where is the tail tip? | not representable | `tail_x, tail_y` |
| Body length this frame? | `w` (rectangle side, over-estimates a curved body) | `‖head − tail‖` |
| Body elongation (carp/carangiform swimming)? | not representable | `‖head − tail‖ / rest length` |
| Heading at full precision? | 45° quantization | `atan2` of the head vector |
| Is the fish nose-down at the surface? | not representable | `head_y` vs water line |
| Tail-beat frequency? | not representable | tail-deviation FFT |

The OBB also carries an **axis with no direction**: `minAreaRect` returns a
long axis, and the code has to *vote* on which end is the front
(`main/direction_class_assignment.py:1023-1041`). Moving to three ordered
keypoints makes head and tail first-class, so the learned model carries the
semantics instead of a downstream heuristic.

---

## 1. Target representation

### 1.1 The keypoint triple

Each fish, in each frame, is represented by three ordered points:

| Keypoint | Index | Meaning | Used for |
|---|---|---|---|
| `head` | 0 | snout / rostrum tip | orientation, feeding, surface contact, entry direction |
| `center` | 1 | body centroid (midpoint of the head–tail axis) | position, speed, schooling geometry |
| `tail` | 2 | caudal-fin tip | body length, tail beat, undulation, tail-first events |

Derived quantities (all continuous, no quantization):

```
body_vector  = head - center
heading_deg  = degrees(atan2(head_x - center_x, -(head_y - center_y)))  in [0, 360)
body_length  = ‖head - tail‖
axis_unit    = (head - tail) / ‖head - tail‖
```

The heading uses the same convention as the rest of the pipeline: `0°`
points up (the `-y` direction of the image) and the angle grows clockwise on
screen, exactly like the eight direction classes of the OBB path
(`angle_to_class`, `main/direction_class_assignment.py:415`). A heading read
from a result CSV can therefore be turned back into a direction vector with
`heading_to_unit_vector` (`main/keypoint_geometry.py`).

`center` is deliberately **not** the midpoint of `head` and `tail`. The
midpoint of a curved fish body lies off the animal; the centroid of the
*segmented body* does not. `main/direction_class_assignment.py` already has
the machinery to compute a true centroid from the blob contour, so the
center keypoint is directly reusable there.

### 1.2 Data model

Replace the per-track buffer tuple with a named record. In
`main/multi_staged_association.py` the buffers are currently
`obb_buf`, `class_buf`, `pos_buf`, `score_buf`, `assign_type_buf`,
`direction_corrected_buf`, `assign_type_history_buf`. The keypoint version
adds:

```python
kp_buf[tid][frame] = {
    "head":  (hx, hy),
    "center": (cx, cy),
    "tail":  (tx, ty),
    "score": float,
    "length": float,      # ‖head − tail‖
    "heading": float,     # degrees, continuous
}
```

and keeps `obb_buf` as a **derived** fallback (the axis-aligned hull of the
three keypoints plus the detected box) so that every consumer that only
needs "where and how big" keeps working unchanged.

### 1.3 CSV contract

Current final CSV (`main/multi_staged_association.py:863-869`):

```
frame, cx0, cy0, w0, h0, heading0, cx1, cy1, w1, h1, heading1, ...
```

Keypoint version — the old columns are **kept** and new columns appended, so
existing analysis scripts keep working:

```
frame,
cx0, cy0, w0, h0, heading0,
hx0, hy0, tx0, ty0, len0,
cx1, cy1, w1, h1, heading1,
hx1, hy1, tx1, ty1, len1,
...
```

`heading` becomes continuous (float degrees). The current quantized value is
recoverable as `round(heading / 45) % 8`, so nothing that depended on the
8-class convention breaks.

---

## 2. Stage-by-stage migration

The pipeline is a sequence of independent processes driven by `config.yaml`
(`main/batch.py:377-462`). Each row below is one stage.

> **Implementation status.** Steps 2, 3, 4, 12 and the first version of 13
> are implemented; see §10. The remaining rows describe the work that is
> still ahead.

### Stage 1 — `initial_tracking.py`
**Unchanged.** Blob contours → provisional tracks.

### Stage 2 — `direction_class_assignment.py`
This is the stage that must change the most, because it is where direction is
*born* and where the synthetic training labels are created.

Current logic (`main/direction_class_assignment.py:959-1080`):
1. Fit an OBB to each blob contour (`main/obb_fitting.py`).
2. Flip axes so consecutive frames agree (`:1007-1009`).
3. Vote with motion: `score_pos/neg = Σ |v|·(û·v̂)`; flip all axes if the
   negative score wins (`:1023-1041`).
4. Bin the surviving axis into 8 classes via `angle_to_class` (`:409-411`).

Keypoint logic:
1. Fit the OBB as today (it is still the right *unoriented* primitive).
2. Project the contour onto the long axis and find the two **extreme
   points** along it → geometric candidate `head` and candidate `tail`.
   Use the distance-transform *width profile* along the axis as the tie
   breaker: for most fish the head end has the larger inscribed radius and
   the tail end tapers. This is a heuristic and it is allowed to be wrong.
3. Compute the body centroid of the filled contour → `center`.
4. Run the **existing motion vote** to decide which extreme is the head:
   the same `score_pos/neg` comparison, applied to the `center → candidate`
   vector instead of to the axis sign.
5. Emit `head`, `center`, `tail` per blob plus a continuous `heading`.
6. Keep the 8-class `class_id` as an auxiliary label for the refine-classifier
   stage (it still works as a coarse consistency check).

**Critical:** the geometric step is only used to *label synthetic training
data*. The learned keypoint model does the semantic head/tail assignment at
inference time, so a wrong geometric guess pollutes training data but does
not propagate directly to the final heading. That is the same failure
containment the current design already relies on for the OBB axis.

### Stage 3 — `direction_class_filtering.py`
Currently trains a small OBB classifier to find blobs whose assigned heading
class disagrees with the model (`main/identify_incorrect_class_labels.py`,
then `main/apply_class_label_filtering.py`).

Keypoint version: train the same "refine" model as a **3-keypoint** model and
delete a blob when the model's predicted `head`/`tail` order disagrees with
the assigned order. Same plumbing (`delete_blobs.csv`), same iteration
driver, only the model head and the comparison change.

### Stages 4 & 5 — `interaction_image_synthesis*.py`
Donor animals are cut, rotated, scaled and pasted onto reference frames.
The donor's label is currently its heading *class*, which survives rotation
via `rotate_heading_class` (`main/create_dataset.py:686`).

Keypoint version: the donor carries three points, and rotation/scale
transforms them exactly the same way the mask is transformed
(`rotate_crop_with_mask_and_obb_scaling`, `main/interaction_image_synthesis.py:489`).
The synthetic label becomes `(class_id, head, center, tail)` and the pasted
keypoints are transformed with the same affine matrix. Because the keypoints
are geometrically exact, this stage can also synthesise the *body-axis
rotation* cases that the 8-class scheme can only approximate.

### Stage 6 — `crop_images.py`
**Unchanged.**

### Stage 7 — `create_dataset.py`
Currently writes YOLO-**OBB** label lines: `cls x1 y1 x2 y2 x3 y3 x4 y4`
(`main/create_dataset.py:646`).

Keypoint version: write YOLO-**pose** label lines
`cls cx cy w h (hx hy) (cx cy) (tx ty)` with the keypoints normalized by
image width/height. The dataset YAML switches from `nc: 8` to
`nc: 1, kpt_shape: [3, 2]` (one class = "fish", three keypoints).
Frame-based train/val splitting is unchanged.

### Stage 8 — `obb_detector_training.py`
Currently an Ultralytics **YOLO-OBB** trainer with a custom
`EpochsOBBTrainer` subclass (`main/obb_detector_training.py:425-520`),
resume/checkpoint reconciliation, OOM retry and a watchdog child process.

Keypoint version: a new `main/keypoint_detector_training.py` that mirrors
this file but uses the Ultralytics **pose** trainer
(`ultralytics.models.yolo.pose.PoseTrainer`) with `kpt_shape: [3, 2]`.
All the robustness machinery (watchdog, `_find_resume_checkpoint`,
`_reconcile_results_csv_for_resume`, `resolve_save_period`,
`batch_utils.auto_batch_size`, `run_with_oom_retry`) is reusable as-is —
that is the main reason to copy the module rather than generalize it.

### Stage 9 — `obb_detection.py`
Currently runs OBB inference and writes `obbs*.csv`.

Keypoint version: `main/keypoint_detection.py` runs pose inference and writes
`keypoints*.csv` with `(frame, score, head, center, tail)`. The
`tracking_video_inputs` / `video_frame_rows` plumbing is reused unchanged.

### Stage 10 — `multi_staged_association.py`
The association cascade is S0–S6 (`main/assign_types.py:24-38`) with three
Kalman filters per track: position, bbox-shape, direction.

Keypoint version:
- **Cost function.** Today the cost blends OBB IoU, motion distance and
  heading distance. The keypoint version replaces OBB IoU with a
  **body-axis IoU** (IoU of the segments `head↔tail`) and adds a
  **head-point distance** term, which is far more discriminative than an
  axis-aligned rectangle when two fish are side by side and parallel.
- **Kalman filters.** Position KF (center), a *length* KF (‖head−tail‖) and a
  *direction* KF (the unit head vector). The bbox-shape KF is replaced by the
  length KF.
- **Stage gates.** S1/S3 use "strong/loose IoU + heading gate" → become
  "strong/loose body-axis overlap + heading gate".
- The forward/backward double pass and the merge
  (`main/multi_staged_association.py:2208-2280`) are unchanged; they are the
  part of the design that is already correct.

### Stage 11 — `refinement.py` (C1–C5)
Currently: C1 Kalman gap fill, C2 overlap gap fill, C3 embedding re-ID,
C4 position/direction spike repair, C5 missing marking.

Keypoint version: each fill interpolates **three** points. A gap fill that
reproduces the center but not the head is a failure, so the C1 KF chord
interpolation must interpolate the head and tail vectors separately (they
rotate during turns — interpolating the head point linearly across a sharp
turn is wrong). The rule: interpolate the **center** linearly, interpolate
the **axis angle** circularly (`circular_interpolate_deg` already exists,
`main/refinement.py:131`), and rebuild `head`/`tail` from center + axis +
length. The embedding stage (C3) is unchanged — it re-identifies fragments
and does not care about the body representation.

### Stage 12 — `create_video.py`
Draws the OBB and a direction triangle at the short-edge outward normal.
Keypoint version draws the `head → center → tail` skeleton, an arrowhead at
`head`, and optionally the body polygon. `main/direction_class_assignment.py`
already has `get_short_edge_candidates` and `outward_normal_for_edge`
(`:1505-1521`) for the current triangle; the keypoint version draws the
arrow directly from the head vector, which is simpler and more accurate.

---

## 3. Head/tail disambiguation — the hard part

Three mechanisms, in increasing order of authority:

1. **Geometry (training data only).** Width profile along the body axis.
   Weak for eel-shaped fish, and for fish that are strongly curved.
2. **Motion (training data only).** The existing `score_pos/neg` vote. Works
   whenever the fish is actually moving; fails on a stationary fish
   (`disp_too_small`, `main/direction_class_assignment.py:1018`).
3. **The learned keypoint model (inference).** The model is trained on data
   labelled by (1) and (2) and learns the semantic assignment directly. This
   is the authority at inference time.
4. **Temporal consistency (post-hoc repair).** A rare model flip produces a
   180° jump in heading. The pipeline already has a Viterbi direction
   post-filter (`ATYPE_VITERBI`, `main/assign_types.py:36`) used to repair
   heading NaN runs. The keypoint version extends it to repair
   head/tail *swaps* — a swap is exactly a 180° heading flip, so the same
   filter applies.

**Guarantee to preserve:** the final `heading` for a fish must be continuous
in time. A single-frame 180° flip is always a model error, never a real
behaviour (a fish cannot reverse instantaneously), which makes the Viterbi
repair safe and cheap.

---

## 4. Multi-ROI / multi-tank support

### 4.1 What exists today

`main/roi.py` already supports **several** polygon ROIs
(`roi_sets`, each with four vertices plus a frame range) and OR-combines them
into one mask (`build_roi_mask_for_frame`, `main/roi.py:90-102`). The
segmentation GUI exposes them as "Additional Outlier" sets.

That model is *mask union*: several regions are excluded from one analysis.
It is **not** the lab scenario, where each tank is an independent unit of
analysis.

Every ROI set may also declare **how many animals it holds**
(`n_animals`) and **how exact that number is** (`population`):

| `population` | Meaning | Effect on the run |
| --- | --- | --- |
| `fixed` | the ROI holds exactly that many animals | the counts of all fixed ROIs are summed into the expected population |
| `estimated` | the ROI holds about that many animals | the count is an expectation; the run uses the variable-population mode |
| `crowded` | the number of animals is unknown and may be large | no expectation; the run uses the variable-population mode |

The sum over the enabled ROIs is the population the run expects, so the
tracker is told how many fish to look for, and the segmentation GUI shows the
count of every ROI on the video. The behaviour analysis writes the
declaration and the number of fish found in each ROI to
`behavior_rois.csv`, so a wrong declaration is visible in the results.

### 4.2 The lab scenario

Three recording setups, all of which must work:

| Setup | Description | Handling |
|---|---|---|
| **A. One camera, several tanks** | a rig sees 4 tanks; one video file | one video, N ROIs, N independent analyses |
| **B. Several cameras, one tank each** | one file per tank (most common) | one session, N videos, one ROI each |
| **C. One camera, one tank, time-sliced** | same tank, several files | already supported via `TRACKING_VIDEO_FILES` |

Setup **B** is already supported structurally (`TRACKING_VIDEO_FILES` is a
list, `main/path_utils.py:16-27`), but there is no *tank roster*: no stable
tank identity, no per-tank calibration, no cross-tank aggregation.

### 4.3 Proposed data model

Add a `TANKS:` section to `config.yaml`:

```yaml
TANKS:
  - id: tank_a
    name: "Tank A (adult zebrafish)"
    view: top                 # top | side
    fish_count: 12
    polygon: [[0.02, 0.05], [0.48, 0.05], [0.48, 0.95], [0.02, 0.95]]
    scale_px_per_mm: 42.5
  - id: tank_b
    name: "Tank B (juveniles)"
    view: side
    fish_count: 30
    polygon: [[0.52, 0.05], [0.98, 0.05], [0.98, 0.95], [0.52, 0.95]]
    scale_px_per_mm: 38.0
```

- `polygon` is in **normalized** coordinates (0–1) so it survives a
  resolution change and can be reused across recordings of the same rig.
- `view` selects the behaviour model (top vs side) in the behavior stage.
- `scale_px_per_mm` enables metric output (mm/s, body lengths/s). Optional:
  without it, speeds are reported in body lengths per second, which is
  scale-free and often what labs want anyway.

A **video→tank assignment** is needed for setup A:

```yaml
VIDEO_TANK_MAP:
  "sessionA_cam1.mp4": ["tank_a", "tank_b"]
  "sessionA_cam2.mp4": ["tank_c"]
```

When a video has more than one tank, the pipeline runs **once per tank**,
cropping to the tank polygon and keeping global pixel coordinates for
re-assembly.

### 4.4 Implementation strategy — per-ROI sub-sessions

The pipeline is already a sequence of independent processes that each read
`config.yaml`. The cheapest way to get multi-tank support without touching
every stage is to **run the existing pipeline once per tank**, with a
per-tank config that:

- injects the tank polygon as the active ROI mask,
- redirects `SESSION_PATH` to `<session>/tanks/<tank_id>/`,
- sets `NUM_OBJECTS` to the tank's `fish_count` (or the global value when
  the tank declares none),
- records `TANK_ID` in the config so results are labelled.

`main/batch.py` gains a tank loop around its existing stage loop; each
iteration writes a derived config into `<session>/tanks/<tank_id>/config.yaml`
and reuses every existing stage unchanged. Results land in
`<session>/tanks/<tank_id>/results/...`.

**All ROIs run at the same time.** A rig with several tanks is analysed in
parallel, not tank by tank: each tank runs the whole pipeline in its own
process, so the stages of different tanks overlap. `TANKS_PARALLEL` (on by
default) switches this on and `TANKS_MAX_WORKERS` limits the number of
tanks that run at once (`0` = every tank). A tank that fails aborts the
whole roster, as any other failing stage does. The derived per-tank config
carries no `TANKS` section, so a tank run never starts a second round of
tank runs.

**Why this is the right first step:** it is additive, it cannot regress the
single-tank path, and it exercises the whole pipeline per tank. Once the
per-tank path is proven, the stages can be refactored to be ROI-native
(injecting a mask instead of a crop), which is a larger change with no
user-visible benefit.

### 4.5 Output naming

Current: `<session>/results/<model>_<dataset>_<run>_<video>.csv`
(`main/multi_staged_association.py:840-859`).

New: `<session>/tanks/<tank_id>/results/<model>_<dataset>_<run>_<video>.csv`,
plus a project-level roll-up:

```
<session>/results/all_tanks_<run>.csv     # long format: tank_id, video, frame, fish_id, ...
```

The long format is what the behavior stage consumes — one row per
(tank, video, frame, fish) with all derived metrics.

---

## 5. Behavior analysis module

New stage, after tracking:

```
main/behavior_analysis.py
```

It consumes the per-fish time series (from the final CSV or directly from
the tracking buffers) and emits **events** and **time series**.

### 5.1 Event model

```python
@dataclass(frozen=True)
class BehaviorEvent:
    tank_id: str
    video: str
    fish_id: int
    kind: str            # "burst", "turn", "surface_visit", ...
    start_frame: int
    end_frame: int
    confidence: float
    metrics: dict        # e.g. {"peak_speed": 3.4, "turn_deg": 78.0}
```

Output: `<session>/results/behavior_events.csv` with one row per event.

### 5.2 Per-fish time series

```
<session>/results/behavior_timeseries.csv
```

Columns: `tank_id, video, frame, fish_id, x, y, speed, accel,
heading, yaw_rate, body_length, elongation, depth, surface_distance,
nearest_neighbor_distance, neighbor_count, polarization`.

### 5.3 Top-view behaviors

The tank plane is (x, y); the water surface is the image plane.

| Behavior | Definition |
|---|---|
| `speed` | `‖Δcenter‖ / dt`, in body lengths/s |
| `acceleration` | `Δspeed / dt` |
| `burst` | `acceleration > τ` sustained for `≥ k` frames (knee detection on the speed trace) |
| `burst_coast` | alternation of burst and coast phases (duty cycle, mean burst length) |
| `turn` | `|yaw_rate| > τ` over `≥ k` frames; report turn angle and handedness (L/R) |
| `sharp_turn` | `|yaw_rate| > 2·τ` |
| `u_turn` | heading change > 150° within a short window |
| `body_elongation` | `‖head−tail‖ / rest_length` — separates carangiform from anguilliform swimming |
| `tail_beat_freq` | dominant frequency of the tail-deviation signal (FFT of the angle between the head–tail axis and the velocity) |
| `wall_contact` | centre within `d` px of the tank boundary polygon |
| `wall_hug` | sustained wall proximity |
| `station_holding` | speed ≈ 0 for `≥ k` frames while the heading varies |
| `bottom_rest` | (needs a side view or a known floor line) prolonged immobility at the tank floor |
| `schooling` | mean nearest-neighbour distance below τ, with local polarization above τ₂ |
| `polarization` | `‖Σ ûᵢ‖ / N` over a neighbourhood |
| `approach` / `follow` | two fish within τ for `≥ k` frames with a stable relative bearing |

### 5.4 Side-view behaviors

The tank cross-section is (x, depth); "up" is the image y axis.

| Behavior | Definition |
|---|---|
| `depth` | depth in mm from the water surface (needs `scale_px_per_mm` and a water line) |
| `depth_band` | occupancy histogram over depth bands (surface / upper / mid / lower / bottom) |
| `surface_visit` | `head_y` within `d` px of the water line |
| `surfacing` | snout breaks the surface |
| `diving` | sustained downward body pitch |
| `vertical_migration` | sustained monotonic depth change over `≥ k` frames |
| `midwater_hold` | low speed at constant depth |
| `bottom_rest` | low speed at the floor |
| `jump` | head above the water line |
| `rolling` | periodic lateral oscillation of the head point — needs a side view with a visible roll |

### 5.5 How the behaviors see the data

The behavior stage should consume the **keypoint** representation, not the
OBB. Concretely:

- `center` → position, speed, acceleration, depth, boundary distance.
- `head` → heading, yaw rate, surface contact, jump, snout-based events.
- `head − tail` → body length, elongation, tail beat, undulation.
- `‖head − center‖ / ‖center − tail‖` → head-to-tail asymmetry, which
  separates "nose-leading" from "tail-leading" moments.

Every behavior is therefore **derived from the keypoints**, which is exactly
why the keypoint migration is a prerequisite rather than an optional extra.

### 5.6 Thresholding

Behavior thresholds must not be hard-coded. Store them per tank in the
config and derive sensible defaults from the data:

```yaml
BEHAVIOR:
  speed_threshold: auto        # auto = median + k·MAD of the observed speed
  turn_threshold: auto
  burst_threshold: auto
  min_event_frames: 5
  enable: [burst, turn, surface_visit, schooling]
```

`auto` thresholds use robust statistics over the run itself, so a threshold
that works for adult zebrafish does not silently break for juveniles.

---

## 6. Implementation order

Each step leaves the app working.

| # | Step | Risk | Notes |
|---|---|---|---|
| 1 | **This rebrand** (done) | low | `AMADEUS` → `FBQ` everywhere; pipeline untouched |
| 2 | Keypoint geometry helpers (done) | low | `main/keypoint_geometry.py`; pure functions, no behaviour change |
| 3 | `TRACK_REPRESENTATION` config switch (done) | low | `obb` (default, current) vs `keypoints`; both paths exercised |
| 4 | Head/center/tail estimation in `direction_class_assignment` (done) | medium | geometric + motion vote; writes the `keypoints.csv` side-car |
| 5 | Pose dataset writer in `create_dataset` | medium | YOLO-pose label format; needs `ultralytics` pose support verified |
| 6 | `keypoint_detector_training` | medium | clone of `obb_detector_training` with the pose trainer |
| 7 | `keypoint_detection` | medium | pose inference → `keypoints*.csv` |
| 8 | Keypoint association | high | new cost function, new Kalman set; validate against the OBB path on the same video |
| 9 | Keypoint refinement | medium | C1–C5 extended to three points |
| 10 | Keypoint rendering | low | new overlay in `create_video` |
| 11 | Keypoint CSV output | low | append columns, keep the old ones |
| 12 | **Tank roster + per-tank sub-sessions** (done) | medium | additive; `main/batch.py` gains a tank loop |
| 13 | `behavior_analysis` (top view) (first version done) | medium | new stage, new outputs |
| 14 | `behavior_analysis` (side view) | medium | depth band + surface events |
| 15 | Calibration (`scale_px_per_mm`) (done) | low | optional per-tank metric conversion |


### Validation strategy for steps 8–11

Run the OBB path and the keypoint path on the **same** video and compare:

- per-frame fish count agreement,
- identity agreement (match by nearest-centre Hungarian match),
- heading difference distribution (should be concentrated near 0°; a mass at
  180° means the head/tail disambiguation failed),
- body-length stability (coefficient of variation of `‖head−tail‖` per fish).

A 180° bimodal heading difference is the canonical failure signature. It must
be caught by the Viterbi repair, not shipped.

---

## 7. Compatibility guarantees

1. The OBB path stays the default. `TRACK_REPRESENTATION: obb` reproduces
   today's behaviour bit-for-bit.
2. Final CSVs keep the existing columns and append the keypoint columns.
3. Existing `config.yaml` files keep working — new sections are optional and
   default to the current single-ROI, single-video behaviour.
4. `TANKS` absent ⇒ exactly today's behaviour, including the session
   directory layout.
5. `docs/REDESIGN.md` is the reference for all of the above; update it when
   the implementation diverges.

---

## 8. Licence and attribution

FBQ is a derivative work of the original AMADEUS project and remains under
the **GNU Affero General Public License v3.0 only** (`AGPL-3.0-only`).

- Every source file keeps its `SPDX-License-Identifier: AGPL-3.0-only`
  header and the original copyright line. **Do not remove them.**
- `LICENSE` is the AGPL-3.0 text; `THIRD_PARTY_NOTICES.md` lists the
  third-party components.
- Upstream AMADEUS is credited in `README.md` and `THIRD_PARTY_NOTICES.md`
  with its authors, citation and DOI.
- Because AGPL-3.0 §5 requires conveying the source of modified versions to
  network users, any hosted or distributed FBQ build must ship the
  corresponding source. The Colab runtime package already carries the
  per-file SHA-256 manifest that proves which source a given run used — keep
  that mechanism.
- The upstream citation must remain in the README and in the published
  papers' reference lists:

  > Yusuke Notomi, Kentarou Matsumura, and Shigeto Dobata (2026).
  > *AMADEUS: Annotation-free multi-animal direction estimation using
  > self-supervised learning.* bioRxiv. https://doi.org/10.64898/2026.09.19.752854

---

## 9. Open questions

1. **Keypoint count.** Three (`head`, `center`, `tail`) is the minimum that
   supports the behaviours above. Adding a `dorsal`/`pectoral` point would
   enable roll detection but multiplies annotation cost and model complexity.
   Decide before step 5, because it changes the pose label format.
2. **Head vs snout.** "Head" is the snout tip, not the head centre. If a lab
   needs the eye position, that is a fourth keypoint, not a redefinition.
3. **Curved bodies.** `‖head − tail‖` underestimates a strongly curved
   fish. Consider a third scalar, `arc_length`, estimated from the contour
   — cheap, since the contour is already available at stage 2.
4. **Multi-camera fusion.** Not in scope now. The tank roster is designed so
   that a later stage can merge per-tank results into one animal table.
5. **Non-fish species.** FBQ's name and behaviour library are fish-specific,
   but the pipeline is species-agnostic. The keypoint semantics
   (`head`/`center`/`tail`) generalise to most swimmers.

---

## 10. What is implemented today

The list below is the state of the code, not a plan. Everything listed here
is additive: the OBB path and the single-tank path behave exactly as before
when the new options are absent.

### 10.1 The representation switch

`main/track_representation.py` holds the switch.

| Key | Values | Default | Effect |
|---|---|---|---|
| `TRACK_REPRESENTATION` | `obb`, `keypoints` | `obb` | On `keypoints`, the heading of a fish comes from the head landmark instead of the oriented-box axis, so it stays continuous instead of being derived from the axis vote. |

The switch is read by `main/direction_class_assignment.py`. The eight-class
`class_id` is still written on both paths, because the refine-classifier
stage and the synthetic-image labels still use it as a coarse consistency
check.

### 10.2 The landmark side-car

`main/direction_class_assignment.py` writes
`<session>/single_animal_images/keypoints.csv` on every run, whichever
representation is selected:

```
frame, blob_index, traj_id, class_id, accepted,
head_x, head_y, center_x, center_y, tail_x, tail_y,
heading_deg, body_length, head_asymmetry, confidence, erase_reasons
```

* `traj_id` is the identity the initial-tracking stage assigned to the fish,
  so the file is per-fish and not per-blob.
* `accepted` is `0` for a blob that the direction stage erased, so the
  behaviour stage can ignore it.
* `heading_deg` uses the pipeline convention described in §1.1.

### 10.3 The tank roster

`main/tank_roster.py` validates the `TANKS:` section and derives one
configuration per tank. `main/batch.py` runs the whole pipeline once per
tank:

| Key | Meaning |
|---|---|
| `TANKS` | list of tanks: `id`, `name`, `view`, `fish_count`, `polygon` (normalized), `scale_px_per_mm` |
| `VIDEO_TANK_MAP` | video file name → list of tank ids that the video covers |
| `ROI_SETS` | derived, not authored: the pixel polygon of the tank of the current run |

Each tank run gets its own session directory
`<session>/tanks/<tank_id>/`, its own `config.yaml` and its own results. The
polygon is applied to the blob records of the initial-tracking and direction
stages, so one segmentation pass serves every tank.

`TANKS` absent ⇒ exactly the old behaviour, including the session layout.

### 10.4 The behaviour stage

`main/behavior_analysis.py` runs once per configuration, after the dataset
is created, and reads the landmark side-car. It writes

| File | Content |
|---|---|
| `<session>/results/behavior/behavior_tracks.csv` | one row per fish per frame: landmarks, heading, elongation, speed, turning rate, distance to the wall, movement state |
| `<session>/results/behavior/behavior_summary.csv` | one row per fish: path length, straightness, mean and maximum speed, turning statistics, body length, time moving and resting, wall proximity, time per heading sector |

The settings live in the `BEHAVIOR` section of the configuration (the
advanced GUI writes the equivalent flat `BEHAVIOR_*` keys):

```yaml
BEHAVIOR:
  ENABLE: true
  MIN_SPEED_PX_S: 1.0        # movement threshold in pixels per second
  TURN_THRESHOLD_DEG: 30.0   # turn threshold in degrees per second
  SCALE_PX_PER_MM: null      # optional; enables mm/s and mm
```

`SCALE_PX_PER_MM` falls back to the scale of the tank
(`TANKS[].scale_px_per_mm`), so a calibrated rig reports metric speeds
without any extra configuration.

The full event model of §5.1 — bursts, schooling, surface visits — is **not**
implemented yet; the stage currently reports the time series and the summary
that those events would be detected on.

