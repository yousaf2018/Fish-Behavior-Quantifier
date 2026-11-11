<p align="center">
  <img src="assets/logo_fbq_splash.png" alt="FBQ" style="width: 60%; height: auto;">
</p>

<br>

---

# Fish Behavior Quantifier FBQ

**Annotation-free multi-fish tracking and behavior quantification for laboratory videos.**

FBQ is a markerless multi-fish tracking system for laboratory tank recordings.
It estimates an oriented bounding box (OBB) and head direction for every fish
while maintaining individual identities over time. No manual training
annotation and no physical marking is required — the model learns directly from
the videos you provide.

This is a **derivative work** of the original AMADEUS project (Notomi,
Matsumura & Dobata, 2026), reworked for fish behavior work. See
[Attribution and upstream](#attribution-and-upstream) and
[License](#license).

## Acknowledgements

This application was developed in the **[Laboratory of Professor Chung-Der Hsiao](https://cdhsiao.weebly.com/pi-cv.html)** in collaboration with **Chung Yuan Christian University, Taiwan 🇹🇼**. Special credit and sincere gratitude are extended to **Professor Hsiao**, who shared his extensive research experience in biology and multiple domains, providing invaluable guidance and supervision throughout the development of this application.

<p align="center">
  <a href="https://www.cycu.edu.tw/">
    <img src="https://raw.githubusercontent.com/yousaf2018/EthoGrid/main/images/cycu.jpg" alt="Chung Yuan Christian University Logo" width="250">
  </a>
</p>

---
## Install and launch

Download and run the installer for your operating system. Initial setup
requires an internet connection: the installer provisions a pinned `uv`, a
Python 3.10–3.12 environment, the PyTorch build matched to your GPU, and a
checksum-pinned FFmpeg.

| Operating system | Launcher |
| --- | --- |
| Windows 10 / 11 | Double-click `FBQ.bat` |
| macOS on Apple Silicon | Double-click `FBQ.command` |
| Linux / WSL2 | Run `bash FBQ.sh` |

After setup, use the desktop shortcut on Windows or run `fbq` (or the short
alias `fq`) in a terminal.

**Hardware.** NVIDIA GPUs on Windows and Linux, Apple Silicon on macOS, and
some AMD GPUs via ROCm are supported. 8 GB or more of VRAM is recommended for
NVIDIA and AMD; operation has been confirmed on a 4 GB NVIDIA laptop GPU.
Requirements scale with resolution and fish count.

## Run tracking

1. Open **Easy Tracking** and select the video.
2. Select **Launch Segmentation**. Configure foreground segmentation so that
   complete, isolated fish are retained as single-fish blobs and unsuitable
   regions are marked as outliers.
3. Specify the number of fish, overlap severity, and whether the fish move
   backward. Select **Processing** to run the workflow.
4. Review the result video and CSV. Use **Refinement** to correct remaining
   errors if needed.

Use **Advanced Tracking** to configure individual stages and parameters, or
**Multi Config Batch** to run several saved configurations.

The full workflow, every parameter and the output format are documented in the
[manual](docs/MANUAL.md).

## Behavior Quantification

The **Behavior Quantification** dialog runs inference on a trained model and
produces per-frame head / center / tail keypoints plus behavioral metrics
(speed, acceleration, angular velocity) for every fish:

- **Head / center / tail keypoints** — head is placed at the leading end of
  movement (fish swims forward, head leads), tail at the trailing end, center
  at the OBB centroid. Orientation follows the AMADEUS direction-estimation
  logic: consecutive OBB long axes are aligned by dot-product sign and the
  global sign is chosen by majority vote against the motion vectors.
- **Direction classes** — 8-way quantized heading (upper, upper_right, right,
  lower_right, lower, lower_left, left, upper_left) matching AMADEUS
  `angle_to_class`.
- **Multi-tank / multi-ROI** — each drawn ROI is tracked independently. Track
  IDs are unique across ROIs and shown as `Tank1-ID1`, `Tank2-ID3`, etc. The
  tank label comes from the ROI name you set in the GUI (default `Tank 1`,
  `Tank 2`, …).
- **Heading arrow** — drawn from tail to head; its length scales with the
  bounding-box size so it stays proportional across fish of different sizes.
- **Configurable display** — arrow size, ID font size, box thickness, keypoint
  radius and font size, and trail length are adjustable in the Behavior
  Quantification dialog under **Display Settings**.
- **Outputs** — an annotated MP4 video (OBB boxes, keypoints, heading arrows,
  trails, speed labels) and a CSV with per-fish per-frame columns:
  `frame, track_id, class_name, cx, cy, w, h, angle, head_x, head_y,
  center_x, center_y, tail_x, tail_y, heading, speed, angular_velocity,
  acceleration`.

## Outputs

The standard Easy Tracking output is a CSV with, per fish and per frame:
identity, OBB center and dimensions, and head direction in degrees. Columns
named `heading` store the head direction. Front and rear keypoints are derived
from OBB geometry; they are not directly detected anatomical landmarks.

```
frame, cx0, cy0, w0, h0, heading0, cx1, cy1, w1, h1, heading1, ...
```

## Recording conditions

FBQ learns directly from the input videos. Recommended conditions:

- a fixed camera viewing the tank from above,
- movement approximately confined to a two-dimensional plane,
- stable illumination and an essentially static background,
- sufficient contrast, resolution and frame rate for reliable per-frame
  tracking,
- a distinguishable body axis with a visually distinguishable front and rear
  (required for head direction),
- enough time with fish separated from one another to generate training data.

Training and analysis videos can be recorded separately under the same
conditions.

## Modes

| Mode | Description | Status |
| --- | --- | --- |
| **Direction estimation** | OBB + head direction + stable identity | default |
| **Without direction estimation** | tracks OBBs and identities, no front/rear assignment | beta, off by default |
| **Variable population** | fish may enter or leave the frame (re-entering fish may receive a new ID) | beta, off by default |
| **Colab** | prepares a Drive bundle + runtime package and opens the notebook on a Google Colab GPU runtime | beta, Linux only |

Check beta results carefully; see the [manual](docs/MANUAL.md) for limitations.

## Roadmap: exact head, center and tail keypoints

The current representation (OBB + an 8-way quantized heading) cannot answer
the questions fish-behavior labs actually ask: where is the snout, where is
the tail tip, how long is the body right now, is the fish nose-down at the
surface, what is the tail-beat frequency.

The next generation replaces the OBB with **three ordered keypoints per
fish — `head`, `center`, `tail`** — and adds a **multi-tank (multi-ROI)
roster** and a **behavior-analysis stage** for top-view and side-view tanks.

The full design, the stage-by-stage migration and the implementation order are
in **[docs/REDESIGN.md](docs/REDESIGN.md)**. The OBB path stays the default
until the keypoint path is validated, so nothing breaks in the meantime.

## Attribution and upstream

FBQ is a derivative work of **AMADEUS — Annotation-free Multi-Animal Direction
Estimation Using Self-supervised learning**.

> Yusuke Notomi, Kentarou Matsumura, and Shigeto Dobata (2026).
> *AMADEUS: Annotation-free multi-animal direction estimation using
> self-supervised learning.* bioRxiv.
> https://doi.org/10.64898/2026.09.19.752854

The upstream project is by Yusuke Notomi and is published at
<https://github.com/jpmyrmecol/AMADEUS>. The original authors retain
copyright in the code they wrote; every source file keeps its
`SPDX-License-Identifier: AGPL-3.0-only` header and its original copyright
line. Please cite the upstream work when you publish results obtained with
this software.

## License

FBQ is distributed under the **GNU Affero General Public License v3.0 only**
(`AGPL-3.0-only`). See [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md).

AGPL-3.0 §5 requires that modified versions convey their source to network
users. Any hosted or distributed FBQ build must ship the corresponding source
code. The Colab runtime package carries a per-file SHA-256 manifest that
records exactly which source a given run used — keep that mechanism intact.
