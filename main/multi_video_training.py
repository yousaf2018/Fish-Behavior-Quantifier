# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Training on more than one video.

A run can learn its detector from several videos instead of one. Every video
is prepared in a session of its own, so the fish crops of one video never mix
with the crops of another, and the dataset of the run is then built from the
pools of all the videos. The training that follows uses every video at once.
"""

from __future__ import annotations

import copy
import os
import shutil
from typing import Callable, List, Optional

import yaml

try:
    from .segmentation_metadata import segmentation_paths_for_session
except ImportError:  # Direct execution of main scripts.
    from segmentation_metadata import segmentation_paths_for_session

#: The configuration keys of the stages that prepare the dataset.
DATASET_STAGE_KEYS = (
    "skip_initial_tracking",
    "skip_trajectory_direction_filtering",
    "skip_refine_blobs_through_tracking",
    "skip_paste_blobs_clustered",
    "skip_paste_blobs_with_crossing",
    "skip_cropping",
)


def training_video_paths(cfg: dict) -> List[str]:
    """Return the training videos of a configuration, the main one first."""
    raw = cfg.get("TRAINING_VIDEO_PATHS") or []
    if isinstance(raw, str):
        raw = [raw]
    videos: List[str] = []
    for value in raw:
        text = str(value).strip()
        if text and text not in videos:
            videos.append(text)
    main_video = str(cfg.get("TRAINING_VIDEO_PATH", "") or "").strip()
    if main_video and main_video not in videos:
        videos.insert(0, main_video)
    return videos


def video_session_path(session_path: str, video: str) -> str:
    """The session of one video of a multi-video run."""
    stem = os.path.splitext(os.path.basename(video))[0] or "video"
    return os.path.join(str(session_path), "training_videos", stem)


def missing_segmentation_outputs(videos: List[str], session_path: str) -> List[str]:
    """Return the videos whose segmentation output is not in the session.

    Every video of a multi-video run is prepared from its own segmentation.
    A video whose segmentation is missing cannot be prepared, and the run
    says which videos those are before it starts.
    """
    missing: List[str] = []
    for video in videos:
        sub_session = video_session_path(session_path, video)
        pickle_path, background_path = segmentation_paths_for_session(sub_session, video)
        if not os.path.isfile(pickle_path) or not os.path.isfile(background_path):
            missing.append(os.path.basename(video))
    return missing


def copy_segmentation_output(session_path: str, sub_session: str, video: str) -> bool:
    """Copy the segmentation of the reference video into the session of that video.

    The reference video was segmented into the session of the run. Its
    video session needs the same files, so the dataset of that video is
    prepared from the very same segmentation.
    """
    source_pickle, source_background = segmentation_paths_for_session(session_path, video)
    target_pickle, target_background = segmentation_paths_for_session(sub_session, video)
    if os.path.isfile(target_pickle) and os.path.isfile(target_background):
        return False
    if not os.path.isfile(source_pickle) or not os.path.isfile(source_background):
        return False
    os.makedirs(os.path.dirname(target_pickle), exist_ok=True)
    shutil.copy2(source_pickle, target_pickle)
    shutil.copy2(source_background, target_background)
    source_config = os.path.join(str(session_path), "segmentation", "segmentation_gui_config.json")
    target_config = os.path.join(os.path.dirname(target_pickle), "segmentation_gui_config.json")
    if os.path.isfile(source_config) and not os.path.isfile(target_config):
        shutil.copy2(source_config, target_config)
    return True


def write_run_config(path: str, run_cfg: dict) -> str:
    """Write a configuration for a stage run and return its path."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(run_cfg, handle, sort_keys=False, allow_unicode=True)
    return path


def run_multi_video_training(
    cfg: dict,
    cfg_path: str,
    run_independent_stages: Callable[[dict, str], None],
) -> Optional[str]:
    """Build the dataset of every training video and combine the results.

    The stages that prepare the dataset run once per video, each in a session
    of its own under ``training_videos``. The dataset stage of the run then
    takes the pools of all the videos, and the training uses all of them.
    """
    videos = training_video_paths(cfg)
    if len(videos) <= 1:
        return None
    session_path = str(cfg["SESSION_PATH"])
    main_video = str(cfg.get("TRAINING_VIDEO_PATH", "") or "").strip()
    other_videos = [video for video in videos if video != main_video]
    missing = missing_segmentation_outputs(other_videos, session_path)
    if missing:
        raise FileNotFoundError(
            "The segmentation of these videos is missing: "
            + ", ".join(missing)
            + ". Segment them with the same settings before the run "
            "(Segment other videos with these settings)."
        )
    source_dirs: List[str] = []
    for index, video in enumerate(videos, start=1):
        sub_session = video_session_path(session_path, video)
        os.makedirs(sub_session, exist_ok=True)
        if video == main_video:
            # The reference video was segmented into the session of the run.
            copy_segmentation_output(session_path, sub_session, video)
        run_cfg = copy.deepcopy(cfg)
        run_cfg["TRAINING_VIDEO_PATH"] = video
        run_cfg["SESSION_PATH"] = sub_session
        # Every video is prepared from its own segmentation, so the paths of
        # the session of that video are the ones the run has to use.
        pickle_path, background_path = segmentation_paths_for_session(sub_session, video)
        run_cfg["PICKLE_PATH"] = pickle_path
        run_cfg["BACKGROUND_PATH"] = background_path
        # The dataset of every video is combined afterwards, so the dataset
        # stage belongs to the main run.
        run_cfg["skip_creating_direction_dataset"] = True
        run_cfg["skip_behavior_analysis"] = True
        run_cfg.pop("TANKS", None)
        run_cfg.pop("CREATE_DATASET_SOURCE_DIRS", None)
        sub_cfg_path = write_run_config(os.path.join(sub_session, "config.yaml"), run_cfg)
        print(
            f"\n[TRAINING VIDEOS] {index}/{len(videos)} preparing the dataset of {video}",
            flush=True,
        )
        run_independent_stages(run_cfg, sub_cfg_path)
        source_dirs.append(os.path.join(sub_session, "single_animal_images"))
    print(
        f"\n[TRAINING VIDEOS] combining the datasets of {len(videos)} videos",
        flush=True,
    )
    cfg["CREATE_DATASET_SOURCE_DIRS"] = source_dirs
    for key in DATASET_STAGE_KEYS:
        cfg[key] = True
    return cfg_path
