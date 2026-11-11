# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Direction-free project configuration; numerical stages live in matching scripts."""

import sys
from pathlib import Path
_MAIN_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_MAIN_DIR), str(_MAIN_DIR.parent)):
    if _path not in sys.path:
        sys.path.append(_path)

from pathlib import Path
import copy
import json
import shutil
import yaml
from path_utils import resolve_config_paths

MODE_DIR = 'without_direction_estimation'
SCHEMA = 'single-animal-obb-horizontal-axial-embedding-v1'

# The direction-free session is a directory of its own, so a stage that ran in
# the parent session has produced nothing here. These two stages produce the
# input of the stages that follow them, so their skip flag is only kept when
# this session already holds the output. Every other skip flag keeps the
# meaning the user gave it.
_REQUIRED_STAGE_ARTIFACTS = (
    ('skip_initial_tracking', (('initial_tracking', 'track_assignments.csv'),)),
    ('skip_trajectory_direction_filtering', (('single_animal_images', 'object_pool', 'manifest.csv'),)),
)


def _artifact_ready(path: Path) -> bool:
    """A stage output counts as present when the file exists or the folder holds files."""
    if path.is_file():
        return True
    if path.is_dir():
        return any(entry.is_file() for entry in path.iterdir())
    return False


def _remap_skip_flags(cfg: dict, target: Path) -> None:
    """Drop the skip flags whose output is missing from the direction-free session.

    The parent session and this session are different directories. A skip flag
    inherited from the parent would leave the next stage without its input, so
    the flag is only kept when this session already holds the output.
    """
    for key, candidates in _REQUIRED_STAGE_ARTIFACTS:
        if not cfg.get(key):
            continue
        if any(_artifact_ready(target.joinpath(*parts)) for parts in candidates):
            continue
        cfg[key] = False
        print(
            f'[without_direction_estimation] {key} was reset: '
            f'{target.joinpath(*candidates[0])} is missing, so the stage runs again here.',
            flush=True,
        )


def prepare_config(path):
    path = Path(path).resolve()
    cfg = resolve_config_paths(yaml.safe_load(path.read_text(encoding='utf-8')))
    if not cfg.get('WITHOUT_DIRECTION_ESTIMATION', False):
        raise ValueError('This entry point requires WITHOUT_DIRECTION_ESTIMATION: true.')
    if cfg.get('_DIRECTIONLESS_SCHEMA') == SCHEMA:
        target = Path(cfg['SESSION_PATH'])
        _remap_skip_flags(cfg, target)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
        return str(path)
    source = Path(cfg['SESSION_PATH']).resolve()
    target = source / MODE_DIR
    target.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg['SESSION_PATH'] = str(target)
    cfg['_DIRECTIONLESS_SCHEMA'] = SCHEMA
    cfg['_SOURCE_SESSION_PATH'] = str(source)
    cfg['YOLO_DATASET_DIR'] = str(target / 'yolo_dataset')
    cfg['training'] = {**cfg.get('training', {}), 'CLASS_NAMES': ['animal'], 'NUM_CLASSES': 1, 'USE_DIRECTION_CLASSES': True}
    # Custom direction-bearing datasets cannot silently cross the mode boundary.
    if cfg.get('CREATE_DATASET_SOURCE_DIRS'):
        raise ValueError('Clear CREATE_DATASET_SOURCE_DIRS before the first direction-free run.')
    # These two small files are consumed by the shared embedding segmenter.
    # Raw videos and segmentation pickles are referenced in place, never edited.
    seg = target / 'segmentation'
    seg.mkdir(exist_ok=True)
    for name in ('background.png', 'segmentation_gui_config.json'):
        src = source / 'segmentation' / name
        if src.is_file():
            shutil.copy2(src, seg / name)
    (target / 'mode.json').write_text(json.dumps({
        'schema': SCHEMA, 'source_session': str(source),
        'class_names': ['animal'], 'embedding_axis': 'horizontal, modulo 180 degrees',
        'final_csv_fields_per_id': ['cx', 'cy', 'w', 'h'],
    }, indent=2), encoding='utf-8')
    output = target / path.name
    _remap_skip_flags(cfg, target)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return str(output)

