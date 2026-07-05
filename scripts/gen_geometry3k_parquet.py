#!/usr/bin/env python
"""Generate full Geometry3K verl parquet with condition_inputs."""
import sys
sys.path.insert(0, '/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/src')

import re
import numpy as np
import pandas as pd
from pathlib import Path
from dual_track_opd.fc_opd.geometry3k_adapter import normalize_geometry3k_record
from dual_track_opd.fc_opd.degradation import (
    materialize_degraded_image,
    precomputed_degraded_transform,
)

SRC = '/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/geometry3k/data/train-00000-of-00001.parquet'
OUT = '/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/geometry3k_full/train.parquet'
DEG = Path('/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/artifacts/fc_opd/degraded_images/geometry3k')
DEG.mkdir(parents=True, exist_ok=True)

IMG_DIR = DEG.parent / 'geometry3k_images'
IMG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(SRC)
print(f"Loaded {len(df)} Geometry3K records")
Path(OUT).parent.mkdir(parents=True, exist_ok=True)

rows = []
for i in range(len(df)):
    row = df.iloc[i].to_dict()

    # Extract image bytes and save to disk
    img_bytes = None
    images_field = row.get('images', [])
    if isinstance(images_field, (list, tuple, np.ndarray)) and len(images_field) > 0:
        first_img = images_field[0]
        if isinstance(first_img, dict) and 'bytes' in first_img:
            img_bytes = first_img['bytes']

    img_path = ''
    if img_bytes:
        img_path = str(IMG_DIR / f'{i}.png')
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

    # Parse question and answer from problem text
    problem = str(row.get('problem', ''))
    answer_raw = row.get('answer', '')

    # Remove <image> tag if present
    problem = problem.replace('<image>', '').strip()

    # Try to extract choices from problem text
    lines = problem.split('\n')
    choices = []
    question_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^[A-D][\.\s\)]', stripped):
            choices.append(stripped)
        else:
            question_lines.append(stripped)

    question = '\n'.join(q for q in question_lines if q).strip()
    if not question:
        question = problem

    answer = str(answer_raw).strip().upper()
    # If answer is numeric, map to choice letter
    if answer.isdigit():
        idx = int(answer)
        if 0 <= idx < len(choices):
            answer = chr(ord('A') + idx)
        elif 1 <= idx <= len(choices):
            answer = chr(ord('A') + idx - 1)

    # Generate paper-faithful degraded image: 10% bilinear downsample + nearest upsample.
    degraded_img = ''
    try:
        if img_path and Path(img_path).exists():
            degraded_img = materialize_degraded_image(
                img_path,
                mode='lowres_10pct_nearest',
                degraded_dir=str(DEG),
            )
    except Exception as e:
        print(f'Warning: failed to create degraded image for {i}: {e}')
    if not degraded_img:
        degraded_img = img_path

    ci = {  # plain dict, no dataclass objects
        'full_image': {'path': img_path},
        'degraded_image': {
            'path': degraded_img,
            'transform': precomputed_degraded_transform('lowres_10pct_nearest'),
        },
        'free_caption': 'A geometry diagram with labeled points and lines.',
        'task_evidence': 'Points, lines, and angles are labeled in the diagram.',
        'task_visible_evidence': 'The diagram shows labeled geometric elements.',
        'task_infer_evidence': 'Geometric relationships can be inferred from the labels.',
        'task_solve_evidence': 'Apply geometric theorems using the labeled elements.',
    }

    # Build images field: list of {'bytes': ..., 'path': ...} (same format as smoke)
    images_field = []
    if img_bytes:
        images_field.append({'bytes': img_bytes, 'path': img_path})

    canonical_question = question
    if choices:
        canonical_question += '\n\nChoices: ' + ' '.join(choices)

    # Build prompt as chat template with <image> tag. VA-OPD should not inject
    # XML/format constraints unless the teacher sees the exact same prompt.
    prompt_content = f'<image>\n{canonical_question}'
    prompt_field = [{'role': 'user', 'content': prompt_content}]

    rows.append({
        'data_source': 'geometry3k',
        'prompt': prompt_field,
        'images': images_field,
        'ability': 'math',
        'reward_model': {'style': 'rule', 'ground_truth': answer},
        'extra_info': {
            'question': canonical_question,
            'choices': choices,
            'answer': answer,
            'condition_inputs': ci,
        },
        'question': canonical_question,
        'condition_inputs': ci,
        'choices': choices,
        'answer': answer,
        'sample_uid': f'geo3k:{i}',
    })

    if (i + 1) % 200 == 0:
        print(f"  processed {i + 1}/{len(df)}")

pd.DataFrame(rows).to_parquet(OUT)
print(f"Wrote {len(rows)} rows -> {OUT}")
