"""Bridge that instantiates verl's MultiTurnSFTDataset against one parquet shard.

Kept as a separate helper so the annotate script and the unit tests can build
the EXACT verl tokenization (per-turn apply_chat_template, generation_prompt
zeroing, assistant-only loss_mask) without importing the whole trainer stack.
The verl backend path is injected from DTOPD_ROOT, mirroring
tests/sft_rl/test_ptd_jsd_loss.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DTOPD_ROOT = os.environ.get(
    "DTOPD_ROOT", "/inspire/hdd/global_user/mengweicheng-240108120092/lzy"
)
VERL_BACKEND = Path(DTOPD_ROOT) / "fc-opd-storage" / "backends" / "verl-qwen35-v090-cu132"


def _ensure_verl_on_path() -> None:
    verl_pkg = str(VERL_BACKEND)
    if verl_pkg not in sys.path:
        sys.path.insert(0, verl_pkg)


def build_verl_dataset(
    parquet_files: list[str] | str,
    tokenizer=None,
    processor=None,
    limit_rows: int = -1,
):
    """Build a MultiTurnSFTDataset with the same config as run_sft_warmup.sh.

    pad_mode=no_padding / truncation=right / max_length=12288 (the trainer's
    data.max_length); messages_key/images_key defaults match the pool schema.
    """
    _ensure_verl_on_path()

    from omegaconf import DictConfig

    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    config = DictConfig(
        {
            "pad_mode": "no_padding",
            "truncation": "right",
            "max_length": 12288,
            "messages_key": "messages",
            "image_key": "images",
            "video_key": "videos",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": None,
            "apply_chat_template_kwargs": {},
            "shuffle": False,
            "seed": None,
            "ignore_input_ids_mismatch": False,
        }
    )
    return MultiTurnSFTDataset(
        parquet_files=parquet_files,
        tokenizer=tokenizer,
        config=config,
        processor=processor,
        max_samples=limit_rows,
    )
