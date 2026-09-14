#!/usr/bin/env python3
"""Patch B17: initialise vLLM engine in GKD rollout worker + fix model path.

Two changes in the recipe's megatron_workers.py:

1) ``init_model`` of the rollout worker – call ``init_engine_sync()`` after
   ``_build_rollout()`` so that ``self.rollout.inference_engine`` is ready
   before the first ``sync_rollout_weights``.

2) ``sync_rollout_weights`` – the model access path
   ``inference_engine.llm_engine.model_executor.driver_worker.worker...``
   is from an older vLLM version.  The compatible checkout's
   WorkerWrapperBase exposes ``.worker.model_runner.model`` instead.

Idempotent.
"""

from __future__ import annotations

import re
from pathlib import Path

# ── patch #1: init_model – inject init_engine_sync after _build_rollout ──────
# We match the line that calls _build_rollout inside the rollout worker's
# init_model (recipe/gkd/megatron_workers.py ~line 707).

INIT_MODEL_BROKEN = (
    r"(?P<indent>\s+)self\._build_rollout\(trust_remote_code=self\.config\.model\.get\(\"trust_remote_code\", False\)\)\n"
    r"\s+self\.rollout_device_mesh = self\.rollout\.device_mesh\n"
    r"\s+log_gpu_memory_usage\(\"After rollout init\", logger=logger\)"
)

INIT_MODEL_FIX = (
    r"\g<indent>self._build_rollout(trust_remote_code=self.config.model.get("
    '"trust_remote_code"'
    r", False))\n"
    r"\g<indent>self.rollout_device_mesh = self.rollout.device_mesh\n"
    r"\g<indent># B17: initialise vLLM inference engine (GKD owns the rollout directly).\n"
    r"\g<indent>from gkd_vllm_helpers import init_engine_sync\n"
    r"\g<indent>init_engine_sync(self.rollout)\n"
    r"\g<indent>log_gpu_memory_usage("
    '"After rollout init"'
    r", logger=logger)"
)

INIT_MODEL_MARKER = "from gkd_vllm_helpers import init_engine_sync"

# ── patch #2: sync_rollout_weights – fix model access path ──────────────────
# Old path (worked with earlier vLLM):
#   self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
# New path (WorkerWrapperBase in vLLM 0.11):
#   self.rollout.inference_engine.worker.model_runner.model

OLD_MODEL_PATH = (
    r"self\.rollout\.inference_engine\.llm_engine\.model_executor\.driver_worker\.worker\.model_runner\.model"
)

NEW_MODEL_PATH = (
    r"self.rollout.inference_engine.worker.model_runner.model"
)

SYNC_MARKER = "self.rollout.inference_engine.worker.model_runner.model"


def patch_source(source: str) -> tuple[str, list[str]]:
    """Apply both patches.  Returns (patched_source, [actions])."""
    actions: list[str] = []
    patched = source

    # Patch #1: init_model engine init
    if INIT_MODEL_MARKER not in patched:
        new, count = re.subn(INIT_MODEL_BROKEN, INIT_MODEL_FIX, patched)
        if count:
            patched = new
            actions.append("init_model:ok")
        else:
            actions.append("init_model:no_match")
    else:
        actions.append("init_model:skip")

    # Patch #2: sync_rollout_weights model path
    if SYNC_MARKER not in patched:
        new, count = re.subn(OLD_MODEL_PATH, NEW_MODEL_PATH, patched)
        if count:
            patched = new
            actions.append("model_path:ok")
        else:
            actions.append("model_path:no_match")
    else:
        actions.append("model_path:skip")

    return patched, actions


def patch_file(path: Path) -> str:
    original = path.read_text()
    patched, actions = patch_source(original)
    status = ", ".join(actions)
    if any(a.endswith(":ok") for a in actions):
        path.write_text(patched)
    return status


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("target", type=Path)
    args = p.parse_args()
    print(f"B17 patch: {patch_file(args.target)} - {args.target}")


if __name__ == "__main__":
    main()
