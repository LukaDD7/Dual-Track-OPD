#!/usr/bin/env python3
"""Compose and validate the exact Hydra config used by the GKD smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REQUIRED = (
    "data.train_files",
    "data.train_batch_size",
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.actor.micro_batch_size",
    # Base MegatronWorker reads these directly before dataclass defaults apply.
    "actor_rollout_ref.actor.ppo_mini_batch_size",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.update_weights_bucket_megabytes",
    "actor_rollout_ref.teacher.server_ip",
    "actor_rollout_ref.teacher.server_port",
    "trainer.total_training_steps",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="on_policy_distill_trainer")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides

    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    OmegaConf.resolve(cfg)

    missing = [key for key in REQUIRED if OmegaConf.select(cfg, key) is None]
    if missing:
        raise SystemExit("missing required GKD config keys: " + ", ".join(missing))

    rollout = cfg.actor_rollout_ref.rollout
    actor = cfg.actor_rollout_ref.actor
    if rollout.name != "vllm" or rollout.mode != "async":
        raise SystemExit("compatible integrated GKD requires rollout name=vllm, mode=async")
    if int(rollout.n) != 1:
        raise SystemExit("text smoke requires exactly one rollout per prompt")
    if int(actor.micro_batch_size) <= 0 or int(actor.ppo_mini_batch_size) <= 0:
        raise SystemExit("actor batch sizes must be positive")
    if int(cfg.trainer.n_gpus_per_node) <= 0 or int(cfg.rollout.n_gpus_per_node) <= 0:
        raise SystemExit("actor and rollout resource pools must each contain a GPU")

    # Confirm that this checkout actually registers the requested rollout pair.
    from verl.workers.rollout import get_rollout_class

    rollout_cls = get_rollout_class(str(rollout.name), str(rollout.mode))
    if rollout_cls.__name__ != "vLLMAsyncRollout" or not hasattr(rollout_cls, "generate_sequences"):
        raise SystemExit(f"expected integrated in-process vLLMAsyncRollout, got {rollout_cls}")

    summary = {
        "rollout_class": f"{rollout_cls.__module__}.{rollout_cls.__name__}",
        "model": cfg.actor_rollout_ref.model.path,
        "train_batch_size": int(cfg.data.train_batch_size),
        "actor_micro_batch_size": int(actor.micro_batch_size),
        "rollout_n": int(rollout.n),
        "actor_pool_gpus": int(cfg.trainer.n_gpus_per_node),
        "rollout_pool_gpus": int(cfg.rollout.n_gpus_per_node),
        "steps": int(cfg.trainer.total_training_steps),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("GKD Hydra preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
