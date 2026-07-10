#!/usr/bin/env python3
"""Sync vLLM engine init helpers for GKD rollout workers.

The compatible verl checkout (d8e97e17) provides vLLMAsyncRollout which is
designed for the async-server pattern: the inference engine is initialised via
ZMQ messages sent by ExternalZeroMQDistributedExecutor.  The GKD recipe does
not launch an async server — it owns the rollout worker directly and expects a
ready, in-process engine.

These helpers bridge that gap without rewriting either codebase.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
import torch.distributed

logger = logging.getLogger(__name__)


def _build_vllm_engine_args(
    rollout_config: Any,          # RolloutConfig
    model_config: Any,            # HFModelConfig
    override_generation_config: dict | None = None,
) -> dict:
    """Build a flat CLI-args dict suitable for AsyncEngineArgs.from_cli_args.

    Mirror vllm_async_server.launch_server as closely as practical.
    """
    cfg = rollout_config
    mc = model_config

    if override_generation_config is None:
        override_generation_config = dict(
            temperature=float(getattr(cfg, "temperature", 1.0)),
            top_k=int(getattr(cfg, "top_k", -1)),
            top_p=float(getattr(cfg, "top_p", 1.0)),
            repetition_penalty=1.0,
            max_new_tokens=int(getattr(cfg, "response_length", 512)),
        )

    args: dict[str, Any] = {
        "dtype": str(getattr(cfg, "dtype", "bfloat16")),
        "load_format": str(getattr(cfg, "load_format", "auto")),
        "skip_tokenizer_init": False,
        "trust_remote_code": bool(getattr(mc, "trust_remote_code", False)),
        "max_model_len": int(getattr(cfg, "max_model_len", getattr(cfg, "prompt_length", 512) + getattr(cfg, "response_length", 512))),
        "max_num_seqs": int(getattr(cfg, "max_num_seqs", 256)),
        "enable_chunked_prefill": bool(getattr(cfg, "enable_chunked_prefill", False)),
        "max_num_batched_tokens": int(getattr(cfg, "max_num_batched_tokens", 8192)),
        "enable_prefix_caching": bool(getattr(cfg, "enable_prefix_caching", False)),
        "enable_sleep_mode": True,
        "disable_custom_all_reduce": True,
        "enforce_eager": bool(getattr(cfg, "enforce_eager", True)),
        "gpu_memory_utilization": float(getattr(cfg, "gpu_memory_utilization", 0.45)),
        "disable_log_stats": bool(getattr(cfg, "disable_log_stats", True)),
        "tensor_parallel_size": int(getattr(cfg, "tensor_model_parallel_size", 1)),
        "seed": int(getattr(cfg, "seed", 0)),
        "override_generation_config": json.dumps(override_generation_config),
        "quantization": getattr(cfg, "quantization", None),
        "hf_overrides": None,
    }

    # Remove None values
    return {k: v for k, v in args.items() if v is not None}


def init_engine_sync(rollout: Any) -> None:
    """Initialise the vLLM inference engine for *rollout* (a vLLMAsyncRollout).

    Must be called after ``_build_rollout()`` (which creates the ZMQ loop and
    binds the IPC socket).  Sends ``init_worker`` / ``init_device`` /
    ``load_model`` through the local ZMQ socket — exactly what
    ExternalZeroMQDistributedExecutor would do, but performed inline so the
    GKD recipe can proceed with its sync path.
    """
    import pickle

    import zmq
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext

    if rollout.inference_engine is not None:
        logger.info("vLLM inference engine already initialized, skipping.")
        return

    # 1. Build CLI args + VllmConfig -------------------------------------------------
    engine_args_dict = _build_vllm_engine_args(rollout.config, rollout.model_config)

    # Parse through vLLM's standard CLI machinery (the way the async server does it).
    # We mimic ``vllm serve <model> --key val ...``.
    cli = ["serve", str(rollout.model_config.local_path)]
    for k, v in engine_args_dict.items():
        if isinstance(v, bool):
            if v:
                cli.append(f"--{k}")
        elif isinstance(v, dict):
            cli.append(f"--{k}")
            cli.append(json.dumps(v))
        else:
            cli.append(f"--{k}")
            cli.append(str(v))

    from vllm.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="vLLM CLI – GKD sync stub")
    # Register the ``serve`` subcommand (required by vLLM's parser).
    import vllm.entrypoints.cli.serve as serve_mod

    subparsers = parser.add_subparsers(required=False, dest="subparser")
    for cmd_mod in [serve_mod]:
        for cmd in cmd_mod.cmd_init():
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)

    parsed = parser.parse_args(args=cli)
    parsed.model = parsed.model_tag  # vLLM normalisation

    engine_args = AsyncEngineArgs.from_cli_args(parsed)
    vllm_config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)

    # 2. Send init commands via ZMQ to *self* ---------------------------------------
    address = rollout.get_zeromq_address()
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(address)

    kwargs = dict(
        vllm_config=vllm_config,
        local_rank=0,
        rank=int(os.environ.get("RANK", 0)),
        distributed_init_method="env://",
        is_driver_worker=True,
    )

    def _zmq_rpc(method: str, args: tuple = (), kw: dict | None = None):
        msg = pickle.dumps((method, args, kw or {}))
        socket.send(msg)
        result = pickle.loads(socket.recv())
        if isinstance(result, Exception):
            raise result
        return result

    try:
        _zmq_rpc("init_worker", args=([kwargs],))
        _zmq_rpc("init_device")
        _zmq_rpc("load_model")
        logger.info("vLLM inference engine initialised via ZMQ loopback (B17).")
    finally:
        socket.close()
        context.term()

    # Verify the engine is ready
    assert rollout.inference_engine is not None, "Engine init failed — inference_engine still None"
