#!/usr/bin/env python3
"""TailSFT ℓ0 pre-annotation: per-example init CE under the BASE model.

TailSFT (arXiv:2608.25756, Algorithm 1) filters ONLINE during SFT using the
margin  m_t(i) = ℓ_t(i) − ℓ_0(i)  where ℓ_t is the CURRENT policy's
length-normalized mean CE over assistant target tokens and ℓ_0 is the same
quantity under the initial policy π_0 (= base Qwen3-VL-8B-Instruct).

This script computes ℓ_0 for every train shard of the MMF-only pool ONCE
(offline, no_grad, bf16) and writes a NEW pool dir with an `init_ce` float
column appended:

  in : $DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft/
       sft_warmup_train__part_%04d.parquet  (113,537 rows, 77 shards)
       sft_warmup_val__part_%04d.parquet    (copied UNCHANGED — val has no
       init_ce, so the trainer's val path falls back to plain CE)
  out: .../sft_rl/mmf_only_sft_tailsft/
       sft_warmup_train__part_%04d.parquet  (same rows + init_ce column)
       sft_warmup_val__part_%04d.parquet    (byte-identical copy)

Tokenization/masking must EXACTLY match verl MultiTurnSFTDataset so ℓ_0 and
ℓ_t are the same functional: per-turn apply_chat_template, generation_prompt
masked to 0, assistant-only loss_mask, NO left-shift here (the shift by one
is applied when both the trainer and this script gather log_probs at
label-aligned positions — see `init_ce_from_logits`).

Single example ≈ one forward of the full sequence under bf16 with logits
materialized only in chunks (log_softmax over 151k vocab is computed via a
chunked gather at target positions to bound memory). Runs on 1 GPU; the
paper's ℓ_0 is a fixed constant per example so this is a one-off cost.

Launch (GPU node, offline):
  CUDA_VISIBLE_DEVICES=0 python scripts/sft_rl/annotate_tailsft_init_ce.py \
    --pool-dir $DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft \
    --out-dir  $DTOPD/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft_tailsft \
    --model    $DTOPD/models/Qwen3-VL-8B-Instruct
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))


def chunked_target_logprobs(
    logits: "torch.Tensor",  # (seq_len, vocab)
    targets: "torch.Tensor",  # (seq_len,) left-shifted labels: targets[i] = input_ids[i+1]
    chunk: int = 2048,
) -> "torch.Tensor":
    """log P(target_i | prefix) for every position, computed in chunks.

    Only the gathered target column of log_softmax is kept (seq_len floats),
    so memory is O(seq_len) beyond the logits chunk itself.
    """
    import torch

    out = torch.empty(targets.shape[0], dtype=torch.float32, device=targets.device)
    for start in range(0, targets.shape[0], chunk):
        end = min(start + chunk, targets.shape[0])
        lp = torch.log_softmax(logits[start:end].float(), dim=-1)
        out[start:end] = lp.gather(-1, targets[start:end].unsqueeze(-1)).squeeze(-1)
    return out


def init_ce_from_logits(
    logits: "torch.Tensor",  # (seq_len, vocab) — positions 0..seq_len-1 predict input_ids[1..seq_len]
    input_ids: "torch.Tensor",  # (seq_len,)
    loss_mask: "torch.Tensor",  # (seq_len,) 1 on assistant target tokens (unshifted)
) -> tuple[float, int]:
    """ℓ = mean CE over assistant target tokens, aligned like verl sft_loss.

    verl computes log_prob[i] = log P(input_ids[i+1] | prefix) and rolls
    loss_mask left by one so rolled_mask[i] selects positions whose NEXT token
    is a target. Mirror that exactly:
      targets[i]   = input_ids[i+1]  (last position has no target)
      valid[i]     = loss_mask[i+1]
    Returns (mean_ce over valid positions, n_valid). (0.0, 0) if no targets.
    """
    import torch

    seq_len = input_ids.shape[0]
    if seq_len < 2:
        return 0.0, 0
    targets = input_ids[1:]
    valid = loss_mask[1:].to(torch.bool)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return 0.0, 0
    lp = chunked_target_logprobs(logits, targets)
    ce = -(lp[valid].float()).mean().item()
    return ce, n_valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-dir",
        default="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft",
        help="source pool (sft_warmup_train/val__part_*.parquet)",
    )
    parser.add_argument(
        "--out-dir",
        default="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_only_sft_tailsft",
    )
    parser.add_argument(
        "--model",
        default="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--max-shards", type=int, default=-1, help="smoke cap")
    parser.add_argument("--limit-rows", type=int, default=-1, help="smoke cap per script run")
    parser.add_argument("--batch", type=int, default=1, help="sequences per forward (memory-bound)")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    pool = Path(args.pool_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_shards = sorted(pool.glob("sft_warmup_train__part_*.parquet"))
    val_shards = sorted(pool.glob("sft_warmup_val__part_*.parquet"))
    if not train_shards:
        raise SystemExit(f"no train shards under {pool}")
    if args.max_shards > 0:
        train_shards = train_shards[: args.max_shards]

    print(f"[tailsft-init-ce] model={args.model}")
    print(f"[tailsft-init-ce] {len(train_shards)} train shards -> {out_dir}")

    # --- val: byte-identical copy (trainer val path has no init_ce -> plain CE)
    for vp in val_shards:
        dst = out_dir / vp.name
        if not dst.exists():
            dst.write_bytes(vp.read_bytes())
            print(f"[tailsft-init-ce] copied val shard {vp.name}")
        else:
            print(f"[tailsft-init-ce] val shard exists, skip {vp.name}")

    # --- train: reuse verl's exact tokenization by instantiating the dataset
    # class against each shard, then forward the base model once per row.
    from verl_utils_bridge import build_verl_dataset  # local helper, see below

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    n_rows = n_done = 0
    sum_ce = 0.0
    for shard_idx, shard in enumerate(train_shards):
        out_path = out_dir / shard.name
        if out_path.exists():
            print(f"[tailsft-init-ce] shard {shard.name} already annotated, skip")
            continue

        table = pq.read_table(shard)
        raw_rows = table.to_pylist()
        if 0 < args.limit_rows < len(raw_rows):
            raw_rows = raw_rows[: args.limit_rows]
        # write pool keeps messages/images/source/image_hash; init_ce appended
        raw_by_key = [
            {k: row[k] for k in ("messages", "images", "source", "image_hash")} for row in raw_rows
        ]

        # MultiTurnSFTDataset requires a REAL tokenizer: its constructor calls
        # extract_system_prompt_and_generation(self.tokenizer, ...) with no None
        # guard, and the real trainer passes model_config.tokenizer. The
        # processor wraps that same tokenizer, so use it here.
        dataset = build_verl_dataset(
            parquet_files=[str(shard)],
            tokenizer=processor.tokenizer,
            processor=processor,
            limit_rows=args.limit_rows,
        )

        init_ces: list[float] = []
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                input_ids = sample["input_ids"].to(device).unsqueeze(0)
                loss_mask = sample["loss_mask"].to(device)
                pos_ids = sample["position_ids"].to(device)
                if pos_ids.dim() == 2:
                    pos_ids = pos_ids.unsqueeze(1)  # (4,seq) -> (4,1,seq): mrope contract is (4,b,s)
                mm = sample.get("multi_modal_inputs", None)

                # mm keys follow the trainer's FLATTENED contract (no batch dim):
                # pixel_values (n_patch,C,H,W), image_grid_thw (n_img,3), ...
                # (the trainer cats across samples via extract_multi_modal_inputs,
                # and NEVER unsqueezes — grid tensors must stay 2-D or the vision
                # tower's `for t, h, w in grid_thw.tolist()` unpacking breaks)
                model_kwargs = {}
                if mm is not None:
                    for k, v in mm.items():
                        v = v.to(device)
                        if "pixel" in k:
                            v = v.to(dtype)
                        model_kwargs[k] = v

                logits = model(
                    input_ids=input_ids,
                    position_ids=pos_ids,
                    use_cache=False,
                    **model_kwargs,
                ).logits[0]  # (seq_len, vocab)
                ce, _ = init_ce_from_logits(logits, input_ids[0], loss_mask)
                init_ces.append(ce)
                n_rows += 1
                sum_ce += ce
                if (i + 1) % 50 == 0:
                    print(
                        f"[tailsft-init-ce] shard {shard_idx} row {i + 1}/{len(dataset)} "
                        f"running mean ℓ0 = {sum_ce / max(n_rows, 1):.4f}",
                        flush=True,
                    )

        schema = table.schema.append(pa.field("init_ce", pa.float64()))
        rows_out = [dict(r, init_ce=ce) for r, ce in zip(raw_by_key, init_ces)]
        pq.write_table(pa.Table.from_pylist(rows_out, schema=schema), str(out_path))
        n_done += len(rows_out)
        print(
            f"[tailsft-init-ce] wrote {out_path.name}: {len(rows_out)} rows, "
            f"shard mean ℓ0 = {sum(init_ces) / max(len(init_ces), 1):.4f}"
        )

    if n_rows:
        print(
            f"[tailsft-init-ce] DONE. rows annotated={n_rows} mean ℓ0={sum_ce / n_rows:.4f}"
        )
    if math.isclose(n_rows, 0) and n_done == 0:
        print("[tailsft-init-ce] nothing to do (all shards already annotated)")


if __name__ == "__main__":
    main()
