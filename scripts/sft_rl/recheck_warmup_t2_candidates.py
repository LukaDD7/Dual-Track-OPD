#!/usr/bin/env python3
"""Re-verify fast-scan candidates through the REAL MultiTurnSFTDataset pipeline.

Fast scan (scan_warmup_t2_exact.py) flags rows by replicating the pipeline's
math (image-header-only pads + literal-pad tokenization). This script takes
those candidates and runs the actual dataset class on only those rows' shards,
so each flagged row is confirmed (or cleared) by the exact code path that
crashed training:

  python recheck_warmup_t2_candidates.py \
      --warmup-dir fc-opd-storage/outputs/fc_opd/sft_rl/warmup_t2 \
      --model models/Qwen3-VL-8B-Instruct \
      --candidates fc-opd-storage/logs/warmup_t2_oversized_fast.json \
      --out fc-opd-storage/logs/warmup_t2_oversized_recheck.json \
      --max-length 12288 --workers 48
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path


def _worker(args) -> dict:
    shard, cand_rows, model, max_length = args
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    cfg = OmegaConf.create({
        "max_length": max_length,
        "truncation": "right",
        "pad_mode": "no_padding",
        "image_key": "images",
        "use_dynamic_bsz": True,
    })
    proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    import json as _json
    cid = int(_json.load(open(f"{model}/config.json"))["image_token_id"])

    sp = str(Path(args_warmup_dir) / shard)  # set via global
    ds = MultiTurnSFTDataset([sp], model, cfg, proc, max_samples=-1)
    rows = pq.read_table(sp).to_pylist()

    confirmed, cleared = [], []
    for c in cand_rows:
        i = c["idx"]
        it = ds[i]
        n_tok = int((it["input_ids"] == cid).sum())
        g = it["multi_modal_inputs"].get("image_grid_thw")
        n_feat = int(sum(int(t) * (int(h) // 2) * (int(w) // 2)
                         for t, h, w in g.tolist())) if g is not None else 0
        seqlen = len(it["input_ids"])
        rec = {
            "shard": shard,
            "idx": i,
            "n_tok": n_tok,
            "n_feat": n_feat,
            "seqlen": seqlen,
            "grids": g.tolist() if g is not None else None,
            "source": rows[i].get("source", "?") if i < len(rows) else "?",
            "fast_kind": c.get("kind"),
            "fast_seqlen": c.get("seqlen"),
        }
        if n_tok != n_feat:
            rec["kind"] = "mismatch"
            confirmed.append(rec)
        elif seqlen >= max_length:
            rec["kind"] = "oversize_truncated"
            confirmed.append(rec)
        else:
            rec["kind"] = "cleared"
            cleared.append(rec)
    return {"shard": shard, "confirmed": confirmed, "cleared": cleared}


args_warmup_dir = ""


def main() -> None:
    global args_warmup_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=12288)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    args_warmup_dir = args.warmup_dir

    cands = json.loads(Path(args.candidates).read_text())
    by_shard: dict[str, list[dict]] = {}
    for c in cands:
        by_shard.setdefault(c["shard"], []).append(c)
    jobs = [(s, rows, args.model, args.max_length)
            for s, rows in sorted(by_shard.items())]

    mp.set_start_method("fork", force=True)
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, jobs)

    confirmed = [r for res in results for r in res["confirmed"]]
    cleared = [r for res in results for r in res["cleared"]]
    out = {
        "confirmed": confirmed,
        "cleared": cleared,
        "n_candidates": len(cands),
        "n_confirmed": len(confirmed),
        "n_cleared": len(cleared),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    from collections import Counter
    print(f"candidates: {len(cands)}  confirmed: {len(confirmed)}  "
          f"cleared: {len(cleared)}")
    print("confirmed by kind:", dict(Counter(r["kind"] for r in confirmed)))
    print("confirmed by source:",
          dict(Counter(r["source"] for r in confirmed).most_common()))


if __name__ == "__main__":
    main()
