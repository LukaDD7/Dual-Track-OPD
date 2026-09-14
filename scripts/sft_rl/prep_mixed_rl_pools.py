#!/usr/bin/env python3
"""Prepare the mixed-difficulty RL training pool (MMF hard + Cauldron broad).

Why: the previous PTD-PO run (qwen3vl_grpo_mmf95k_ptd_sft6939_coef5e4_steps50)
trained on the MMF hint pool, which is 100% pass_rate=0 hardest samples. Eval
evidence (docs/sft_rl_full_results_ptdpo_20260831.md) shows the RL gain
concentrated on long-CoT benchmarks (ReMI exact 0.3623, best of four arms)
while MCQ short-answer benchmarks regressed (ViewSpatial 0.4231->0.2272,
MMMU-Pro 0.3960->0.2445, MMBench 85.1->73.0, GQA 0.6163->0.5542 vs base).
verl's sampler shuffles pool-level, so the pool mix is the batch mix; making
Cauldron's broad-coverage MCQ rows a first-class share of the pool lets GRPO /
PTD-PO see reward on exactly the distribution that regressed.

Inputs (both already on disk):
  mmf_hint      hint_pools/mmf_hint/mmf_hint__part_*.parquet  (95,128 rows,
                images inline as {bytes, path}, carries hint /
                prompt_with_hint / hint_reason from the 8010 teacher pipeline)
  cauldron_hint hint_pools/cauldron_hint/cauldron_hint__part_*.parquet
                (Cauldron rows WITH generated hints, images external
                image_url references -> read bytes from disk when inlining)

What this script does:
  1. Filter both pools to rule-verifiable GTs under the FIXED reward gates
     (gt strip trailing period; letter A-H; yesno; pure numeric) — the same
     logic as scripts/sft_rl/mmf_reward.py so the train pool and the reward
     agree.
  2. Deterministically sample per-source rows:
       - Cauldron: cap per closed-form subset (default 1500/subset, drop
         subsets with < min-per-subset rule-ok rows).
       - MMF: take a share of hard rows (default 12,000) stratified over
         source to keep the hard-tail diversity of the pool.
  3. Inline Cauldron image bytes (read from sft_full/images) into
     {bytes, path} image structs — verl requires non-None bytes on the dict
     path (rl_dataset.py opens BytesIO(image["bytes"]) whenever the key is
     present) and concatenate_datasets needs a single unified schema.
  4. Keep MMF hint columns; Cauldron rows carry their hint columns too.
     Rows with empty hint simply degrade to plain GRPO under
     algorithm.ptd.threshold=1.0 (agent_loop zero-fills hint tensors).
  5. Write <=1500-row zstd shards + a stats JSON.

Usage:
  python3 scripts/sft_rl/prep_mixed_rl_pools.py \
    --out-dir fc-opd-storage/outputs/fc_opd/sft_rl/rl_mixed_20k
  (defaults: --mmf-rows 12000 --cauldron-per-subset 1500)
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def gt_rule_kind(ground_truth: Any) -> str | None:
    """Classify GT exactly like mmf_reward.compute_score's gates.

    Returns "letter" | "yesno" | "number" | "latex" | None ("other" = drop).
    "latex" rows (has_number / free-form containing digits but non-numeric)
    stay eligible only when --keep-latex is passed; they route to the
    mathruler grader at reward time, same as the MMF pool does.
    """
    gt = str(ground_truth).strip() if ground_truth is not None else ""
    gt = re.sub(r"\.+$", "", gt).strip()
    if not gt:
        return None
    if re.fullmatch(r"\(?([A-Ha-h])\)?\.?", gt):
        return "letter"
    if re.fullmatch(r"(yes|no|true|false)", gt.lower()):
        return "yesno"
    if re.fullmatch(r"-?\d+(\.\d+)?", gt):
        return "number"
    if re.search(r"\d", gt):
        return "latex"
    return "other"


IMAGES_INLINE_SCHEMA = pa.schema([
    pa.field("data_source", pa.string()),
    pa.field("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    pa.field("ability", pa.string()),
    pa.field("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
    pa.field("extra_info", pa.struct([
        ("question", pa.string()), ("source", pa.string()),
        ("gt_source", pa.string()), ("original_answer", pa.string()),
    ])),
    pa.field("question", pa.string()),
    pa.field("sample_uid", pa.string()),
    pa.field("hint", pa.string()),
    pa.field("prompt_with_hint", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    pa.field("hint_reason", pa.string()),
])


def _row_kind(row: dict[str, Any]) -> str | None:
    rm = row.get("reward_model") or {}
    return gt_rule_kind(rm.get("ground_truth"))


def load_pool(shard_glob: str) -> list[dict[str, Any]]:
    files = sorted(glob.glob(shard_glob))
    if not files:
        raise SystemExit(f"no shards matching {shard_glob}")
    rows: list[dict[str, Any]] = []
    for f in files:
        t = pq.read_table(f)
        rows.extend(t.to_pylist())
    return rows


CAULDRON_GLOBS = (
    "cauldron_hint__part_*.parquet",       # post-hint shards (preferred)
    "cauldron_hint_pool__part_*.parquet",  # raw pool shards (no hint columns yet)
)


def load_cauldron_pool(cauldron_dir: str) -> list[dict[str, Any]]:
    for pat in CAULDRON_GLOBS:
        rows = None
        files = sorted(glob.glob(f"{cauldron_dir}/{pat}"))
        if files:
            rows = []
            for f in files:
                rows.extend(pq.read_table(f).to_pylist())
            return rows
    raise SystemExit(
        f"no cauldron shards matching {cauldron_dir}/{{{','.join(CAULDRON_GLOBS)}}}"
    )


def select_stratified(
    rows: list[dict[str, Any]], per_group: int, rng: random.Random,
    min_group: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deterministically pick <=per_group rows from each data_source group."""
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_src[str(r.get("data_source", ""))].append(r)
    selected: list[dict[str, Any]] = []
    taken: dict[str, int] = {}
    for src in sorted(by_src):
        grp = by_src[src]
        if len(grp) < min_group:
            continue
        if len(grp) > per_group:
            idx = sorted(rng.sample(range(len(grp)), per_group))
            grp = [grp[i] for i in idx]
        taken[src] = len(grp)
        selected.extend(grp)
    return selected, taken


def select_stratified_by_share(
    rows: list[dict[str, Any]], total: int, rng: random.Random,
    min_group: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deterministically sample `total` rows, stratified proportionally to each
    source's eligible count.

    A flat per-source cap (select_stratified) under-fills the pool against the
    target and crushes the dominant hard source (MMR1 is 55% of the MMF pool but
    would get the same cap as a 75-row source). Largest-remainder proportional
    allocation keeps large sources' share, caps every quota at actual
    availability, and reflows leftover rows into the largest headroom so the
    final count hits min(total, eligible).
    """
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_src[str(r.get("data_source", ""))].append(r)
    srcs = {s: g for s, g in by_src.items() if len(g) >= min_group}
    total_elig = sum(len(g) for g in srcs.values())
    target = min(total, total_elig)
    if target <= 0:
        return [], {}

    # Exact proportional quota, floored, then largest-remainder for the rest.
    quota = {s: int(len(g) * target / total_elig) for s, g in srcs.items()}
    rem = target - sum(quota.values())
    order = sorted(
        srcs, key=lambda s: (len(srcs[s]) * target / total_elig) - quota[s],
        reverse=True,
    )
    for s in order[:rem]:
        quota[s] += 1

    # Cap at availability; carry any unfillable quota into residual for reflow.
    picked: dict[str, set[int]] = {}
    taken: dict[str, int] = {}
    residual = 0
    for s in sorted(srcs):
        grp = srcs[s]
        q = min(quota[s], len(grp))
        idx = (set(rng.sample(range(len(grp)), q)) if q < len(grp)
               else set(range(len(grp))))
        picked[s] = idx
        taken[s] = len(idx)
        residual += quota[s] - q

    # Reflow residual into the sources with the most headroom.
    if residual:
        for s in sorted(srcs, key=lambda s: len(srcs[s]) - len(picked[s]),
                        reverse=True):
            grp = srcs[s]
            headroom = len(grp) - len(picked[s])
            if headroom <= 0:
                continue
            add = min(residual, headroom)
            pool = [i for i in range(len(grp)) if i not in picked[s]]
            picked[s].update(rng.sample(pool, add))
            taken[s] = len(picked[s])
            residual -= add
            if residual == 0:
                break

    selected: list[dict[str, Any]] = []
    for s in sorted(srcs):
        selected.extend(srcs[s][i] for i in sorted(picked[s]))
    return selected, taken


def inline_cauldron_bytes(row: dict[str, Any]) -> bool:
    """Convert image_url refs to inline {bytes, path} structs.

    Returns False (row dropped) if any referenced file is missing — verl's
    dict image path would crash on bytes=None anyway (the struct always
    materializes both keys via pyarrow).
    """
    out_imgs: list[dict[str, Any]] = []
    for img in row.get("images") or []:
        url = (img or {}).get("image_url")
        b = (img or {}).get("bytes")
        if b is not None:
            out_imgs.append({"bytes": b, "path": img.get("path")})
            continue
        if not url:
            return False
        try:
            data = Path(url).read_bytes()
        except OSError:
            return False
        if not data:
            return False
        out_imgs.append({"bytes": data, "path": url})
    if not out_imgs:
        return False
    row["images"] = out_imgs
    return True


def write_shards(rows: list[dict[str, Any]], out_dir: Path, prefix: str,
                 shard_rows: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i in range(0, len(rows), shard_rows):
        p = out_dir / f"{prefix}__part_{i // shard_rows:04d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows[i : i + shard_rows], schema=IMAGES_INLINE_SCHEMA),
            p, compression="zstd",
        )
        paths.append(str(p))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    DTOPD = Path("/inspire/hdd/global_user/mengweicheng-240108120092/lzy")
    base = DTOPD / "fc-opd-storage/outputs/fc_opd/sft_rl"
    ap.add_argument("--out-dir", type=Path, default=base / "rl_mixed_20k")
    ap.add_argument("--mmf-dir", default=str(base / "hint_pools/mmf_hint"),
                    help="dir with mmf_hint__part_*.parquet (hint-carrying)")
    ap.add_argument("--cauldron-dir", default=str(base / "hint_pools/cauldron_pool_100k"),
                    help="dir with cauldron_hint (or raw cauldron_hint_pool) shards")
    ap.add_argument("--mmf-rows", type=int, default=12000,
                    help="total MMF hard rows (stratified per source)")
    ap.add_argument("--cauldron-per-subset", type=int, default=1500,
                    help="max rows per Cauldron subset")
    ap.add_argument("--keep-latex", action="store_true",
                    help="keep GTs with digits that are not pure numbers "
                         "(mathruler-graded); default: letter/yesno/number only")
    ap.add_argument("--min-per-subset", type=int, default=1,
                    help="drop Cauldron subsets with fewer rule-ok rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-rows", type=int, default=1500)
    ap.add_argument("--prefix", default="rl_mixed_train")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ---- MMF hard rows ---------------------------------------------------
    mmf_rows = load_pool(f"{args.mmf_dir}/mmf_hint__part_*.parquet")
    mmf_elig = [r for r in mmf_rows if _row_kind(r) in ("letter", "yesno", "number", "latex")]
    mmf_kinds = Counter(_row_kind(r) for r in mmf_rows)
    mmf_sel, mmf_taken = select_stratified_by_share(mmf_elig, args.mmf_rows, rng)

    # ---- Cauldron broad rows ---------------------------------------------
    cld_rows = load_cauldron_pool(args.cauldron_dir)
    kinds = {"letter", "yesno", "number"} | ({"latex"} if args.keep_latex else set())
    cld_elig = [r for r in cld_rows if _row_kind(r) in kinds]
    cld_kinds = Counter(_row_kind(r) for r in cld_rows)
    cld_sel, cld_taken = select_stratified(
        cld_elig, args.cauldron_per_subset, rng, min_group=args.min_per_subset
    )

    dropped_missing_img = 0
    kept_cld: list[dict[str, Any]] = []
    for r in cld_sel:
        if inline_cauldron_bytes(r):
            kept_cld.append(r)
        else:
            dropped_missing_img += 1

    # ---- normalize hint columns on rows that lack them -------------------
    all_rows = mmf_sel + kept_cld
    for r in all_rows:
        r.setdefault("hint", "")
        r.setdefault("hint_reason", "pool_passthrough" if not r.get("hint") else "")
        if "prompt_with_hint" not in r:
            q = str(r.get("question", ""))
            h = str(r.get("hint") or "")
            r["prompt_with_hint"] = (
                [{"role": "user", "content": f"{q}\n\n{h.strip()}"}] if h.strip() else []
            )
        r.setdefault("ability", "reasoning")
        ei = r.setdefault("extra_info", {})
        ei.setdefault("question", str(r.get("question", "")))
        ei.setdefault("source", str(r.get("data_source", "")))
        ei.setdefault("gt_source", "")
        ei.setdefault("original_answer", "")
        rm = r.setdefault("reward_model", {})
        rm.setdefault("style", "rule")
        rm.setdefault("ground_truth", "")
        r.setdefault("data_source", str(ei.get("source", "")))
        r.setdefault("prompt", [{"role": "user", "content": str(r.get("question", ""))}])
        r.setdefault("sample_uid", f"{r.get('data_source','')}:{id(r)}")

    total = len(all_rows)
    hint_ok = sum(1 for r in all_rows if str(r.get("hint") or "").strip())
    mmf_hint_ok = sum(1 for r in mmf_sel if str(r.get("hint") or "").strip())
    cld_hint_ok = sum(1 for r in kept_cld if str(r.get("hint") or "").strip())

    paths = write_shards(all_rows, args.out_dir, args.prefix, args.shard_rows)

    stats = {
        "pool": "rl_mixed",
        "mmf_dir": str(args.mmf_dir),
        "cauldron_dir": str(args.cauldron_dir),
        "seed": args.seed,
        "keep_latex": args.keep_latex,
        "mmf_total": len(mmf_rows),
        "mmf_rule_ok": len(mmf_elig),
        "mmf_gt_kinds": dict(mmf_kinds),
        "mmf_selected": len(mmf_sel),
        "mmf_per_source": {k: v for k, v in sorted(mmf_taken.items())},
        "cauldron_total": len(cld_rows),
        "cauldron_rule_ok": len(cld_elig),
        "cauldron_gt_kinds": dict(cld_kinds),
        "cauldron_selected": len(kept_cld),
        "cauldron_per_subset": {k: v for k, v in sorted(cld_taken.items())},
        "cauldron_dropped_missing_image": dropped_missing_img,
        "total_rows": total,
        "mmf_share": round(len(mmf_sel) / total, 4) if total else 0.0,
        "cauldron_share": round(len(kept_cld) / total, 4) if total else 0.0,
        "rows_with_hint": hint_ok,
        "mmf_rows_with_hint": mmf_hint_ok,
        "cauldron_rows_with_hint": cld_hint_ok,
        "shard_rows": args.shard_rows,
        "shards": paths,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rl_mixed_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "shards"},
                     ensure_ascii=False, indent=2))
    print(f"wrote {len(paths)} shards under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
