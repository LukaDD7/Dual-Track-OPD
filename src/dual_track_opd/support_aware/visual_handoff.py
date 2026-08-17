"""Visual Handoff Diagnostic (frozen plan Question A, GPU).

For each selected teacher trace:

1. forced-forward the *same* teacher response tokens under full and degraded
   images for teacher and student -> full-vocabulary Jensen-Shannon visual
   attribution ``V_t^T`` / ``V_t^S`` and ``DeltaV_t = V_t^T - V_t^S``;
2. aggregate to reasoning blocks -> ``DeltaV_i``;
3. find the one-downward BIC change point on the block sequence -> candidate
   ``h_V`` (cheap localization only, never a claim ``h_V == h*``);
4. causally validate with student continuation at ``h_V - 1``, ``h_V``,
   ``h_V + 1`` blocks (K=4, K=8 if ambiguous) and at ``h_V`` under the
   degraded image.

Outputs one JSONL record per prompt; the HPC launcher shards across GPU
pairs and merges into the Visual Handoff Report.  No training, no RL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from .causal_image import build_image_conditions
from .causal_runtime import _prompt_inputs, load_runtime_models, response_chunk_logits
from .diagnostic import build_prompt, extract_image
from .prefix_intervention import generate_continuation
from .reachability_proxy import teacher_trace_id
from .reasoning_blocks import blocks_with_token_spans
from .verifier import verify_answer


SCHEMA_VERSION = "support-aware-visual-handoff-v3"


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _selected_teacher_trace(
    retained_rows: Sequence[Mapping[str, Any]], sample_uid: str
) -> dict[str, Any]:
    candidates = [
        row
        for row in retained_rows
        if str(row.get("sample_uid")) == sample_uid
        and row.get("correct") is True
        and row.get("retained_for_fkl") is True
    ]
    if not candidates:
        raise ValueError(f"{sample_uid}: no correct retained_for_fkl teacher proposal")
    candidates.sort(
        key=lambda row: (
            int(row.get("reachability_rank") or 10**9),
            int(row.get("proposal_id") or 0),
        )
    )
    return candidates[0]


def _load_inputs(
    *,
    cohort_dir: str,
    cohort_parquet_path: str,
    proposal_dirs: Sequence[str],
    rescue_dirs: Sequence[str],
    shard_index: int,
    num_shards: int,
    max_prompts: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import pandas as pd

    cohort_path = Path(cohort_parquet_path).expanduser().resolve()
    if not cohort_path.is_file():
        cohort_path = Path(cohort_dir).expanduser() / "cohort.parquet"
    frame = pd.read_parquet(cohort_path)
    frame.index = frame["sample_uid"].astype(str)

    retained_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in proposal_dirs:
        for row in _read_jsonl(Path(source) / "retained_proposals.jsonl"):
            retained_by_uid[str(row.get("sample_uid"))].append(row)

    minimal_horizon: dict[str, int] = {}
    for source in rescue_dirs:
        for row in _read_jsonl(Path(source) / "minimal_rescue_prefixes.jsonl"):
            if row.get("meets_preregistered_rescue_rule") is True:
                minimal_horizon.setdefault(str(row["sample_uid"]), int(row["horizon"]))

    uids: list[str] = []
    for uid, rows in retained_by_uid.items():
        if uid not in frame.index:
            continue
        try:
            _selected_teacher_trace(rows, uid)
        except ValueError:
            continue
        uids.append(uid)
    uids.sort()
    start = len(uids) * shard_index // num_shards
    end = len(uids) * (shard_index + 1) // num_shards
    shard_uids = uids[start:end]
    if max_prompts is not None:
        shard_uids = shard_uids[:max_prompts]

    inputs: list[dict[str, Any]] = []
    for uid in shard_uids:
        cohort_row = frame.loc[uid].to_dict()
        cohort_row["sample_uid"] = uid
        inputs.append(
            {
                "uid": uid,
                "cohort": cohort_row,
                "trace": _selected_teacher_trace(retained_by_uid[uid], uid),
            }
        )
    return inputs, minimal_horizon


def _jensen_shannon_from_logits(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
) -> torch.Tensor:
    """Exact per-position JS over the full vocabulary.

    The old diagnostic only compared the log-probability of the realized
    teacher token.  That scalar can miss a large counterfactual redistribution
    elsewhere in the vocabulary.  Chunking is handled by ``_condition_js`` so
    this full-distribution calculation stays bounded in memory.
    """

    if first_logits.shape != second_logits.shape:
        raise ValueError(
            f"JS logits must have identical shapes: {tuple(first_logits.shape)} "
            f"!= {tuple(second_logits.shape)}"
        )
    first_logp = torch.log_softmax(first_logits.float(), dim=-1)
    second_logp = torch.log_softmax(second_logits.float(), dim=-1)
    mixture_logp = torch.logaddexp(first_logp, second_logp) - math.log(2.0)
    first_kl = torch.sum(first_logp.exp() * (first_logp - mixture_logp), dim=-1)
    second_kl = torch.sum(second_logp.exp() * (second_logp - mixture_logp), dim=-1)
    return 0.5 * (first_kl + second_kl)


@torch.inference_mode()
def _condition_js(
    model: Any,
    processor: Any,
    *,
    prompt_text: str,
    images: Any,
    response_ids: Sequence[int],
    device: str,
    chunk_size: int,
    null_control: bool,
) -> dict[str, list[float]]:
    """Per-token full-vocabulary JS(full || counterfactual), chunked."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    conditions = [("degraded", images.degraded)]
    if null_control:
        conditions.append(("null", images.null))
    prompt_inputs = {
        "full": _prompt_inputs(processor, images.full, prompt_text),
        **{
            name: _prompt_inputs(processor, image, prompt_text)
            for name, image in conditions
        },
    }
    scores: dict[str, list[float]] = {name: [] for name, _ in conditions}
    full_logits, _ = response_chunk_logits(
        model,
        prompt_inputs["full"],
        response_ids,
        start=0,
        end=len(response_ids),
        device=device,
    )
    for name, _ in conditions:
        counterfactual_logits, _ = response_chunk_logits(
            model,
            prompt_inputs[name],
            response_ids,
            start=0,
            end=len(response_ids),
            device=device,
        )
        for start in range(0, len(response_ids), chunk_size):
            end = min(len(response_ids), start + chunk_size)
            js = _jensen_shannon_from_logits(
                full_logits[start:end],
                counterfactual_logits[start:end],
            )
            scores[name].extend(float(value) for value in js.cpu().tolist())
            del js
        del counterfactual_logits
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    del full_logits
    return scores


def _block_signals(
    *,
    response_ids: Sequence[int],
    blocks: Sequence[Any],
    teacher_js: Mapping[str, Sequence[float]],
    student_js: Mapping[str, Sequence[float]],
    null_control: bool,
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        start, end = block.start_token, block.end_token
        if end <= start:
            continue
        v_t = float(np.mean(teacher_js["degraded"][start:end]))
        v_s = float(np.mean(student_js["degraded"][start:end]))
        row: dict[str, Any] = {
            "block_index": index,
            "block_type": block.block_type,
            "start_token": start,
            "end_token": end,
            "V_T_i": v_t,
            "V_S_i": v_s,
            "DeltaV_i": v_t - v_s,
            "score_type": "full_vocab_jensen_shannon",
            "text_excerpt": block.text[:120],
        }
        if null_control:
            v_t_null = float(np.mean(teacher_js["null"][start:end]))
            v_s_null = float(np.mean(student_js["null"][start:end]))
            row["V_T_null_i"] = v_t_null
            row["V_S_null_i"] = v_s_null
            row["DeltaV_null_i"] = v_t_null - v_s_null
        signals.append(row)
    return signals


def _visual_change_point(values: Sequence[float]) -> dict[str, Any]:
    """Robust one-downward BIC change point (Prefix Teach, Suffix Fade style).

    Guards against single-block noisy spikes: winsorize the block series to
    the 5th/95th percentiles, median-filter with window 3, and require at
    least 3 blocks per regime.  Only a *sustained* high -> low visual-gap
    transition qualifies as ``h_V``.
    """

    n = len(values)
    result: dict[str, Any] = {"tau_block": None, "bic0": None, "bic1": None, "significant": False}
    if n < 6:
        return result
    array = np.asarray(values, dtype=np.float64)
    lower, upper = np.percentile(array, [5.0, 95.0])
    if upper - lower <= 1e-12:
        return result
    array = np.clip(array, lower, upper)
    array = np.asarray(
        [float(np.median(array[max(0, index - 1) : index + 2])) for index in range(n)],
        dtype=np.float64,
    )
    total_mean = float(array.mean())
    rss0 = float(np.sum((array - total_mean) ** 2))
    if rss0 <= 1e-12:
        return result
    bic0 = n * math.log(rss0 / n) + 1 * math.log(n)
    best_tau: int | None = None
    best_bic1 = math.inf
    for tau in range(3, n - 2):
        pre = array[:tau]
        post = array[tau:]
        mu_pre = float(pre.mean())
        mu_post = float(post.mean())
        if mu_pre <= mu_post:
            continue
        rss1 = float(np.sum((pre - mu_pre) ** 2) + np.sum((post - mu_post) ** 2))
        if mu_pre - mu_post < 0.5 * max(float(array.std()), 1e-6):
            continue
        bic1 = n * math.log(max(rss1 / n, 1e-12)) + 3 * math.log(n)
        if bic1 < best_bic1:
            best_bic1 = bic1
            best_tau = tau
    if best_tau is None:
        return result
    significant = best_bic1 < bic0
    result.update(
        {
            # tau is the first post-change block.  It is deliberately absent
            # when BIC does not support the extra regime.
            "tau_block": best_tau if significant else None,
            "candidate_tau_block": best_tau,
            "bic0": bic0,
            "bic1": best_bic1,
            "significant": significant,
        }
    )
    return result


def _handoff_block_index(change: Mapping[str, Any]) -> int | None:
    """Return the last pre-change block for a significant boundary."""

    tau = change.get("tau_block")
    if change.get("significant") is not True or tau is None:
        return None
    tau = int(tau)
    return tau - 1 if tau > 0 else None


def _judge_continuation(
    model: Any,
    processor: Any,
    *,
    image: Image.Image,
    prompt_text: str,
    prefix_ids: Sequence[int],
    gold_answer: str,
    k: int,
    seed: int,
    max_continuation_tokens: int,
    device: str,
) -> dict[str, Any]:
    correct = 0
    for index in range(k):
        generation = generate_continuation(
            model,
            processor,
            image=image,
            prompt_text=prompt_text,
            prefix_ids=prefix_ids,
            max_continuation_tokens=max_continuation_tokens,
            temperature=0.7,
            top_p=0.95,
            seed=seed + index,
            device=device,
        )
        record = generation["generation"]
        response_text = getattr(record, "response_text_display", None)
        if response_text is None and isinstance(record, dict):
            response_text = record.get("response_text_display")
        verdict = verify_answer(str(response_text or ""), gold_answer)
        if verdict.get("correct"):
            correct += 1
    return {"k": k, "correct": correct, "pass_rate": correct / k}


def _block_containing_token(blocks: Sequence[Any], token_index: int) -> int | None:
    for index, block in enumerate(blocks):
        if block.start_token <= token_index < block.end_token:
            return index
    return None


def run_visual_handoff(
    *,
    student_model_path: str,
    teacher_model_path: str,
    student_device: str,
    teacher_device: str,
    dtype: str,
    cohort_dir: str,
    cohort_parquet_path: str,
    proposal_dirs: Sequence[str],
    rescue_dirs: Sequence[str],
    output_dir: str,
    response_format: str,
    degraded_mode: str,
    null_control: bool,
    chunk_size: int,
    continuation_k: int,
    continuation_k_ambiguous: int,
    max_continuation_tokens: int,
    shard_index: int,
    num_shards: int,
    max_prompts: int | None,
    seed: int,
) -> dict[str, Any]:
    inputs, minimal_horizon = _load_inputs(
        cohort_dir=cohort_dir,
        cohort_parquet_path=cohort_parquet_path,
        proposal_dirs=proposal_dirs,
        rescue_dirs=rescue_dirs,
        shard_index=shard_index,
        num_shards=num_shards,
        max_prompts=max_prompts,
    )
    models = load_runtime_models(
        student_model_path=student_model_path,
        student_device=student_device,
        dtype=dtype,
        teacher_model_path=teacher_model_path,
        teacher_device=teacher_device,
    )
    student = models.student_model
    teacher = models.teacher_model
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / f"visual_handoff_records.s{shard_index}.jsonl"

    records: list[dict[str, Any]] = []
    for item in inputs:
        uid = item["uid"]
        cohort_row = item["cohort"]
        trace = item["trace"]
        response_ids = list(trace.get("response_token_ids") or [])
        if not response_ids:
            continue
        try:
            image = extract_image(cohort_row).convert("RGB")
            images = build_image_conditions(image, degraded_mode=degraded_mode)
            question = str(cohort_row.get("question") or "").strip()
            gold_answer = str(cohort_row.get("answer") or "")
            prompt_text = build_prompt(question, response_format=response_format)
            tokenizer = models.student_processor.tokenizer
            _, blocks = blocks_with_token_spans(tokenizer, response_ids)

            teacher_js = _condition_js(
                teacher, models.teacher_processor,
                prompt_text=prompt_text, images=images,
                response_ids=response_ids, device=teacher_device, chunk_size=chunk_size,
                null_control=null_control,
            )
            student_js = _condition_js(
                student, models.student_processor,
                prompt_text=prompt_text, images=images,
                response_ids=response_ids, device=student_device, chunk_size=chunk_size,
                null_control=null_control,
            )
            block_signals = _block_signals(
                response_ids=response_ids,
                blocks=blocks,
                teacher_js=teacher_js,
                student_js=student_js,
                null_control=null_control,
            )
            delta_v = [float(row["DeltaV_i"]) for row in block_signals]
            change = _visual_change_point(delta_v)
            change_null = (
                _visual_change_point([float(row["DeltaV_null_i"]) for row in block_signals])
                if null_control
                else None
            )
            h_star = minimal_horizon.get(uid)
            h_star_block = _block_containing_token(blocks, h_star) if h_star is not None else None
            handoff_block = _handoff_block_index(change)
            handoff_null_block = (
                _handoff_block_index(change_null) if change_null is not None else None
            )
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "prompt_id": uid,
                "trace_id": teacher_trace_id(trace),
                "h_star": h_star,
                "h_star_block": h_star_block,
                "h_V": None,
                "h_V_block": handoff_block,
                "change_point": change,
                "change_point_null": change_null,
                "h_V_null": (
                    blocks[handoff_null_block].end_token
                    if handoff_null_block is not None
                    else None
                ),
                "h_V_null_block": handoff_null_block,
                "block_signals": block_signals,
                "continuation": {},
                "verdict": None,
                "degraded_mode": degraded_mode,
                "visual_score": "full_vocab_jensen_shannon",
            }
            if (
                handoff_block is not None
                and handoff_block - 1 >= 0
                and handoff_block + 1 < len(blocks)
            ):
                candidates = [handoff_block - 1, handoff_block, handoff_block + 1]
                continuations: dict[str, Any] = {}
                for block_index in candidates:
                    prefix_pos = blocks[block_index].end_token
                    prefix_pos = max(1, min(prefix_pos, len(response_ids)))
                    outcome = _judge_continuation(
                        student, models.student_processor,
                        image=images.full,
                        prompt_text=prompt_text,
                        prefix_ids=response_ids[:prefix_pos],
                        gold_answer=gold_answer,
                        k=continuation_k,
                        seed=seed + hash(uid) % 10**6,
                        max_continuation_tokens=max_continuation_tokens,
                        device=student_device,
                    )
                    continuations[str(block_index)] = {
                        "prefix_token_pos": prefix_pos,
                        "outcome": outcome,
                    }
                    if outcome["pass_rate"] == 0.5 and continuation_k_ambiguous > continuation_k:
                        outcome_k8 = _judge_continuation(
                            student, models.student_processor,
                            image=images.full,
                            prompt_text=prompt_text,
                            prefix_ids=response_ids[:prefix_pos],
                            gold_answer=gold_answer,
                            k=continuation_k_ambiguous - continuation_k,
                            seed=seed + hash(uid) % 10**6 + 1000,
                            max_continuation_tokens=max_continuation_tokens,
                            device=student_device,
                        )
                        outcome["k"] = continuation_k_ambiguous
                        outcome["correct"] += outcome_k8["correct"]
                        outcome["pass_rate"] = outcome["correct"] / outcome["k"]
                degraded_outcome = _judge_continuation(
                    student, models.student_processor,
                    image=images.degraded,
                    prompt_text=prompt_text,
                    prefix_ids=response_ids[: blocks[handoff_block].end_token],
                    gold_answer=gold_answer,
                    k=continuation_k,
                    seed=seed + hash(uid) % 10**6 + 2000,
                    max_continuation_tokens=max_continuation_tokens,
                    device=student_device,
                )
                record["h_V"] = blocks[handoff_block].end_token
                record["continuation"] = {
                    "full_image": continuations,
                    "degraded_at_hV": degraded_outcome,
                }
                if h_star_block is not None:
                    record["verdict"] = abs(handoff_block - h_star_block) <= 1
            records.append(record)
        except Exception as exc:  # keep the shard alive; audit failures
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "prompt_id": uid,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "shard": shard_index,
        "prompts": len(records),
        "records_path": str(records_path),
    }
    return summary


def run_visual_handoff_merge(*, shard_dir: str, output_dir: str) -> dict[str, Any]:
    """Merge shard records into the Visual Handoff Report."""

    shard_root = Path(shard_dir).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for path in sorted(shard_root.glob("visual_handoff_records.s*.jsonl")):
        records.extend(_read_jsonl(path))
    schemas = {record.get("schema_version") for record in records}
    if len(schemas) > 1:
        raise ValueError(
            f"mixed schema versions across shards: {sorted(schemas)}; "
            "stale v1 records must be cleared before merging"
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    merged_path = output / "visual_handoff_records.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    ok = [record for record in records if "error" not in record]
    with_hv = [record for record in ok if record.get("h_V_block") is not None]
    with_gold = [record for record in with_hv if record.get("h_star") is not None]
    verdict_true = [record for record in with_gold if record.get("verdict") is True]
    pass_rates: list[float] = []
    pass_rates_at_tau: list[float] = []
    degraded_rates: list[float] = []
    for record in with_hv:
        continuation = record.get("continuation") or {}
        full_image = continuation.get("full_image") or {}
        for key, value in full_image.items():
            pass_rates.append(float(value["outcome"]["pass_rate"]))
            if record.get("h_V_block") is not None and key == str(record["h_V_block"]):
                pass_rates_at_tau.append(float(value["outcome"]["pass_rate"]))
        degraded = continuation.get("degraded_at_hV")
        if degraded:
            degraded_rates.append(float(degraded["pass_rate"]))
    with_null = [record for record in with_hv if record.get("h_V_null_block") is not None]
    with_null_gold = [record for record in with_gold if record.get("h_V_null_block") is not None]
    verdict_null_true = [
        record for record in with_null_gold
        if abs(record["h_V_null_block"] - record["h_star_block"]) <= 1
    ]
    report = {
        "schema_version": SCHEMA_VERSION + "-report",
        "coverage": {
            "records": len(records),
            "errors": len(records) - len(ok),
            "with_change_point": len(with_hv),
            "with_gold_and_change_point": len(with_gold),
            "schema_version": sorted(schemas)[0] if schemas else None,
            "degraded_modes": sorted(
                {record.get("degraded_mode") for record in ok if record.get("degraded_mode")}
            ),
        },
        "change_point": {
            "significant_count": sum(
                1 for record in with_hv if record.get("change_point", {}).get("significant")
            ),
            "hV_within_1_block_of_hstar": sum(1 for record in with_gold if record.get("verdict") is True),
            "enrichment": (
                (sum(1 for record in with_gold if record.get("verdict") is True) / len(with_gold))
                if with_gold
                else None
            ),
        },
        "change_point_null": {
            "count": len(with_null),
            "hV_null_within_1_block_of_hstar": len(verdict_null_true),
            "enrichment": (
                len(verdict_null_true) / len(with_null_gold) if with_null_gold else None
            ),
        },
        "continuation": {
            "full_image_at_hV_mean_pass_rate": (
                float(np.mean(pass_rates)) if pass_rates else None
            ),
            "full_image_at_hV_tau_mean_pass_rate": (
                float(np.mean(pass_rates_at_tau)) if pass_rates_at_tau else None
            ),
            "degraded_at_hV_mean_pass_rate": (
                float(np.mean(degraded_rates)) if degraded_rates else None
            ),
        },
        "records_path": str(merged_path),
    }
    report_path = output / "visual_handoff_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run")
    merge = sub.add_parser("merge")
    for action in (run, merge):
        action.add_argument("--output-dir", required=True)
    run.add_argument("--student-model", default="${DTOPD_MODEL_ROOT}/Qwen3-VL-4B-Instruct")
    run.add_argument("--teacher-model", default="${DTOPD_MODEL_ROOT}/Qwen3-VL-32B-Instruct")
    run.add_argument("--student-device", default="cuda:0")
    run.add_argument("--teacher-device", default="cuda:1")
    run.add_argument("--dtype", default="bfloat16")
    run.add_argument("--cohort-dir", required=True)
    run.add_argument("--cohort-parquet-path")
    run.add_argument("--proposal-dirs", nargs="+", required=True)
    run.add_argument("--rescue-dirs", nargs="+", required=True)
    run.add_argument("--response-format", default="legacy_answer")
    run.add_argument("--degraded-mode", default="lowres_20_bilinear_nearest")
    run.add_argument("--no-null-control", action="store_true", help="skip blank-image null condition")
    run.add_argument("--chunk-size", type=int, default=64)
    run.add_argument("--continuation-k", type=int, default=4)
    run.add_argument("--continuation-k-ambiguous", type=int, default=8)
    run.add_argument("--max-continuation-tokens", type=int, default=256)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--max-prompts", type=int)
    run.add_argument("--seed", type=int, default=20260816)
    merge.add_argument("--shard-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "merge":
        report = run_visual_handoff_merge(
            shard_dir=args.shard_dir,
            output_dir=args.output_dir,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    summary = run_visual_handoff(
        student_model_path=os.path.expandvars(args.student_model),
        teacher_model_path=os.path.expandvars(args.teacher_model),
        student_device=args.student_device,
        teacher_device=args.teacher_device,
        dtype=args.dtype,
        cohort_dir=args.cohort_dir,
        cohort_parquet_path=args.cohort_parquet_path or "",
        proposal_dirs=args.proposal_dirs,
        rescue_dirs=args.rescue_dirs,
        output_dir=args.output_dir,
        response_format=args.response_format,
        degraded_mode=args.degraded_mode,
        null_control=not args.no_null_control,
        chunk_size=args.chunk_size,
        continuation_k=args.continuation_k,
        continuation_k_ambiguous=args.continuation_k_ambiguous,
        max_continuation_tokens=args.max_continuation_tokens,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        max_prompts=args.max_prompts,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
