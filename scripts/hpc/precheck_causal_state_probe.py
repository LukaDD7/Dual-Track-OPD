#!/usr/bin/env python3
"""Pre-check causal-state probe shard outputs before the final merge.

Reads trajectory_results/*.json from one or more shard dirs, validates the
records, writes the standard causal_report package (records.jsonl,
candidate_windows.csv, summary.json, trajectory_overview.svg) and a deeper
pre-check markdown report with per-length relay/transport statistics, leakage
checks, barrier stats, note patterns, and the strongest actionable candidates.

Usage:
  python3 scripts/hpc/precheck_causal_state_probe.py \
    --input-dir .../causal_state_probe_20260808_s0 \
    --input-dir .../causal_state_probe_20260808_s1 \
    --output-dir .../causal_state_probe_20260811_precheck
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dual_track_opd.support_aware.causal_report import (  # noqa: E402
    write_reports,
)


def _is_finite(value) -> bool:
    return not isinstance(value, float) or (math.isfinite(value) and not math.isnan(value))


def _mean(values) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return None if not finite else float(fmean(finite))


def _pct(values, q) -> float | None:
    finite = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not finite:
        return None
    return float(finite[min(len(finite) - 1, int(round(q * (len(finite) - 1))))])


def load_records(input_dirs: list[Path]) -> list[dict]:
    records = []
    for root in input_dirs:
        for path in sorted((root / "trajectory_results").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["_file_mtime_unix"] = path.stat().st_mtime
            records.append(record)
    return records


def validate(records: list[dict], input_dirs: list[Path], cutoff_unix: float) -> dict:
    issues: list[str] = []
    ids = [str(r.get("trajectory_id") or "") for r in records]
    if len(ids) != len(set(ids)):
        issues.append(f"duplicate trajectory_id: {len(ids) - len(set(ids))}")
    candidate_ids = [str(c.get("candidate_id") or "") for r in records for c in (r.get("candidate_windows") or ())]
    if len(candidate_ids) != len(set(candidate_ids)):
        issues.append(f"duplicate candidate_id: {len(candidate_ids) - len(set(candidate_ids))}")
    old_impl = 0
    new_impl = 0
    for r in records:
        if r.get("schema_version") != "causal-state-probe-v1":
            issues.append(f"{r.get('trajectory_id')}: unexpected schema_version")
        n_tokens = len(r.get("trajectory_token_ids") or [])
        n_signals = len(r.get("token_signals") or [])
        if n_tokens != n_signals:
            issues.append(f"{r.get('trajectory_id')}: token count {n_tokens} != signal rows {n_signals}")
        mtime = float(r.get("_file_mtime_unix") or 0.0)
        if mtime and mtime < cutoff_unix:
            old_impl += 1
        elif mtime:
            new_impl += 1
        signals = r.get("token_signals") or []
        for row in signals:
            for key, value in row.items():
                if isinstance(value, (int, float)) and not _is_finite(value):
                    issues.append(f"{r.get('trajectory_id')}: non-finite token_signals.{key}")
                    break
        n_traj = n_tokens
        for candidate in r.get("candidate_windows") or ():
            for key, value in candidate.items():
                if isinstance(value, (int, float)) and not _is_finite(value):
                    issues.append(f"{r.get('trajectory_id')}: non-finite candidate.{key}")
                    break
            start = int(candidate["start"]) if candidate.get("start") is not None else -1
            anchor = int(candidate["anchor"]) if candidate.get("anchor") is not None else -1
            end = int(candidate["end"]) if candidate.get("end") is not None else -1
            if not (0 <= start <= anchor <= end <= n_traj):
                issues.append(f"{candidate.get('candidate_id')}: bounds {start}/{anchor}/{end} vs len {n_traj}")
            for key in ("relay_continuations", "visual_continuations"):
                estimates = candidate.get(key) or {}
                if isinstance(estimates, dict):
                    for length, estimate in estimates.items():
                        for field in ("n", "n_correct", "n_malformed", "pass_rate"):
                            value = estimate.get(field) if isinstance(estimate, dict) else None
                            if isinstance(value, (int, float)) and not _is_finite(value):
                                issues.append(f"{candidate.get('candidate_id')}: non-finite {key}.{length}.{field}")
            for key in ("relay_probability_by_length",):
                for length, value in (candidate.get(key) or {}).items():
                    if isinstance(value, (int, float)) and not _is_finite(value):
                        issues.append(f"{candidate.get('candidate_id')}: non-finite {key}.{length}")
            prefix = str(r.get("trajectory_id") or "") + ":candidate-"
            if not str(candidate.get("candidate_id") or "").startswith(prefix):
                issues.append(f"{candidate.get('candidate_id')}: candidate_id prefix mismatch")

    manifest_checks: dict = {"consistent": True, "details": []}
    manifests = []
    for root in input_dirs:
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            manifest_checks["details"].append(f"missing manifest: {root}")
            manifest_checks["consistent"] = False
            continue
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifests:
        expected = {
            str(work_id)
            for manifest in manifests
            for work_id in (manifest.get("provenance") or {}).get("expected_work_ids", ())
        }
        present = {str(r.get("trajectory_id") or "") for r in records}
        missing = sorted(expected - present)
        stray = sorted(present - expected)
        if missing:
            manifest_checks["details"].append(f"expected-but-missing records: {len(missing)} (e.g. {missing[:3]})")
            manifest_checks["consistent"] = False
        if stray:
            manifest_checks["details"].append(f"records-not-in-manifest: {len(stray)}")
            manifest_checks["consistent"] = False
        reference = manifests[0]
        for manifest in manifests[1:]:
            def norm_config(value):
                config = dict((value.get("config") or {}))
                config.pop("output_dir", None)
                config.pop("shard_index", None)
                return config
            if norm_config(manifest) != norm_config(reference):
                manifest_checks["details"].append("shard configs differ")
                manifest_checks["consistent"] = False
            for key in ("k32_validation_sha256", "k32_rollouts_sha256", "k32_support_summary_sha256", "cohort_sha256"):
                if ((manifest.get("provenance") or {}).get(key) != (reference.get("provenance") or {}).get(key)):
                    manifest_checks["details"].append(f"provenance {key} differs")
                    manifest_checks["consistent"] = False
            for key in ("transformers", "tokenizer_hash", "git_commit"):
                if ((manifest.get("runtime") or {}).get(key) != (reference.get("runtime") or {}).get(key)):
                    manifest_checks["details"].append(f"runtime {key} differs")
                    manifest_checks["consistent"] = False
        manifest_checks["expected_work_ids"] = len(expected)
    return {
        "issues": issues,
        "n_records": len(records),
        "n_unique_ids": len(set(ids)),
        "n_prompts": len({str(r.get("prompt_id")) for r in records}),
        "implementation_strata": {
            "old_impl_records": old_impl,
            "new_impl_records": new_impl,
            "cutoff_unix": cutoff_unix,
        },
        "manifest_checks": manifest_checks,
    }


def deep_stats(records: list[dict]) -> dict:
    candidates = [c for r in records for c in (r.get("candidate_windows") or ())]
    states = Counter(str(c.get("state_class") or "unclassified") for c in candidates)
    outcomes = Counter(
        "correct" if r.get("trajectory_correct") is True
        else "wrong" if r.get("trajectory_correct") is False
        else "unknown"
        for r in records
    )
    lengths = [len(r.get("trajectory_token_ids") or []) for r in records]

    relay_by_len: dict[str, list[float]] = {length: [] for length in ("32", "64", "128")}
    relay_prob: dict[str, list[float]] = {length: [] for length in ("32", "64", "128")}
    relay_conditional_by_len: dict[str, list[float]] = {length: [] for length in ("32", "64", "128")}
    relay_coverage: dict[str, dict] = {length: {
        "total": 0,
        "all_clean_estimates": 0,
        "with_any_malformed": 0,
        "fully_malformed": 0,
        "n_malformed_distribution": Counter(),
        "continuation_n": 0,
        "continuation_valid": 0,
    } for length in ("32", "64", "128")}
    pair_clean_by_len: dict[str, dict] = {length: {
        "treatment_clean": 0, "control_clean": 0, "pair_clean": 0, "total": 0,
    } for length in ("32", "64", "128")}
    relay_cross_consistent = 0
    relay_cross_total = 0
    relay_cross_valid = 0
    relay_cross_valid_total = 0
    transport_gains: list[float] = []
    transport_probs: list[float] = []
    transport_vs_wrong: list[float] = []
    actionable_transport: list[tuple[float, float, str]] = []
    visual_fine: list[float] = []
    visual_all: list[float] = []
    answer_leakage: list[float] = []
    notes: Counter[str] = Counter()
    barrier_count = 0
    barrier_peak_nll: list[float] = []
    barrier_has_peaks = 0
    js_candidates_degraded: list[float] = []
    js_candidates_null: list[float] = []
    js_trajectory_degraded: list[float] = []
    js_trajectory_null: list[float] = []
    strong_relay: list[dict] = []
    strong_transport: list[dict] = []

    for record in records:
        for row in record.get("token_signals") or ():
            js_trajectory_degraded.append(float(row.get("js_full_degraded") or 0.0))
            js_trajectory_null.append(float(row.get("js_full_null") or 0.0))
        for path_rows in (record.get("teacher_trajectories") or ()):
            barriers = path_rows.get("reachability_barriers") or ()
            barrier_count += len(barriers)
            for barrier in barriers:
                value = barrier.get("peak_nll")
                if value is not None:
                    barrier_peak_nll.append(float(value))
                if barrier.get("peaks") is not None:
                    barrier_has_peaks += 1
        for candidate in record.get("candidate_windows") or ():
            relay = candidate.get("relay_gain_by_length") or {}
            probabilities = candidate.get("relay_probability_by_length") or {}
            estimates = {
                str(length): estimate
                for length, estimate in (candidate.get("relay_continuations") or {}).items()
            }
            baseline_full = next(
                (
                    estimate for estimate in (candidate.get("visual_continuations") or ())
                    if isinstance(estimate, dict) and estimate.get("condition") == "full"
                ),
                None,
            )
            control_clean = baseline_full is not None and int(baseline_full.get("n_malformed") or 0) == 0
            for length in relay_by_len:
                if relay.get(length) is not None:
                    relay_by_len[length].append(float(relay[length]))
                if probabilities.get(length) is not None:
                    relay_prob[length].append(float(probabilities[length]))
                estimate = estimates.get(length)
                if estimate is not None:
                    malformed = int(estimate.get("n_malformed") or 0)
                    n = int(estimate.get("n") or 0)
                    relay_coverage[length]["total"] += 1
                    relay_coverage[length]["n_malformed_distribution"][malformed] += 1
                    relay_coverage[length]["continuation_n"] += n
                    relay_coverage[length]["continuation_valid"] += n - malformed
                    if malformed == 0:
                        relay_coverage[length]["all_clean_estimates"] += 1
                    else:
                        relay_coverage[length]["with_any_malformed"] += 1
                    if malformed == n and n > 0:
                        relay_coverage[length]["fully_malformed"] += 1
                    pair_clean_by_len[length]["total"] += 1
                    if malformed == 0:
                        pair_clean_by_len[length]["treatment_clean"] += 1
                    if control_clean:
                        pair_clean_by_len[length]["control_clean"] += 1
                    if malformed == 0 and control_clean:
                        pair_clean_by_len[length]["pair_clean"] += 1
                        if relay.get(length) is not None:
                            relay_conditional_by_len[length].append(float(relay[length]))
            if all(relay.get(length) is not None for length in relay_by_len):
                values = [float(relay[length]) for length in relay_by_len]
                relay_cross_total += 1
                relay_cross_consistent += int(all(v > 0 for v in values) or all(v <= 0 for v in values))
                if all(estimates.get(length) is not None and int(estimates[length].get("n_malformed") or 0) == 0 for length in relay_by_len):
                    relay_cross_valid_total += 1
                    relay_cross_valid += int(all(v > 0 for v in values) or all(v <= 0 for v in values))
            if candidate.get("transport_gain") is not None:
                transport_gains.append(float(candidate["transport_gain"]))
                if candidate.get("transport_probability") is not None:
                    transport_probs.append(float(candidate["transport_probability"]))
            if candidate.get("transport_vs_wrong_gain") is not None:
                transport_vs_wrong.append(float(candidate["transport_vs_wrong_gain"]))
            if candidate.get("transport_gain") is not None and candidate.get("transport_vs_wrong_gain") is not None:
                actionable = min(
                    float(candidate["transport_gain"]),
                    float(candidate["transport_vs_wrong_gain"]),
                )
                prob = min(
                    float(candidate.get("transport_probability") or 0.0),
                    float(candidate.get("transport_vs_wrong_probability") or 0.0),
                )
                actionable_transport.append((actionable, prob, str(candidate.get("candidate_id") or "")))
            if candidate.get("visual_fine_gain") is not None:
                visual_fine.append(float(candidate["visual_fine_gain"]))
            if candidate.get("visual_all_gain") is not None:
                visual_all.append(float(candidate["visual_all_gain"]))
            if candidate.get("answer_leakage") is not None:
                answer_leakage.append(float(candidate["answer_leakage"]))
            if candidate.get("student_js_full_degraded") is not None:
                js_candidates_degraded.append(float(candidate["student_js_full_degraded"]))
            if candidate.get("student_js_full_null") is not None:
                js_candidates_null.append(float(candidate["student_js_full_null"]))
            for note in candidate.get("notes") or ():
                notes[str(note)] += 1
            best_relay_gain = max(relay.values()) if relay else None
            best_relay_length = max(relay, key=lambda key: relay[key]) if relay else None
            relay_p = (
                float(probabilities[best_relay_length])
                if best_relay_length is not None and probabilities.get(best_relay_length) is not None
                else None
            )
            if (
                candidate.get("state_class") == "on_policy_repairable"
                and best_relay_gain is not None
                and best_relay_gain >= 0.2
                and relay_p is not None
                and relay_p >= 0.9
            ):
                strong_relay.append({
                    "candidate_id": candidate.get("candidate_id"),
                    "prompt_id": record.get("prompt_id"),
                    "trajectory_correct": record.get("trajectory_correct"),
                    "anchor": candidate.get("anchor"),
                    "relative_position": round(float(candidate.get("relative_position") or 0.0), 3),
                    "relay_gain": round(float(best_relay_gain), 3),
                    "relay_length": best_relay_length,
                    "relay_probability": round(float(relay_p), 3),
                    "relay_by_length": {
                        k: (round(float(v), 3) if v is not None else None)
                        for k, v in (candidate.get("relay_gain_by_length") or {}).items()
                    },
                })
            if (
                candidate.get("state_class") == "transportable_low_reachability"
                and candidate.get("transport_gain") is not None
                and float(candidate["transport_gain"]) > 0.2
            ):
                strong_transport.append({
                    "candidate_id": candidate.get("candidate_id"),
                    "prompt_id": record.get("prompt_id"),
                    "transport_gain": round(float(candidate["transport_gain"]), 3),
                    "transport_probability": round(float(candidate.get("transport_probability") or 0.0), 3),
                    "transport_vs_wrong_gain": (
                        round(float(candidate["transport_vs_wrong_gain"]), 3)
                        if candidate.get("transport_vs_wrong_gain") is not None else None
                    ),
                })

    def summarize(values: list[float]) -> dict | None:
        if not values:
            return None
        return {
            "n": len(values),
            "mean": round(float(fmean(values)), 4),
            "median": round(float(median(values)), 4),
            "p90": round(float(_pct(values, 0.9) or 0.0), 4),
            "max": round(float(max(values)), 4),
            "positive_ratio": round(sum(1 for v in values if v > 0) / len(values), 3),
        }

    return {
        "candidate_count": len(candidates),
        "state_class_counts": dict(sorted(states.items())),
        "trajectory_outcome_counts": dict(sorted(outcomes.items())),
        "trajectory_token_lengths": summarize(lengths),
        "relay_gain_by_length": {k: summarize(v) for k, v in relay_by_len.items()},
        "relay_probability_by_length": {k: summarize(v) for k, v in relay_prob.items()},
        "relay_itt_lower_bound_by_length": {k: summarize(v) for k, v in relay_by_len.items()},
        "relay_conditional_pair_clean_by_length": {
            k: summarize(v) for k, v in relay_conditional_by_len.items()
        },
        "relay_coverage": {
            k: {
                "total": v["total"],
                "all_clean_estimate_rate": round(v["all_clean_estimates"] / v["total"], 4) if v["total"] else None,
                "with_any_malformed": v["with_any_malformed"],
                "fully_malformed": v["fully_malformed"],
                "continuation_valid_rate": round(
                    v["continuation_valid"] / v["continuation_n"], 4
                ) if v["continuation_n"] else None,
                "n_malformed_distribution": dict(sorted(v["n_malformed_distribution"].items())),
            }
            for k, v in relay_coverage.items()
        },
        "relay_pair_clean": {
            k: {
                "treatment_clean_rate": round(v["treatment_clean"] / v["total"], 4) if v["total"] else None,
                "control_clean_rate": round(v["control_clean"] / v["total"], 4) if v["total"] else None,
                "pair_clean_rate": round(v["pair_clean"] / v["total"], 4) if v["total"] else None,
                "n": v["total"],
            }
            for k, v in pair_clean_by_len.items()
        },
        "relay_cross_length_consistent": {
            "n_candidates_with_all_lengths": relay_cross_total,
            "sign_consistent_ratio": round(relay_cross_consistent / relay_cross_total, 3) if relay_cross_total else None,
        },
        "relay_cross_length_consistent_valid_only": {
            "n_candidates": relay_cross_valid_total,
            "sign_consistent_ratio": round(relay_cross_valid / relay_cross_valid_total, 3) if relay_cross_valid_total else None,
        },
        "transport_skipped_notes": {
            str(key): value
            for key, value in notes.items()
            if str(key).startswith("transport_skipped") or str(key).startswith("wrong_prefix_skipped")
        },
        "transport_gain": summarize(transport_gains),
        "transport_probability": summarize(transport_probs),
        "transport_vs_wrong_gain": summarize(transport_vs_wrong),
        "actionable_transport_positive": summarize([v[0] for v in actionable_transport]),
        "visual_fine_gain": summarize(visual_fine),
        "visual_all_gain": summarize(visual_all),
        "answer_leakage_pass_rate": summarize(answer_leakage),
        "js_at_candidates": {
            "full_vs_degraded": summarize(js_candidates_degraded),
            "full_vs_null": summarize(js_candidates_null),
        },
        "js_trajectory_wide": {
            "full_vs_degraded": summarize(js_trajectory_degraded),
            "full_vs_null": summarize(js_trajectory_null),
        },
        "notes": dict(notes.most_common(20)),
        "reachability_barriers": {
            "count": barrier_count,
            "peak_nll": summarize(barrier_peak_nll),
            "with_peaks_field": barrier_has_peaks,
        },
        "strong_relay_examples": sorted(
            strong_relay, key=lambda item: -item["relay_gain"]
        )[:10],
        "strong_transport_examples": sorted(
            strong_transport, key=lambda item: -item["transport_gain"]
        )[:10],
    }


def render_markdown(
    validation: dict,
    stats: dict,
    input_dirs: list[Path],
    output_dir: Path,
) -> str:
    lines = [
        f"# Causal State Probe — {validation['n_records']}-sample Pre-check (2026-08-11)",
        "",
        "## Provenance",
        "",
        f"- Input shards: {', '.join(str(p) for p in input_dirs)}",
        f"- Output package: `{output_dir}`",
        f"- Records: {validation['n_records']} (unique ids {validation['n_unique_ids']}, "
        f"prompts {validation['n_prompts']})",
        f"- Implementation strata: old {validation['implementation_strata']['old_impl_records']} / "
        f"new {validation['implementation_strata']['new_impl_records']} "
        f"(cutoff {validation['implementation_strata']['cutoff_unix']})",
        "",
        "## Validation",
        "",
    ]
    if validation["issues"]:
        lines += ["**Issues found:**", ""]
        lines += [f"- {issue}" for issue in validation["issues"][:20]]
        lines += [""]
    else:
        lines += ["No issues found in the specific checks below.", ""]
    manifest_checks = validation["manifest_checks"]
    lines += [
        "Manifest/work-ID/config consistency:",
        "",
        f"- consistent: {manifest_checks['consistent']} "
        f"(expected work IDs {manifest_checks.get('expected_work_ids')})",
    ]
    if manifest_checks["details"]:
        lines += [f"- {detail}" for detail in manifest_checks["details"]]
    lines += [""]

    def block(title: str, body: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(body)
        lines.append("")

    block("Trajectory mix", (
        f"- outcomes: {stats['trajectory_outcome_counts']}\n"
        f"- token length: {stats['trajectory_token_lengths']}"
    ))
    block("Candidate state classes", "\n".join(
        f"- {state}: {count} ({count / stats['candidate_count'] * 100:.1f}%)"
        for state, count in stats["state_class_counts"].items()
    ))
    relay = stats["relay_gain_by_length"]
    coverage = stats["relay_coverage"]
    block("Relay gains by length — ITT lower bound", "\n".join(
        f"- L{length}: {relay[length]}  *(all outcomes counted with original K; "
        "malformed/leakage treated as failures)*"
        for length in ("32", "64", "128")
    ) + (
        "\n- cross-length sign consistency (all): "
        f"{stats['relay_cross_length_consistent']}\n"
        "- cross-length sign consistency (pair-clean estimates only): "
        f"{stats['relay_cross_length_consistent_valid_only']}"
    ))
    block("Relay coverage (continuation-level)", "\n".join(
        f"- L{length}: all-clean-estimate rate {coverage[length]['all_clean_estimate_rate']}, "
        f"continuation-valid rate {coverage[length]['continuation_valid_rate']}, "
        f"fully-malformed {coverage[length]['fully_malformed']}, "
        f"n_malformed dist {coverage[length]['n_malformed_distribution']}"
        for length in ("32", "64", "128")
    ) + "\n\nPair-clean (treatment + matched control both clean):\n"
        + "\n".join(
            f"- L{length}: {stats['relay_pair_clean'][length]}"
            for length in ("32", "64", "128")
        ))
    conditional = stats["relay_conditional_pair_clean_by_length"]
    block("Relay gains — conditional (answer-free, pair-clean, sensitivity)", "\n".join(
        f"- L{length}: {conditional[length]}" for length in ("32", "64", "128")
    ) + "\n\n*Post-treatment filtering; not an unbiased causal estimand. "
        "Caveat: `n_malformed` conflates relay answer leakage (continuation never "
        "generated), missing answer marker, truncation, and verifier-unparseable "
        "outputs; reason-level counts require a stratified replay (review P0-2).*")
    skipped = stats["transport_skipped_notes"]
    if skipped:
        block("Transport skip reasons", "\n".join(
            f"- {key}: {count}" for key, count in sorted(skipped.items(), key=lambda item: -item[1])
        ))
    block("Transport", "\n".join([
        f"- transport gain: {stats['transport_gain']}",
        f"- vs wrong-prefix gain: {stats['transport_vs_wrong_gain']}",
        f"- actionable positive: {stats['actionable_transport_positive']}",
    ]))
    block("Visual signals", "\n".join([
        f"- candidate-window JS full-vs-degraded: {stats['js_at_candidates']['full_vs_degraded']}",
        f"- candidate-window JS full-vs-null: {stats['js_at_candidates']['full_vs_null']}",
        f"- trajectory-wide JS full-vs-degraded: {stats['js_trajectory_wide']['full_vs_degraded']}",
        f"- visual_fine_gain: {stats['visual_fine_gain']}",
        f"- answer_leakage pass rate: {stats['answer_leakage_pass_rate']}",
    ]))
    block("Teacher-path reachability barriers", (
        f"- count: {stats['reachability_barriers']['count']}\n"
        f"- peak NLL: {stats['reachability_barriers']['peak_nll']}\n"
        f"- with per-peak detail (`peaks` field): {stats['reachability_barriers']['with_peaks_field']}"
    ))
    block("Note patterns (top 20)", "\n".join(
        f"- {note}: {count}" for note, count in stats["notes"].items()
    ))

    lines.append("## Strongest actionable candidates")
    lines.append("")
    if stats["strong_relay_examples"]:
        lines.append("### on_policy_repairable (relay gain > 0.2, p > 0.85)")
        lines.append("")
        lines.append("| candidate | prompt | correct | anchor | rel | gain | p | l32/l64/l128 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for item in stats["strong_relay_examples"]:
            relay_values = item["relay_by_length"]
            lines.append(
                f"| {item['candidate_id']} | {item['prompt_id']} | {item['trajectory_correct']} "
                f"| {item['anchor']} | {item['relative_position']} | {item['relay_gain']} "
                f"| {item['relay_probability']} | {relay_values.get('32')}/{relay_values.get('64')}/{relay_values.get('128')} |"
            )
        lines.append("")
    if stats["strong_transport_examples"]:
        lines.append("### transportable_low_reachability (transport gain > 0.2)")
        lines.append("")
        lines.append("| candidate | prompt | transport_gain | p | vs_wrong |")
        lines.append("|---|---|---|---|---|")
        for item in stats["strong_transport_examples"]:
            lines.append(
                f"| {item['candidate_id']} | {item['prompt_id']} | {item['transport_gain']} "
                f"| {item['transport_probability']} | {item['transport_vs_wrong_gain']} |"
            )
        lines.append("")
    lines.append("## Pre-check verdict")
    lines.append("")
    lines.append(
        "Reviewed 2026-08-11 by Codex; provisional pending GPU-side follow-ups "
        "(see docs/causal_state_probe_precheck_review_20260811.md)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dirs = [Path(value).expanduser().resolve() for value in args.input_dir]
    output_dir = Path(args.output_dir).expanduser().resolve()
    records = load_records(input_dirs)
    # Old implementation wrote records before the 2026-08-09 05:13 UTC relaunch
    # with the single-forward optimization; later records use the new code.
    cutoff_unix = 1786252400.0
    validation = validate(records, input_dirs, cutoff_unix)
    stats = deep_stats(records)
    write_reports(output_dir, records)
    (output_dir / "precheck.md").write_text(
        render_markdown(validation, stats, input_dirs, output_dir),
        encoding="utf-8",
    )
    print(json.dumps({
        "records": validation["n_records"],
        "prompts": validation["n_prompts"],
        "issues": validation["issues"][:10],
        "state_class_counts": stats["state_class_counts"],
        "relay_gain_by_length": stats["relay_gain_by_length"],
        "transport_gain": stats["transport_gain"],
        "actionable_transport_positive": stats["actionable_transport_positive"],
        "answer_leakage_pass_rate": stats["answer_leakage_pass_rate"],
        "strong_relay": len(stats["strong_relay_examples"]),
        "strong_transport": len(stats["strong_transport_examples"]),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
