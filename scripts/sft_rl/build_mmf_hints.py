#!/usr/bin/env python3
"""Offline privileged-hint construction for PTD-PO on the MMF RL pool.

Mirrors PTD-PO (arXiv:2606.07000) Appendix C.1: given (image, question,
ground-truth answer), a strong model writes a concise, structured, answer-free
hint (spatial grounding + high-level reasoning direction + distractor
suppression). The answer is PRIVILEGED: it is only shown to the hint generator
offline and must never appear in the hint.

Output schema = input schema + three columns:
  - hint:            raw hint text ("" when QC hard-rejects and retry fails)
  - prompt_with_hint: verl chat messages for the hint-augmented context
                      (same <image> placeholders + question + hint); [] = no hint
  - hint_reason:     "" when accepted; else the QC reason(s) comma-joined

Idempotent: re-running resumes from existing output shards by sample_uid.

Usage:
  bash scripts/sft_rl/serve_hint_gen.sh                # start vLLM server
  python3 scripts/sft_rl/build_mmf_hints.py \
    --input-dir fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_3k \
    --out-dir  fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_3k_hint \
    --api-base http://127.0.0.1:8010/v1 \
    --model Qwen3-VL-235B-Instruct
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from geo3k_reward import _normalize_latex, _numeric_equal  # noqa: E402

DEFAULT_SYSTEM_PROMPT = """You are an expert multimodal reasoning tutor. Given an image, a question, and the
verified correct answer (PRIVATE — never reveal it), produce a CONCISE, structured
hint that teaches the reasoning path WITHOUT exposing the answer.

Hard rules:
1. Solution-consistent: point to the visual evidence and the reasoning steps that
   lead to the correct solution; align with the verified reasoning direction.
2. Zero-spoiler: NEVER output the final answer, exact intermediate numerical
   results, or object names that uniquely identify the answer. Do NOT write the
   solution trace or a chain of thought. For multiple-choice questions, NEVER
   mention any option letter (A, B, C, D, E...) or the phrases "correct option",
   "the answer is X", "option X is correct" — refer to choices by their content
   ("the shaded triangle", "the circuit with the ammeter") instead.
3. Distractor suppression: explicitly list which visual elements or textual cues
   are irrelevant or traps and should be ignored.
4. Multi-part answers: if the answer has several parts (multiple blanks,
   coordinates, or parameter settings), NEVER reveal or restate ANY individual
   part, formula, or parameter list. Describe only where to find each piece and
   how to derive it.

Format: imperative bullets, 2-6 bullets, 40-150 words. Give slightly more detail
only for visually ambiguous or logically difficult steps. Be terse; no summaries,
no "in conclusion", no code blocks.

Example (multiple-choice; the correct choice is PRIVATE, here "C"):
  - Compare the two figures' left column: the pattern alternates between rotation
    and color change across rows.
  - The third figure inherits the clockwise shift of the second figure; ignore the
    background grid and the text labels.
  - The choice that follows this rule is the one whose shape is rotated 90 degrees
    clockwise and recolored — describe it by content, never by letter."""

TRIVIAL_NUMBERS = {
    "0", "1", "2", "3", "4", "5", "6", "10", "100", "0.0", "1.0", "0.5", "50",
}
SPATIAL_WORDS = (
    "image", "chart", "graph", "diagram", "figure", "left", "right", "top",
    "bottom", "region", "area", "shape", "object", "axis", "label", "between",
    "above", "below", "corner", "center", "colour", "color", "row", "column",
    "grid", "angle", "line", "point", "section", "quadrant",
)
COT_DEGENERATE_RE = re.compile(
    r"(final answer|the answer is|therefore,?\s+the answer|correct response|"
    r"the response (?:is|should be)|正确答案|应选|答案是|答案[:：])",
    re.IGNORECASE,
)

HARD_REASONS = frozenset(
    {
        "empty",
        "too_long",
        "cot_degenerate",
        "answer_leak_letter",
        "answer_leak_word",
        "answer_leak_numeric",
        "answer_leak_substr",
        "answer_leak_component_numeric",
        "answer_leak_component_word",
        "degenerate_output",
    }
)


def _is_hard_rejected(reason: str) -> bool:
    if not reason:
        return False
    return any(k in HARD_REASONS for k in reason.split(","))


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _strip_latex(s: str) -> str:
    s = _normalize_latex(s)
    return re.sub(r"\\", "", s)


def _hint_numbers(s: str) -> list[str]:
    # integers / decimals / negatives; ignore LaTeX command args
    return re.findall(r"-?\d+(?:\.\d+)?", s)


def _answer_kind(gt: str) -> str:
    gt = gt.strip()
    if re.fullmatch(r"\(?([A-Ea-e])\)?\.?", gt):
        return "mcq"
    if gt.lower() in ("yes", "no", "true", "false"):
        return "yesno"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", gt):
        return "numeric"
    return "other"


def _mcq_letter(gt: str) -> str:
    m = re.fullmatch(r"\(?([A-Ea-e])\)?\.?", gt.strip())
    return m.group(1).upper() if m else ""


def _answer_components(gt: str) -> list[str]:
    """Split multi-part answers like '50, 57.6, 292' or 'A; C'."""
    return [p.strip() for p in re.split(r"[,;、；]", gt) if p.strip()]


def qc_hint(
    hint: str,
    answer: str,
    original_answer: str = "",
    question: str = "",
    max_chars: int = 2048,
    max_decisive_overlap: int = 3,
) -> tuple[bool, list[str], list[str]]:
    """Return (ok, hard_reasons, soft_reasons).

    hard reasons -> regenerate; soft reasons -> recorded + human audit.
    """
    hard: list[str] = []
    soft: list[str] = []
    hint = hint.strip()
    gt = (answer or "").strip()

    if not hint:
        hard.append("empty")
        return False, hard, soft
    if len(hint) > max_chars:
        hard.append("too_long")
    # Degenerate low-entropy output (e.g. 768x "!" from a broken FP8-MoE
    # serving). These passed every leak/regex check on 2026-08-25 and were
    # wrongly accepted, so this is a hard rejection, not a soft flag.
    stripped = re.sub(r"\s+", "", hint)
    if len(stripped) >= 16 and len(set(stripped)) <= 3:
        hard.append("degenerate_output")
    if COT_DEGENERATE_RE.search(hint):
        hard.append("cot_degenerate")

    kind = _answer_kind(gt)
    hint_compact = _compact(hint)
    if kind == "mcq":
        letter = _mcq_letter(gt)
        if letter and re.search(rf"\b[{letter}]\b", hint, re.IGNORECASE):
            hard.append("answer_leak_letter")
    elif kind == "yesno":
        if re.search(rf"\b{re.escape(gt.lower())}\b", hint, re.IGNORECASE):
            hard.append("answer_leak_word")
    elif kind == "numeric":
        # numbers already present in the question are public context: a hint may
        # cite them without leaking the answer (fixes the dominant false-positive).
        question_numbers = set(_hint_numbers(question))
        for num in _hint_numbers(hint):
            if _numeric_equal(num, gt) and num not in question_numbers:
                hard.append("answer_leak_numeric")
                break
    # general textual containment (covers expressions / latex / raw strings)
    gt_variants = {
        _compact(gt),
        _compact(_strip_latex(gt)),
        _compact(re.sub(r"[^A-Za-z0-9./=+-]", "", gt)),
    }
    for v in gt_variants:
        if v and len(v) >= 2 and v in hint_compact:
            hard.append("answer_leak_substr")
            break
    # fraction forms: "\frac{1}{4}" -> "1/4" (and direct "1/4" in gt_variants)
    frac_forms: set[str] = set()
    for m in re.finditer(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", gt):
        frac_forms.add(_compact(f"{m.group(1)}/{m.group(2)}"))
    if "/" in gt:
        frac_forms.add(_compact(gt))
    for ff in frac_forms:
        if ff and ff in hint_compact:
            hard.append("answer_leak_substr")
            break

    # component-wise leakage for multi-part answers (e.g. "50, 57.6, 292"):
    # the full-string check above misses a hint that reveals only one part
    # ("total = 32 / 0.64 = 50 days"). Also cover numbers embedded in "other"
    # answers ("80 square units").
    components = _answer_components(gt)
    hint_nums = _hint_numbers(hint)
    # numbers that appear as the result of an arithmetic expression in the hint
    # ("= 50", "0.64 -> 50"): these are computed values, so the "number also
    # appears in the question" exemption must not apply.
    computed_nums = set(
        re.findall(r"[=÷×xX*/+\-]\s*(-?\d+(?:\.\d+)?)", hint)
    )
    check_components = components if len(components) > 1 else (
        [gt] if kind == "other" else []
    )
    question_numbers = set(_hint_numbers(question))
    for comp in check_components:
        comp_kind = _answer_kind(comp)
        comp_nums = [comp] if comp_kind == "numeric" else _hint_numbers(comp)
        leaked_num = False
        for cn in comp_nums:
            for num in hint_nums:
                if _numeric_equal(num, cn) and (
                    num not in question_numbers or num in computed_nums
                ):
                    hard.append("answer_leak_component_numeric")
                    leaked_num = True
                    break
            if leaked_num:
                break
        if not leaked_num and len(comp) >= 4 and re.fullmatch(r"[A-Za-z][A-Za-z \-]+", comp):
            if re.search(rf"\b{re.escape(comp)}\b", hint, re.IGNORECASE):
                hard.append("answer_leak_component_word")

    # decisive intermediate numbers (soft; heuristic)
    if original_answer:
        orig_nums = {
            n for n in _hint_numbers(original_answer) if n not in TRIVIAL_NUMBERS
        }
        hint_nums = set(_hint_numbers(hint))
        overlap = orig_nums & hint_nums
        if len(overlap) >= max_decisive_overlap:
            soft.append(f"decisive_overlap:{sorted(overlap)[:5]}")

    if not any(w in hint.lower() for w in SPATIAL_WORDS):
        soft.append("no_spatial")

    return (not hard), hard, soft


def build_prompt_with_hint(question: str, hint: str) -> list[dict[str, str]]:
    if not hint.strip():
        return []
    return [{"role": "user", "content": f"{question}\n\n{hint.strip()}"}]


def _img_to_data_url(imgs: list[dict[str, Any]]) -> list[str]:
    urls = []
    for img in imgs or []:
        b = img.get("bytes")
        if b is None and img.get("image_url"):
            # Cauldron pools store images as external file references.
            try:
                b = Path(img["image_url"]).read_bytes()
            except OSError:
                continue
        if b is None:
            continue
        # sniff mime so vLLM gets the right content type for pooled stores
        # that use extension-less .img files
        if b.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif b.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif b.startswith(b"RIFF") and b[8:12] == b"WEBP":
            mime = "image/webp"
        elif b.startswith(b"GIF87a") or b.startswith(b"GIF89a"):
            mime = "image/gif"
        elif b.startswith(b"BM"):
            mime = "image/bmp"
        else:
            mime = "image/png"
        if isinstance(b, bytes):
            urls.append(f"data:{mime};base64,{base64.b64encode(b).decode()}")
    return urls


def _call_hint_api(
    client: Any,
    model: str,
    system_prompt: str,
    question: str,
    image_urls: list[str],
    answer: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    user_parts: list[dict[str, Any]] = []
    for url in image_urls:
        user_parts.append({"type": "image_url", "image_url": {"url": url}})
    user_parts.append(
        {
            "type": "text",
            "text": (
                f"{question}\n\n"
                f"Ground-truth answer (PRIVATE — for hint construction only, "
                f"never reveal it): {answer}"
            ),
        }
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_parts},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        # qwen3.5/3.6 chat templates default to thinking mode, which emits a
        # full worked solution ("The user wants a hint...") instead of the
        # concise zero-spoiler hint; prefill an empty <think> block instead.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or ""


def read_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(input_dir.glob("*.parquet")):
        # never mix val rows into the hint training set (train/val must stay disjoint)
        if "val" in p.name.lower():
            continue
        t = pq.read_table(str(p))
        rows.extend(t.to_pylist())
    return rows


def write_shards(
    rows: list[dict[str, Any]],
    out_dir: Path,
    shard_rows: int = 1500,
    prefix: str = "mmf_rl_train_hint",
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_schema = pa.schema(
        [
            pa.field("data_source", pa.string()),
            pa.field("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            # image struct is pool-dependent: MMF pools carry (bytes, path);
            # Cauldron pools carry external (image_url) references.
            _images_field(rows),
            pa.field("ability", pa.string()),
            pa.field("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            pa.field(
                "extra_info",
                pa.struct(
                    [
                        ("question", pa.string()),
                        ("source", pa.string()),
                        ("gt_source", pa.string()),
                        ("original_answer", pa.string()),
                    ]
                ),
            ),
            pa.field("question", pa.string()),
            pa.field("sample_uid", pa.string()),
        ]
    )
    schema = base_schema.append(
        pa.field("hint", pa.string())
    ).append(
        pa.field(
            "prompt_with_hint",
            pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
        )
    ).append(pa.field("hint_reason", pa.string()))

    paths: list[Path] = []
    n_shards = (len(rows) + shard_rows - 1) // shard_rows
    for i in range(n_shards):
        chunk = rows[i * shard_rows : (i + 1) * shard_rows]
        out = out_dir / f"{prefix}__part_{i:04d}.parquet"
        table = pa.Table.from_pylist(chunk, schema=schema)
        pq.write_table(table, out, compression="zstd")
        paths.append(out)
    return paths


def _images_field(rows: list[dict[str, Any]]) -> pa.Field:
    for r in rows:
        imgs = r.get("images") or []
        if imgs:
            keys = [k for k in ("bytes", "path", "image_url") if k in imgs[0]]
            if "image_url" in keys:
                return pa.field("images", pa.list_(pa.struct([("image_url", pa.string())])))
            return pa.field(
                "images",
                pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            )
    return pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())])))


def _shard_prefix_of(shards: list[Path]) -> str:
    """Infer the original shard prefix so --recheck keeps pool-specific naming."""
    if not shards:
        return "mmf_rl_train_hint"
    m = re.match(r"^(.*)__part_\d+\.parquet$", shards[0].name)
    return m.group(1) if m else "mmf_rl_train_hint"


def load_existing(out_dir: Path) -> dict[str, tuple[str, str]]:
    """sample_uid -> (hint, reason) from previously written shards."""
    existing: dict[str, tuple[str, str]] = {}
    if not out_dir.exists():
        return existing
    for p in sorted(out_dir.glob("*.parquet")):
        try:
            t = pq.read_table(str(p), columns=["sample_uid", "hint", "hint_reason"])
            for uid, hint, reason in zip(
                t.column("sample_uid").to_pylist(),
                t.column("hint").to_pylist(),
                t.column("hint_reason").to_pylist(),
            ):
                existing[str(uid)] = (str(hint), str(reason))
        except Exception:
            continue
    return existing


def recheck_existing(args: argparse.Namespace) -> int:
    """Re-run QC on already-generated hints with the current rules.

    No API calls: newly hard-rejected hints are emptied in place (PTD then
    simply does not activate on those rows). Original shards are backed up
    under ``hint_recheck_backup_<timestamp>/`` before rewriting.
    """
    import shutil

    shards = sorted(args.out_dir.glob("*.parquet"))
    if not shards:
        print(f"FATAL: no parquet shards under {args.out_dir}", file=sys.stderr)
        return 2

    backup_dir = args.out_dir / f"hint_recheck_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for p in shards:
        shutil.copy2(p, backup_dir / p.name)

    rows: list[dict[str, Any]] = []
    total = kept = emptied = already_empty = 0
    reason_counter: dict[str, int] = {}
    for p in shards:
        t = pq.read_table(str(p))
        for row in t.to_pylist():
            total += 1
            hint = str(row.get("hint") or "")
            if not hint:
                already_empty += 1
                rows.append(row)
                continue
            answer = str((row.get("reward_model") or {}).get("ground_truth", ""))
            original_answer = str((row.get("extra_info") or {}).get("original_answer", ""))
            question = str(row.get("question", ""))
            ok, hard, soft = qc_hint(
                hint,
                answer,
                original_answer,
                question=question,
                max_chars=args.max_chars,
                max_decisive_overlap=args.max_decisive_overlap,
            )
            if ok:
                kept += 1
                if soft:
                    row["hint_reason"] = ",".join(soft)
                rows.append(row)
            else:
                emptied += 1
                row["hint"] = ""
                row["prompt_with_hint"] = []
                row["hint_reason"] = ",".join(hard + soft)
                for r_ in hard:
                    reason_counter[r_] = reason_counter.get(r_, 0) + 1
                rows.append(row)

    # drop stale shards, then rewrite with the same shard size / naming
    for p in shards:
        p.unlink()
    write_shards(rows, args.out_dir, shard_rows=args.shard_rows,
                 prefix=_shard_prefix_of(shards))

    stats = {
        "rechecked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "kept": kept,
        "emptied": emptied,
        "already_empty": already_empty,
        "new_hard_reasons": reason_counter,
        "backup_dir": str(backup_dir),
    }
    (args.out_dir / "hint_recheck_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[recheck] total={total} kept={kept} emptied={emptied} "
        f"already_empty={already_empty} reasons={reason_counter}",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--api-base", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--model", default="Qwen3-VL-235B-Instruct")
    ap.add_argument("--system-prompt-file", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--retry-temperature", type=float, default=0.9)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N rows")
    ap.add_argument("--retry-rejected", action="store_true",
                    help="re-generate hints only for hard-rejected rows; "
                         "accepted rows are kept as-is")
    ap.add_argument("--shard-rows", type=int, default=1500)
    ap.add_argument("--shard-prefix", default="mmf_rl_train_hint",
                    help="output shard filename prefix (e.g. mmf_hint / cauldron_hint)")
    ap.add_argument("--max-chars", type=int, default=2048)
    ap.add_argument("--max-decisive-overlap", type=int, default=3)
    ap.add_argument("--include-reference-reasoning", action="store_true",
                    help="append original_answer (private) to the generator input")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-api", action="store_true",
                    help="dry-run mode: run QC only on placeholder hints (for tests)")
    ap.add_argument("--recheck", action="store_true",
                    help="re-run QC on existing output shards with the current "
                         "rules; newly hard-rejected hints are emptied in place "
                         "(no API calls, originals backed up)")
    args = ap.parse_args()

    if args.recheck:
        return recheck_existing(args)

    if args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    prompt_version = hashlib.sha256(system_prompt.encode()).hexdigest()[:12]

    rows = read_rows(args.input_dir)
    if args.limit > 0:
        rows = rows[: args.limit]
    existing = load_existing(args.out_dir)
    if args.retry_rejected:
        todo = [
            r
            for r in rows
            if str(r.get("sample_uid", "")) not in existing
            or _is_hard_rejected(existing[str(r.get("sample_uid", ""))][1])
        ]
    else:
        todo = [r for r in rows if str(r.get("sample_uid", "")) not in existing]
    print(f"[hints] total={len(rows)} existing={len(existing)} todo={len(todo)}", flush=True)

    stats: dict[str, int] = {"ok": 0}
    reasons_counter: dict[str, int] = {}
    rejected_samples: list[str] = []
    progress = {"processed": 0, "ok": 0, "accepted": 0, "hard_reject": 0, "reasons": {}}
    sample_dump: set[str] = set()

    def write_progress() -> None:
        progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "hint_gen_progress.json").write_text(
            json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # dump a few raw samples for manual quality review (accepted + hard-rejected)
        samples_path = args.out_dir / "hint_samples.jsonl"
        with samples_path.open("a", encoding="utf-8") as f:
            for r in out_rows:
                uid = str(r.get("sample_uid", ""))
                if uid in sample_dump:
                    continue
                is_ok = r.get("_qc_ok", False)
                if is_ok and len(sample_dump) >= 10:
                    continue
                if not is_ok and len(sample_dump) >= 5:
                    continue
                f.write(
                    json.dumps(
                        {
                            "sample_uid": uid,
                            "question": str(r.get("question", ""))[:400],
                            "answer": str((r.get("reward_model") or {}).get("ground_truth", ""))[:80],
                            "hint": str(r.get("hint", ""))[:800],
                            "hard": r.get("_qc_hard", []),
                            "reason": r.get("hint_reason", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sample_dump.add(uid)
                if len(sample_dump) >= 15:
                    break

    if not args.no_api:
        try:
            from openai import OpenAI
        except ImportError:
            print("FATAL: openai client not installed in this env", file=sys.stderr)
            return 2
        client = OpenAI(base_url=args.api_base, api_key="EMPTY", timeout=args.timeout)

    def process(row: dict[str, Any]) -> dict[str, Any]:
        uid = str(row.get("sample_uid", ""))
        question = str(row.get("question", ""))
        answer = str((row.get("reward_model") or {}).get("ground_truth", ""))
        original_answer = str((row.get("extra_info") or {}).get("original_answer", ""))
        images = row.get("images") or []

        hint = ""
        reason = ""
        tries = [args.temperature, args.retry_temperature]
        for attempt, temp in enumerate(tries):
            if args.no_api:
                hint = ""  # placeholder; QC will reject -> exercise reject path
            else:
                try:
                    for retry in range(args.max_retries):
                        try:
                            hint = _call_hint_api(
                                client,
                                args.model,
                                system_prompt,
                                question,
                                _img_to_data_url(images),
                                answer if not args.include_reference_reasoning else (
                                    f"{answer}\n\nReference reasoning (PRIVATE): {original_answer}"
                                ),
                                args.max_new_tokens,
                                temp,
                                args.timeout,
                            )
                            break
                        except Exception as e:  # noqa: BLE001
                            if retry == args.max_retries - 1:
                                raise
                            time.sleep(2 ** (retry + 1))
                except Exception as e:  # noqa: BLE001
                    reason = f"api_error:{type(e).__name__}"
                    hint = ""
                    break
            ok, hard, soft = qc_hint(
                hint,
                answer,
                original_answer,
                question=question,
                max_chars=args.max_chars,
                max_decisive_overlap=args.max_decisive_overlap,
            )
            row["_qc_ok"] = ok
            row["_qc_hard"] = hard
            row["_qc_soft"] = soft
            if ok:
                reason = ",".join(soft)
                break
            reason = ",".join(hard)
            if attempt == len(tries) - 1:
                hint = ""
        row["hint"] = hint
        row["prompt_with_hint"] = build_prompt_with_hint(question, hint)
        row["hint_reason"] = reason
        return row

    out_rows: list[dict[str, Any]] = []
    if existing and args.retry_rejected:
        # keep only accepted rows from disk; hard-rejected rows are regenerated
        for r in rows:
            uid = str(r.get("sample_uid", ""))
            if uid in existing and not _is_hard_rejected(existing[uid][1]):
                hint, reason = existing[uid]
                r["hint"] = hint
                r["prompt_with_hint"] = build_prompt_with_hint(
                    str(r.get("question", "")), hint
                )
                r["hint_reason"] = reason
                r["_qc_ok"] = True
                r["_qc_hard"] = []
                r["_qc_soft"] = [] if not reason else reason.split(",")
                out_rows.append(r)
        # NOTE: hard-rejected rows must only enter via the todo-processing
        # loop below; appending them here duplicates the batch.
    elif existing:
        for r in rows:
            uid = str(r.get("sample_uid", ""))
            if uid in existing:
                hint, reason = existing[uid]
                r["hint"] = hint
                r["prompt_with_hint"] = build_prompt_with_hint(
                    str(r.get("question", "")), hint
                )
                r["hint_reason"] = reason
                r["_qc_ok"] = not reason
                r["_qc_hard"] = [] if not reason else reason.split(",")
                r["_qc_soft"] = [] if not reason else []
                out_rows.append(r)
    else:
        # fresh run: every row enters via the todo-processing loop below only;
        # pre-seeding here would duplicate the batch.
        out_rows = []

    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
            futs = {ex.submit(process, r): r for r in todo}
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:  # noqa: BLE001
                    row = futs[fut]
                    row["hint"] = ""
                    row["prompt_with_hint"] = []
                    row["hint_reason"] = f"worker_error:{type(e).__name__}"
                out_rows.append(row)
                done += 1
                reason = row.get("hint_reason", "")
                progress["processed"] = done
                if not reason:
                    progress["ok"] += 1
                if row.get("_qc_ok", False):
                    progress["accepted"] += 1
                else:
                    progress["hard_reject"] += 1
                    for key in row.get("_qc_hard", []):
                        progress["reasons"][key] = progress["reasons"].get(key, 0) + 1
                if done % 50 == 0 or done == len(todo):
                    print(
                        f"[hints] {done}/{len(todo)} accepted={progress['accepted']} "
                        f"hard_reject={progress['hard_reject']} "
                        f"reject={json.dumps(progress['reasons'], ensure_ascii=False)}",
                        flush=True,
                    )
                    write_progress()

    if todo:
        write_progress()

    # order rows by original index for deterministic shards
    order = {id(r): i for i, r in enumerate(rows)}
    out_rows.sort(key=lambda r: order[id(r)])

    for r in out_rows:
        reason = r.get("hint_reason", "")
        if not reason:
            stats["ok"] += 1
        else:
            for key in r.get("_qc_hard", []):
                reasons_counter[key] = reasons_counter.get(key, 0) + 1
            rejected_samples.append(str(r.get("sample_uid", "")))

    paths = write_shards(out_rows, args.out_dir, shard_rows=args.shard_rows,
                         prefix=args.shard_prefix)
    config = {
        "input_dir": str(args.input_dir),
        "out_dir": str(args.out_dir),
        "api_base": args.api_base,
        "model": args.model,
        "system_prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "retry_temperature": args.retry_temperature,
        "max_chars": args.max_chars,
        "max_decisive_overlap": args.max_decisive_overlap,
        "include_reference_reasoning": args.include_reference_reasoning,
        "seed": args.seed,
        "n_rows": len(rows),
        "hint_ratio": round(stats["ok"] / len(rows), 4) if rows else 0.0,
    }
    (args.out_dir / "hint_gen_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out_dir / "hint_gen_stats.json").write_text(
        json.dumps(
            {"stats": stats, "reasons": reasons_counter,
             "rejected_samples": rejected_samples[:500]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_progress()
    print(f"[hints] done: shards={[str(p) for p in paths]} stats={stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
