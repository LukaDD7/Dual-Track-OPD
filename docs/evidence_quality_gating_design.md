# Evidence Quality Gating: Constrained Decoding + Rejection Sampling + Resampling

## Problem Statement

4B Qwen3-VL generates `task_infer` evidence that leaks the final answer ("solve_like").
The 572f737 prompt fix reduced solve_like from 31/32 → 2/32 records, but:

1. **2 records still fail** (4/64 rollout-rows lost; `visual_text_inference` receives zero signal from those samples)
2. **Current rejection drops entire records** — no retry, no resampling, silently reduces effective sample count
3. **4B scale needs reliability** — every lost record is a waste of scarce training signal

## Goals

1. **Hard guarantee**: zero solve_like evidence entering offline scoring
2. **Full sample utilization**: every selected record contributes K valid rollouts
3. **Observability**: log every rejection and retry for debugging

## Architecture: Two-Layer Gating

```
┌─────────────────────────────────────────────────────────┐
│ LAYER 1: Evidence Generation (evidence_generation.py)    │
│                                                         │
│  For each record:                                       │
│    attempt = 0                                          │
│    while attempt < max_attempts:                        │
│      task_infer = model.generate(                       │
│        prompt,                                          │
│        logits_processors=[ForbidAnswerTokens()]  ← NEW  │
│      )                                                  │
│      cls = classify(task_infer)                         │
│      if cls == "clean_infer":                           │
│        write to cache; break                            │
│      else:                                              │
│        log rejection; attempt++  ← NEW (retry)          │
│                                                         │
│    if exhausted:                                        │
│      mark record as "evidence_failed"                   │
│      → will be dropped at scoring time (no retry)       │
│                                                         │
│ LAYER 1.5: Evidence Cache Validation (4c builder load)  │
│                                                         │
│  validate_evidence_row() — unchanged                    │
│  strict_condition_validation: drop solve_like rows      │
│  (should be near-zero after Layer 1 fix)               │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ LAYER 2: Rollout Generation (4c offline builder)        │
│                                                         │
│  For each record that passed evidence validation:       │
│    for rollout_id in range(K):                          │
│      attempt = 0                                        │
│      while attempt < max_rollout_attempts:              │
│        response = rollout_generator.generate()          │
│        chunks = parse_chunks(response)                  │
│        if chunks is valid (has answer, non-empty):      │
│          write row; break                               │
│        else:                                            │
│          log rejection; attempt++  ← NEW (retry)        │
│                                                         │
│      if exhausted:                                      │
│        write fallback row (malformed outcome, low gate) │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Layer 1: Constrained Decoding

### Token Forbid List

When generating `task_infer` evidence, block these tokens at the logits-processor level:

```
Forbid set = {A, B, C, D, E,  A., B., C., D.,  (A), (B), (C), (D),
              answer, Answer, ANSWER,
              "the answer is", "The answer is",
              "因此", "所以答案是", "故选"}
```

### Implementation

```python
# src/dual_track_opd/fc_opd/constrained_decode.py  (NEW FILE)

import torch
from transformers import LogitsProcessor

class ForbidAnswerTokensLogitsProcessor(LogitsProcessor):
    """Set logits of forbidden token IDs to -inf before sampling."""
    
    def __init__(self, forbidden_token_ids: list[int]):
        self.forbidden_ids = torch.tensor(forbidden_token_ids, dtype=torch.long)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.forbidden_ids] = -float("inf")
        return scores


def build_forbidden_token_ids(tokenizer, extra_strings: list[str] | None = None) -> list[int]:
    """Build the set of token IDs to forbid during evidence generation.
    
    Includes:
    - Upper/lower case option letters A-D (single tokens)
    - Variants: 'A.', ' A', '(A)', ' A.', etc.
    - Common answer-prefix tokens found empirically
    """
    forbid_strings = [
        # Option letters (various formats)
        "A", "B", "C", "D", "E",
        " A", " B", " C", " D",
        "A.", "B.", "C.", "D.",
        " A.", " B.", " C.", " D.",
        "(A)", "(B)", "(C)", "(D)",
        "a", "b", "c", "d",
        # Answer prefixes
        "answer", "Answer", "ANSWER",
        # Chinese answer patterns
        "答案", "故选", "因此",
    ]
    if extra_strings:
        forbid_strings.extend(extra_strings)
    
    token_ids = set()
    for s in forbid_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        token_ids.update(ids)
    
    return sorted(token_ids)
```

### Integration Point

`evidence_generation.py:_generate_single()` — add `logits_processors` kwarg:

```python
def _generate_single(
    self,
    prompt: str,
    image_path: str,
    *,
    max_new_tokens: int = 512,
    logits_processors: list | None = None,  # NEW
) -> str:
    ...
    outputs = self.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        logits_processors=logits_processors,  # NEW
        ...
    )
```

Only apply constrained decoding for `task_infer` evidence generation — NOT for `free_caption`, `task_visible`, `task_solve`, or student rollouts.

## Layer 1: Rejection Sampling + Resampling

### Config

```python
@dataclass
class EvidenceGenerationConfig:
    ...
    # NEW fields
    max_evidence_attempts: int = 3       # retries per record for task_infer
    enable_constrained_decoding: bool = True
    constrained_decode_condition: str = "task_infer"  # which condition to constrain
```

### Resample Loop

In `_build_evidence_row()` (or equivalent per-record function):

```python
def _generate_task_infer_with_retry(
    generator, prompt, image_path, choices, config,
) -> tuple[str, dict]:
    """Generate task_infer evidence with retry on solve_like."""
    
    logits_processor = None
    if config.enable_constrained_decoding:
        forbidden_ids = build_forbidden_token_ids(generator.tokenizer)
        logits_processor = ForbidAnswerTokensLogitsProcessor(forbidden_ids)
    
    for attempt in range(config.max_evidence_attempts):
        raw = generator.generate(
            prompt=prompt,
            image_path=image_path,
            logits_processors=[logits_processor] if logits_processor else None,
        )
        classification = classify_task_infer_evidence(raw, choices)
        
        if classification["class"] == "clean_infer":
            return raw, {
                "attempts": attempt + 1,
                "constrained_decoding_used": config.enable_constrained_decoding,
            }
        
        log.warning(
            f"task_infer rejection attempt={attempt+1}/{config.max_evidence_attempts} "
            f"class={classification['class']} flags={classification['flags']}"
        )
    
    # Exhausted retries
    return raw, {
        "attempts": config.max_evidence_attempts,
        "exhausted_retries": True,
        "final_class": classification["class"],
    }
```

### Output Row Changes

Each evidence cache row gets new fields:

```json
{
  "task_infer_retry_attempts": 1,
  "task_infer_constrained_decoding_used": true,
  "task_infer_exhausted_retries": false
}
```

## Layer 2: Rollout Resampling

### Motivation

The student 4B model may occasionally produce empty or unparseable rollouts. Currently these are dropped (counted as `rollout_empty_count`). We should retry to ensure K valid rollouts per record.

### Config

```python
@dataclass
class FourConditionOfflineBuilderConfig:
    ...
    max_rollout_attempts: int = 3  # NEW: retries per rollout slot
```

### Resample Loop

In `_build_rows()`, replace the single `rollout_generator.generate()` call:

```python
for rollout_id in range(config.rollouts_per_prompt):
    response_text = None
    for attempt in range(config.max_rollout_attempts):
        seed = rollout_seed(base_seed=config.seed, source_index=source_index, 
                           rollout_id=rollout_id, attempt=attempt)
        try:
            response_text = rollout_generator.generate(
                question=question, image_path=image_path, 
                prompt_text=prompt.text, seed=seed
            )
        except Exception:
            continue
        
        if response_text and response_text.strip():
            break  # valid rollout
        response_text = None
    
    if response_text is None:
        # Fallback: write row with malformed outcome
        diagnostics.rollout_exhausted_count += 1
        row = _build_fallback_row(record, evidence_row, rollout_uid, ...)
    else:
        diagnostics.rollout_success_count += 1
        row = _build_row_from_response(response_text, ...)
    
    rows.append(row)
```

## Diagnostics Additions

### Evidence Builder Summary

```json
{
  "constrained_decoding_enabled": true,
  "task_infer_retry_stats": {
    "total_attempts": 70,
    "total_retries": 6,
    "exhausted_retries": 1,
    "attempt_distribution": {"1": 28, "2": 3, "3": 1}
  }
}
```

### Offline Builder Diagnostics

```python
@dataclass
class OfflineBuilderDiagnostics:
    ...
    rollout_exhausted_count: int = 0      # NEW: rollouts that failed all retries
    rollout_retry_count: int = 0           # NEW: total retries across all rollouts
    evidence_exhausted_records: int = 0    # NEW: records that failed evidence retries
```

### Summary Output

```json
{
  "rollout_exhausted_count": 1,
  "rollout_retry_count": 5,
  "evidence_exhausted_records": 0,
  "skipped_records_by_reason": {
    "evidence_exhausted_retries": 0,
    "evidence_validation": 0,
    "rollout_exhausted": 1,
    ...
  }
}
```

## Files to Modify

| File | Change |
|------|--------|
| `src/dual_track_opd/fc_opd/constrained_decode.py` | **NEW** — `ForbidAnswerTokensLogitsProcessor`, `build_forbidden_token_ids()` |
| `src/dual_track_opd/fc_opd/evidence_generation.py` | Add `logits_processors` to model.generate(); add retry loop in per-record generation; new config fields; new summary fields |
| `src/dual_track_opd/fc_opd/four_condition_offline_builder.py` | Add `max_rollout_attempts` config; retry loop in `_build_rows()`; new diagnostics counters; fallback row builder |
| `scripts/hpc/build_fc_opd_4c_evidence_cache.py` | Pass new CLI args: `--max-evidence-attempts`, `--no-constrained-decoding` |
| `scripts/hpc/build_fc_opd_geometry3k_4c_offline_scores.py` | Pass new CLI arg: `--max-rollout-attempts` |

## Expected Impact

| Metric | Before (572f737) | After (Layer 1+2) |
|--------|------------------|-------------------|
| Evidence solve_like rate | 6.25% (4/64) | **0%** |
| Records contributing full K rollouts | 30/32 | **32/32** |
| Rollouts lost to empty/malformed | variable | **0** (fallback rows instead) |
| `visual_text_inference` weight | 4,678 (limit=32) | **~5,000+** (all records contribute) |
| Remaining friction | Prompt weakness | **Token-level guarantee** |

## Implementation Order (for codex)

1. **Write `constrained_decode.py`** — forbid-list logits processor
2. **Wire into evidence generation** — `_generate_single()` accepts `logits_processors`; only applied for `task_infer`
3. **Add retry loop in evidence builder** — `while attempt < max_attempts` around task_infer generation
4. **Add retry loop in offline builder** — `while attempt < max_rollout_attempts` around rollout generation
5. **Add fallback row builder** — when rollout is exhausted, write a row that gates low
6. **Update CLIs** — new args exposed in shell scripts
7. **Smoke test** — limit=32 constrained decode → verify 32/32 clean, 32×K rows
