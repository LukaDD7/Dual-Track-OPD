# FC-OPD Codex Next Steps

Written 2026-06-28.  Summarises what is done, what is still missing, and
concrete implementation tasks for the next Codex round.  This document is
the single source of truth for remaining FC-OPD integration work.

**2026-06-29 correction:** Use
`docs/fc_opd_gpu_first_validation.md` as the GPU-first validation checklist.
CPU fallback is no longer an acceptable proof of FC-OPD feasibility.  The verl
smoke must instantiate a real GPU student scorer or fail fast.

## 1. Current State (What Changed Since Codex's Last Push)

### 1.1  Branch `codex/failure-calibrated-opd` at `b2bad5b`

| What | Status |
|------|--------|
| Core FC-OPD algorithm (`conditions`, `loss`, `router`, `verifier`, `signal_decomposer`) | ✅  Done |
| Teacher service (HTTP, transformers backend) | ✅  Done |
| Offline score builders (diagnostics only, NOT for training) | ✅  Done |
| Online batch assembler (`online_batch.py`) — framework-independent | ✅  Done |
| verl tensor adapter (`verl_integration.py`) — shapes `[B,C,T,K]` | ✅  Done |
| verl patches (26+177 lines, cleanly applicable) | ✅  Done |
| Post-rollout hook (`verl_post_rollout_hook.py`) — wires batch → online batch | ✅  Done |
| Sparse KD math (`verl_sparse_kd.py`) — consumed by actor patch | ✅  Done |
| CPU-level smoke tests (195 passed, 2 skipped) | ✅  Done |
| GPU student scorer (`student_scorer.py`) — FQN-callable, standalone | ✅  Added this round |
| Multi-sample batched teacher scoring (`score_teacher_conditions_multi_sample`) | ✅  Added this round |
| End-to-end GPU training smoke (verl + patches applied + GPU) | 🟡  Partial (loss=0.211, param delta=4.33e-08, but infra issues block clean run) |
| Remove-padding + Ulysses SP path in actor patch | ❌  NOT DONE |
| Hydra config for FC-OPD training | 🟡  CLI overrides work; structured config not yet loadable via `--config-name` |

### 1.2  This Round's Additions

1.  **`src/dual_track_opd/fc_opd/student_scorer.py`** — Standalone GPU student
    forced-scorer implementing `StudentForcedScorer` protocol.  Loadable via
    FQN: `dual_track_opd.fc_opd.student_scorer.StudentScorer`.  Accepts
    `model_path`, `device`, `dtype`, `top_k` as constructor kwargs (passed
    via `student_scorer_kwargs` in verl config).  This is the required path for
    the next GPU feasibility smoke.

2.  **`score_teacher_conditions_multi_sample()`** in `teacher_client.py` —
    Sends B×C teacher requests in a single HTTP POST instead of B sequential
    calls.

3.  **GPU-first `verl_post_rollout_hook.py`** — When `student_scorer_fqn` is not
    configured, the hook now fails fast.  CPU fallback does not validate real
    Qwen vocabulary alignment, student logits, GPU memory, or actor loss wiring,
    so it must not be used as the main smoke path.

4.  **`configs/experiment/verl_fc_opd_smoke.yaml`** — Reference verl Hydra
    config for FC-OPD training.

5.  **`scripts/hpc/run_verl_fc_opd_smoke.sh`** — GPU launch script that
    starts Ray, verifies patches, and runs one verl PPO step with FC-OPD
    enabled.

6.  **Minimal verl parquet dataset** at
    `fc-opd-storage/outputs/fc_opd/verl_smoke/train.parquet` — 2-sample
    Geometry3K dataset with FC-OPD fields (`question`, `condition_inputs`,
    `choices`, `answer_metadata`).

7.  **Exports** — `StudentScorer`, `score_teacher_conditions_multi_sample`
    exported in `__init__.py`.

### 1.3  GPU Smoke Troubleshooting Log

One verl PPO step with FC-OPD **did succeed once** (loss=0.211, parameter
delta=4.33e-08, all 6 conditions active, verifier_outcome=correct), proving
the actor patch works end-to-end.  However, due to GPU pod restarts,
subsequent attempts hit infrastructure issues:

| Issue | Symptom | Fix |
|-------|---------|-----|
| **CC unset / wrong path** | `No such file or directory: '.../bin/gcc'` | vLLM v1 + Triton JIT needs `CC` pointing to a valid C compiler. If user manually `export CC=$CONDA_PREFIX/bin/gcc` but the env lacks `gcc_linux-64`, the worker crashes. Fix: script now validates `CC` is executable, falls back to `gcc` on PATH. |
| **Teacher model path** | HF hub timeout (GPU pod has no internet) | Use local path: `--model /path/to/models/Qwen3-VL-32B-Instruct`, NOT `--model Qwen/Qwen3-VL-32B-Instruct` |
| **FlashAttention2 not installed** | `attn_implementation='flash_attention_2'` not found | Override: `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` |
| **Hydra struct error** | `algorithm.fc_opd` not a recognized key in `AlgoConfig` | All `algorithm.fc_opd.*` overrides need `+` prefix for new keys |
| **Missing `log_prob_micro_batch_size_per_gpu`** | verl validation error | Must be set explicitly (`=2`) |
| **vLLM fused kernels** | Assertion failure with `use_fused_kernels=false` | FC-OPD needs live logits → `use_fused_kernels=false` required, but may conflict with vLLM v1. Try `enforce_eager=true` |

**Current blocker**: vLLM v1 worker crashes because `CC` resolves to a
non-existent path.  The smoke script now validates `CC` before passing it
to the worker.  Run with `unset CC` or let the script fall back to `gcc`.

## 2. Remaining Work — Prioritized Task List

### P0 — Blocks First Real Training Run

#### Task 1: End-to-End GPU Training Smoke

Apply the two verl patches, start a teacher service, and run one PPO step
with `fc_opd_coef > 0`.

**Command** (on the 4×H200 GPU pod):

```bash
# 1. Activate env and ensure CC is valid
conda activate /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/envs/fc-opd-verl071-cu128
unset CC  # let smoke script auto-detect

# 2. Start teacher on a free GPU (32B model ~65GB, needs 1 GPU)
CUDA_VISIBLE_DEVICES=0 python -m dual_track_opd.fc_opd.teacher_service \
    --backend transformers \
    --model /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct \
    --port 18080 --top-k 32 --dtype bfloat16 --device cuda:0 &
# Wait for model load, verify:
sleep 15 && curl -s http://127.0.0.1:18080/health

# 3. Run one-step PPO with FC-OPD (on a different GPU)
ray stop -f
CUDA_VISIBLE_DEVICES=1 bash scripts/hpc/run_verl_fc_opd_smoke.sh
```

**Acceptance criteria:**
- `student_scorer_fqn` is configured and the real GPU scorer is instantiated
- `actor/fc_opd_loss > 0` (not collapsed to zero)
- `actor/fc_opd_coef` present in metrics
- Model parameters change after one step
- No crash

**Why Codex is needed:** You have the GPU nodes and understand verl's
launch infrastructure (Ray cluster, FSDP config).  The smoke test script
(`scripts/hpc/run_fc_opd_online_train_step_smoke.py`) can serve as a
single-GPU reference, but the real smoke needs the full verl stack.

#### Task 2: Remove-Padding + Ulysses SP Path

The actor patch (`fc_opd_fsdp_actor_aux_kd.patch`) currently raises
`RuntimeError` when `use_remove_padding=True` AND Ulysses sequence parallel
is enabled:

```python
# In the patch, around dp_actor.py line ~450:
if use_remove_padding and ulysses_sp_size > 1:
    raise RuntimeError(
        "FC-OPD sparse-KD with remove-padding + Ulysses SP is not implemented yet"
    )
```

**What to do:**
- Understand how remove-padding restores the full `[B, T, vocab]` logits
  from packed sequences.
- Verify `fc_teacher_topk_indices` indices are correctly aligned with
  the un-padded positions.
- Remove the guard and test.

**Acceptance criteria:**
- Training runs with `use_remove_padding=True` and Ulysses SP without
  crashing.
- `actor/fc_opd_loss` values match the non-remove-padding path within 1e-6
  for the same batch.

### P1 — Important For Usability

#### Task 3: Hydra Config for FC-OPD Training

Create `recipe/fc_opd/config/fc_opd_trainer.yaml` in the verl submodule
(or as a referenced config from the project root).  Minimum contents:

```yaml
# @package _global_
algorithm:
  fc_opd:
    post_rollout_hook: dual_track_opd.fc_opd.verl_post_rollout_hook.fc_opd_post_rollout_hook
    student_scorer_fqn: dual_track_opd.fc_opd.student_scorer.StudentScorer
    student_scorer_kwargs:
      model_path: ${actor_rollout_ref.model.path}
      device: cuda
      dtype: bfloat16
      top_k: 32
    teacher_url: http://127.0.0.1:18080
    conditions: [full, degraded, free, task_visible, task_infer, task_solve]
    verifier_fqn: dual_track_opd.fc_opd.verifier.verify_geometry3k_response
    fc_opd_coef: 0.1

actor_rollout_ref:
  actor:
    use_fused_kernels: false  # REQUIRED — FC-OPD needs live logits
    fsdp_config:
      param_offload: false
```

**Acceptance criteria:**
- `python -m verl.trainer.main_ppo --config-name=fc_opd_trainer` starts
  without config errors.
- All `algorithm.fc_opd.*` fields are consumed without warnings.

#### Task 4: Batch-Aware Teacher Scoring in Hook

The post-rollout hook currently calls the teacher scorer **per sample**
inside `compute_online_fc_opd_batch`.  Use the new
`score_teacher_conditions_multi_sample()` to batch all samples into a
single HTTP call.

**What to change in `verl_post_rollout_hook.py`:**

Option A — replace the `_build_teacher_scorer` closure with a multi-sample
variant that collects all samples, calls
`score_teacher_conditions_multi_sample()`, and partitions results back.

Option B — modify `compute_online_fc_opd_batch` to accept a multi-sample
teacher scorer and process all teacher requests upfront.

Option A is simpler and doesn't change the `online_batch.py` interface.

**Acceptance criteria:**
- For a batch of 32 samples, exactly 1 HTTP POST to `/score` (verified
  via teacher service log).
- Per-sample teacher scores identical to the sequential path.

### P2 — Nice To Have

#### Task 5: Port Teacher Service to ZMQ + vLLM

Follow GKD's `recipe/gkd/megatron/teacher/` pattern:
- Replace HTTP with ZMQ for lower latency
- Use vLLM async engine for batched teacher forward passes
- Support dynamic batching of teacher requests across workers

#### Task 6: Verify Distributed Training (4×H200, 2 Steps)

Scale the smoke to multi-GPU distributed training and verify:
- FSDP sharding doesn't corrupt `fc_*` tensor fields
- Metrics are consistent across ranks
- Loss doesn't collapse to zero over multiple steps

#### Task 7: Condition Evidence Generator as Ray Actor

Currently evidence generation (`build_fc_opd_4c_evidence_cache.py`) runs
offline.  For truly online operation, port the generator to a Ray actor
so `free`/`task` evidence is generated on-the-fly during training.

## 3. Architecture Notes — Decisions Already Made

### 3.1  Patch Strategy (Not Fork, Not Recipe)

We patch verl in-place with 2 thin patches (203 lines total).  We do NOT
create a `recipe/fc_opd/` directory inside verl.  The rationale:

- Patches are explicit, versioned, re-applicable
- All research logic lives in `src/dual_track_opd/fc_opd/` — importable,
  testable, independent of verl
- A fork would require maintaining a parallel verl repo
- A recipe would couple FC-OPD to verl's internal module structure

The one exception: we DO need a **thin Hydra config** (Task 3 above) so
`main_ppo` can find the `algorithm.fc_opd.*` settings.  This config only
references project-side modules; it contains zero research logic.

### 3.2  Hook Placement

The post-rollout hook fires at exactly one point in `RayPPOTrainer.fit()`:
after rollout responses are unioned into `batch` and `response_mask` is
computed, but before advantage estimation.  This is the right place because:

- Rollout tokens exist (teacher/student can score them)
- `response_mask` exists (we know which tokens are model-generated)
- Advantages haven't been computed yet (we can still modify the batch)

The trainer patch is **26 lines** because this insertion point is stable
across verl versions.

### 3.3  Student Scorer Strategy

The hook needs a `student_scorer_fqn` that resolves to a callable matching
`StudentForcedScorer`.  Two deployment modes:

| Mode | Memory | Complexity | When to Use |
|------|--------|------------|-------------|
| Standalone (`StudentScorer`) | 2× student model | Low | Smoke tests, single-GPU, debugging |
| Actor-reuse (custom FQN) | 1× student model | High | Production training — reuses actor's live model |

The standalone `StudentScorer` shipped this round is the reference
implementation.  For production, write a thin FQN wrapper that calls the
actor worker's forward pass directly, avoiding a second model load.

### 3.4  Why Teacher Uses HTTP

The teacher is intentionally a separate process (or container) accessed
via HTTP:

- Teacher (32B) may not fit on the same GPU as the actor (4B/8B)
- HTTP lets us scale the teacher independently
- The protocol is simple and debuggable
- ZMQ/vLLM upgrade path is clear (Task 5)

## 4. Verification Checklist

Before declaring FC-OPD "ready for training":

- [ ] Patches apply cleanly to verl `bec9ef74` (verified)
- [ ] `pytest -q` passes all tests (195 passed, 2 skipped)
- [ ] Teacher service starts and responds to `/health`
- [ ] Single-GPU smoke (`run_fc_opd_online_train_step_smoke.py`) succeeds
- [ ] **Task 1**: Full verl PPO step with `fc_opd_coef > 0` succeeds
- [ ] **Task 1**: `actor/fc_opd_loss > 0`
- [ ] **Task 1**: Model parameters change after one step
- [ ] **Task 2**: Remove-padding + Ulysses SP path works
- [ ] **Task 3**: Hydra config exists and is loadable
- [ ] **Task 4**: Batched multi-sample teacher scoring in hook

## 5. File Inventory — Key Files For Codex To Touch

| File | Action | Priority |
|------|--------|----------|
| `third_party/verl/verl/trainer/ppo/ray_trainer.py` | Apply trainer patch (hook insertion) | P0 |
| `third_party/verl/verl/workers/actor/dp_actor.py` | Apply actor patch (FC-OPD loss + remove-padding guard) | P0 |
| `recipe/fc_opd/config/fc_opd_trainer.yaml` (new, inside verl) | Create Hydra config | P1 |
| `src/dual_track_opd/fc_opd/verl_post_rollout_hook.py` | Option A: add multi-sample teacher scoring | P1 |
| `patches/verl/fc_opd_fsdp_actor_aux_kd.patch` | Remove Ulysses guard (Task 2) | P0 |

## 6. Quick Reference — Running Pieces

### Start Teacher Service

```bash
python -m dual_track_opd.fc_opd.teacher_service \
    --backend transformers --model Qwen/Qwen3-VL-32B-Instruct \
    --port 18080 --top-k 32 --dtype bfloat16 --device cuda
```

### Build Evidence Cache

```bash
DATASET=/path/to/Geometry3K DTOPD_OUTPUT_ROOT=/path/to/outputs LIMIT=8 \
bash scripts/hpc/build_fc_opd_4c_evidence_cache.sh
```

### Run Single-GPU Smoke

```bash
python scripts/hpc/run_fc_opd_online_train_step_smoke.py \
    --evidence-jsonl /path/to/evidence_cache.jsonl \
    --student-model Qwen/Qwen3-VL-4B-Instruct \
    --teacher-url http://127.0.0.1:18080 \
    --device cuda --dtype bfloat16
```

### Run Tests

```bash
pytest -q
```
