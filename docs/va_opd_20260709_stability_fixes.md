# VA-OPD Stability Fixes — 2026-07-09

## Summary

Three root causes of VA-OPD training failure were identified and addressed:

1. **NCCL deadlock** — FSDP2 allgather on non-power-of-2 training ranks
2. **Mode collapse** — Pure reverse KL (mode-seeking) → entropy explosion / deterministic collapse
3. **CUDA OOM** — Peak memory exceeded 141 GB H200 limit at 4-rank (less FSDP sharding)

## 1. NCCL Deadlock: GPU Topology Fix

### Root Cause

NCCL 2.27.3 + CUDA 12.8 + 3/5/6-rank FSDP2 topology triggers probabilistic
`_ALLGATHER_BASE` deadlock (NumelOut=502065195, Timeout=1800s).
All runs with 3/5/6 training ranks crashed between steps 2–30.
4-rank power-of-2 never triggered deadlock.

### Fix: `--teacher-gpus` / `--train-gpus` (commit `a0d501c`)

Decoupled teacher and training GPU allocation:
- **Default**: 1 teacher (GPU 0) + 4 training ranks (GPUs 1,2,3,4)
- **Power-of-2 guard**: Fatal on non-power-of-2 unless `--allow-nonpower2`
- **Legacy `--gpus N`** still works with deprecation warning

```bash
# Recommended command
bash scripts/hpc/run_verl_fc_opd_overnight.sh \
    --teacher-gpus 0 --train-gpus 1,2,3,4 \
    --name t1_train4 --background
```

### Also added: `--test-fix replicate` (commit `b53d637`)

`fsdp_size=1` — eliminates FSDP parameter sharding entirely, converting to
near-replicated data-parallel.  vLLM GPU util reduced to 0.25 to compensate.
Use to verify whether a hang is NCCL allgather-related vs memory-related.

```bash
bash scripts/hpc/run_verl_fc_opd_overnight.sh \
    --teacher-gpus 0 --train-gpus 1,2,3,4 \
    --test-fix replicate --name t1_train4_repl --background
```

## 2. Mode Collapse: Reverse KL → JSD

### Root Cause

Pure reverse KL `KL(P_S || P_T)` is mode-seeking: the student only needs to
match one teacher mode.  Without a reward anchor (no GRPO), this drives the
student to either:
- **Deterministic collapse**: entropy → 0.003, all outputs identical (seen with `<answer>` prompt)
- **Random collapse**: entropy → 2.47, outputs become uniform (seen with `\boxed{}` prompt)

Both paths give score=0 and fc_opd_loss→0 (the loss "succeeds" at matching
the teacher distribution, but the student produces garbage).

### Fix: JSD (commit — uncommitted, in working tree)

Replaced reverse KL with Jensen-Shannon Divergence:

```
JSD(P_T, P_S) = β·KL(P_T || M) + (1-β)·KL(P_S || M)
M = β·P_T + (1-β)·P_S
```

JSD is symmetric, bounded in [0, ln 2 ≈ 0.693], and avoids both
mode-seeking collapse and mode-covering noise.  Default β=0.5.

**Results** (8-step comparison, same config):

| Metric | Reverse KL | JSD |
|--------|-----------|-----|
| fc_opd_loss | 10.6 → 0.13 (collapsed) | 0.56–0.61 (stable) |
| Entropy | 0.54 → 2.47 (exploded) | 0.37–0.53 (healthy) |
| Score | 0 → 0 (dead) | 0.05 → 0.58 (learning) |
| Grad norm | 3.9 → 0.3 | 0.04–0.13 (stable) |

### Files changed

- `src/dual_track_opd/fc_opd/verl_sparse_kd.py` — added `compute_verl_sparse_jsd()`
- `src/dual_track_opd/fc_opd/va_opd_loss.py` — added `loss_type` / `jsd_beta` params
- `src/dual_track_opd/fc_opd/verl_actor_loss.py` — added `va_opd_jsd` mode dispatch
- `src/dual_track_opd/fc_opd/verl_post_rollout_hook.py` — JSD log message
- `scripts/hpc/run_verl_fc_opd_overnight.sh` — `--loss-mode` flag, default `va_opd_jsd`

## 3. CUDA OOM: Token Budget Reduction

### Root Cause

4-rank FSDP2 shards parameters across fewer GPUs than 5/6-rank → each GPU
holds more parameters.  With `ppo_max_token_len_per_gpu=10240`, peak memory
reached 102.7 GB, then backward pass requested 46 GB more → OOM at step 16.

### Fix (commits `a581c61`, `b501c36`)

Reduced token budget:
- `max_prompt_length`: 8192 → 6144
- `max_model_len`: 10240 → 8192
- `ppo_max_token_len_per_gpu`: 10240 → 8192

Constraints: `ppo_max_token_len_per_gpu >= max_model_len >= max_prompt + max_response`

**Results**:
- Before: peak 102.7 GB → OOM at step 16
- After: peak 65–95 GB → stable through step 27 (memory still creeping)

### Remaining risk

Memory still crept from 65 GB (step 2) to 95 GB (step 26) over time.
Step 27 hung at 95.5 GB — possibly another OOM or NCCL hang at higher
allgather sizes with larger sequences.

## Experiment Log

| Run | Layout | Loss | Token | Steps | Stop | Cause |
|-----|--------|------|-------|-------|------|-------|
| pow2_bs5 | 2T+5R (7GPU) | RKL | 10240 | 68 | Step ~15 | Mode collapse (entropy=2.47) |
| train_jsd_bs5 #1 | 2T+5R | JSD | 10240 | 10 | Hang | NCCL deadlock (500M allgather) |
| train_jsd_bs5 #2 | 2T+5R | JSD | 10240 | 10 | Hang | NCCL deadlock |
| t1_train4 #1 | 1T+4R | JSD | 10240 | 16 | OOM | 102.7 GB + 46 GB alloc |
| t1_train4 #2 | 1T+4R | JSD | 8192 | 0 | Assertion | max_token < max_model |
| t1_train4 #3 | 1T+4R | JSD | 8192 | 27 | Hang | 95.5 GB, silent after |

## Next Steps

1. **Test `--test-fix replicate`** to determine if step-27 hang is NCCL or memory
2. **Reduce token budget further** (7168) if memory creep confirmed
3. **Commit JSD changes** once full-run validated
4. **Consider verl GKD recipe** — Megatron-only, but has forward KL + schedulers

## verl On-Policy Distillation Reference

The `third_party/verl/recipe/gkd/` directory contains a maintained
on-policy knowledge distillation recipe with 4 loss variants (KL, RKL,
KL+RKL, JSD) and async schedulers.  Currently Megatron-only.
Key files:
- `megatron_distill_losses.py` — TP-aware KL/RKL/JSD implementations
- `docs/advance/async-on-policy-distill.md` — full architecture doc
