# VA-OPD → GKD Migration Gates

Each gate must pass in order before proceeding. Do NOT skip gates.
If a gate fails, stop and report — do NOT attempt to patch around it without explicit approval.

---

## Gate 0: Environment Audit — No cu129/cu130, torch cuda=12.8

**Script**: `scripts/hpc/audit_nccl_stack.sh`

**Pass criteria**:
- `nvidia-smi` reports driver 570.x
- `nvcc --version` (if present) reports CUDA ≤ 12.8
- `python -c "import torch; print(torch.version.cuda)"` outputs `12.8`
- `pip freeze | grep -E 'torch|vllm|ray|flash|nccl|nvidia-cudnn|transformer-engine|megatron'` contains **zero** occurrences of `cu129` or `cu130`
- All wheel CUDA tags resolve to `cu128` or `cu12`

**Failure**: ANY cu129/cu130 string → fatal. Fix conda env before continuing.

---

## Gate 1: nccl-tests all_reduce_perf — 4 GPUs Continuous

**Script**: `scripts/hpc/audit_nccl_stack.sh` (nccl-tests section)

**Pass criteria**:
- `all_reduce_perf -b 8M -e 2G -f 2 -g 4` completes without timeout or hang
- Bus bandwidth within expected range for H200 NVLink
- No NCCL WARN/ERROR in output

**Failure**: Hang or timeout → NCCL environment is broken at the collective level. Do not proceed to training.

---

## Gate 2: GKD Text Smoke — 10 Steps

**Script**: `scripts/hpc/run_gkd_text_smoke.sh`

**Pass criteria**:
- Teacher server starts and responds to health check
- Ray cluster initializes with correct GPU count
- Training completes 10 steps without hang, crash, or OOM
- KL loss and grad_norm are finite (not NaN/Inf)
- No NCCL timeout or collective hang

**Failure**: Any hang, crash, or NaN loss → GKD environment is unstable. Debug before continuing.

---

## Gate 3: GKD Text Smoke — 200 Steps

**Script**: `scripts/hpc/run_gkd_text_smoke.sh` (with `--steps 200`)

**Pass criteria**:
- Training completes 200 steps without hang, crash, or OOM
- KL loss stable (not diverging to infinity)
- grad_norm within reasonable range (no explosion)
- Memory usage stable (no creeping OOM)
- At least one checkpoint saved successfully

**Failure**: Late-step hang (like legacy step 25-27) → NCCL collective issue persists in Megatron path. OOM → reduce batch size or enable recompute.

---

## Gate 4: Qwen3.5-0.8B Megatron Actor Load

**Script**: `scripts/hpc/run_gkd_qwen35_probe.sh`

**Pass criteria**:
- **Probe A** (Transformers load): `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")` succeeds
- **Probe B** (vLLM serve): vLLM can load and serve Qwen3.5-0.8B, single inference passes
- **Probe C** (Megatron actor): GKD actor model provider loads Qwen3.5-0.8B without `qwen3_5 architecture unsupported` or `ModelLayerSpec missing` errors

**Failure — Expected blocker**:
If Probe C fails with architecture-not-supported errors, this is documented as an **expected gap** in the GKD Megatron model provider. Do NOT attempt to patch Megatron model layers without explicit approval — this is a non-trivial engineering task (custom `ModelLayerSpec`, TP-aware attention, etc.).

---

## Gate 5: Qwen3.5-4B Student + Qwen3.5-27B Teacher — 50 Steps

**Script**: (to be created after Gate 4 passes)

**Pass criteria**:
- Student model: Qwen3.5-4B loaded via Megatron actor
- Teacher model: Qwen3.5-27B served via vLLM
- Training completes 50 steps without hang, crash, or OOM
- JSD loss (if available) or KL loss stable
- Score/reward metric shows learning signal (not flat zero)

---

## Gate 6 (Future): VA-OPD Loss Migration

**Prerequisite**: Gates 0–5 ALL passed.

Only after all gates 0–5 pass should VA-OPD loss migration begin:
1. Port `compute_va_opd_loss()` to GKD's Megatron loss interface
2. Add dual-condition (full/degraded) teacher query support
3. Add visual advantage weighting in the GKD trainer
4. Run 50-step VA-OPD smoke on geometry3k

---

## Summary Table

| Gate | What | Steps | Key Risk |
|------|------|-------|----------|
| 0 | Env audit | — | cu129/cu130 wheels |
| 1 | nccl-tests | — | NCCL collective hang |
| 2 | Text smoke | 10 | GKD env integration |
| 3 | Text smoke long | 200 | Late-step NCCL hang |
| 4 | Qwen3.5 probe | — | Megatron arch unsupported |
| 5 | 4B+27B train | 50 | Scale to real models |
| 6 | VA-OPD migrate | 50+ | Loss port correctness |
