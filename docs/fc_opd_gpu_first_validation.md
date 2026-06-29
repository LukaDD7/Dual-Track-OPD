# FC-OPD GPU-First Validation Plan

Written 2026-06-29.  This note corrects the validation direction after the
server-side Claude Code changes in commit `2885d8e`.

## Direction

The next milestone is not a CPU fallback smoke.  The goal is to quickly test
whether online FC-OPD is feasible with real GPU student logits inside the verl
training path.

CPU fallback is only useful for local interface tests.  It must not be treated
as evidence that FC-OPD training works because it does not validate:

- Qwen tokenizer/vocabulary alignment between teacher top-k ids and student
  logits
- real forced-scoring logits from the current student model
- GPU memory pressure from teacher, actor, rollout, and student scoring
- whether `actor/fc_opd_loss` reaches the actor patch with meaningful weights
- whether one optimizer step changes model parameters under FC-OPD

The verl hook now fails fast when `algorithm.fc_opd.student_scorer_fqn` is
missing.  The GPU smoke script configures
`dual_track_opd.fc_opd.student_scorer.StudentScorer` explicitly.

## Current State

- Core FC-OPD components exist under `src/dual_track_opd/fc_opd/`.
- Teacher HTTP service and client exist, with exact token-alignment checks.
- The post-rollout hook extracts current rollout tokens and builds verl tensors.
- `StudentScorer` can be loaded by FQN and performs standalone GPU forced
  scoring.
- `scripts/hpc/run_verl_fc_opd_smoke.sh` is the GPU-first one-step smoke entry.

Known limitations:

- The standalone `StudentScorer` loads a second copy of the student model.  This
  is acceptable for the first feasibility smoke on a large GPU instance, but it
  is not the final efficient design.
- The teacher multi-sample helper exists, but the verl hook still calls the
  teacher per sample.  This is a throughput issue, not the first feasibility
  blocker.
- A better final integration should reuse the actor worker's live model/logits
  instead of loading a duplicate student scorer.

## GPU Smoke Command

Run this from the repo root on the GPU instance after applying the verl patches.
Use separate visible GPUs for teacher and the training smoke when possible.

```bash
conda activate /inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/envs/fc-opd-verl071-cu128
unset CC

CUDA_VISIBLE_DEVICES=3 python -m dual_track_opd.fc_opd.teacher_service \
    --backend transformers \
    --model /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct \
    --port 18080 \
    --top-k 32 \
    --dtype bfloat16 \
    --device cuda:0 &

sleep 15
curl -s http://127.0.0.1:18080/health

ray stop -f
CUDA_VISIBLE_DEVICES=1 bash scripts/hpc/run_verl_fc_opd_smoke.sh
```

If the training smoke OOMs because the standalone scorer loads another student
copy, reduce batch size first.  If it still OOMs, the next engineering step is
to replace standalone `StudentScorer` with an actor-worker scorer that reuses
the live model.

## Acceptance Criteria

The smoke is useful only if all of these are true:

- the run uses `student_scorer_fqn=dual_track_opd.fc_opd.student_scorer.StudentScorer`
- the hook does not enter CPU fallback
- teacher token alignment passes
- `fc_opd/hook_loss` is finite and non-zero in hook metrics
- `actor/fc_opd_loss` appears in verl actor metrics
- `fc_condition_weights` has non-zero active mass
- one training step completes without crash

## What To Fix Next

1. Run the GPU-first smoke and capture the first failure.
2. If OOM: shrink smoke batch, then plan actor-live-logit scorer.
3. If token alignment fails: inspect response tokenization and teacher service
   request construction before changing loss code.
4. If hook tensors attach but actor loss is absent: inspect the verl actor patch
   config key path and tensor names.
5. After one-step success: wire batched teacher scoring into the hook to reduce
   HTTP overhead.
