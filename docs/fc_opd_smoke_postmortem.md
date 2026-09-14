# FC-OPD Smoke Post-Mortem & Next Steps

Written 2026-06-29 after the first successful end-to-end GPU smoke.

## 1. Smoke Result

One verl PPO step with FC-OPD completed on 2×H200 (GPU 1=verl, GPU 2=StudentScorer).

```
fc_opd/hook_loss:           0.295
fc_opd/hook_active_weight: 57.7
actor/fc_opd_loss:          0.536
actor/fc_opd_coef:          0.1
actor/grad_norm:            8.78
actor/pg_loss:             -0.999
```

**All six conditions scored non-zero teacher logits.** The router assigned
non-zero weights to `full`, `task_visible`, `task_infer`, and `task_solve`.
`degraded` and `free` had zero selection — see §4.

## 2. What We Fixed (12 issues → 0)

### A. Missing verl config (4 issues)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| KV cache OOM | Default `max_model_len=262144`, gpu_mem=0.3 | `max_model_len=2048` |
| Chunk assert | Default `agent.num_workers=8`, batch=2 | `agent.num_workers=2` |
| Reward `NotImplementedError` | `data_source=geometry3k` not in verl | Custom reward function via `file://` |
| StudentScorer needs GPU | `ray start --num-gpus=1` | `--num-gpus=2` |

**Lesson:** CLI overrides are fragile. A Hydra YAML (like GKD's
`on_policy_distill_trainer.yaml`) would catch these at config-validation time.

### B. Data pipeline — verl drops non-standard columns (4 issues)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| `condition_inputs` not in batch | verl AgentLoop only preserves `extra_info` | Store in `extra_info`; read from `extra_info` in hook |
| `question` becomes full prompt text | Prompt-tensor fallback decode leaked `<\|image_pad\|>` | Read question from `extra_info` first |
| `choices` / `answer` missing | Same verl filtering | Read from `extra_info` |
| Verifier returns `malformed=true` | No choices/answer → can't match | Fixed by reading from `extra_info` |

**Lesson:** verl's RLHFDataset → AgentLoop → rollout pipeline preserves only
a fixed set of non-tensor columns. All FC-OPD-specific data MUST go through
`extra_info`. A custom `RLHFDataset` subclass could formalize this.

### C. Architecture — TaskRunner is CPU-only (3 issues)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| StudentScorer `No CUDA` | `task_runner_class = ray.remote(num_cpus=1)` | Wrap StudentScorer in a Ray GPU actor |
| CUDA tensor serialization | GPU actor returns tensors on CUDA | `.detach().cpu()` inside actor before return |
| `condition_ids` shape `[C]` vs batch `[B]` | TensorDict requires matching batch dim | Store in `non_tensor_batch` as np.array |

**Lesson:** The standalone StudentScorer Ray actor loads a duplicate 4B model
and requires CUDA↔CPU tensor marshalling. The GKD pattern — scoring inside
the actor worker that already has the model — eliminates all of this.

### D. verl actor patch gap (1 issue)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| `fc_opd_coef=0.0` | Actor config doesn't see `algorithm.fc_opd.loss_coef` | Hook writes `fc_opd_coef` as `[B]` tensor into batch; actor reads from `micro_batch.batch` |

## 3. Architecture Assessment

### What works well (keep as-is)

- **Teacher HTTP service** — 32B model on separate GPU, correct top-k scoring,
  exact token-alignment check. Protocol is clean and debuggable.
- **Core FC-OPD math** — `conditions.py`, `loss.py`, `router.py`,
  `signal_decomposer.py`, `verl_sparse_kd.py`. All unit-tested (195 passed).
- **Online batch assembly** — `online_batch.py` is framework-independent.
- **Patch strategy** — 2 thin verl patches (~203 lines total). Clean, versioned,
  re-applicable.

### What needs refactoring (align with GKD)

| Current | Problem | Target (GKD pattern) |
|---------|---------|---------------------|
| Standalone StudentScorer via Ray actor | Duplicate 4B model, CUDA↔CPU serialization | Score inside actor worker (reuse actor model) |
| 50+ CLI overrides | Easy to miss parameters | Single Hydra YAML config |
| `extra_info` as ad-hoc channel | Implicit, fragile | Custom `RLHFDataset` subclass with explicit FC-OPD fields |
| Per-sample teacher HTTP calls | 6 HTTP calls per sample | `score_teacher_conditions_multi_sample` (already implemented, not wired) |
| CPU fallback removed (fail-fast) | Correct for validation | Keep fail-fast; actor-reuse scorer eliminates need for fallback |

## 4. Known Gaps

### 4.1 Zero selection for `degraded` and `free`

The default router (`mode=chunk`) maps XML response chunks to conditions:
`visual_evidence → TASK_VISIBLE`, `reasoning → FULL`, `answer → FULL`.
`degraded` and `free` are never assigned.

**Options:**
- Switch router to `mode=uniform_all_conditions` for debugging
- Implement `mode=student_deficit_chunk_gated` (uses capability scores)
- Accept that some conditions get zero weight when response format matches

### 4.2 Remove-padding + Ulysses SP

Actor patch raises `RuntimeError` when `use_remove_padding=True` and
`ulysses_sp_size > 1`. This guard exists because the current implementation
assumes contiguous token positions.

### 4.3 Image transform not applied

`degraded_image.transform` metadata (`{type: lowres_nearest, scale: 0.1}`)
is recorded but the teacher service opens the original image file directly.
A degraded-image cache or on-the-fly transform would be needed for
meaningful `degraded` condition scores.

### 4.4 Multi-sample teacher batching

`score_teacher_conditions_multi_sample` exists in `teacher_client.py` but
the hook still calls `score_teacher_conditions` per sample.

## 5. Priority Roadmap

### P0 — Production-readiness blockers

1. **Move student scoring into actor worker.**
   - Remove `ray_student_scorer.py` and standalone `StudentScorer` Ray actor.
   - In the actor patch (`dp_actor.py`), after computing student logits for
     policy loss, compute the per-condition forced log-probs by re-running
     the forward pass with each condition's prompt.
   - This eliminates the duplicate model load, the CUDA serialization hack,
     and the extra GPU requirement.
   - Reference: GKD's `_async_get_teacher_knowledge` pattern.

2. **Hydra YAML config.**
   - Create `configs/experiment/verl_fc_opd.yaml` with all FC-OPD settings.
   - Port all CLI overrides from the smoke script into the YAML.
   - Verify `python -m verl.trainer.main_ppo --config-name=verl_fc_opd` works.

### P1 — Important for correctness

3. **Wire multi-sample teacher batching.**
   - Replace per-sample `score_teacher_conditions` with
     `score_teacher_conditions_multi_sample` in the hook.
   - Verify: one HTTP POST per hook invocation regardless of batch size.

4. **Remove-padding + Ulysses SP support.**
   - Understand verl's remove-padding token index mapping.
   - Verify `fc_teacher_topk_indices` alignment with un-padded positions.
   - Remove the guard and test with `use_remove_padding=True`.

5. **Degraded-image transform.**
   - Either pre-compute degraded images as artifacts (like the existing
     `artifacts/fc_opd/degraded_images/` directory pattern) or apply
     transforms on-the-fly in the teacher service.

### P2 — Nice to have

6. **Custom RLHFDataset subclass.**
   - Add explicit FC-OPD fields (`fc_condition_inputs`, `fc_question`,
     `fc_choices`, `fc_answer`) that are preserved through the pipeline.
   - Removes the implicit `extra_info` dependency.

7. **Distributed training verification.**
   - Scale smoke to 4×H200, 2+ steps.
   - Verify FSDP doesn't corrupt `fc_*` tensors.
   - Verify metrics consistency across ranks.

8. **Teacher service: ZMQ + vLLM backend.**
   - Follow GKD's `recipe/gkd/megatron/teacher/` pattern.
   - Replace HTTP with ZMQ for lower latency.
   - Use vLLM async engine for batched teacher forward passes.

## 6. Running the Smoke

```bash
# Prerequisites (all on NFS):
#   1. Teacher on GPU 0, port 18080
#   2. Parquet at fc-opd-storage/outputs/fc_opd/verl_smoke/train.parquet
#   3. Student model at models/Qwen3-VL-4B-Instruct

# Start teacher
CUDA_VISIBLE_DEVICES=0 python -m dual_track_opd.fc_opd.teacher_service \
    --backend transformers \
    --model /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct \
    --port 18080 --top-k 32 --dtype bfloat16 --device cuda:0 &

# Run smoke
ray stop -f && rm -rf /tmp/ray/*
CUDA_VISIBLE_DEVICES=1,2 bash scripts/hpc/run_verl_fc_opd_smoke.sh

# Expected output:
#   actor/fc_opd_loss > 0
#   fc_opd/hook_active_weight > 0
#   One step completes without crash
```

## 7. File Inventory

| File | Role | Status |
|------|------|--------|
| `src/dual_track_opd/fc_opd/conditions.py` | Condition definitions + inputs | ✅ |
| `src/dual_track_opd/fc_opd/loss.py` | FC-OPD loss math | ✅ |
| `src/dual_track_opd/fc_opd/router.py` | Condition weight routing | ✅ |
| `src/dual_track_opd/fc_opd/signal_decomposer.py` | Teacher top-K signal decomposition | ✅ |
| `src/dual_track_opd/fc_opd/verl_sparse_kd.py` | Sparse KD for actor | ✅ |
| `src/dual_track_opd/fc_opd/online_batch.py` | Online batch assembly | ✅ |
| `src/dual_track_opd/fc_opd/teacher_service.py` | Teacher HTTP service | ✅ |
| `src/dual_track_opd/fc_opd/teacher_client.py` | Teacher HTTP client | ✅ |
| `src/dual_track_opd/fc_opd/teacher_transformers.py` | Teacher HF backend | ✅ |
| `src/dual_track_opd/fc_opd/student_scorer.py` | Standalone StudentScorer | 🔄 Replace with actor-reuse |
| `src/dual_track_opd/fc_opd/ray_student_scorer.py` | Ray GPU actor wrapper | 🔄 Remove when actor-reuse done |
| `src/dual_track_opd/fc_opd/verl_post_rollout_hook.py` | Post-rollout hook | ✅ (needs multi-sample batching) |
| `src/dual_track_opd/fc_opd/verl_integration.py` | Verl tensor adapter | ✅ |
| `src/dual_track_opd/fc_opd/smoke_reward.py` | Dummy reward for smoke | ✅ (temporary) |
| `scripts/hpc/run_verl_fc_opd_smoke.sh` | Smoke launch script | ✅ |
| `third_party/verl/verl/trainer/ppo/ray_trainer.py` | Trainer patch (hook insertion) | ✅ |
| `third_party/verl/verl/workers/actor/dp_actor.py` | Actor patch (FC-OPD loss) | ✅ |
