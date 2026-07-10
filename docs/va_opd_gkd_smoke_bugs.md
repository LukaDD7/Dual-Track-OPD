# GKD Text Smoke Bug Report

Run date: 2026-07-10. Environment: `vaopd-gkd-cu128` (CUDA 12.8, torch 2.8, vLLM 0.11, Ray 2.56).
GPU node: 8×H200, NFS shared with CPU node. No internet on GPU node.

All bugs are version-mismatch between:
- **verl checkout**: `external/verl_gkd/verl` @ `bcb638649a50e58494a8ddd92085ad1174f674b8`
- **recipe submodule**: `external/verl_gkd/verl/recipe` @ `ba246418f4de12b845a09bba975f1a5242adc898`
- **GKD megatron recipe**: `recipe/gkd/megatron/`

The GKD recipe was designed for an older verl version; our verl checkout is newer with API changes.

---

## Bugs Fixed

### B1. proxy.py stdout buffering — `scripts/hpc/run_gkd_text_smoke.sh`
- **Symptom**: proxy log empty after 60s, readiness check times out
- **Fix**: `python -u proxy.py` (unbuffered output)
- **Commit**: `ea50018`

### B2. Raw TCP readiness check fails — `scripts/hpc/run_gkd_text_smoke.sh`
- **Symptom**: python `socket.connect()` to ZMQ port times out even though port is listening
- **Fix**: Use `ss -Hltn sport = :PORT` instead of Python TCP check
- **Commit**: `a42fdb6`

### B3. Ray dashboard not ready — `scripts/hpc/run_gkd_text_smoke.sh`
- **Symptom**: `ray job submit` fails because dashboard on port 8265 never starts
- **Fix**: Bypass ray job submit entirely; run `python -m` directly (FC-OPD pattern). Export env vars from `runtime_env.yaml` manually.
- **Commit**: `d13eba0`

### B4. `_STEP_COUNT` grep bug — `scripts/hpc/run_gkd_text_smoke.sh`
- **Symptom**: `grep -c ... || echo 0` captures both "0\n" (grep output) and "0" (echo output) → `_STEP_COUNT="0\n0"` → bash integer comparison fails
- **Fix**: Move `|| _STEP_COUNT=0` outside `$()`
- **Commit**: `d7b98cc`

### B5. `ModuleNotFoundError: No module named 'recipe.gkd.ray_trainer'`
- **Symptom**: `from recipe.gkd.ray_trainer import OnPolicyDistillTrainer` fails
- **Root cause**: Recipe was refactored (`ad471ea`): files moved from `recipe/gkd/` to `recipe/gkd/megatron/` but import paths in `main_gkd.py` were never updated. `ray_trainer.py` is at `recipe/gkd/megatron/ray_trainer.py` but import expects `recipe/gkd/ray_trainer.py`.
- **Fix**: Create symlinks in `recipe/gkd/`: `ray_trainer.py → megatron/ray_trainer.py`, `teacher_utils.py → megatron/teacher_utils.py`, `teacher → megatron/teacher`. Also set `PYTHONPATH` to include `external/verl_gkd/verl/`.
- **Commit**: `85e976a` (smoke script) + `cac2233` (recipe symlinks)
- **Also needed**: Install 13 missing verl runtime deps (`codetiming`, `torchdata`, `datasets`, etc.)

### B6. `ImportError: cannot import name 'Profiler' from 'verl.utils.profiler.profile'`
- **Symptom**: GKD recipe imports `Profiler` but this verl version renamed it to `DistProfiler`
- **Fix**: Add no-op `Profiler` class to `verl/utils/profiler/profile.py` (all usages in recipe are commented out)
- **Commit**: `ce0adf93` (verl)

### B7. GPU resource pool mismatch (2 available vs 4 desired)
- **Symptom**: `ValueError: Total available GPUs 2.0 is less than total desired GPUs 4`
- **Root cause**: GKD creates separate Ray resource pools for actor (Megatron) and rollout (vLLM). With 2 train GPUs, `trainer.n_gpus_per_node=2` + `rollout.n_gpus_per_node=2` = 4 total desired.
- **Fix**: Split GPUs equally: `_POOL_GPUS = floor(n_gpus / 2)`, minimum 1. 2 GPUs → 1+1=2 total.
- **Commit**: `69bc324`

### B8. `TypeError: RolloutConfig.__init__() got an unexpected keyword argument 'update_weights_bucket_megabytes'`
- **Symptom**: GKD recipe accesses `self.config.rollout.update_weights_bucket_megabytes` directly, but verl has it nested under `checkpoint_engine`
- **Fix**: Add flat alias field to `RolloutConfig` in `verl/workers/config/rollout.py`
- **Commit**: `089b01aa` (verl)

### B9. `ValueError: Rollout mode 'sync' has been removed`
- **Symptom**: GKD config defaults to `mode: sync` but verl removed sync mode
- **Fix**: Override `actor_rollout_ref.rollout.mode=async` in smoke test
- **Commit**: `796860d`

### B10. `ConfigAttributeError: Key 'router_replay' is not in struct`
- **Symptom**: GKD config (plain dict) lacks `router_replay` field expected by base MegatronWorker
- **Fix**: Safe access with `.get()` for dict / `getattr()` for struct
- **Commit**: `1995ea28` (verl)

### B11. OmegaConf nested dict serialization (systemic)
- **Symptom**: `ConfigAttributeError` / `object_type=dict` on any attribute access → `config.rollout.n`, `config.actor.ppo_mini_batch_size`, etc.
- **Root cause**: `ray.get(runner.run.remote(config))` serializes OmegaConf DictConfig → nested plain dicts. All downstream attribute accesses fail.
- **Fixes**:
  - `main_gkd.py`: `config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))` after Ray deserialization — `f7903c3` (recipe)
  - `megatron_workers.py` (recipe): Same rebuild in worker `__init__` — `9b2dea2`, `4af521c` (recipe)
  - `megatron_workers.py` (verl): Safe access to `rollout.n` with `.get()` — `609f4f1b` (verl)

### B12. `AssertionError: When dataclass_type is not provided, config must contain _target_`
- **Symptom**: `omega_conf_to_dataclass(self.config.model)` fails because model config is plain dict without `_target_`
- **Fix**: Pass `dataclass_type=HFModelConfig` explicitly; same for `RolloutConfig`
- **Commit**: `928e862d` (verl)

### B13. `AssertionError` in `batch.pop()` — missing `input_ids`, `attention_mask`, `position_ids`
- **Symptom**: GKD recipe expects pre-tokenized `input_ids` but verl `RLHFDataset.__getitem__` only returns `raw_prompt` (AgentLoop pattern)
- **Fix**: Add tokenization bridge in `_async_gen_next_batch()`: tokenize `raw_prompt` with `apply_chat_template`, pad to max length, create `input_ids`/`attention_mask`/`position_ids`
- **Commit**: `916c319`, `cf79377` (recipe)
- **Note**: `apply_chat_template` with list treats it as single conversation; must tokenize individually then pad.

### B14. `AttributeError: 'ServerAdapter' object has no attribute 'inference_engine'`
- **Symptom**: Rollout worker accesses `self.rollout.inference_engine.llm_engine...` but vLLM now uses `ServerAdapter` (client-server pattern) without direct engine access
- **Fix**: Detect `ServerAdapter` via `hasattr(self.rollout, "update_weights")`; use `self.rollout.update_weights(tensors)` via async loop instead of `inference_model.load_weights(tensors)`
- **Commit**: `6fc8011`, `a40c0e9` (recipe)

---

## Current Bug (FIX PREPARED; GPU VALIDATION PENDING)

### B15. `RuntimeError: There is no current event loop in thread 'MainThread'`

**Location**: `recipe/gkd/megatron/megatron_workers.py:850` — `sync_rollout_weights()`

**Trace**:
```
File ".../megatron_workers.py", line 850, in sync_rollout_weights
    loop = asyncio.get_event_loop_policy().get_event_loop()
RuntimeError: There is no current event loop in thread 'MainThread'.
```

**Root cause**: `sync_rollout_weights()` is called from a Ray worker in a non-async context (synchronous method via `register(blocking=False)` dispatch). There's no running event loop, and `get_event_loop()` in this context raises because no loop was set.

ServerAdapter's `update_weights()` is an async coroutine and needs an event loop. But the sync_rollout_weights method is synchronous (runs on Ray's main thread).

**Possible fixes**:
1. Create a new event loop: `loop = asyncio.new_event_loop()` + `loop.run_until_complete(...)` — but this may conflict with Ray's internal event loop
2. Run the weight update in a separate thread with its own event loop
3. Restructure the sync_rollout_weights to not call async ServerAdapter methods directly — perhaps the weight sync should happen via NCCL broadcast + IPC (as ServerAdapter expects) rather than calling update_weights from the rollout worker
4. Check how the base verl code handles weight sync with ServerAdapter — there may be a different sync mechanism (`sync_rollout_weights` may not be needed with ServerAdapter at all)

**Implemented compatibility fix**: `scripts/hpc/patch_gkd_b15_event_loop.py`
strictly replaces the implicit-loop lookup with a scoped Python 3.12
`asyncio.Runner`. The smoke launcher applies it idempotently before any GPU
process starts and refuses to modify an unrecognized backend revision. This is
the narrowest change consistent with the existing B14 bridge; the real GPU
smoke must still verify Ray ObjectRef and ZMQ behavior under the scoped loop.

**Reproducibility warning**: B6--B14 currently exist only as commits in the two
server-side detached checkouts. Their commit objects/diffs are not present in
this repository, so a fresh setup cannot reproduce the state required by this
B15 patch. Export both backend commit series with `git format-patch` (or commit
their combined diffs under `patches/verl/`) before treating Gate 2 as
reproducible.

**Key files involved**:
- `external/verl_gkd/verl/recipe/gkd/megatron/megatron_workers.py` (lines ~782-850) — GKD rollout worker sync_rollout_weights
- `external/verl_gkd/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py` (line 155) — ServerAdapter.update_weights
- `external/verl_gkd/verl/recipe/gkd/ray_trainer.py` (line 329) — calls sync_rollout_weights

---

## Repositories Modified

1. **Our repo** (`Dual-Track-OPD`, branch `codex/va-opd`): `scripts/hpc/run_gkd_text_smoke.sh`
2. **verl checkout** (`external/verl_gkd/verl/`, detached HEAD): `verl/workers/megatron_workers.py`, `verl/workers/config/rollout.py`, `verl/utils/profiler/profile.py`
3. **recipe submodule** (`external/verl_gkd/verl/recipe/`, detached HEAD): `gkd/megatron/main_gkd.py`, `gkd/megatron/megatron_workers.py`, `gkd/megatron/ray_trainer.py`, symlinks in `gkd/`

---

## How to Reproduce

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
bash scripts/hpc/run_gkd_text_smoke.sh --teacher-gpu 1 --train-gpus 2,3 --steps 10
```

GPU node already has all fixes applied (NFS shared). No git pull needed.

---

## Run Logs

Latest run: `runs/gkd_smoke/gkd_smoke_20260710_034930/`
