from pathlib import Path


SMOKE = Path("scripts/hpc/run_gkd_text_smoke.sh")
VALIDATOR = Path("scripts/hpc/validate_gkd_smoke_config.py")
PREPARE = Path("scripts/setup/prepare_gkd_compatible_checkout.sh")
TEACHER_PATCH = Path("scripts/hpc/patch_gkd_teacher_memory.py")
TENSORDICT_PATCH = Path("scripts/hpc/patch_gkd_b23_locked_tensordict.py")


def test_smoke_uses_official_gkd_namespaces_and_explicit_upstream_defaults():
    source = SMOKE.read_text(encoding="utf-8")

    assert '"+actor_rollout_ref.rollout.n=1"' in source
    assert '"+actor_rollout_ref.actor.ppo_mini_batch_size=4"' in source
    assert '"actor_rollout_ref.actor.micro_batch_size=1"' in source
    assert '"actor_rollout_ref.teacher.server_ip=127.0.0.1"' in source
    assert '"actor_rollout_ref.rollout.load_format=dummy_megatron"' in source
    assert '"actor_rollout_ref.rollout.top_k=-1"' in source

    assert "+teacher.server_ip" not in source
    assert "+algorithm.use_kl_in_reward" not in source
    assert "actor_rollout_ref.actor.use_kl_loss" not in source


def test_same_override_array_drives_preflight_and_training():
    source = SMOKE.read_text(encoding="utf-8")
    assert source.count('"${GKD_OVERRIDES[@]}"') == 2
    assert "validate_gkd_smoke_config.py" in source

    validator = VALIDATOR.read_text(encoding="utf-8")
    assert '"actor_rollout_ref.rollout.n"' in validator
    assert '"actor_rollout_ref.actor.ppo_mini_batch_size"' in validator
    assert 'rollout_cls.__name__ != "vLLMRollout"' in validator
    assert '"raise NotImplementedError" in generate_source' in validator


def test_compatible_checkout_restores_sync_rollout_from_pre_retirement_commit():
    source = PREPARE.read_text(encoding="utf-8")

    assert 'GKD_SYNC_ROLLOUT_COMMIT="ab0705220a95952219111409d8f971872002c193"' in source
    assert "vllm_rollout/vllm_rollout_spmd.py" in source
    assert "verl/workers/rollout/base.py" in source
    assert "recipe/gkd/megatron_workers.py" in source


def test_smoke_exports_complete_cuda_jit_environment_before_ray():
    source = SMOKE.read_text(encoding="utf-8")

    assert 'CUDA_TOOLCHAIN="${CUDA_TOOLCHAIN:-' in source
    assert '"${CUDA_TOOLCHAIN}/bin/nvcc"' in source
    assert '"${CUDA_TOOLCHAIN}/lib"' in source
    assert '"${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib"' in source
    assert 'export LIBRARY_PATH="${CUDA_RUNTIME_LIB}' in source
    assert 'export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}' in source
    assert source.index('export CUDA_HOME="${CUDA_TOOLCHAIN}"') < source.index("=== Starting Ray")


def test_teacher_readiness_requires_engine_and_end_to_end_inference():
    source = SMOKE.read_text(encoding="utf-8")

    assert "patch_gkd_teacher_memory.py" in source
    assert "GKD_TEACHER_GPU_MEMORY_UTILIZATION" in source
    assert "grep -q '^worker started\\.\\.\\.'" in source
    assert "Teacher end-to-end inference warmup passed" in source
    assert source.index("Teacher end-to-end inference warmup passed") < source.index("=== Starting Ray")


def test_teacher_process_group_cleanup_includes_vllm_children():
    source = SMOKE.read_text(encoding="utf-8")

    assert "trap cleanup_teacher_processes EXIT" in source
    assert 'nohup setsid env CUDA_VISIBLE_DEVICES=""' in source
    assert 'nohup setsid env CUDA_VISIBLE_DEVICES="${TEACHER_GPU}"' in source
    assert 'kill -TERM -- "-${pid}"' in source
    assert 'kill -KILL -- "-${pid}"' in source
    assert "worker.log (last 100 lines)" in source


def test_smoke_rejects_skipped_teacher_batches_as_fake_progress():
    source = SMOKE.read_text(encoding="utf-8")

    assert "INFO: update actor done\\." in source
    assert "Teacher batch skips" in source
    assert "actor/kl_loss" in source
    assert "${EXIT_OK} && ${STEPS_OK} && ${TEACHER_OK} && ${LOSS_OK}" in source


def test_teacher_memory_patch_is_idempotent():
    import importlib.util

    spec = importlib.util.spec_from_file_location("patch_gkd_teacher_memory", TEACHER_PATCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    original = (
        "import argparse\n\n"
        "value = gpu_memory_utilization=0.7,\n"
        "outputs = self.llm.generate("
        "prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)\n"
    )
    patched, status = module.patch_source(original)
    assert status == "ok"
    assert "import os" in patched
    assert "GKD_TEACHER_GPU_MEMORY_UTILIZATION" in patched
    assert '[{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids]' in patched
    assert module.patch_source(patched) == (patched, "skip")


def test_teacher_patch_upgrades_generate_after_memory_patch_was_already_applied():
    import importlib.util

    spec = importlib.util.spec_from_file_location("patch_gkd_teacher_memory", TEACHER_PATCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    partially_patched = (
        "import argparse\nimport os\n\n"
        'value = gpu_memory_utilization=float(os.environ.get('
        '"GKD_TEACHER_GPU_MEMORY_UTILIZATION", "0.7")),\n'
        "outputs = self.llm.generate("
        "prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)\n"
    )
    patched, status = module.patch_source(partially_patched)
    assert status == "ok"
    assert module.OLD_GENERATE not in patched
    assert module.GENERATE_MARKER in patched


def test_locked_tensordict_patch_unlocks_batch_for_microbatch_construction():
    import importlib.util

    spec = importlib.util.spec_from_file_location("patch_gkd_b23_locked_tensordict", TENSORDICT_PATCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    original = (
        "        data.batch[\"attention_mask\"] = "
        "data.batch[\"attention_mask\"].to(bool)\n"
    )
    patched, status = module.patch_source(original)
    assert status == "ok"
    assert "        data.batch.unlock_()\n" in patched
    assert "with data.batch.unlock_():" not in patched
    assert patched.count("unlock_()") == 1
    assert module.patch_source(patched) == (patched, "skip")


def test_locked_tensordict_patch_upgrades_scoped_b23_fix():
    import importlib.util

    spec = importlib.util.spec_from_file_location("patch_gkd_b23_locked_tensordict", TENSORDICT_PATCH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    scoped = (
        "        with data.batch.unlock_():\n"
        "            data.batch[\"attention_mask\"] = "
        "data.batch[\"attention_mask\"].to(bool)\n"
    )
    patched, status = module.patch_source(scoped)
    assert status == "ok"
    assert "with data.batch.unlock_():" not in patched
    assert module.MARKER in patched


def test_smoke_applies_locked_tensordict_patch_before_ray():
    source = SMOKE.read_text(encoding="utf-8")

    assert "patch_gkd_b23_locked_tensordict.py" in source
    assert source.index("patch_gkd_b23_locked_tensordict.py") < source.index("=== Starting Ray")


def test_training_epochs_cover_requested_steps_instead_of_stopping_at_32():
    source = SMOKE.read_text(encoding="utf-8")

    assert "TRAIN_BATCH_SIZE=4" in source
    assert "pyarrow.parquet" in source
    assert "_STEPS_PER_EPOCH=$(( _TRAIN_ROW_COUNT / TRAIN_BATCH_SIZE ))" in source
    assert "_TOTAL_EPOCHS=$(( (NUM_STEPS + _STEPS_PER_EPOCH - 1) / _STEPS_PER_EPOCH ))" in source
    assert '"trainer.total_epochs=${_TOTAL_EPOCHS}"' in source
    assert '"trainer.total_epochs=1"' not in source
