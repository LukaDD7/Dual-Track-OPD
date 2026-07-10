from pathlib import Path


SMOKE = Path("scripts/hpc/run_gkd_text_smoke.sh")
VALIDATOR = Path("scripts/hpc/validate_gkd_smoke_config.py")
PREPARE = Path("scripts/setup/prepare_gkd_compatible_checkout.sh")


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
