from pathlib import Path


PROBE = Path("scripts/hpc/run_gkd_qwen35_probe.sh")


def test_gate4_uses_compatible_checkout_and_real_weight_loads():
    source = PROBE.read_text(encoding="utf-8")

    assert "external/verl_gkd_compatible/verl" in source
    assert "AutoModelForCausalLM.from_pretrained" in source
    assert "device_map='cpu'" in source
    assert "run_probe_c_legacy_introspection" not in source
    assert "run_gkd_text_smoke.sh" in source
    assert "--steps 1" in source
    assert "model_impl='transformers'" in source
    assert 'GKD_TEACHER_MODEL_IMPL="transformers"' in source
    assert 'GKD_ENV="${GKD_ENV}"' in source
    assert 'VERL_GKD_DIR="${VERL_GKD_DIR}"' in source


def test_gate4_has_explicit_nonoverlapping_gpu_selection():
    source = PROBE.read_text(encoding="utf-8")

    assert "--vllm-gpu" in source
    assert "--teacher-gpu" in source
    assert "--train-gpus" in source
    assert "overlaps train GPUs" in source
    assert "at least two train GPUs" in source
    assert 'setsid env CUDA_VISIBLE_DEVICES="${VLLM_GPU}"' in source
    assert 'kill -KILL -- "-${probe_b_pid}"' in source


def test_gate4_failures_return_nonzero():
    source = PROBE.read_text(encoding="utf-8")

    assert 'log "  Gate 4 requested stage(s): FAIL"' in source
    assert "Exit 0 even on expected blocker" not in source
    assert source.rstrip().endswith("exit 1")
