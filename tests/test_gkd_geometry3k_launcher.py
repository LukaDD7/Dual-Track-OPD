from pathlib import Path


LAUNCHER = Path("scripts/hpc/run_gkd_geometry3k_qwen3vl.sh")
PREPARE = Path("scripts/setup/prepare_fc_opd_verl.sh")
ACTOR_PATCH = Path("patches/verl/fc_opd_fsdp_actor_aux_kd.patch")


def test_launcher_is_online_gkd_by_default_and_has_va_opd_switch():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'OBJECTIVE="gkd"' in source
    assert 'gkd) CONDITIONS="[full]"' in source
    assert 'va_opd|va_opd_jsd) CONDITIONS="[full,degraded]"' in source
    assert "fc_opd_post_rollout_hook" in source
    assert "+algorithm.fc_opd.loss_mode=${OBJECTIVE}" in source
    assert "trainer.total_training_steps=${STEPS}" in source
    assert 'grep -c "training/global_step:"' in source
    assert "preflight_gkd_geometry3k.py" in source


def test_launcher_uses_real_qwen3vl_geometry3k_defaults_and_teacher_warmup():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "Qwen3-VL-4B-Instruct" in source
    assert "Qwen3-VL-32B-Instruct" in source
    assert "train-00000-of-00001.parquet" in source
    assert "warmup_gkd_geometry3k_teacher.py" in source
    assert source.index("Teacher end-to-end image warmup") < source.index("Starting Ray")
    assert "teacher_alignment_diagnostic.json" in source


def test_backend_overlay_treats_gkd_and_jsd_as_pure_distillation():
    patch = ACTOR_PATCH.read_text(encoding="utf-8")
    assert '"gkd", "gkd_forward", "vgg_opd", "va_opd", "va_opd_jsd"' in patch
    prepare = PREPARE.read_text(encoding="utf-8")
    assert "patch_verl_pure_distill_modes.py" in prepare
    assert "--recount" in prepare
    assert "patch_verl_fc_opd_global_norm.py" in prepare
    assert "fc_opd_global_normalizer" in patch


def test_launcher_defaults_to_official_topk_and_exposes_eos_diagnostic():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "TOP_K=256" in source
    assert "IGNORE_EOS=false" in source
    assert "--diagnostic-ignore-eos" in source
    assert "actor_rollout_ref.rollout.ignore_eos=${IGNORE_EOS}" in source
