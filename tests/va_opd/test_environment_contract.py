from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def pinned_constraints() -> dict[str, str]:
    path = REPO_ROOT / "configs/environment/verl_va_opd_e003_cu128.constraints.txt"
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        pins[name] = version
    return pins


def test_native_va_opd_matrix_matches_backend_and_vllm_metadata() -> None:
    pins = pinned_constraints()

    assert pins["torch"] == "2.9.0"
    assert pins["torchvision"] == "0.24.0"
    assert pins["torchaudio"] == "2.9.0"
    assert pins["vllm"] == "0.12.0"
    assert pins["transformers"] == "4.57.3"
    assert pins["flashinfer-python"] == "0.5.3"
    assert pins["flash-attn"] == "2.8.3"
    assert pins["numpy"] == "2.2.6"

    # vLLM 0.12.0 publishes transformers>=4.56,<5, while the exact verl
    # backend publishes vllm>=0.8.5,<=0.12.0 and numpy>=2.0.0.
    transformer_major, transformer_minor, _ = map(int, pins["transformers"].split("."))
    assert transformer_major == 4
    assert transformer_minor >= 56
    assert tuple(map(int, pins["vllm"].split("."))) <= (0, 12, 0)
    assert int(pins["numpy"].split(".", 1)[0]) >= 2


def test_setup_builds_pinned_vllm_with_managed_cu128_toolchain() -> None:
    setup = (REPO_ROOT / "scripts/hpc/setup_va_opd_native_env.sh").read_text(encoding="utf-8")

    assert 'VLLM_COMMIT="4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e"' in setup
    assert 'PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"' in setup
    assert 'export VLLM_VERSION_OVERRIDE="0.12.0"' in setup
    assert 'cuda-toolkit=12.8 gcc_linux-64=12 gxx_linux-64=12' in setup
    assert 'export TORCH_CUDA_ARCH_LIST="9.0"' in setup
    assert '"${PYTHON}" use_existing_torch.py' in setup
    assert 'pip wheel --no-build-isolation --no-deps' in setup
    assert 'pip install --dry-run --constraint' in setup
    assert '"${PYTHON}" -m pip check' in setup
    assert "cpu-source-build-cu128-h200-sm90" in setup
    assert "pip install --constraint \"${CONSTRAINTS}\" vllm==" not in setup
    assert "VA_OPD_SKIP_FLASH_ATTN_BUILD" not in setup


def test_all_native_entrypoints_use_canonical_conda_env_root() -> None:
    entrypoints = (
        REPO_ROOT / "scripts/hpc/setup_va_opd_native_env.sh",
        REPO_ROOT / "scripts/hpc/run_va_opd_native.sh",
        REPO_ROOT / "scripts/hpc/collect_va_opd_gpu_facts.sh",
    )
    for path in entrypoints:
        text = path.read_text(encoding="utf-8")
        assert "DTOPD_CONDA_ENVS_ROOT" in text, path
        assert "va-opd-native-e003-cu128-r595-v1" in text, path
        assert "fc-opd-storage/envs/va-opd" not in text, path


def test_preflight_requires_source_build_provenance() -> None:
    preflight = (REPO_ROOT / "scripts/hpc/preflight_va_opd_native.py").read_text(encoding="utf-8")

    assert '("2.9.0", "12.8", "0.12.0+cu128", "4.57.3")' in preflight
    assert '"flash-attn": "2.8.3"' in preflight
    assert "va_opd_environment_manifest.json" in preflight
    assert "EXPECTED_VLLM_SOURCE_COMMIT" in preflight


def test_cu132_setup_fails_closed_and_registry_marks_it_quarantined() -> None:
    setup = (REPO_ROOT / "scripts/hpc/setup_va_opd_native_env_cu132.sh").read_text(encoding="utf-8")
    registry = (REPO_ROOT / "docs/environment_registry.md").read_text(encoding="utf-8")

    assert "must not be rebuilt" in setup
    assert "exit 2" in setup
    assert "va-opd-verl-e003-cu132" in registry
    assert "`quarantined`" in registry
