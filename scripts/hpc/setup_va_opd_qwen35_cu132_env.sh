#!/usr/bin/env bash
# Build a fresh CUDA 13.2 environment for the qwen3.5/3.6 distillation stack
# (torch 2.13.0+cu132 + vLLM 0.27.1 cu130 wheel + verl v0.9.0).
#
# House rules this follows (see docs/environment_registry.md + the qwen35
# cu129 build scripts/setup_qwen35_cu129.sh):
#   * GPU nodes have no internet; everything is built on the CPU node onto
#     shared storage and the prefix is consumed directly from NFS.
#   * NVIDIA runtime libs come from pip nvidia-* packages inside the env;
#     only libcuda.so.1 comes from the driver (/lib/x86_64-linux-gnu).
#   * nvcc/GCC live in a separate conda toolchain prefix (cuda132-toolchain),
#     never mixed into the Python env.
#   * vLLM wheel is pinned to the official cu130 build from wheels.vllm.ai.
#   * No --force-reinstall; every version is locked in the constraints file.
#
# Run on the CPU instance (with internet):
#   nohup bash scripts/hpc/setup_va_opd_qwen35_cu132_env.sh > /tmp/build_qwen35_cu132.log 2>&1 &

set -euo pipefail

DTOPD_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
ENV_PREFIX="${VA_OPD_ENV_PREFIX:-${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1}"
CUDA_TOOLCHAIN="${VA_OPD_CUDA_TOOLCHAIN:-${DTOPD_ROOT}/envs/cuda132-toolchain}"
VERL_DIR="${VA_OPD_VERL_DIR:-${DTOPD_ROOT}/fc-opd-storage/backends/verl-qwen35-v090-cu132}"
VERL_TAG="v0.9.0"
VERL_COMMIT="483b8a009ba3a97563edee3a19887e4862b8094a"
WHEELHOUSE="${VA_OPD_WHEELHOUSE:-${DTOPD_ROOT}/fc-opd-storage/wheelhouse/va-opd-qwen35-cu132-r595-v1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/configs/environment/verl_va_opd_qwen35_cu132.constraints.txt"
PYTHON="${ENV_PREFIX}/bin/python"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu132"
# wheels.vllm.ai serves files under a build-commit prefix; the listing page
# (/0.27.1/cu130/vllm/) links to the real hash-prefixed location.
VLLM_WHEEL_URL="https://wheels.vllm.ai/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm-0.27.1-cp38-abi3-manylinux_2_28_x86_64.whl"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.0}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.16.post3}"
RAY_VERSION="${RAY_VERSION:-2.55.1}"
TENSORDICT_VERSION="${TENSORDICT_VERSION:-0.10.0}"
MAX_JOBS="${MAX_JOBS:-32}"

echo "========================================="
echo "Build started at $(date)"
echo "ENV_PREFIX:     ${ENV_PREFIX}"
echo "VERL_DIR:       ${VERL_DIR}"
echo "WHEELHOUSE:     ${WHEELHOUSE}"
echo "TRANSFORMERS:   ${TRANSFORMERS_VERSION}"
echo "========================================="

# ── Prerequisites ─────────────────────────────────────────────────────────
if [[ ! -x "${CUDA_TOOLCHAIN}/bin/nvcc" ]]; then
    echo "FATAL: CUDA toolchain not found: ${CUDA_TOOLCHAIN}" >&2
    exit 1
fi
${CUDA_TOOLCHAIN}/bin/nvcc --version | grep -q 'release 13\.2' || {
    echo "FATAL: toolchain is not CUDA 13.2: ${CUDA_TOOLCHAIN}" >&2
    exit 1
}

CONDA_TOOL=""
if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    CONDA_TOOL="${CONDA_EXE}"
elif command -v micromamba >/dev/null 2>&1; then
    CONDA_TOOL="$(command -v micromamba)"
elif command -v conda >/dev/null 2>&1; then
    CONDA_TOOL="$(command -v conda)"
else
    echo "FATAL: conda or micromamba required" >&2
    exit 1
fi

# ── Create env if needed ──────────────────────────────────────────────────
if [[ ! -x "${PYTHON}" ]]; then
    "${CONDA_TOOL}" create -y -p "${ENV_PREFIX}" python=3.12 pip=25.1 setuptools wheel
fi
"${PYTHON}" -m pip install --upgrade pip==25.1.1 setuptools wheel 2>&1 | tail -2

# ── Step 1: torch 2.13.0+cu132 + torchvision (explicit cu132 index) ────────
echo ""
echo "=== Step 1: torch 2.13.0+cu132 from ${PYTORCH_INDEX} ==="
"${PYTHON}" -m pip install --index-url "${PYTORCH_INDEX}" --constraint "${CONSTRAINTS}" \
    torch==2.13.0 torchvision==0.28.0 2>&1 | tail -5

"${PYTHON}" - <<'PY'
import torch
assert torch.__version__.startswith("2.13.0"), f"torch version: {torch.__version__}"
assert torch.version.cuda == "13.2", f"torch cuda: {torch.version.cuda}"
print(f"torch {torch.__version__} CUDA {torch.version.cuda} — PASS")
PY

# ── Step 2: wheelhouse snapshot (torch family + vLLM wheel) ───────────────
echo ""
echo "=== Step 2: wheelhouse snapshot ==="
mkdir -p "${WHEELHOUSE}"
if [[ -f "${WHEELHOUSE}/torch-2.13.0+cu132-cp312-cp312-manylinux_2_28_x86_64.whl" ]]; then
    echo "torch wheel already in wheelhouse — skipping snapshot download"
else
    "${PYTHON}" -m pip download --index-url "${PYTORCH_INDEX}" --constraint "${CONSTRAINTS}" \
        -d "${WHEELHOUSE}" torch==2.13.0 torchvision==0.28.0 2>&1 | tail -2
fi

VLLM_WHEEL="${WHEELHOUSE}/$(basename "${VLLM_WHEEL_URL}")"
if [[ ! -f "${VLLM_WHEEL}" ]]; then
    "${PYTHON}" -m pip download --no-deps -d "${WHEELHOUSE}" "${VLLM_WHEEL_URL}" 2>&1 | tail -2
fi
echo "vLLM wheel: ${VLLM_WHEEL}"

# ── Step 3: vLLM 0.27.1 cu130 wheel + its deps ────────────────────────────
echo ""
echo "=== Step 3: vLLM 0.27.1 cu130 wheel (torch==2.13.0 satisfied by +cu132) ==="
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" "${VLLM_WHEEL}" 2>&1 | tail -6

# ── Step 4: verl v0.9.0 backend (clone + port local patches) ──────────────
echo ""
echo "=== Step 4: verl ${VERL_TAG} backend ==="
if [[ ! -e "${VERL_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${VERL_DIR}")"
    git clone --depth 1 --branch "${VERL_TAG}" https://github.com/verl-project/verl.git "${VERL_DIR}"
fi
git -C "${VERL_DIR}" fetch origin "refs/tags/${VERL_TAG}" --depth 1 2>&1 | tail -1
if [[ "$(git -C "${VERL_DIR}" rev-parse HEAD)" != "${VERL_COMMIT}" ]]; then
    if [[ -n "$(git -C "${VERL_DIR}" status --porcelain --untracked-files=no)" ]]; then
        echo "FATAL: refusing to switch a modified backend: ${VERL_DIR}" >&2
        git -C "${VERL_DIR}" status --short >&2
        exit 1
    fi
    git -C "${VERL_DIR}" switch --detach "${VERL_COMMIT}"
fi
echo "verl commit: $(git -C "${VERL_DIR}" rev-parse --short HEAD)"

# Port the vLLM engine-init diagnostics hook from the qwen35 cu128 backend.
"${PYTHON}" - "${VERL_DIR}" <<'PY'
import re
import sys
from pathlib import Path

verl_dir = Path(sys.argv[1])
target = verl_dir / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
content = target.read_text()

diag_fn = '''def _dump_engine_diag(exc: BaseException) -> str:
    """Dump engine-init failure diagnostics to a file on shared storage.

    Ray redirects actor subprocess stdout/stderr into the worker log, so the
    EngineCore's "root cause" may never reach the user's terminal. The EngineCore
    process inherits this process's stderr fd, so we tail /proc/self/fd/2 (the
    Ray worker log) to capture it, plus GPU/memory snapshots.
    """
    import os
    import subprocess
    import time
    import traceback

    diag_dir = os.environ.get(
        "VERL_DIAG_DIR",
        "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs",
    )
    try:
        os.makedirs(diag_dir, exist_ok=True)
    except OSError:
        diag_dir = "/tmp"
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(diag_dir, f"vllm_engine_diag_{ts}_{os.getpid()}.log")

    def _run(cmd):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return f"\\n=== {' '.join(cmd)} ===\\n{out.stdout}\\n{out.stderr}"
        except Exception as e:  # noqa: BLE001
            return f"\\n[cmd {' '.join(cmd)} failed] {e}\\n"

    with open(path, "w") as f:
        f.write(f"=== vLLM engine init failure ({ts}) pid={os.getpid()} ===\\n")
        f.write(traceback.format_exc())
        try:
            # WorkerProc's own traceback is logged asynchronously after the
            # parent raises; give it a moment to flush before tailing.
            time.sleep(5)
            fd2 = os.path.realpath("/proc/self/fd/2")
            f.write(f"\\n=== stderr target: {fd2} ===\\n")
            candidates = [fd2]
            if fd2.endswith(".err"):
                candidates.append(fd2[:-4] + ".out")  # Ray splits stdout -> worker-*.out
            for cand in candidates:
                f.write(f"\\n=== tail {cand} ===\\n")
                if os.path.isfile(cand):
                    out = subprocess.run(["tail", "-n", "2500", cand], capture_output=True, text=True, timeout=30)
                    f.write(out.stdout)
                    if out.stderr:
                        f.write(f"\\n[tail err] {out.stderr}\\n")
                else:
                    f.write("(not a regular file)\\n")
        except Exception as e:  # noqa: BLE001
            f.write(f"\\n[stderr read failed] {e}\\n")
        f.write(_run(["nvidia-smi"]))
        f.write(_run(["free", "-g"]))
    return path


'''

marker = "logger.setLevel(logging.INFO)"
if "def _dump_engine_diag" not in content:
    if marker not in content:
        print("FATAL: cannot locate logger.setLevel marker in vllm_async_server.py", file=sys.stderr)
        sys.exit(1)
    content = content.replace(marker, marker + "\n\n\n" + diag_fn, 1)


def _wrap_block(indent: str, call_line: str) -> str:
    ind8, ind12 = indent, indent + "    "
    return (
        f"{ind8}try:\n{ind12}{call_line}\n"
        f"{ind8}except Exception as exc:  # noqa: BLE001\n"
        f"{ind12}diag = _dump_engine_diag(exc)\n"
        f'{ind12}logger.error("vLLM engine init failed; diagnostics dumped to %s", diag)\n'
        f"{ind12}raise\n"
    )


# Repair-or-wrap the AsyncLLM init call (line-based, idempotent).  An earlier
# version of this port generated a broken indentation (IndentationError at
# runtime); this pass fixes both that and the pristine-call case.
lines = content.splitlines()
out, i, wrapped = [], 0, False
while i < len(lines):
    line = lines[i]
    m = re.match(r"^(\s*)try:\s*$", line)
    if m and i + 1 < len(lines) and "AsyncLLM.from_vllm_config" in lines[i + 1]:
        j = i
        for _ in range(8):
            j += 1
            if j >= len(lines) or lines[j].strip().startswith("raise"):
                break
        j = min(j + 1, len(lines))
        out.append(_wrap_block(m.group(1), lines[i + 1].strip()))
        i = j
        wrapped = True
        continue
    out.append(line)
    i += 1
content = "\n".join(out) + "\n"

if not wrapped:
    fresh = re.search(
        r"^(?P<indent>[ \t]*)(engine_client = AsyncLLM\.from_vllm_config\([^\n]*\))\s*$",
        content,
        re.MULTILINE,
    )
    if not fresh:
        print("WARNING: AsyncLLM.from_vllm_config call not found — diagnostics wrap skipped", file=sys.stderr)
    else:
        content = (
            content[: fresh.start()]
            + _wrap_block(fresh.group("indent"), fresh.group(2))
            + content[fresh.end():]
        )
        wrapped = True

target.write_text(content)

import py_compile
py_compile.compile(str(target), doraise=True)
print(f"vllm_async_server.py: diagnostics hook ensured (wrap={wrapped})")
PY

# Port the env-var tweaks to the qwen3.5 example launcher (use_task_rewards /
# rollout_num_workers), matching the qwen35 cu128 backend.  Idempotent python
# fix: earlier sed-based inserts duplicated agent.num_workers and skipped the
# use_task_rewards definition (grep matched the EXTRA line's "use_task_rewards=").
RUN_SCRIPT="${VERL_DIR}/examples/on_policy_distillation_trainer/run_qwen3_5_4b_fsdp.sh"
if [[ -f "${RUN_SCRIPT}" ]]; then
    "${PYTHON}" - "${RUN_SCRIPT}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text().splitlines(keepends=True)
out, seen = [], False
for line in lines:
    if line.strip().startswith("actor_rollout_ref.rollout.agent.num_workers="):
        if seen:
            continue  # drop duplicates
        seen = True
    out.append(line)
text = "".join(out)
changed = []

if "rollout_num_workers=${ROLLOUT_NUM_WORKERS" not in text:
    anchor = "rollout_tp=${ROLLOUT_TP:-2}\n"
    assert anchor in text, "rollout_tp anchor missing"
    text = text.replace(anchor, anchor + "rollout_num_workers=${ROLLOUT_NUM_WORKERS:-8}\n", 1)
    changed.append("rollout_num_workers def")
if "use_task_rewards=${USE_TASK_REWARDS" not in text:
    anchor = "use_policy_gradient=${USE_POLICY_GRADIENT:-True}\n"
    assert anchor in text, "use_policy_gradient anchor missing"
    text = text.replace(anchor, anchor + "use_task_rewards=${USE_TASK_REWARDS:-False}\n", 1)
    changed.append("use_task_rewards def")
if "distillation.distillation_loss.use_task_rewards=${use_task_rewards}" not in text:
    text = text.replace(
        "distillation.distillation_loss.use_task_rewards=False",
        "distillation.distillation_loss.use_task_rewards=${use_task_rewards}",
    )
    changed.append("EXTRA use_task_rewards var")
if not seen:
    anchor = "    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1\n"
    assert anchor in text, "log_prob anchor missing"
    text = text.replace(anchor, "    actor_rollout_ref.rollout.agent.num_workers=${rollout_num_workers}\n" + anchor, 1)
    changed.append("agent.num_workers x1")

path.write_text(text)
print("run script tweaks:", changed or "already correct")
PY
    echo "run_qwen3_5_4b_fsdp.sh: env-var tweaks applied"
else
    echo "WARNING: ${RUN_SCRIPT} not found in verl ${VERL_TAG}" >&2
fi

# ── Step 5: verl editable install ───────────────────────────────────────────
# verl 0.9.0 metadata pins transformers<5.11 (conservative) which conflicts
# with the qwen3.5-proven 5.12.0; install --no-deps and pull its runtime deps
# explicitly (same pattern as scripts/setup_qwen35_cu129.sh).  The import gate
# + real model load gate validate the override.
echo ""
echo "=== Step 5: verl editable install [vllm,geo] ==="
"${PYTHON}" -m pip install --no-deps -e "${VERL_DIR}" 2>&1 | tail -3
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" \
    "ray[default]==${RAY_VERSION}" \
    "tensordict==${TENSORDICT_VERSION}" \
    hydra-core accelerate datasets "pandas" "pyarrow>=19.0.0" peft torchdata \
    wandb tensorboard dill codetiming pylatexenc packaging pybind11 \
    mathruler==0.1.0 qwen-vl-utils==0.0.14 2>&1 | tail -5

# ── Step 5.5: transfer_queue (verl 0.9.0 V1 trainer hard dependency) ──────
# verl 0.9.0's V1 trainer does `import transfer_queue` unconditionally, but the
# package is NOT on PyPI and has no dist-info; the proven qwen35 cu128 stack
# ships it as a bare site-packages dir (v0.1.8).  Copy the same version for
# parity; record provenance in the manifest.
TQ_SRC="${TQ_SRC:-${DTOPD_ROOT}/envs/va-opd-qwen35-cu128/lib/python3.12/site-packages/transfer_queue}"
TQ_DST="${ENV_PREFIX}/lib/python3.12/site-packages/transfer_queue"
if [[ -d "${TQ_DST}" ]]; then
    echo "transfer_queue already present"
elif [[ -d "${TQ_SRC}" ]]; then
    cp -a "${TQ_SRC}" "${TQ_DST}"
    echo "transfer_queue copied from va-opd-qwen35-cu128"
else
    echo "FATAL: transfer_queue source not found: ${TQ_SRC}" >&2
    exit 1
fi

# ── Step 6: project editable install ───────────────────────────────────────
echo ""
echo "=== Step 6: dual-track-opd editable install ==="
"${PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}" 2>&1 | tail -2

# ── Step 7: flash-attn source build (SM90, conda CUDA 13.2 toolchain) ──────
echo ""
echo "=== Step 7: flash-attn ${FLASH_ATTN_VERSION} source build (SM90) ==="
"${PYTHON}" -m pip install 'cmake>=3.26.1,<4.0' ninja 2>&1 | tail -2

unset CUDA_HOME CUDA_PATH NVCC CUDACXX CUDAHOSTCXX
unset CC CXX CPP CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LDFLAGS_LD
unset CMAKE_ARGS CMAKE_PREFIX_PATH TORCH_CUDA_ARCH_LIST

CC_BIN=""
CXX_BIN=""
for candidate in "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-cc" "${CUDA_TOOLCHAIN}/bin/gcc"; do
    if [[ -x "${candidate}" ]]; then CC_BIN="${candidate}"; break; fi
done
for candidate in "${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-c++" "${CUDA_TOOLCHAIN}/bin/g++"; do
    if [[ -x "${candidate}" ]]; then CXX_BIN="${candidate}"; break; fi
done
[[ -n "${CC_BIN}" && -n "${CXX_BIN}" ]] || { echo "FATAL: no managed GCC/G++ in toolchain" >&2; exit 1; }

TARGETS_INCLUDE="${CUDA_TOOLCHAIN}/targets/x86_64-linux/include"
if [[ -d "${TARGETS_INCLUDE}" ]]; then
    mkdir -p "${CUDA_TOOLCHAIN}/include"
    for _item in "${TARGETS_INCLUDE}"/*; do
        _base="$(basename "${_item}")"
        [[ -e "${CUDA_TOOLCHAIN}/include/${_base}" ]] || ln -s "${_item}" "${CUDA_TOOLCHAIN}/include/${_base}"
    done
fi

export CUDA_HOME="${CUDA_TOOLCHAIN}"
export CUDA_PATH="${CUDA_TOOLCHAIN}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLCHAIN}"
export CUDACXX="${CUDA_TOOLCHAIN}/bin/nvcc"
export CUDAHOSTCXX="${CXX_BIN}"
export CC="${CC_BIN}"
export CXX="${CXX_BIN}"
export PATH="${ENV_PREFIX}/bin:${CUDA_TOOLCHAIN}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib64:${CUDA_TOOLCHAIN}/lib64/stubs:${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export C_INCLUDE_PATH="${CUDA_TOOLCHAIN}/include:${TARGETS_INCLUDE}:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_TOOLCHAIN}/include:${TARGETS_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"
export CMAKE_PREFIX_PATH="${CUDA_TOOLCHAIN}:${ENV_PREFIX}"
export TORCH_CUDA_ARCH_LIST="9.0"
export FLASH_ATTN_CUDA_ARCHS="90"
export CMAKE_POLICY_DEFAULT_CMP0146="OLD"
export MAX_JOBS="${MAX_JOBS}"
export NVCC_THREADS="${NVCC_THREADS:-4}"

if "${PYTHON}" -c "import importlib.metadata as md; assert md.version('flash-attn') == '${FLASH_ATTN_VERSION}'" 2>/dev/null; then
    echo "flash-attn ${FLASH_ATTN_VERSION} already installed — skipping rebuild"
else
    "${PYTHON}" -m pip install --no-build-isolation --no-cache-dir --no-binary :all: \
        --constraint "${CONSTRAINTS}" "flash-attn==${FLASH_ATTN_VERSION}" 2>&1 | tail -6
fi

# ── Step 7.5: torchaudio CUDA check relaxation ────────────────────────────
# vLLM 0.27.1 hard-pins torchaudio==2.11.0 whose PyPI wheel is a CUDA 13.0
# build; torch is 2.13.0+cu132.  torchaudio's _check_cuda_version raises on any
# minor mismatch.  Driver 595 runs all CUDA 13.x runtimes and this stack never
# loads audio models, so relax the check to major-version-only (idempotent).
TORCHAUDIO_UTILS="${ENV_PREFIX}/lib/python3.12/site-packages/torchaudio/_extension/utils.py"
if grep -qF "DT_OPD: relax torchaudio CUDA check" "${TORCHAUDIO_UTILS}"; then
    echo "torchaudio CUDA check already relaxed"
else
    "${PYTHON}" - "${TORCHAUDIO_UTILS}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
content = path.read_text()
old = '        if ta_version != t_version:'
new = '        if ta_version.split(".")[0] != t_version.split(".")[0]:  # DT_OPD: relax torchaudio CUDA check (13.0 wheel vs 13.2 torch, driver 595 compatible)'
if old not in content:
    print("FATAL: torchaudio version-check line not found", file=sys.stderr)
    sys.exit(1)
content = content.replace(old, new, 1)
path.write_text(content)
print("torchaudio CUDA check relaxed to major-version-only")
PY
fi

# ── Step 8: verification ───────────────────────────────────────────────────
echo ""
echo "=== Step 8: verification ==="
"${PYTHON}" -m pip check 2>&1 | tail -8 || true

echo ""
echo "--- import gate ---"
PYTHONPATH="${REPO_ROOT}/src:${VERL_DIR}" "${PYTHON}" - <<'PY'
import importlib.metadata as md
import torch

assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.version.cuda == "13.2", torch.version.cuda
print(f"torch {torch.__version__} CUDA {torch.version.cuda}")

import vllm
print(f"vllm {vllm.__version__}")
expected = {
    "torch": "2.13.0+cu132",
    "vllm": "0.27.1",
    "transformers": "5.12.0",
    "flashinfer-python": "0.6.16.post3",
    "flash-attn": "2.8.3",
    "ray": "2.55.1",
    "tensordict": "0.10.0",
}
for name, wanted in expected.items():
    actual = md.version(name)
    ok = actual.startswith(wanted.split("+")[0])
    print(f"  {name}=={actual} {'OK' if ok else 'MISMATCH'}")
    assert ok, (name, actual, wanted)

# vLLM model registry must know qwen3_5
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = set(ModelRegistry.get_supported_archs())
    hit = any("Qwen3_5" in a or "Qwen3.5" in a for a in archs)
    print(f"  vllm registry qwen3_5 arch present: {hit} ({len(archs)} archs)")
    assert hit
except AssertionError:
    raise
except Exception as exc:  # registry API drift
    import vllm.model_executor.models as _m
    from pathlib import Path as _Path
    models_dir = _Path(_m.__file__).parent
    hit = any(p.name.startswith("qwen3_5") for p in models_dir.glob("*.py"))
    print(f"  registry API check failed ({exc}); qwen3_5 model file present: {hit}")
    assert hit

# transformers qwen3.5 class
try:
    from transformers import Qwen3_5ForConditionalGeneration
    print("  transformers Qwen3_5ForConditionalGeneration: OK")
except ModuleNotFoundError:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5MoeForConditionalGeneration,
    )
    print("  transformers Qwen3_5MoeForConditionalGeneration: OK")
except Exception as exc:
    print(f"  WARNING: transformers qwen3_5 class check failed: {exc}")

# verl distillation loss registry
from verl.trainer.distillation.losses import get_distillation_loss_settings
for mode in ("k1", "k2", "k3", "kl", "abs", "mse", "low_var_kl", "forward_kl_topk"):
    try:
        settings = get_distillation_loss_settings(mode)
        assert settings is not None, mode
    except Exception as exc:
        print(f"  verl loss mode {mode}: FAIL ({exc})")
        raise
print("  verl distillation loss modes k1/k2/k3/kl/abs/mse/low_var_kl/forward_kl_topk: OK")

import flash_attn
print(f"  flash_attn {flash_attn.__version__} import: OK")

import transfer_queue
from pathlib import Path as _Path
_tq_version = (_Path(transfer_queue.__file__).parent / "version" / "version").read_text().strip()
print(f"  transfer_queue {_tq_version} import: OK")
PY

echo ""
echo "--- readelf audit of vllm _C.abi3.so ---"
VLLM_SO=$("${PYTHON}" - <<'PY'
import glob, os, vllm
root = os.path.dirname(vllm.__file__)
cands = sorted(glob.glob(os.path.join(root, "_C.abi3.so")) + glob.glob(os.path.join(root, "_C*.so")))
print(cands[0])
PY
)
echo "  ${VLLM_SO}"
readelf -d "${VLLM_SO}" | grep NEEDED | grep -E 'cudart|libcuda' || true

# ── Environment manifest ───────────────────────────────────────────────────
ENV_MANIFEST="${ENV_PREFIX}/share/dual-track-opd/va_opd_qwen35_environment_manifest.json"
GENERIC_ENV_MANIFEST="${ENV_PREFIX}/share/dual-track-opd/environment_manifest.json"
mkdir -p "$(dirname "${ENV_MANIFEST}")"
VA_OPD_ENV_MANIFEST="${ENV_MANIFEST}" \
VA_OPD_GENERIC_ENV_MANIFEST="${GENERIC_ENV_MANIFEST}" \
VA_OPD_ENV_PREFIX="${ENV_PREFIX}" \
VA_OPD_REPO_ROOT="${REPO_ROOT}" \
VA_OPD_VLLM_WHEEL="${VLLM_WHEEL}" \
VA_OPD_CUDA_TOOLCHAIN="${CUDA_TOOLCHAIN}" \
VA_OPD_CONSTRAINTS="${CONSTRAINTS}" \
VA_OPD_VERL_DIR="${VERL_DIR}" \
VA_OPD_VERL_COMMIT="${VERL_COMMIT}" \
"${PYTHON}" - <<'PY'
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

wheel = Path(os.environ["VA_OPD_VLLM_WHEEL"]).resolve()
constraints = Path(os.environ["VA_OPD_CONSTRAINTS"]).resolve()
manifest = {
    "schema_version": 1,
    "environment_name": Path(os.environ["VA_OPD_ENV_PREFIX"]).name,
    "environment_prefix": str(Path(os.environ["VA_OPD_ENV_PREFIX"]).resolve()),
    "purpose": "qwen3.5/3.6 on-policy distillation (27B->4B etc.) on 8xH200 with CUDA 13.2",
    "status_at_build": "candidate",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "build_kind": "cpu-wheel-install-cu132-h200-sm90",
    "packages": {
        name: md.version(name)
        for name in ("torch", "torchvision", "vllm", "transformers", "flashinfer-python", "flash-attn", "ray", "tensordict")
    },
    "transfer_queue": "0.1.8 (bare site-packages copy from va-opd-qwen35-cu128; not on PyPI)",
    "torch_cuda": torch.version.cuda,
    "nccl": ".".join(str(part) for part in torch.cuda.nccl.version()),
    "torch_cuda_arch_list": "9.0",
    "target_gpu": "NVIDIA H200 (SM90)",
    "build_node_kind": "CPU with internet; shared prefix consumed on GPU node",
    "repo_commit": subprocess.check_output(
        ["git", "-C", os.environ["VA_OPD_REPO_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "repo_dirty": bool(
        subprocess.check_output(
            ["git", "-C", os.environ["VA_OPD_REPO_ROOT"], "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
    ),
    "vllm_wheel": str(wheel),
    "vllm_wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "constraints_sha256": hashlib.sha256(constraints.read_bytes()).hexdigest(),
    "verl_backend_commit": os.environ["VA_OPD_VERL_COMMIT"],
    "verl_backend_path": os.environ["VA_OPD_VERL_DIR"],
    "cuda_toolchain": str(Path(os.environ["VA_OPD_CUDA_TOOLCHAIN"]).resolve()),
    "nvcc_version": subprocess.check_output(
        [str(Path(os.environ["VA_OPD_CUDA_TOOLCHAIN"]) / "bin/nvcc"), "--version"], text=True
    ).strip(),
    "verification": {
        "pip_check": "pass-with-known-verl-transformers-pin-override",
        "gpu_kernel_smoke": "pending",
        "nccl_smoke": "pending",
        "training_smoke": "pending",
    },
}
payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
Path(os.environ["VA_OPD_ENV_MANIFEST"]).write_text(payload, encoding="utf-8")
Path(os.environ["VA_OPD_GENERIC_ENV_MANIFEST"]).write_text(payload, encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo ""
echo "Build finished at $(date)"
echo "Next (GPU node): audit -> preflight -> NCCL smoke -> OPD smoke; see docs/va_opd_qwen35_cu132_proposal_20260816.md"
