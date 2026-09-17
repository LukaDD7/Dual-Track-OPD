#!/usr/bin/env bash
# Rebuild the pinned native VA-OPD vLLM 0.12.0 cu128 wheel and repair its manifest.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HPC_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
ENV_PREFIX="${VA_OPD_ENV_PREFIX:-${HPC_ROOT}/envs/va-opd-native-e003-cu128-r595-v1}"
VLLM_SOURCE="${VA_OPD_VLLM_SOURCE:-${HPC_ROOT}/fc-opd-storage/backends/vllm-va-opd-v0120}"
CUDA_TOOLCHAIN="${VA_OPD_CUDA_TOOLCHAIN:-${HPC_ROOT}/envs/cuda128-toolchain}"
WHEELHOUSE="${VA_OPD_WHEELHOUSE:-${HPC_ROOT}/fc-opd-storage/wheelhouse/va-opd-cu128-r595-v1}"
CONSTRAINTS="${REPO_ROOT}/configs/environment/verl_va_opd_e003_cu128.constraints.txt"
VLLM_COMMIT="4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e"

PYTHON="${ENV_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || { echo "FATAL: missing Python ${PYTHON}" >&2; exit 1; }
[[ -d "${VLLM_SOURCE}/.git" ]] || { echo "FATAL: missing vLLM source ${VLLM_SOURCE}" >&2; exit 1; }
[[ -x "${CUDA_TOOLCHAIN}/bin/nvcc" ]] || { echo "FATAL: missing CUDA toolchain" >&2; exit 1; }

if [[ "$(git -C "${VLLM_SOURCE}" rev-parse HEAD)" != "${VLLM_COMMIT}" ]]; then
    echo "FATAL: vLLM source is not at ${VLLM_COMMIT}" >&2
    exit 1
fi
# The source checkout may retain requirement edits from a previous
# ``use_existing_torch.py`` run. The temporary build clone below starts from the
# committed tree and reapplies that transformation itself, so dirty status is
# informational rather than fatal.
if [[ -n "$(git -C "${VLLM_SOURCE}" status --porcelain --untracked-files=no)" ]]; then
    echo "NOTE: vLLM source has prior use_existing_torch.py requirement edits" >&2
fi

CC_BIN="${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-cc"
CXX_BIN="${CUDA_TOOLCHAIN}/bin/x86_64-conda-linux-gnu-c++"
[[ -x "${CC_BIN}" ]] || CC_BIN="${CUDA_TOOLCHAIN}/bin/gcc"
[[ -x "${CXX_BIN}" ]] || CXX_BIN="${CUDA_TOOLCHAIN}/bin/g++"
[[ -x "${CC_BIN}" && -x "${CXX_BIN}" ]] || { echo "FATAL: managed GCC/G++ missing" >&2; exit 1; }

mkdir -p "${WHEELHOUSE}"
EXISTING_WHEEL="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name 'vllm-0.12.0*.whl' -print -quit)"
if [[ -n "${EXISTING_WHEEL}" && "${VA_OPD_FORCE_REBUILD:-0}" != "1" ]]; then
    ENV_MANIFEST="${ENV_PREFIX}/share/dual-track-opd/va_opd_environment_manifest.json" \
    VA_OPD_VLLM_WHEEL="${EXISTING_WHEEL}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

wheel = Path(os.environ["VA_OPD_VLLM_WHEEL"]).resolve()
manifest_path = Path(os.environ["ENV_MANIFEST"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
if manifest.get("vllm_wheel") == str(wheel) and manifest.get("vllm_wheel_sha256") == wheel_hash:
    print(f"Existing vLLM wheel and manifest are valid: {wheel}")
else:
    manifest["vllm_wheel"] = str(wheel)
    manifest["vllm_wheel_sha256"] = wheel_hash
    manifest["vllm_wheel_rebuilt_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["vllm_wheel_rebuild_reason"] = "reused existing source-built wheel and repaired manifest"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Updated manifest for existing wheel: {manifest_path}")
PY
    exit 0
fi

BUILD_PARENT="$(mktemp -d /tmp/va-opd-vllm-build-XXXXXX)"
cleanup() {
    rm -rf "${BUILD_PARENT}"
}
trap cleanup EXIT INT TERM
BUILD_ROOT="${BUILD_PARENT}/vllm"
git clone --shared --no-checkout "${VLLM_SOURCE}" "${BUILD_ROOT}"
git -C "${BUILD_ROOT}" checkout --detach "${VLLM_COMMIT}"

export CUDA_HOME="${CUDA_TOOLCHAIN}"
export CUDA_PATH="${CUDA_TOOLCHAIN}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLCHAIN}"
export CUDACXX="${CUDA_TOOLCHAIN}/bin/nvcc"
export CUDAHOSTCXX="${CXX_BIN}"
export CC="${CC_BIN}"
export CXX="${CXX_BIN}"
export PATH="${ENV_PREFIX}/bin:${CUDA_TOOLCHAIN}/bin:${PATH}"
export C_INCLUDE_PATH="${CUDA_TOOLCHAIN}/include:${CUDA_TOOLCHAIN}/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_TOOLCHAIN}/include:${CUDA_TOOLCHAIN}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${CUDA_TOOLCHAIN}:${ENV_PREFIX}"
export TORCH_CUDA_ARCH_LIST="9.0"
export CMAKE_POLICY_DEFAULT_CMP0146="OLD"
export MAX_JOBS="${VA_OPD_MAX_JOBS:-16}"
export NVCC_THREADS="${VA_OPD_NVCC_THREADS:-2}"
export VLLM_TARGET_DEVICE="cuda"
export VLLM_VERSION_OVERRIDE="0.12.0"

cd "${BUILD_ROOT}"
"${PYTHON}" use_existing_torch.py
"${PYTHON}" -m pip install 'cmake>=3.26.1,<4.0'
"${PYTHON}" -m pip install --constraint "${CONSTRAINTS}" -r requirements/build.txt
sed -i '/cmake_args += \[f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}\/bin\/nvcc"\]/a\            cmake_args += [f"-DCUDA_TOOLKIT_ROOT_DIR={CUDA_HOME}"]' setup.py 2>/dev/null || true
export VLLM_PYTHON_EXECUTABLE="${PYTHON}"
"${PYTHON}" -m pip wheel --no-build-isolation --no-deps --wheel-dir "${WHEELHOUSE}" .

WHEEL="$(find "${WHEELHOUSE}" -maxdepth 1 -type f -name 'vllm-0.12.0*.whl' -print -quit)"
[[ -n "${WHEEL}" ]] || { echo "FATAL: wheel build produced no vllm-0.12.0 wheel" >&2; exit 1; }
echo "WHEEL: ${WHEEL}"
sha256sum "${WHEEL}"

ENV_MANIFEST="${ENV_PREFIX}/share/dual-track-opd/va_opd_environment_manifest.json"
VA_OPD_ENV_MANIFEST="${ENV_MANIFEST}" VA_OPD_VLLM_WHEEL="${WHEEL}" \
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(os.environ["VA_OPD_ENV_MANIFEST"])
wheel = Path(os.environ["VA_OPD_VLLM_WHEEL"]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["vllm_wheel"] = str(wheel)
manifest["vllm_wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
manifest["vllm_wheel_rebuilt_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest["vllm_wheel_rebuild_reason"] = "original source-built wheel was missing from shared storage"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Updated manifest: {manifest_path}")
PY

echo "NATIVE VA-OPD vLLM WHEEL REBUILD PASSED"
