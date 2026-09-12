#!/usr/bin/env bash
# Read-only GPU-node inventory for the VA-OPD CPU-CC -> human -> GPU feedback loop.

set -u

HPC_ROOT="${DTOPD_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_PREFIX="${VA_OPD_ENV_PREFIX:-${HPC_ROOT}/envs/va-opd-native-e003-cu128-r595-v1}"
VERL_DIR="${VERL_VA_OPD_DIR:-${HPC_ROOT}/fc-opd-storage/backends/verl-va-opd-e0031631-clean}"
VLLM_SOURCE="${VA_OPD_VLLM_SOURCE:-${HPC_ROOT}/fc-opd-storage/backends/vllm-va-opd-v0120}"
STUDENT_MODEL="${VA_OPD_STUDENT_MODEL:-${HPC_ROOT}/models/Qwen3-VL-4B-Instruct}"
TEACHER_MODEL="${VA_OPD_TEACHER_MODEL:-${HPC_ROOT}/models/Qwen3-VL-32B-Instruct}"
FACT_ROOT="${VA_OPD_FACT_ROOT:-${HPC_ROOT}/fc-opd-storage/diagnostics/va_opd_gpu_facts}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
FACT_DIR="${FACT_ROOT}/${TIMESTAMP}"
REPORT="${FACT_DIR}/gpu_facts.txt"

mkdir -p "${FACT_DIR}"
exec > >(tee "${REPORT}") 2>&1

section() {
    printf '\n===== %s =====\n' "$1"
}

try_run() {
    local label="$1"
    shift
    section "${label}"
    "$@"
    local status=$?
    if (( status != 0 )); then
        printf 'COMMAND_EXIT=%s\n' "${status}"
    fi
    return 0
}

section "identity"
printf 'timestamp=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
printf 'hostname=%s\n' "$(hostname 2>/dev/null || true)"
printf 'user=%s\n' "$(id -un 2>/dev/null || true)"
printf 'repo_root=%s\n' "${REPO_ROOT}"
printf 'hpc_root=%s\n' "${HPC_ROOT}"
printf 'env_prefix=%s\n' "${ENV_PREFIX}"
printf 'verl_dir=%s\n' "${VERL_DIR}"
printf 'vllm_source=%s\n' "${VLLM_SOURCE}"
printf 'student_model=%s\n' "${STUDENT_MODEL}"
printf 'teacher_model=%s\n' "${TEACHER_MODEL}"

try_run "kernel" uname -a
if [[ -r /etc/os-release ]]; then
    try_run "os-release" sed -n '1,120p' /etc/os-release
fi
try_run "limits" bash -c 'ulimit -a'
try_run "cpu-memory" bash -c 'command -v lscpu >/dev/null && lscpu; command -v free >/dev/null && free -h'

section "scheduler-safe-environment"
env | grep -E '^(SLURM_(JOB_ID|JOB_NAME|JOB_NODELIST|JOB_NUM_NODES|GPUS|GPUS_ON_NODE|CPUS_ON_NODE)|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|RAY_ADDRESS)=' | sort || true

try_run "filesystem-capacity" df -h "${HPC_ROOT}"
try_run "filesystem-inodes" df -i "${HPC_ROOT}"
try_run "nvidia-smi-summary" nvidia-smi
try_run "nvidia-smi-list" nvidia-smi -L
try_run "nvidia-smi-query" nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,driver_version,pstate,temperature.gpu,power.draw,power.limit,mig.mode.current --format=csv
try_run "nvidia-smi-compute-processes" nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
try_run "nvidia-topology" nvidia-smi topo -m
try_run "nvidia-errors" nvidia-smi -q -d ECC,ERROR

section "toolchain-discovery-do-not-use-system-nvcc-for-builds"
NVCC_PATH="$(command -v nvcc 2>/dev/null || true)"
printf 'nvcc_on_path=%s\n' "${NVCC_PATH:-NONE}"
if [[ -n "${NVCC_PATH}" ]]; then
    printf 'nvcc_realpath=%s\n' "$(realpath "${NVCC_PATH}" 2>/dev/null || true)"
    "${NVCC_PATH}" --version || true
fi
printf 'configured_cuda_toolchain=%s\n' "${VA_OPD_CUDA_TOOLCHAIN:-UNSET}"

section "path-readiness"
for path in \
    "${REPO_ROOT}" \
    "${ENV_PREFIX}/bin/python" \
    "${ENV_PREFIX}/bin/torchrun" \
    "${ENV_PREFIX}/share/dual-track-opd/va_opd_environment_manifest.json" \
    "${VERL_DIR}/.git" \
    "${VLLM_SOURCE}/.git" \
    "${STUDENT_MODEL}/config.json" \
    "${TEACHER_MODEL}/config.json"; do
    if [[ -r "${path}" ]]; then
        printf 'READABLE %s\n' "${path}"
    else
        printf 'MISSING_OR_UNREADABLE %s\n' "${path}"
    fi
done

if [[ -r "${ENV_PREFIX}/share/dual-track-opd/va_opd_environment_manifest.json" ]]; then
    section "environment-build-provenance"
    sed -n '1,240p' "${ENV_PREFIX}/share/dual-track-opd/va_opd_environment_manifest.json"
fi

section "git-identities"
git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true
git -C "${REPO_ROOT}" status --short --ignore-submodules=all 2>/dev/null || true
git -C "${VERL_DIR}" rev-parse HEAD 2>/dev/null || true
git -C "${VERL_DIR}" status --short --untracked-files=no 2>/dev/null || true
git -C "${VLLM_SOURCE}" rev-parse HEAD 2>/dev/null || true
git -C "${VLLM_SOURCE}" status --short --untracked-files=no 2>/dev/null || true

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    section "python-runtime"
    "${ENV_PREFIX}/bin/python" - <<'PY'
import importlib.metadata as md
import json
import platform

result = {
    "python": platform.python_version(),
    "packages": {},
}
for name in (
    "torch",
    "torchvision",
    "vllm",
    "transformers",
    "ray",
    "tensordict",
    "flashinfer-python",
    "flash-attn",
    "numpy",
    "pyarrow",
):
    try:
        result["packages"][name] = md.version(name)
    except md.PackageNotFoundError:
        result["packages"][name] = None

try:
    import torch

    result["torch_cuda_runtime"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    result["visible_gpu_count"] = torch.cuda.device_count()
    result["visible_gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    result["nccl_version"] = torch.cuda.nccl.version() if torch.cuda.is_available() else None
except Exception as exc:  # inventory must report rather than hide import/runtime errors
    result["torch_probe_error"] = repr(exc)

print(json.dumps(result, indent=2, sort_keys=True))
PY
else
    section "python-runtime"
    printf 'ENVIRONMENT_PYTHON_MISSING\n'
fi

try_run "loaded-cuda-nccl-libraries" bash -c "command -v ldconfig >/dev/null && ldconfig -p | grep -E 'lib(nccl|cuda|cudart)\\.so' | head -n 120"
try_run "ray-status" "${ENV_PREFIX}/bin/ray" status

section "result"
printf 'FACT_DIR=%s\n' "${FACT_DIR}"
printf 'REPORT=%s\n' "${REPORT}"
printf 'Send this report to the CPU-instance CC. It is diagnostic output outside Git.\n'
