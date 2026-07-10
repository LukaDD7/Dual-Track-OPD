#!/usr/bin/env bash
# run_gkd_qwen35_probe.sh — Qwen3.5 support probes (Gate 4)
#
# Staged probes:
#   --stage config    CPU-safe: AutoConfig, AutoTokenizer, AutoProcessor
#   --stage vllm      GPU required: vLLM load + single inference
#   --stage megatron  3 GPUs: one-step GKD smoke with real Megatron actor load
#   --stage all       Run all stages (CPU-safe parts on CPU, GPU parts need GPU)
#
# GPU node has no internet — must use local model paths.
#
# Usage:
#   bash scripts/hpc/run_gkd_qwen35_probe.sh --stage config
#   bash scripts/hpc/run_gkd_qwen35_probe.sh --stage all
#   bash scripts/hpc/run_gkd_qwen35_probe.sh --stage vllm --qwen35-08b-path /path/to/model
#   bash scripts/hpc/run_gkd_qwen35_probe.sh --stage megatron --teacher-gpu 0 --train-gpus 1,2

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="gkd_qwen35_probe_${TIMESTAMP}"
OUTPUT_DIR="${REPO_ROOT}/runs/gkd_qwen35_probe/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
PROBE_LOG="${OUTPUT_DIR}/probe.log"

# ── env var overrides ─────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/miniconda3}"
MODEL_ROOT="${MODEL_ROOT:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models}"
GKD_ENV="${GKD_ENV:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vaopd-gkd-cu128}"
PYTHON="${GKD_ENV}/bin/python"
CUDA_TOOLCHAIN="${CUDA_TOOLCHAIN:-/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/cuda128-toolchain}"
VERL_GKD_DIR="${VERL_GKD_DIR:-${REPO_ROOT}/external/verl_gkd_compatible/verl}"

# ── defaults ──────────────────────────────────────────────────────────────
STAGE="all"
QWEN35_08B_PATH="${MODEL_ROOT}/Qwen3.5-0.8B"
QWEN35_4B_PATH="${MODEL_ROOT}/Qwen3.5-4B"
VLLM_GPU=0
TEACHER_GPU=0
TRAIN_GPU_LIST="1,2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)            STAGE="${2:?--stage needs config|vllm|megatron|all}"; shift 2 ;;
        --qwen35-08b-path)  QWEN35_08B_PATH="${2:?--qwen35-08b-path needs a path}"; shift 2 ;;
        --qwen35-4b-path)   QWEN35_4B_PATH="${2:?--qwen35-4b-path needs a path}"; shift 2 ;;
        --env-path)         GKD_ENV="${2:?--env-path needs a value}"; PYTHON="${GKD_ENV}/bin/python"; shift 2 ;;
        --vllm-gpu)         VLLM_GPU="${2:?--vllm-gpu needs a GPU index}"; shift 2 ;;
        --teacher-gpu)      TEACHER_GPU="${2:?--teacher-gpu needs a GPU index}"; shift 2 ;;
        --train-gpus)       TRAIN_GPU_LIST="${2:?--train-gpus needs two or more comma-separated indices}"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() {
    echo "$@" | tee -a "${PROBE_LOG}"
}

fatal() {
    log "FATAL: $*"
    exit 1
}

# ── results tracking ──────────────────────────────────────────────────────
PROBE_A_EXIT=-1
PROBE_B_EXIT=-1
PROBE_C_EXIT=-1

log "════════════════════════════════════════════════════════"
log "  Qwen3.5 Probe — Gate 4"
log "  Run ID:   ${RUN_ID}"
log "  Stage:    ${STAGE}"
log "  Output:   ${OUTPUT_DIR}"
log "  Model 08B: ${QWEN35_08B_PATH}"
log "  Model 4B:  ${QWEN35_4B_PATH}"
log "  vLLM GPU:   ${VLLM_GPU}"
log "  GKD GPUs:   teacher=${TEACHER_GPU}, train=${TRAIN_GPU_LIST}"
log "════════════════════════════════════════════════════════"
log ""

# ── preamble ──────────────────────────────────────────────────────────────
if [[ ! -x "${PYTHON}" ]]; then
    fatal "Python not found at ${PYTHON}. Run setup_vaopd_gkd_cu128.sh --execute first."
fi

if [[ -x "${CUDA_TOOLCHAIN}/bin/nvcc" ]]; then
    export CUDA_HOME="${CUDA_TOOLCHAIN}"
    export CUDA_PATH="${CUDA_TOOLCHAIN}"
    export PATH="${CUDA_TOOLCHAIN}/bin:${PATH}"
    for _candidate in "${CUDA_TOOLCHAIN}/lib" "${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib"; do
        if [[ -f "${_candidate}/libcudart.so" ]]; then
            export LIBRARY_PATH="${_candidate}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
            export LD_LIBRARY_PATH="${_candidate}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            break
        fi
    done
fi

log "=== Environment ==="
"${PYTHON}" -c "
import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')
" 2>&1 | tee -a "${PROBE_LOG}" || true

"${PYTHON}" -c "
import vllm; print(f'vllm={vllm.__version__}')
" 2>&1 | tee -a "${PROBE_LOG}" || true
log ""


# ═══════════════════════════════════════════════════════════════════════════
# Probe A: Config-only (CPU-safe)
# ═══════════════════════════════════════════════════════════════════════════
run_probe_a() {
    log "=== Probe A: Config-Only Load (CPU-safe) ==="
    log "    Required 0.8B: AutoConfig + tokenizer/processor + AutoModelForCausalLM weights"

    # Only test 0.8B config on CPU; 4B is too heavy for config-only on CPU
    local models_to_test=("${QWEN35_08B_PATH}")
    if [[ -d "${QWEN35_4B_PATH}" ]]; then
        # Optional 4B config check; Gate 4 requires the 0.8B weight load.
        models_to_test+=("${QWEN35_4B_PATH}")
    else
        log "  Optional 4B config: SKIP (path not found)"
    fi

    local all_ok=true
    for model_path in "${models_to_test[@]}"; do
        if [[ ! -d "${model_path}" ]]; then
            log "  SKIP: Model path not found: ${model_path}"
            log "        Download on CPU node first: huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ${model_path}"
            all_ok=false
            continue
        fi
        log ""
        log "--- Loading config for ${model_path} ---"

        set +e
        "${PYTHON}" -c "
import sys
model_id = '${model_path}'
required_weight_load = model_id == '${QWEN35_08B_PATH}'
print(f'Loading config from {model_id}...')

# AutoConfig
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print(f'  AutoConfig: {cfg.model_type}')

# AutoTokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f'  AutoTokenizer: {type(tok).__name__}')

# AutoProcessor (if available)
try:
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f'  AutoProcessor: {type(proc).__name__}')
except Exception as e:
    print(f'  AutoProcessor: not available — {e}')

if required_weight_load:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        low_cpu_mem_usage=True,
        device_map='cpu',
    )
    print(f'  AutoModelForCausalLM: {type(model).__name__} (weights loaded on CPU)')
    del model

print(f'  Probe A config for {model_id}: PASS')
" 2>&1 | tee -a "${PROBE_LOG}"
        local exit_code=$?
        set -e

        if [[ ${exit_code} -ne 0 ]]; then
            log "  Probe A config: FAIL for ${model_path}"
            all_ok=false
        fi
    done

    if ${all_ok}; then
        PROBE_A_EXIT=0
        log "Probe A: PASS"
    else
        PROBE_A_EXIT=1
        log "Probe A: FAIL (one or more models failed config load)"
    fi
    log ""
}


# ═══════════════════════════════════════════════════════════════════════════
# Probe B: vLLM serve/import (GPU required)
# ═══════════════════════════════════════════════════════════════════════════
run_probe_b() {
    log "=== Probe B: vLLM Import + Serve Qwen3.5-0.8B (GPU required) ==="

    if ! command -v nvidia-smi &>/dev/null; then
        log "  SKIP: nvidia-smi not found — Probe B requires GPU. Use --cpu-ok on audit_nccl_stack.sh."
        return
    fi

    if [[ ! -d "${QWEN35_08B_PATH}" ]]; then
        fatal "Model not found at ${QWEN35_08B_PATH}. GPU node has no internet — prepare model on CPU node first to NFS."
    fi

    local probe_b_process_log="${OUTPUT_DIR}/probe_b_process.log"
    local probe_b_pid
    set +e
    setsid env CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${PYTHON}" -c "
import torch
import sys

model_id = '${QWEN35_08B_PATH}'
print(f'Testing vLLM import + load for {model_id}...')

try:
    from vllm import LLM, SamplingParams
    print('  vLLM import OK')
except ImportError as e:
    print(f'  vLLM import FAIL: {e}')
    sys.exit(1)

try:
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        gpu_memory_utilization=0.25,
        max_model_len=512,
        tensor_parallel_size=1,
        enforce_eager=True,
        model_impl='transformers',
    )
    print(f'  vLLM LLM created OK')
    prompts = ['What is 2 + 2?']
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16))
    print(f'  Inference output: {outputs[0].outputs[0].text[:100]}')
    print('  Probe B: PASS')
    del llm
    torch.cuda.empty_cache()
except Exception as e:
    print(f'  Probe B: FAIL — {e}')
    sys.exit(1)
" > "${probe_b_process_log}" 2>&1 &
    probe_b_pid=$!
    wait "${probe_b_pid}"
    PROBE_B_EXIT=$?
    # vLLM 0.11 may leave EngineCore alive when initialization fails.
    kill -TERM -- "-${probe_b_pid}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${probe_b_pid}" 2>/dev/null || true
    cat "${probe_b_process_log}" | tee -a "${PROBE_LOG}"
    set -e

    log "Probe B exit code: ${PROBE_B_EXIT}"
    if [[ ${PROBE_B_EXIT} -eq 0 ]]; then
        log "Probe B: PASS"
    else
        log "Probe B: FAIL"
    fi
    log ""
}


# ═══════════════════════════════════════════════════════════════════════════
# Probe C: GKD/Megatron actor model provider (GPU required)
# ═══════════════════════════════════════════════════════════════════════════


# Probe C exercises the real actor build/load path. Registry grep or module
# introspection alone is insufficient evidence of architecture support.
run_probe_c() {
    log "=== Probe C: Real GKD/Megatron Actor Load Qwen3.5-0.8B (GPU required) ==="
    log "    Runs one complete GKD optimizer step; success is stronger than actor-load-only."

    if ! command -v nvidia-smi &>/dev/null; then
        PROBE_C_EXIT=1
        log "Probe C: FAIL — nvidia-smi not found"
        return
    fi
    if [[ ! -d "${VERL_GKD_DIR}" ]]; then
        PROBE_C_EXIT=1
        log "Probe C: FAIL — compatible verl checkout not found at ${VERL_GKD_DIR}"
        log "Run: bash scripts/setup/prepare_gkd_compatible_checkout.sh"
        return
    fi
    if [[ ! -d "${QWEN35_08B_PATH}" ]]; then
        PROBE_C_EXIT=1
        log "Probe C: FAIL — model not found at ${QWEN35_08B_PATH}"
        return
    fi
    if [[ ",${TRAIN_GPU_LIST}," == *",${TEACHER_GPU},"* ]]; then
        PROBE_C_EXIT=1
        log "Probe C: FAIL — teacher GPU ${TEACHER_GPU} overlaps train GPUs ${TRAIN_GPU_LIST}"
        return
    fi
    local train_gpu_count
    train_gpu_count=$(echo "${TRAIN_GPU_LIST}" | tr ',' '\n' | wc -l)
    if [[ ${train_gpu_count} -lt 2 ]]; then
        PROBE_C_EXIT=1
        log "Probe C: FAIL — GKD needs at least two train GPUs, got ${TRAIN_GPU_LIST}"
        return
    fi

    set +e
    GKD_TEACHER_MODEL_IMPL="transformers" \
        GKD_ENV="${GKD_ENV}" \
        VERL_GKD_DIR="${VERL_GKD_DIR}" \
        bash "${REPO_ROOT}/scripts/hpc/run_gkd_text_smoke.sh" \
        --steps 1 \
        --model-path "${QWEN35_08B_PATH}" \
        --teacher-gpu "${TEACHER_GPU}" \
        --train-gpus "${TRAIN_GPU_LIST}" \
        2>&1 | tee -a "${PROBE_LOG}"
    PROBE_C_EXIT=${PIPESTATUS[0]}
    set -e

    if [[ ${PROBE_C_EXIT} -eq 0 ]]; then
        log "Probe C: PASS — Qwen3.5 Megatron actor loaded and completed one update"
    else
        log "Probe C: FAIL — inspect the nested GKD smoke traceback above"
    fi
    log ""
}


# ═══════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════
case "${STAGE}" in
    config)
        run_probe_a
        ;;
    vllm)
        run_probe_b
        ;;
    megatron)
        run_probe_c
        ;;
    all)
        run_probe_a
        run_probe_b
        run_probe_c
        ;;
    *)
        fatal "Unknown stage: ${STAGE}. Use config|vllm|megatron|all."
        ;;
esac

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════"
log "  Probe complete."
log "  Results:  ${PROBE_LOG}"
log ""
_probe_a_label() {
    if [[ ${PROBE_A_EXIT} -eq 0 ]]; then echo "PASS"; elif [[ ${PROBE_A_EXIT} -eq -1 ]]; then echo "NOT RUN"; else echo "FAIL"; fi
}
_probe_b_label() {
    if [[ ${PROBE_B_EXIT} -eq 0 ]]; then echo "PASS"; elif [[ ${PROBE_B_EXIT} -eq -1 ]]; then echo "NOT RUN"; else echo "FAIL"; fi
}
_probe_c_label() {
    if [[ ${PROBE_C_EXIT} -eq 0 ]]; then echo "PASS"; elif [[ ${PROBE_C_EXIT} -eq -1 ]]; then echo "NOT RUN"; else echo "FAIL"; fi
}
log "  Probe A (Config):    $(_probe_a_label)"
log "  Probe B (vLLM):      $(_probe_b_label)"
log "  Probe C (Megatron):  $(_probe_c_label)"
log ""
GATE4_OK=false
case "${STAGE}" in
    config)   [[ ${PROBE_A_EXIT} -eq 0 ]] && GATE4_OK=true ;;
    vllm)     [[ ${PROBE_B_EXIT} -eq 0 ]] && GATE4_OK=true ;;
    megatron) [[ ${PROBE_C_EXIT} -eq 0 ]] && GATE4_OK=true ;;
    all)      [[ ${PROBE_A_EXIT} -eq 0 && ${PROBE_B_EXIT} -eq 0 && ${PROBE_C_EXIT} -eq 0 ]] && GATE4_OK=true ;;
esac
if ${GATE4_OK}; then
    log "  Gate 4 requested stage(s): PASS"
else
    log "  Gate 4 requested stage(s): FAIL"
fi
log "════════════════════════════════════════════════════════"

if ${GATE4_OK}; then
    exit 0
fi
exit 1
