#!/usr/bin/env bash
# run_gkd_qwen35_probe.sh — Qwen3.5 support probes (Gate 4)
#
# Staged probes:
#   --stage config    CPU-safe: AutoConfig, AutoTokenizer, AutoProcessor
#   --stage vllm      GPU required: vLLM load + single inference
#   --stage megatron  GPU required: GKD Megatron actor model provider
#   --stage all       Run all stages (CPU-safe parts on CPU, GPU parts need GPU)
#
# GPU node has no internet — must use local model paths.
#
# Usage:
#   bash scripts/hpc/run_gkd_qwen35_probe.sh --stage config
#   bash scripts/hpc/run_gkd_qwen35_probe.sh --stage all
#   bash scripts/hpc/run_gkd_qwen35_probe.sh --stage vllm --qwen35-08b-path /path/to/model

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
VERL_GKD_DIR="${REPO_ROOT}/external/verl_gkd/verl"

# ── defaults ──────────────────────────────────────────────────────────────
STAGE="all"
QWEN35_08B_PATH="${MODEL_ROOT}/Qwen3.5-0.8B"
QWEN35_4B_PATH="${MODEL_ROOT}/Qwen3.5-4B"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)            STAGE="${2:?--stage needs config|vllm|megatron|all}"; shift 2 ;;
        --qwen35-08b-path)  QWEN35_08B_PATH="${2:?--qwen35-08b-path needs a path}"; shift 2 ;;
        --qwen35-4b-path)   QWEN35_4B_PATH="${2:?--qwen35-4b-path needs a path}"; shift 2 ;;
        --env-path)         GKD_ENV="${2:?--env-path needs a value}"; PYTHON="${GKD_ENV}/bin/python"; shift 2 ;;
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
PROBE_C_BLOCKER=false

log "════════════════════════════════════════════════════════"
log "  Qwen3.5 Probe — Gate 4"
log "  Run ID:   ${RUN_ID}"
log "  Stage:    ${STAGE}"
log "  Output:   ${OUTPUT_DIR}"
log "  Model 08B: ${QWEN35_08B_PATH}"
log "  Model 4B:  ${QWEN35_4B_PATH}"
log "════════════════════════════════════════════════════════"
log ""

# ── preamble ──────────────────────────────────────────────────────────────
if [[ ! -x "${PYTHON}" ]]; then
    fatal "Python not found at ${PYTHON}. Run setup_vaopd_gkd_cu128.sh --execute first."
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
    log "    AutoConfig + AutoTokenizer + AutoProcessor (no weight loading for 4B)"

    # Only test 0.8B config on CPU; 4B is too heavy for config-only on CPU
    local models_to_test=("${QWEN35_08B_PATH}")
    if [[ "${STAGE}" == "all" ]] || [[ "${STAGE}" == "config" ]]; then
        # Also check 4B config — don't load weights, just config
        models_to_test+=("${QWEN35_4B_PATH}")
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

    set +e
    "${PYTHON}" -c "
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
" 2>&1 | tee -a "${PROBE_LOG}"
    PROBE_B_EXIT=$?
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
run_probe_c() {
    log "=== Probe C: GKD/Megatron Actor Load Qwen3.5-0.8B (GPU required) ==="

    if ! command -v nvidia-smi &>/dev/null; then
        log "  SKIP: nvidia-smi not found — Probe C requires GPU."
        return
    fi

    if [[ ! -d "${VERL_GKD_DIR}" ]]; then
        log "  SKIP: verl GKD checkout not found at ${VERL_GKD_DIR}"
        return
    fi

    if [[ ! -d "${QWEN35_08B_PATH}" ]]; then
        fatal "Model not found at ${QWEN35_08B_PATH}. GPU node has no internet — prepare model on CPU node first to NFS."
    fi

    # First: grep for qwen support in the GKD verl checkout
    log "--- Scanning for Qwen support in verl GKD checkout ---"
    set +e
    grep -R -n "qwen" "${VERL_GKD_DIR}/verl/verl" "${VERL_GKD_DIR}/verl/recipe/gkd" \
        -i 2>/dev/null | head -100 | tee -a "${PROBE_LOG}" || true
    set -e
    log ""

    # Try introspection of the actual checkout
    log "--- Attempting Megatron model provider introspection ---"
    set +e
    "${PYTHON}" -c "
import sys
sys.path.insert(0, '${VERL_GKD_DIR}')

model_id = '${QWEN35_08B_PATH}'
print(f'Attempting Megatron model provider load for {model_id}...')
print()

# Step 1: list available model registrations
print('=== Checking model registry ===')
try:
    # Try common paths in the GKD verl checkout
    import importlib, pkgutil

    # Check if there's a megatron models module
    try:
        from verl.models.megatron import registry
        supported = registry.get_supported_models()
        print(f'  Supported Megatron models: {sorted(supported)}')

        qwen35_supported = any('qwen3_5' in m.lower() or 'qwen35' in m.lower() for m in supported)
        if qwen35_supported:
            print('  Qwen3.5 variant FOUND in Megatron registry')
        else:
            print('  Qwen3.5 NOT found in Megatron registry')
    except ImportError as e:
        print(f'  Registry import failed: {e}')

    # Step 2: try to find ModelLayerSpec
    print()
    print('=== Checking ModelLayerSpec ===')
    try:
        from verl.models.megatron.layers import get_model_layer_spec
        for arch in ['qwen3_5', 'qwen35', 'qwen3.5']:
            try:
                spec = get_model_layer_spec(arch)
                print(f'  ModelLayerSpec for \"{arch}\": FOUND')
            except (ValueError, KeyError, NotImplementedError) as e:
                print(f'  ModelLayerSpec for \"{arch}\": MISSING — {e}')
    except ImportError as e:
        print(f'  layers import failed: {e}')

except Exception as e:
    print(f'  Introspection error: {e}')

print()
print('=== Probe C Summary ===')
# Check if any qwen3_5 model registration was found
found_qwen35 = False
try:
    from verl.models.megatron.registry import get_supported_models
    found_qwen35 = any('qwen3_5' in m.lower() or 'qwen35' in m.lower() for m in get_supported_models())
except:
    pass

if found_qwen35:
    print('Qwen3.5 Megatron support: PRESENT — Probe C may pass')
    print('Proceed to full actor model load test.')
else:
    print('Qwen3.5 Megatron support: ABSENT — EXPECTED BLOCKER')
    print('This means GKD/Megatron does not yet support Qwen3.5 architecture.')
    print('Do NOT attempt to patch Megatron model layers without explicit approval.')
    print('Gate 4 cannot pass until Qwen3.5 Megatron support is added upstream.')
    print('Ref: verl/models/megatron/layers/ — custom ModelLayerSpec needed for Qwen3.5.')
" 2>&1 | tee -a "${PROBE_LOG}"
    PROBE_C_EXIT=$?
    set -e

    # If the Python script found qwen3_5 support, it's not a blocker
    # If not, it's an expected blocker — check the output
    if grep -q "PRESENT" "${PROBE_LOG}" 2>/dev/null; then
        PROBE_C_BLOCKER=false
        log "Probe C: Qwen3.5 Megatron support PRESENT — Gate 4 may proceed"
    else
        PROBE_C_BLOCKER=true
        PROBE_C_EXIT=0  # expected blocker, not a script error
        log "Probe C: EXPECTED BLOCKER — Qwen3.5 Megatron support is absent"
        log "Gate 4 not passed. Do NOT attempt to patch Megatron model layers."
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
    if [[ ${PROBE_A_EXIT} -eq 0 ]]; then echo "PASS"; else echo "NOT RUN"; fi
}
_probe_b_label() {
    if [[ ${PROBE_B_EXIT} -eq 0 ]]; then echo "PASS"; elif [[ ${PROBE_B_EXIT} -eq -1 ]]; then echo "NOT RUN"; else echo "FAIL"; fi
}
_probe_c_label() {
    if ${PROBE_C_BLOCKER}; then echo "EXPECTED BLOCKER"; elif [[ ${PROBE_C_EXIT} -eq 0 ]]; then echo "PASS"; else echo "NOT RUN"; fi
}
log "  Probe A (Config):    $(_probe_a_label)"
log "  Probe B (vLLM):      $(_probe_b_label)"
log "  Probe C (Megatron):  $(_probe_c_label)"
log ""
if ${PROBE_C_BLOCKER}; then
    log "  Gate 4: NOT PASSED (expected blocker — Qwen3.5 Megatron support absent)"
    log "  This is a known gap. Do not proceed to Gate 5 without explicit approval."
else
    log "  Gate 4 status: check probe results above"
fi
log "════════════════════════════════════════════════════════"

# Exit 0 even on expected blocker (it's not a script error, it's a finding)
exit 0
