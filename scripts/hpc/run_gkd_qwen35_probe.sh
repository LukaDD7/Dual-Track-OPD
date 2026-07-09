#!/usr/bin/env bash
# run_gkd_qwen35_probe.sh — Qwen3.5 support probes (Gate 4)
#
# Three probes to determine if Qwen3.5 can be used with the GKD/Megatron stack:
#   Probe A: Transformers load Qwen3.5-0.8B + Qwen3.5-4B
#   Probe B: vLLM serve/import Qwen3.5-0.8B
#   Probe C: GKD/Megatron actor model provider load Qwen3.5-0.8B
#
# If Probe C fails with architecture-unsupported, record as expected blocker.
# Do NOT attempt to patch Megatron model layers without explicit approval.
#
# Usage:
#   bash scripts/hpc/run_gkd_qwen35_probe.sh

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="gkd_qwen35_probe_${TIMESTAMP}"
OUTPUT_DIR="${REPO_ROOT}/runs/gkd_qwen35_probe/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

CONDA_BASE="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/miniconda3"
GKD_ENV="${CONDA_BASE}/envs/vaopd-gkd-cu128"
PYTHON="${GKD_ENV}/bin/python"
VERL_GKD_DIR="${REPO_ROOT}/external/verl_gkd/verl"

QWEN35_08B="Qwen/Qwen3.5-0.8B"
QWEN35_4B="Qwen/Qwen3.5-4B"

log() {
    echo "$@" | tee -a "${OUTPUT_DIR}/probe.log"
}

fatal() {
    log "FATAL: $*"
    exit 1
}

log "════════════════════════════════════════════════════════"
log "  Qwen3.5 Probe — Gate 4"
log "  Run ID:  ${RUN_ID}"
log "  Output:  ${OUTPUT_DIR}"
log "════════════════════════════════════════════════════════"
log ""

# ── preamble ──────────────────────────────────────────────────────────────
if [[ ! -x "${PYTHON}" ]]; then
    fatal "Python not found at ${PYTHON}. Run setup_vaopd_gkd_cu128.sh --execute first."
fi

log "=== Environment ==="
"${PYTHON}" -c "
import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda}')
import vllm; print(f'vllm={vllm.__version__}')
" 2>&1 | tee -a "${OUTPUT_DIR}/probe.log"
log ""

# ═══════════════════════════════════════════════════════════════════════════
# Probe A: Transformers load
# ═══════════════════════════════════════════════════════════════════════════
log "=== Probe A: Transformers Load ==="

for MODEL in "${QWEN35_08B}" "${QWEN35_4B}"; do
    log ""
    log "--- Loading ${MODEL} with transformers ---"
    "${PYTHON}" -c "
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

print(f'Loading {${MODEL@Q}}...')
try:
    model = AutoModelForCausalLM.from_pretrained(
        '${MODEL}',
        torch_dtype=torch.bfloat16,
        device_map='cpu',
        trust_remote_code=True,
    )
    print(f'  Model loaded: {type(model).__name__}')
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')
    print(f'  Config: {model.config.model_type}')
    print('  Probe A: PASS')
except Exception as e:
    print(f'  Probe A: FAIL — {e}')
    exit(1)
" 2>&1 | tee -a "${OUTPUT_DIR}/probe.log"
    PROBE_A_EXIT=$?
    if [[ ${PROBE_A_EXIT} -ne 0 ]]; then
        log "Probe A FAILED for ${MODEL}"
        # Don't exit — try the next model
    fi
done
log ""

# ═══════════════════════════════════════════════════════════════════════════
# Probe B: vLLM serve/import
# ═══════════════════════════════════════════════════════════════════════════
log "=== Probe B: vLLM Import/Serve Qwen3.5-0.8B ==="
"${PYTHON}" -c "
import torch
import sys

print(f'Testing vLLM import for ${QWEN35_08B}...')
try:
    from vllm import LLM, SamplingParams
    print('  vLLM import OK')
except ImportError as e:
    print(f'  vLLM import FAIL: {e}')
    sys.exit(1)

# Dry-run: try to create LLM instance (this downloads if needed)
# We use a small GPU memory fraction to avoid OOM
try:
    llm = LLM(
        model='${QWEN35_08B}',
        trust_remote_code=True,
        gpu_memory_utilization=0.25,
        max_model_len=512,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    print(f'  vLLM LLM created: {llm}')
    # Run a single inference
    prompts = ['Hello, world!']
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16))
    print(f'  Inference output: {outputs[0].outputs[0].text[:100]}')
    print('  Probe B: PASS')
    del llm
    torch.cuda.empty_cache()
except Exception as e:
    print(f'  Probe B: FAIL — {e}')
    sys.exit(1)
" 2>&1 | tee -a "${OUTPUT_DIR}/probe.log"
PROBE_B_EXIT=$?
log "Probe B exit code: ${PROBE_B_EXIT}"
log ""

# ═══════════════════════════════════════════════════════════════════════════
# Probe C: GKD/Megatron actor model provider load
# ═══════════════════════════════════════════════════════════════════════════
log "=== Probe C: GKD/Megatron Actor Load Qwen3.5-0.8B ==="

if [[ ! -d "${VERL_GKD_DIR}" ]]; then
    log "WARNING: verl GKD checkout not found at ${VERL_GKD_DIR} — skipping Probe C"
else
    "${PYTHON}" -c "
import sys
sys.path.insert(0, '${VERL_GKD_DIR}')

print(f'Attempting Megatron model provider load for ${QWEN35_08B}...')

# Check if Qwen3.5 architecture is in the supported model registry
try:
    from verl.models.megatron.registry import get_supported_models
    supported = get_supported_models()
    print(f'  Supported Megatron models: {sorted(supported)}')

    qwen35_supported = any('qwen3_5' in m.lower() or 'qwen35' in m.lower() for m in supported)
    if qwen35_supported:
        print('  Qwen3.5 variant found in Megatron registry')
    else:
        print('  Qwen3.5 NOT found in Megatron registry')
except Exception as e:
    print(f'  Registry check FAIL: {e}')

# Try to find ModelLayerSpec for Qwen3.5
try:
    from verl.models.megatron.layers import get_model_layer_spec
    print('  Attempting get_model_layer_spec for Qwen3.5...')
    # This will likely fail — that's the expected outcome
    try:
        spec = get_model_layer_spec('qwen3_5')
        print(f'  ModelLayerSpec found: {spec}')
    except (ValueError, KeyError, NotImplementedError) as e:
        print(f'  EXPECTED BLOCKER: Qwen3.5 ModelLayerSpec missing — {e}')
        print('  This means GKD/Megatron does not yet support Qwen3.5 architecture.')
        print('  Do NOT attempt to patch Megatron model layers without explicit approval.')
        # This is NOT a fatal error — it's a documented expected gap
except Exception as e:
    print(f'  Import check FAIL: {e}')

print()
print('=== Probe C Summary ===')
print('If ModelLayerSpec is missing, this is an EXPECTED BLOCKER for Gate 4.')
print('Migration cannot proceed past Gate 4 until Qwen3.5 Megatron support is added upstream.')
print('Ref: verl/models/megatron/layers/ — custom ModelLayerSpec needed for Qwen3.5.')
" 2>&1 | tee -a "${OUTPUT_DIR}/probe.log"
    PROBE_C_EXIT=$?
    log "Probe C exit code: ${PROBE_C_EXIT}"
fi
log ""

# ═══════════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════"
log "  Probe complete."
log "  Results:  ${OUTPUT_DIR}/probe.log"
log "  Probe A (Transformers): $([ ${PROBE_A_EXIT:-1} -eq 0 ] && echo 'PASS' || echo 'FAIL')"
log "  Probe B (vLLM):         $([ ${PROBE_B_EXIT:-1} -eq 0 ] && echo 'PASS' || echo 'FAIL')"
log "  Probe C (Megatron):     $([ ${PROBE_C_EXIT:-1} -eq 0 ] && echo 'PASS' || echo 'EXPECTED BLOCKER')"
log "════════════════════════════════════════════════════════"
