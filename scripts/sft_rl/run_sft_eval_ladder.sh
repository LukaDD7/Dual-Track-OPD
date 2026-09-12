#!/usr/bin/env bash
#
# SFT warmup 评测入口（GPU 实例一键跑，manuscript 第 3 步前置）
#   A. 保留集 pass@1 / pass@8（规则判分，无需 judge）:
#      - geo3k 官方测试集 601 行（held-out，训练重叠 ~0.3%）
#      - MMF RL val 500 行（22 sources 规则可判子集）
#   B. 项目六项 benchmark（manuscript 第 3 步，GQA/DynaMath/ViewSpatial/
#      MMMU-Pro/ReMI 规则 + MMBench judge）
#
# 用法（GPU 实例，SFT 完成后 8 卡空闲时）:
#   bash scripts/sft_rl/run_sft_eval_ladder.sh                    # 最终 ckpt 6939
#   EVAL_CKPT=<hf-dir> EVAL_TAG=epoch1 bash scripts/sft_rl/run_sft_eval_ladder.sh
#   SMOKE=1 bash scripts/sft_rl/run_sft_eval_ladder.sh            # 冒烟（各 8 行/8 条）
#
# 产物:
#   pass@k:  $R/fc-opd-storage/outputs/fc_opd/sft_rl/eval/<tag>/{geo3k,mmf}_*.jsonl + summary
#   benchmark: $R/eval_runs/vision_opd_project_baseline/<tag>_nojudge|_judged
set -euo pipefail

# 每个 A 段的完整输出落独立日志（tail 管道会吞掉 vLLM 报错根因）
DTOPD_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
REPO_ROOT="${DTOPD_ROOT}/projects/Dual-Track-OPD"
# 训练 env（vllm 0.27.1 + mathruler）——eval_geo3k 的规则判分依赖 mathruler，
# qwen3vl-cu128-vllm 没装 mathruler。
EVAL_ENV="${EVAL_ENV:-${DTOPD_ROOT}/envs/va-opd-qwen35-v090-cu132-r595-v1}"
PY="${EVAL_ENV}/bin/python"

# vLLM flashinfer JIT 需要 nvcc + gcc + ninja（GPU 实例无系统 CUDA）：
# nvcc/gcc 来自 cuda132-toolchain；ninja 在评测 env 的 bin/（训练脚本靠
# conda activate 拿到它，这里不 activate，必须显式加 PATH——否则 JIT 编译
# sampling 模块时报 FileNotFoundError: 'ninja'，A1 即死于此）
CUDA_TOOLCHAIN="${DTOPD_ROOT}/envs/cuda132-toolchain"
export CUDA_HOME="${CUDA_TOOLCHAIN}"
export PATH="${EVAL_ENV}/bin:${CUDA_TOOLCHAIN}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib64:${CUDA_TOOLCHAIN}/lib64/stubs:${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_TOOLCHAIN}/lib:${CUDA_TOOLCHAIN}/targets/x86_64-linux/lib:${EVAL_ENV}/lib/python3.12/site-packages/torch/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${DTOPD_ROOT}/.cache/flashinfer"

EVAL_CKPT="${EVAL_CKPT:-${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/ckpt/qwen3vl_sft_warmup_20260826_1324/global_step_6939/huggingface}"
EVAL_TAG="${EVAL_TAG:-sft_6939}"
EVAL_GPU="${EVAL_GPU:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
BENCH_EVAL_PORT="${BENCH_EVAL_PORT:-8000}"
BENCH_JUDGE_PORT="${BENCH_JUDGE_PORT:-8001}"
OUT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/eval/${EVAL_TAG}"
SMOKE="${SMOKE:-0}"

[ -f "${EVAL_CKPT}/model.safetensors" ] || compgen -G "${EVAL_CKPT}/model-*.safetensors" >/dev/null \
  || { echo "FATAL: no model weights in ${EVAL_CKPT}"; exit 1; }
mkdir -p "${OUT_DIR}"

GEO3K_DATA="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_grpo_val_test.parquet"
MMF_DATA="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_20k/mmf_rl_val__part_0000.parquet"

# pass@8（temp 0.7 采样 8 次）+ pass@1（temp 0 单次）各跑一遍；
# --rollouts 1 => temp 0；--rollouts 8 => temp 0.7（脚本内置）
ROWS_ARG=""
BENCH_LIMIT=""
if [[ "${SMOKE}" == "1" ]]; then
  ROWS_ARG="--max-rows 8"
  BENCH_LIMIT="SFT_RL_SMOKE=1"
  EVAL_TAG="${EVAL_TAG}_smoke"
fi

echo "== [A1] geo3k pass@1 (601 rows, temp 0) =="
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" scripts/sft_rl/eval_geo3k.py \
  --model "${EVAL_CKPT}" --data "${GEO3K_DATA}" \
  --out "${OUT_DIR}/geo3k_p1.jsonl" ${ROWS_ARG} \
  --max-response-length 2048 --batch 16 --reward geo3k \
  > "${OUT_DIR}/a1_geo3k_p1.log" 2>&1; tail -3 "${OUT_DIR}/a1_geo3k_p1.log"

echo "== [A2] geo3k pass@8 (601 rows, temp 0.7 x8) =="
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" scripts/sft_rl/eval_geo3k.py \
  --model "${EVAL_CKPT}" --data "${GEO3K_DATA}" \
  --out "${OUT_DIR}/geo3k_p8.jsonl" ${ROWS_ARG} --rollouts 8 \
  --max-response-length 2048 --batch 8 --reward geo3k \
  > "${OUT_DIR}/a2_geo3k_p8.log" 2>&1; tail -3 "${OUT_DIR}/a2_geo3k_p8.log"

echo "== [A3] MMF val pass@1 (500 rows, temp 0) =="
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" scripts/sft_rl/eval_geo3k.py \
  --model "${EVAL_CKPT}" --data "${MMF_DATA}" \
  --out "${OUT_DIR}/mmf_p1.jsonl" ${ROWS_ARG} \
  --max-response-length 4096 --batch 8 --reward mmf --max-model-len 16384 \
  > "${OUT_DIR}/a3_mmf_p1.log" 2>&1; tail -3 "${OUT_DIR}/a3_mmf_p1.log"

echo "== [A4] MMF val pass@8 (500 rows, temp 0.7 x8) =="
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" scripts/sft_rl/eval_geo3k.py \
  --model "${EVAL_CKPT}" --data "${MMF_DATA}" \
  --out "${OUT_DIR}/mmf_p8.jsonl" ${ROWS_ARG} --rollouts 8 \
  --max-response-length 4096 --batch 4 --reward mmf --max-model-len 16384 \
  > "${OUT_DIR}/a4_mmf_p8.log" 2>&1; tail -3 "${OUT_DIR}/a4_mmf_p8.log"

# 格式合格率：直接从 pass@1 输出统计（<answer>/boxed/ANSWER: 抽取成功率 = 格式合格率）
"${PY}" - "${OUT_DIR}" <<'EOF'
import json, sys, re
out = sys.argv[1]
pat = re.compile(r"\\boxed\{|<answer>|answer\s*[:：]", re.I)
for name in ("geo3k_p1", "mmf_p1"):
    try:
        rows = [json.loads(l) for l in open(f"{out}/{name}.jsonl")]
    except FileNotFoundError:
        continue
    fmt_ok = sum(1 for r in rows if pat.search(r.get("solution", "")))
    acc = sum(r["score"] for r in rows) / len(rows)
    print(f"[{name}] n={len(rows)} pass@1={acc:.4f} format_compliance={fmt_ok/len(rows):.4f}")
EOF

# B 段可通过 SFT_RL_BENCHMARKS / SFT_RL_JUDGE_BENCHMARKS 传入（两者都未设置时跑默认
# 六项；显式置空其一且另一个也为空则跳过 B 段只跑 A 段 pass@k —— 多 ckpt 快速对比用）
if [[ -z "${SFT_RL_BENCHMARKS+x}${SFT_RL_JUDGE_BENCHMARKS+x}" || -n "${SFT_RL_BENCHMARKS-}" || -n "${SFT_RL_JUDGE_BENCHMARKS-}" ]]; then
echo ""
echo "== [B] 项目六项 benchmark（judge=Qwen3-VL-32B-FP8 on GPU${JUDGE_GPU}）=="
if [[ -n "${BENCH_LIMIT}" ]]; then eval "export ${BENCH_LIMIT}"; fi
cd "${REPO_ROOT}"
# B 段起 vLLM server 需要空闲 GPU：与其他 ladder run 并行时错开 eval/judge GPU 和端口
SFT_RL_MODEL_HF="${EVAL_CKPT}" \
SFT_RL_RUN_NAME="${EVAL_TAG}" \
SFT_RL_EVAL_GPU="${EVAL_GPU}" SFT_RL_JUDGE_GPU="${JUDGE_GPU}" \
SFT_RL_EVAL_PORT="${BENCH_EVAL_PORT}" SFT_RL_JUDGE_PORT="${BENCH_JUDGE_PORT}" \
  bash scripts/eval/run_target_benchmarks.sh
fi

echo "== DONE. pass@k: ${OUT_DIR}; benchmarks: ${DTOPD_ROOT}/eval_runs/vision_opd_project_baseline/${EVAL_TAG}_* =="
