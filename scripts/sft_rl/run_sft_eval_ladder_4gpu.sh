#!/usr/bin/env bash
#
# SFT-RL 评测阶梯（4 卡加速版）——与 run_sft_eval_ladder.sh 严格同口径，仅 A 段并行化。
#   A. 保留集 pass@1 / pass@8（规则判分，无需 judge）:
#      - geo3k 官方测试集 601 行（held-out）
#      - MMF RL val 500 行
#      四个子任务（geo3k_p1 / geo3k_p8 / mmf_p1 / mmf_p8）由 run_sft_eval_ladder.sh
#      在同一张卡上串行跑；本脚本把它们分摊到 4 张卡并行（进程内 vLLM，tp=1，
#      每次请求独立，无跨卡依赖），接近 4× 加速——pass@8 是耗时大头，收益最高。
#   B. 项目六项 benchmark（GQA/DynaMath/ViewSpatial/MMMU-Pro/ReMI 规则 + MMBench judge）:
#      沿用原 2 卡架构（1 卡 8B eval :8000 + 1 卡 32B judge :8001）。六项对同一个
#      eval server 顺序跑、--workers 8 已打满单卡吞吐，且 8B 上 tensor-parallel>1 通常
#      更慢，故 B 段保持 2 卡原样，A 段跑完（wait 返回）后再起，避免抢卡。
#
# 用法（GPU 实例，4 卡 0,1,2,3）:
#   EVAL_CKPT=<hf-dir> EVAL_TAG=ptd_grpo_sft6939_coef5e4 \
#     bash scripts/sft_rl/run_sft_eval_ladder_4gpu.sh
#   SMOKE=1 ... bash scripts/sft_rl/run_sft_eval_ladder_4gpu.sh   # 冒烟（各 8 行/8 条）
#   A_BLOCK_GPUS=0,1,2,3 EVAL_GPU=0 JUDGE_GPU=1 ...                # 显式指定卡位
#
# 产物（与 run_sft_eval_ladder.sh 完全同名，不影响四臂对表口径）:
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
# A 段 4 卡卡位（本机只有 0-3；不要写 4,5,6,7）
A_BLOCK_GPUS="${A_BLOCK_GPUS:-0,1,2,3}"
# B 段 2 卡卡位（eval 8B / judge 32B）
EVAL_GPU="${EVAL_GPU:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
BENCH_EVAL_PORT="${BENCH_EVAL_PORT:-8000}"
BENCH_JUDGE_PORT="${BENCH_JUDGE_PORT:-8001}"
OUT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/eval/${EVAL_TAG}"
SMOKE="${SMOKE:-0}"

IFS=',' read -r -a A_GPU <<< "${A_BLOCK_GPUS}"
[ "${#A_GPU[@]}" -ge 4 ] || { echo "FATAL: A_BLOCK_GPUS 需至少 4 个卡号，实际「${A_BLOCK_GPUS}」"; exit 1; }

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
  OUT_DIR="${DTOPD_ROOT}/fc-opd-storage/outputs/fc_opd/sft_rl/eval/${EVAL_TAG}"
  mkdir -p "${OUT_DIR}"
fi

declare -a PIDS=()

echo "== [A] 4 卡并行：A1/A2 geo3k(卡${A_GPU[0]}/${A_GPU[1]})  A3/A4 mmf(卡${A_GPU[2]}/${A_GPU[3]}) =="

echo "== [A1] geo3k pass@1 (601 rows, temp 0) — GPU${A_GPU[0]} =="
CUDA_VISIBLE_DEVICES="${A_GPU[0]}" "${PY}" scripts/sft_rl/eval_geo3k.py \
  --model "${EVAL_CKPT}" --data "${GEO3K_DATA}" \
  --out "${OUT_DIR}/geo3k_p1.jsonl" ${ROWS_ARG} \
  --max-response-length 2048 --batch 16 --reward geo3k \
  > "${OUT_DIR}/a1_geo3k_p1.log" 2>&1 &
PIDS+=($!)

echo "== [A2] geo3k pass@8 (601 rows, temp 0.7 x8) — GPU${A_GPU[1]} =="
CUDA_VISIBLE_DEVICES="${A_GPU[1]}" "${PY}" scripts/sft_rl/eval_geo3k.py \
  --model "${EVAL_CKPT}" --data "${GEO3K_DATA}" \
  --out "${OUT_DIR}/geo3k_p8.jsonl" ${ROWS_ARG} --rollouts 8 \
  --max-response-length 2048 --batch 8 --reward geo3k \
  > "${OUT_DIR}/a2_geo3k_p8.log" 2>&1 &
PIDS+=($!)

echo "== [A3] MMF val pass@1 (500 rows, temp 0) — GPU${A_GPU[2]} =="
CUDA_VISIBLE_DEVICES="${A_GPU[2]}" "${PY}" scripts/sft_rl/eval_geo3k.py \
  --model "${EVAL_CKPT}" --data "${MMF_DATA}" \
  --out "${OUT_DIR}/mmf_p1.jsonl" ${ROWS_ARG} \
  --max-response-length 4096 --batch 8 --reward mmf --max-model-len 16384 \
  > "${OUT_DIR}/a3_mmf_p1.log" 2>&1 &
PIDS+=($!)

echo "== [A4] MMF val pass@8 (500 rows, temp 0.7 x8) — GPU${A_GPU[3]} =="
CUDA_VISIBLE_DEVICES="${A_GPU[3]}" "${PY}" scripts/sft_rl/eval_geo3k.py \
  --model "${EVAL_CKPT}" --data "${MMF_DATA}" \
  --out "${OUT_DIR}/mmf_p8.jsonl" ${ROWS_ARG} --rollouts 8 \
  --max-response-length 4096 --batch 4 --reward mmf --max-model-len 16384 \
  > "${OUT_DIR}/a4_mmf_p8.log" 2>&1 &
PIDS+=($!)

# 逐个 wait + 报告失败 job（flashinfer 首次多模态请求的 SM90 JIT 冷启动可能撞车；
# 谁报错单跑该条命令即可，--out 会覆盖重写，不污染数据）
names=(a1_geo3k_p1 a2_geo3k_p8 a3_mmf_p1 a4_mmf_p8)
dead=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "ALERT: ${names[$i]} 非零退出，重跑该条即可（tail ${OUT_DIR}/${names[$i]}.log）" >&2
    dead=1
  fi
done
[ "$dead" = "0" ] && echo "== A 段 4 卡全部完成 =="

for i in "${!PIDS[@]}"; do
  echo "--- ${names[$i]} ---"; tail -3 "${OUT_DIR}/${names[$i]}.log"
done

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
# 注意：A 段刚把 4 张卡都用满，B 段在此处才起 vLLM server，不抢卡。
if [[ -z "${SFT_RL_BENCHMARKS+x}${SFT_RL_JUDGE_BENCHMARKS+x}" || -n "${SFT_RL_BENCHMARKS-}" || -n "${SFT_RL_JUDGE_BENCHMARKS-}" ]]; then
echo ""
echo "== [B] 项目六项 benchmark（judge=Qwen3-VL-32B-FP8 on GPU${JUDGE_GPU}，eval GPU${EVAL_GPU}）=="
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