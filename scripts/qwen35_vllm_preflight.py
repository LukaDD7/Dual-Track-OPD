"""GPU 预检：torch CUDA + vLLM 0.23 qwen3_5 引擎（vLLM spawn 需要真实文件作为 __main__）。

环境变量覆盖（均为可选）：
  PREFLIGHT_MODEL        默认 /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3.5-4B
  PREFLIGHT_TP           默认 1
  PREFLIGHT_GPU_MEM      默认 0.3
  PREFLIGHT_MAX_MODEL_LEN 默认 2049
  PREFLIGHT_MAX_TOKENS   默认 8
"""

import os
import sys

MODEL = os.environ.get(
    "PREFLIGHT_MODEL", "/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3.5-4B"
)
TP = int(os.environ.get("PREFLIGHT_TP", "1"))
GPU_MEM = float(os.environ.get("PREFLIGHT_GPU_MEM", "0.3"))
MAX_MODEL_LEN = int(os.environ.get("PREFLIGHT_MAX_MODEL_LEN", "2049"))
MAX_TOKENS = int(os.environ.get("PREFLIGHT_MAX_TOKENS", "8"))


def main() -> None:
    import torch

    assert torch.cuda.is_available(), "torch.cuda.is_available()=False"
    print(f"[preflight] torch {torch.__version__} cuda ok: {torch.cuda.get_device_name(0)}")

    # verl 的 unpad 路径必需 flash_attn（与 attn_implementation 无关）
    import flash_attn
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
    print(f"[preflight] flash_attn {flash_attn.__version__} bert_padding ok")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )
    out = llm.generate(["1+1=?"], SamplingParams(max_tokens=MAX_TOKENS))
    print(f"[preflight] vllm qwen3_5 engine OK ({MODEL}, TP={TP}, mem={GPU_MEM}, max_len={MAX_MODEL_LEN}):",
          out[0].outputs[0].text[:30])


if __name__ == "__main__":
    sys.exit(main())
