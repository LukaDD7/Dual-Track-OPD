# Native Qwen3-VL real-image 4-step canary — 2026-08-06

## 0. 结论

**PASS**。`va-opd-native-e003-cu128-r595-v1` + `verl-va-opd-e0031631-clean`
在真实 Geometry3K 图像上完成 4 步反向更新:exit 0、无 failure、loss/grad
全部 finite。STP-OPD 四臂 mechanics pilot 的前置训练 gate 已清除,
该环境从 `candidate` 转 `active`(registry 已更新)。

## 1. 运行信息

| 项 | 值 |
|---|---|
| 命令 | `run_va_opd_native.sh --objective opd --profile smoke --steps 4 --name canary_20260806 --visible-gpus 0,1,2,3,4,5 --allow-busy-gpus` |
| env / backend | `va-opd-native-e003-cu128-r595-v1`(torch 2.9.0+cu128、vLLM 0.12.0 source、TF 4.57.3)/ `verl-va-opd-e0031631-clean`(e0031631) |
| 模型 | student Qwen3-VL-4B-Instruct / teacher Qwen3-VL-32B-Instruct(TP2) |
| 数据 | `geometry3k_gkd/train.parquet`(1901 题,真实 images),prompt batch 4 × K=4 |
| repo commit | `0ff6edd`(dirty=false) |
| run 目录 | `fc-opd-storage/runs/va_opd_native/qwen3vl_geometry3k_native_opd_smoke_canary_20260806_20260806_095422/` |
| checkpoint | `fc-opd-storage/checkpoints/va_opd_native/qwen3vl_geometry3k_native_opd_smoke_canary_20260806_20260806_095422/` |

## 2. result.json 关键值

| 指标 | 值 |
|---|---:|
| completed_steps | 4 |
| exit_code | 0 |
| failures | [] |
| finite_distillation_loss_count | 8 |
| finite_gradient_count | 4 |
| last_distillation_loss | 0.1546 |
| last_actor_entropy | 0.275 |
| last_gradient_norm | 8.17 |
| last_response_length_clip_ratio | 0.25 |
| last_response_length_mean | 1015.5 |

## 3. 前置修复(preflight 曾阻断,已修并 push 至 `0ff6edd`)

`preflight_va_opd_native.py` 三处:`EXPECTED_BACKEND_CHANGES` 未定义
(删除早于 manifest 读取的错误检查);vllm/torch/transformers `+cuXXX` 后缀
归一化;manifest 未记录包(如 tensordict)跳过。修复后 preflight 在 CPU 与
GPU 实例均 PASS。

## 4. 下一步

1. 实现 `rescue_screen.py`(新 prompt 两阶段自适应 horizon 探测,32B teacher);
2. 235B 两阶段管线(`teacher_proposal_generate` 8 卡 TP + `student_proposal_rescore`),
   等 235B 下载完成(约 237.6GB,进行中);
3. STP-OPD 四臂 mechanics pilot(A0/A1/A2/A3,现有 7 个 rescue-positive prompt),
   把 `prefix_scaffold.py` 原语接入 verl 训练。
