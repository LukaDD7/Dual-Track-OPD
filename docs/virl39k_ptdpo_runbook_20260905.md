# ViRL39K PTD-PO（RL-only，基模型起步）复现 Runbook（2026-09-05）

复现 arXiv:2606.07000（PTD-PO: Privileged Tutoring Distillation Policy
Optimization）于 ViRL39K，直接从基模型 Qwen3-VL-8B-Instruct 起步（RL-only，
无 SFT warmup）。hint 由本地 Qwen3.6-35B-A3B teacher 离线生成（论文未公开
hint 数据，须自生成）。

- 数据：train = FULL `virl39k/`（26 shard / 38,348 行，GT 四类与
  `mmf_reward.py` 四门完全匹配）；val = PAPO_MMK12_test（B 方案，2,000 行
  全 MCQ letter，对齐论文 MMK12 评测口径，train/val 构造性不重叠）
- 超参（论文 8B 值）：`algorithm.ptd coef=5e-2 top_k=100 threshold=1.0
  kl_direction=jsd_kl use_ref_teacher=True`；lr=1e-6；batch=16 / mini=8 /
  token=24576；rollout_n=8；save_freq=25 / test_freq=10；total_epochs=1
- 执行入口：`scripts/sft_rl/run_grpo_mmf_ptd.sh`（GPU 命令清单：
  `$DTOPD/manuscript-sft-rl-gpu`；hint 生成管线见该清单第 1–2 步）

本文件记录 2026-09-05 的两次启动事故（0131 死亡 + 0736 resume 陷阱 +
**0736 于 10:53 重复同一死法**）、r3 的两次事故（13:59 narwhals import
崩 + 16:27 step 9 冻结死亡、实例回收）、根因结论、以及 r4 重启方案
（v2：nohup + watchdog 调度 + 健康金丝雀）。

## 事故时间线（两个 run）

| 时间 (09-05) | 事件 |
|---|---|
| 01:31 | run A 启动：`qwen3vl_virl39k_ptd_base_4gpu_20260905_0131`（名下简称 **0131**） |
| 02:12 | 0131 过 init，step:1 开始训练 |
| 03:51 | 0131 存 `global_step_25`（tracker=25，checkpoint 完整） |
| 04:27:02 | 0131 step:35 banner，全指标正常（GPU 116/150GB、host 内存 308/1658GB、response 1352 tok） |
| 04:27–04:29 | **Ray 控制面（GCS/raylet）无声死亡**（详见下节）；driver 挂起 |
| 05:39:27 | 0131 log 末次 flush（与 TailSFT 启动序列的 `ray stop --force` 吻合——收尸者，非根因）；171MB 日志冻结 |
| 07:36 | run B 启动：`..._resume_20260905_0736`（名下简称 **0736**；命令带新实验名 = 触发 resume 陷阱，详见下节） |
|---|---|
| 07:57 | 0736 log 第 192294 行：`Checkpoint tracker file does not exist` → **`Training from scratch`** |
| 08:12 | 0736 过 init，step:1 + 首轮 val 完成；此后健康训练（~155–170s/step，0 ERROR） |
| 08:33 | 0736 到 step:9；`critic/score/mean` 0.57–0.96 波动、`ptd/mask_ratio` 0.16–0.43、response ~1700 tok —— 均正常量级 |
| 09:23:33 | 0736 一次 Ray dashboard blip（`Cannot reach the node` + raylet 心跳失联 2 条），**自愈**，训练不受影响 |
| 10:17:45 | 0736 第二次同型 blip，自愈 |
| 10:18:11 | 0736 step:20 banner + val（`val-core/MMK12_test/reward/mean@1=0.6599`；本轮 val 3910s 异常偏长，见下节） |
| 10:18:45–10:53:34 | **0736 重复 0131 死法：Ray 控制面无声死亡，driver 挂起**。末条带时间戳日志 10:18:45；此后 35 分钟只有无时间戳的 Ray event-stats 堆栈静默写入，10:53:34 完全冻结。**全 run 零 checkpoint（save_freq=25 从未到达，ckpt 目录不存在）** |
| 11:10+ | Claude 侧确认（三次 60/30/30s mtime 采样全冻结）：log 停止增长，`training/global_step` 停在 20 |
| 13:2x | 根因判别闭环（块 1–4）：cgroup OOM / 重启 / Ray 自身崩溃全部排除 → 外部 SIGKILL 级杀手点名 raylet+gcs_server（详见「根因判别结果」表） |
| 13:59:42 | **r3 首启数秒即崩**：`ModuleNotFoundError: No module named 'narwhals'`（transformers→sklearn 1.9.0→narwhals import 链；8.8KB 日志单栈，Ray 未起）——**非 SIGKILL 死法**。根因 = 训练 env 13:04–14:10 被 pip 会话持续装 ~25 个包（sklearn 13:45 → narwhals 14:03 → threadpoolctl 14:10），r3 恰好撞进 sklearn-已升级/narwhals-未装的破窗。Claude 14:22–14:32 全链验证 import 全 OK、env 14:10 后无新写入 → 同名重跑即恢复（事故记录：`docs/VA-OPD_BUG_TRACKER.md` #23） |
| 15:04:06 | **r3 二次启动成功**：`qwen3vl_virl39k_ptd_base_4gpu_20260905_r3`（简称 **r3**，新实验名保住前 run 日志）。"Training from scratch"（前任全零/废 ckpt）。init val（step:0）= **0.6473**；此后健康训练 ~207 s/step |
| 15:04–16:27 | r3 健康：真实 step 1–9（须用 `training/global_step:N` 解析，banner 里 `timing_s/step:213` 是每步耗时不是步号）；score 0.55–0.95、`ptd/mask_ratio` 0.17–0.25、response 1700–2900 tok、host 内存口径稳定 286–291 GB、日志恒定 36 行/分钟。**watchdog 从未安装** |
| 16:27:10 | **r3 死于 step 9，零 ckpt**（差 16 步到 save_freq=25）。死法同签名：末条时间戳 16:27:10 后只有 event-stats 尾巴（仅 ~2 秒，0736 是 35 分钟）+ 零 ERROR/SIGTERM。尾巴极短 → 不排除 16:27 那一刻是平台直接回收整机 |
| 09-06 晨 | **实例晚间被平台自动回收**（r3 死后 GPU 空转触发；用户确认）。死亡链：死 → 无 watchdog 无人重启 → GPU 空转 → 整机被回收，全损。→ v2 重启清单加入 watchdog 必装 + 健康金丝雀（`$DTOPD/manuscript` 块 0b/2） |
| 09-06 | 用户反馈两点：① 历史上类似症状后来排查是自己代码的问题 → 根因判别重开为待仲裁（金丝雀 RSS/oom_kill 曲线裁决"自己代码泄漏 vs 外部杀手"）；② GPU 实例**无 tmux、无 crontab** → 启动改 nohup、watchdog 调度改 nohup 死循环（cron 亦为实例本地，换实例必重装） |

## 0131 为什么停（终版结论）

**Ray 控制面（GCS/raylet）在 04:27–04:29 之间无声死亡，训练侧零错误。**
证据链（`logs/qwen3vl_virl39k_ptd_base_4gpu_20260905_0131.log`，171MB）：

1. 04:27:02 step:35 banner 全指标正常，资源远离上限
   （`perf/cpu_memory_used_gb`=308 是 **host 全机口径**，见
   `engine_workers.py:221` 用 `psutil.virtual_memory()`，非进程用量）；
2. 04:27:48 起 `reporter_agent.py:932/940 -- Failed to get worker/agents
   pids from raylet`（raylet 心跳失联）；
3. 04:29:02 最后一条 `ERROR (dashboard) node_head.py:499 -- Cannot reach
   the node ... after timeout 14`，之后日志冻结；
4. **无 SIGTERM 痕迹**——对比 09-04 07:07 那次 `ray stop` 杀进程有明确
   `raylet received SIGTERM` 字样，这次一个都没有 → SIGKILL 级死法；
5. driver 挂起 ~70 分钟后，05:39:27 末次 flush 与 TailSFT 启动序列
   （其 launcher 开头 `ray stop --force`）时间吻合 → **TailSFT 只是收尸者，
   非根因**（TailSFT 本身健康跑到 08:04 才因 GPU3 OOM 崩，另一条独立事故）。

### 根因判别结果（2026-09-05 13:00，用户在盒上执行块 1–4 后闭环）

| 检查 | 结果 | 排除/指向 |
|---|---|---|
| `/sys/fs/cgroup/memory.events` | low/high/max/oom/oom_kill **全零** | 排除 cgroup OOM kill |
| `uptime -s` | 2026-07-24 08:09（43 天） | 排除容器/宿主重启 |
| `ps aux` | raylet 与 gcs_server 均为 `<defunct>` 僵尸，main_ppo 挂起未死 | 两个控制面进程被 **SIGKILL 级强杀**，driver 只是受害者 |
| session `logs/` | 无 raylet.err/gcs_server.err（Ray 0.9 控制面日志走 stdout，已查训练日志：末条 raylet 04:28:44 / gcs 04:28:03，无崩溃栈） | 排除 Ray 自身崩溃 bug 的直接证据 |
| 块 4 收尸 | `pkill` + `ray stop --force` 清 8 进程，GPU 4–7 降至 ~3.1GB 残留 | 盒面已清干净，可重启 |

**定性：盒上存在周期性发作（40–90 分钟一次）的外部杀手，精确点名
raylet + gcs_server（SIGKILL，非 OOM、非重启、非 Ray bug）。** 两次死亡
（04:27 / 10:17–10:53）+ 四次自愈 blip（02:51/03:10/04:20/09:23/10:17）
同签名。最大嫌疑 = 平台 per-进程资源上限（RSS/线程数），raylet/gcs 恰是
进程树里 RSS 最高的两个常驻进程。dmesg 在容器内无权限（无 CAP_SYSLOG），
无法拿到内核级 kill 记录做最终实锤。

未闭环的可选拼图：两个 dead session 的 `logs/old/`（Ray 日志轮转目录）
与 `debug_state.txt` 尚未查；若 `old/` 里有轮转的 raylet.err 尾巴，
可见死前最后几行。命令已写入 `$DTOPD/manuscript` 块 5。

## 0736 也死了（10:53，同型死法，终版结论）

**0736 在 step:20 之后（10:18:45–10:53:34）重复了 0131 的 Ray 控制面
无声死亡，driver 挂起，全 run 零 checkpoint。** 证据链
（`logs/qwen3vl_virl39k_ptd_base_4gpu_resume_20260905_0736.log`，110MB）：

1. 10:18:11 step:20 banner + val 全指标正常
   （`val-core/MMK12_test/reward/mean@1=0.6599`，score 0.64，
   response 2777 tok，step 170.6s，GPU 114/137GB alloc）；
2. 末条带时间戳日志 **10:18:45**（`Plasma store debug dump: 0.31/200 GB`，
   0 pending objects —— 无 object-store 压力）；
3. 此后 35 分钟只有**无时间戳**的 Ray event-stats 堆栈静默写入
   （2548 行，全为 worker 侧 `Global stats` dump —— 这是 Ray 心跳/查
   询线程在控制面失联后的固定输出模式），10:53:34 完全冻结；
4. 全 run 无任何 SIGTERM/OOM/Traceback 级致命错误（3 条 ERROR 全为
   09:23/10:17 的自愈型 dashboard blip）；
5. **ckpt 目录不存在** —— `save_freq=25`，死在 step 20，差 5 步到
   首个 checkpoint；
6. Claude 侧三次 mtime 采样（11:10/11:11/11:14，间隔 60/30/30s）
   全冻结，`training/global_step` 停在 20。

**两次死亡相隔 ~6 小时，同一签名（raylet/GCS 无声死亡 + 无 SIGTERM +
event-stats 堆栈尾巴），且 0736 的两次 09:23/10:17 自愈 blip 与 0131
健康期 02:51/03:10/04:20 的 blip 同型 → 这是周期性发作的宿主级问题，
不是一次性事故。** 每次发作间隔 40–90 分钟。在根因（盒上 dmesg//tmp/ray
两查）闭环之前，任何重启都大概率在 1–3 小时内再次死亡，且死点随机
（0736 死在首个 ckpt 之前 = 全损）。

### 与 val 的相关性（弱线索，非结论）

两次死亡时 driver 都处于或刚过 val 相关阶段：0131 死在 step 35（下一
个 val 在 step 40）；0736 死在 step 20 val 完成、进入 step 21 之后。
且 0736 死前那轮 val 耗时 3910s（0131 同期 step 10/20/30 的 val 只要
617–740s），5 倍偏长 —— val 期间 vLLM 并发生成会拉高控制面 RPC 压力。
但 0131 的死亡点（step 35）不在 val 中，故 val 只能算放大器候选，
不是充要条件。

### 重启决策（2026-09-05 13:2x 修订版，含磁盘预算）

**新发现的硬约束——磁盘预算**：实测 0131 的 `global_step_25` 单个 ckpt
= **98G**（actor：8.2G×4 model 分片 + 17G×4 optimizer 分片 + 12M
huggingface/ 只有 config/tokenizer 无权重）；盘剩 792G / 10T（93% used）；
`run_grpo_mmf_ptd.sh` 原先**没有任何 ckpt 保留上限**——2397 步全跑完
save_freq=25 会写 ~96 份 ≈ 9.2TB，必然撑爆盘。原路线 2 的
`TRAINER_SAVE_FREQ=5` 更糟（~step 40 即爆）。

修订版路线 2（已写入 `$DTOPD/manuscript`，推荐）：

1. `TRAINER_SAVE_FREQ=25` 不变（每 25 步 ≈80 分钟一个 ckpt）；
2. **新增 `trainer.max_actor_ckpt_to_keep=2`**——verl 0.9 后端原生支持
   （`checkpoint_manager.py` 的 `ensure_checkpoint_capacity`/
   `register_checkpoint` 实现删除逻辑，`ray_trainer.py:1011` 从
   `config.trainer.max_actor_ckpt_to_keep` 读取；`ppo_trainer.yaml:249`
   默认 null = 不清理）。补丁方式：sed 给
   `run_grpo_mmf_ptd.sh` 的 hydra 覆写块插一行（manuscript 块 6，
   幂等）。峰值占用 = 2×98G = ~196G，磁盘安全；任一时刻都有完整
   ckpt 可 resume，死亡窗口全损上限 = 25 步 ≈80 分钟；
   ★ 注意：`previous_saved_paths` 初始为空、resume 不回填，保留计数
   从本次进程的首次 save 起算——重启后第 1、2 个 ckpt 累积，第 3 个
   save 时才触发删旧，属预期行为；
3. watchdog（cron 每分钟查 log mtime，冻结 >10 分钟 → 杀 + 同名重启，
   `resume_mode=auto` 从最新 ckpt 续训）；
4. 评测注意事项：RL ckpt 的 `huggingface/` 子目录**只有
   config/tokenizer 没有权重**——必须用
   `scripts/sft_rl/export_hf_from_grpo_ckpt.sh`（model_merger merge）
   从 FSDP 分片导出 HF 权重后再跑 eval ladder（此前 grpo50 等 RL 臂
   评测即此流程）。

若平台确认 per-进程资源上限（如 raylet RSS 上限），找平台调高或换
实例（原路线 3）；换节点前可先用修订路线 2 硬扛验证。

## 0736 命中的 resume 陷阱（教训）

`trainer.resume_mode=auto` 的查找逻辑（`ray_trainer.py:1057 _load_checkpoint`
→ `checkpoint_manager.py:219 find_latest_ckpt_path`）**只读当前实验名目录下
的 `latest_checkpointed_iteration.txt`**。换实验名重启 = tracker 不存在 =
"Training from scratch"，不会去找旧 run 的 checkpoint。

- 0736 的 `SFT_RL_NAME=..._resume_20260905_0736` 触发该陷阱 → 从 step 0
  全新重跑（新 lineage：独立 ckpt 目录、优化器/step 计数从零；起点/数据/
  超参与 0131 完全相同，仅浪费 25 步 ≈80 分钟，科学上无差别）。
- 用户拍板（2026-09-05 08:35）：**让 0736 继续跑完**，结束后统一改文件/目录名。
- 0131 的 `global_step_25` 作废弃处理（不再续跑，留作事故证据）。

## 0736 收尾清单（已于 10:53 死亡，清单改写为善后版）

> 08:35 的"让它跑完+改名"决策前提已失效：0736 没跑到 25 步首个
> checkpoint 就死了，没有可改名的训练产物。原清单第 1/2/4 条作废，
> 当前状态 = 零 checkpoint + 110MB 死日志。

1. ~~健康结论记录~~（作废：无终态可记。死前终值 = step 20，
   val 0.6599，score mean 0.64，response 2777 tok，mask_ratio 0.297）。
2. ~~改名~~（作废：ckpt 目录从未创建；log 若想去 `resume_` 可直接
   `mv`，无科学影响）。
3. **0131 + 0736 残留处置**（已完成）：块 4 收尸已执行（pkill +
   `ray stop --force`，8 进程清理，GPU 4–7 降至 ~3.1GB）。两个 run 的
   log 留作事故证据；0736 的 log 是第二次同型死亡证据，勿删。
   0131 的 `global_step_25`（98G）可删省盘，但建议**等 r3 首个 ckpt
   落地后再删**（它是目前唯一的「PTD-PO save 路径完整性」实锤样本）。
4. ~~盒上两查~~（已完成，结论见「0131 为什么停」的根因判别结果表；
   剩余可选拼图 = session `logs/old/`，命令在 manuscript 块 5）。
5. **重启**：按修订路线 2 执行（manuscript 块 5–9：可选尸检 →
   sed 补丁 max_actor_ckpt_to_keep=2 → 启动 r3 → 挂 watchdog）。
6. **评测**：r3 跑完先用
   `scripts/sft_rl/export_hf_from_grpo_ckpt.sh` 导 HF 权重，再跑
   eval ladder（对表论文 PTD-PO 8B 71.86 / GRPO 68.78）。

## 产物清单

| 产物 | 路径 |
|---|---|
| 0736 train ckpt | **不存在**（死于 step 20，save_freq=25 未到达，全损） |
| 0736 train log | `$DTOPD/fc-opd-storage/logs/qwen3vl_virl39k_ptd_base_4gpu_resume_20260905_0736.log`（110MB，事故证据 #2） |
| 0131 事故 ckpt | `.../ckpt/qwen3vl_virl39k_ptd_base_4gpu_20260905_0131/global_step_25`（98G，作废弃但暂留：r3 首个 ckpt 落地后可删） |
| 0131 事故 log | `$DTOPD/fc-opd-storage/logs/qwen3vl_virl39k_ptd_base_4gpu_20260905_0131.log`（171MB，事故证据 #1） |
| hint 数据 | `.../sft_rl/virl39k_hint/virl39k_rl_train_hint__part_*.parquet`（26 分片） |
| val 数据 | `.../sft_rl/virl39k_mmk12_test/virl39k_rl_val__part_0000.parquet`（2,000 行） |
| GPU 命令清单 | `$DTOPD/manuscript`（13:2x 版 = 根因结论 + 修订路线 2 重启块 5–10）；`$DTOPD/manuscript-sft-rl-gpu`（三步走原始版） |

## 2026-09-08 r4 实际进度与恢复点

### 终态审计

| 项目 | 结果 |
|---|---|
| 当前活跃进程 | 无（Ray/vLLM/main_ppo 均不在跑） |
| 最后观察到的训练 step | **399**（`training/global_step:399`；step 400/val 未完成） |
| 最新完整 checkpoint | **`global_step_390`**（`latest_checkpointed_iteration.txt=390`） |
| checkpoint 体积 | `global_step_390` ≈ **98G**（4×8.77G model 分片 + 4×17.54G optimizer 分片 + 元数据） |
| 训练冻结时间 | 2026-09-07 **17:32:51**；watchdog/canary 末条 17:32:41 |
| 日志尾部形态 | Ray event-stats / autoscaler 输出，无正常完成、Traceback、SIGTERM 或 OOM 栈 |
| 最近 val | step 390 `val-core/MMK12_test/reward/mean@1=0.6776316` |
| 本段最好 val | step 190 `0.7231781`（step 250–390 在 0.676–0.699 区间波动） |
| 预期总步数 | ~2397（38,348 rows / batch 16，1 epoch；`TOTAL_TRAINING_STEPS=null`） |
| 完成度 | 观察到 399/2397 ≈ **16.6%**；可恢复训练从 390/2397 ≈ **16.3%** |

r4 lineage 不是从 1 连续跑到 399：现有 dead log 显示 2026-09-06 23:56 进程从
`global_step_25` 恢复；2026-09-07 01:59 进程从 `global_step_170` 恢复，随后
继续到 step 399。checkpoint 管理器在保存 390 后删除了 370 的 actor 权重
（目录仅剩 ~129K 元数据），当前可用恢复点只有 390。该行为与
`max_actor_ckpt_to_keep=2` 的清理逻辑及“目录保留标记/旧目录清理不同步”的后端
实现一致；恢复时必须读 tracker，不要按目录名猜。

### r4 训练 recipe（已提交入口）

执行入口为 `scripts/sft_rl/run_grpo_mmf_ptd.sh`。r4 的关键 resolved 配置：

| 组件 | 值 |
|---|---|
| student | `$DTOPD/models/Qwen3-VL-8B-Instruct`（base，无 SFT warmup） |
| GPUs | 4×（`SFT_RL_GPUS=0,1,2,3`；Ray 单机 4 卡） |
| backend | `verl-qwen35-v090-cu132` @ `483b8a0`，工作区 dirty；diff SHA256 `542bdc28…` |
| runtime | torch `2.13.0+cu132`，transformers `5.12.0`，Ray `2.55.1`，vLLM `0.27.1` |
| train | `virl39k_hint/virl39k_rl_train_hint__part_0000..0025.parquet`（26 shard / 38,348 rows / hint ratio 0.8009） |
| val | `virl39k_mmk12_test/virl39k_rl_val__part_0000.parquet`（2,000 rows，全 MCQ letter） |
| batch | `TRAIN_BATCH_SIZE=16`，`PPO_MINI_BATCH_SIZE=8`，`ROLLOUT_N=8` |
| length | prompt 2048，response 12288，actor token budget 24576 |
| optimizer | actor lr `1e-6`；KL loss coef `0.01`，`low_var_kl` |
| PTD | `enable=true`，`coef=5e-2`，`top_k=100`，`threshold=1.0`，`kl_direction=jsd_kl`，`all_trajectories=false`，`use_ref_teacher=true` |
| rollout | vLLM TP=1，`gpu_memory_utilization=0.45`，free cache engine |
| checkpoint | `save_freq=25`，`max_actor_ckpt_to_keep=2`，`test_freq=10`，`resume_mode=auto` |
| fused kernels | `SFT_RL_FUSED_KERNELS=0`（PTD top-K 需要 eager logits 输出） |

同名恢复命令（必须保持 experiment name 不变；改名会触发 “Training from
scratch” resume 陷阱）：

```bash
SFT_RL_NAME=qwen3vl_virl39k_ptd_base_4gpu_20260906_r4 \
SFT_RL_MODEL=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct \
SFT_RL_TRAIN=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/virl39k_hint/virl39k_rl_train_hint__part_*.parquet \
SFT_RL_VAL=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/virl39k_mmk12_test/virl39k_rl_val__part_0000.parquet \
SFT_RL_GPUS=0,1,2,3 \
TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=8 ROLLOUT_N=8 \
MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=12288 SFT_RL_ACTOR_TOKEN_BUDGET=24576 \
TOTAL_EPOCHS=1 TRAINER_SAVE_FREQ=25 TRAINER_TEST_FREQ=10 \
PTD_ENABLE=true PTD_COEF=5e-2 PTD_TOP_K=100 PTD_THRESHOLD=1.0 \
PTD_KL_DIRECTION=jsd_kl PTD_ALL_TRAJECTORIES=0 PTD_USE_REF_TEACHER=1 \
SFT_RL_FUSED_KERNELS=0 \
bash scripts/sft_rl/run_grpo_mmf_ptd.sh
```

评测前必须先从 FSDP 分片导出 HF 权重：
`scripts/sft_rl/export_hf_from_grpo_ckpt.sh`；RL ckpt 内的
`huggingface/` 子目录只有 tokenizer/config，没有完整权重。

### 数据与 hint 处理 recipe

1. **ViRL39K train 转换**：`scripts/sft_rl/convert_virl39k.py`
   - 输入：`$DTOPD/dataset/ViRL39K/39Krelease.parquet` + extracted `images/`
     tree；
   - 答案先去外层 `\boxed{...}`，再按 `letter / yesno / pure_number /
     has_number / other` 分类，默认丢 `other`；
   - 校验图片存在，读取图片 bytes，图片多于 `<image>` placeholder 时在
     question 前补 placeholder；placeholder 多于图片则丢样本；
   - 结果：38,870 → **38,348** kept rows，26 shard，`gt_other_dropped=522`，
     `placeholder_injected=850`，`missing_image=0`；
   - GT 分布：letter 12,389、pure_number 18,037、has_number 7,498、
     yesno 424、other 522；source 最大项 MMK12 12,661、Processed 12,652。

2. **MMK12 test val 转换**：`scripts/sft_rl/convert_mmk12_test.py`
   - 输入：`PAPOGalaxy/PAPO_MMK12_test` 的 `train-00000-of-00001.parquet`
     （HF split 名为 `train`，但语义是 test）；
   - 图片已内联为 `{bytes,path}`；沿用与 train 完全相同的 GT 分类和 verl
     RL schema；
   - 结果：**2,000** rows，全 letter，0 dropped；train/val 构造性不重叠。

3. **离线 hint 生成**：`scripts/sft_rl/serve_hint_gen.sh` +
   `scripts/sft_rl/build_mmf_hints.py`
   - teacher：本地 `qwen3.6-35B-A3B`，vLLM OpenAI-compatible server；
   - system prompt 版本 `b35bd247dd00`：只给空间/推理方向、抑制 distractor，
     严禁泄漏答案、选项字母、关键中间数字或 chain-of-thought；
   - 生成参数：`max_new_tokens=768`，temperature 0.6，retry temperature 0.9，
     `max_chars=2048`，`max_decisive_overlap=3`，seed 42；
   - 硬 QC 会拒绝并重试：empty、too long、CoT degenerate、answer letter/
     word/numeric/substring/component leak、degenerate output；
   - 最终 hint ratio **0.8009**（约 30,713/38,348 rows 带可用 hint）；
   - 输出 schema = 原 RL schema + `hint`、`prompt_with_hint`、
     `hint_reason`；PTD 训练时学生仍用 question-only rollout，teacher/ref
     在 hint-augmented context 下计算 top-K 目标。

小体积可审计 manifest（仅路径/行数/SHA256，不含数据本体）：
`data/manifests/ptdpo_virl39k_r4_20260908.jsonl`。原始 ViRL39K/MMK12、
hint parquet、训练 log 和 98G checkpoint 均在 `fc-opd-storage`，不入 Git。
