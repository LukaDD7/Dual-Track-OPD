# 候选训练集备选池：下载 / 转换 / 校验状态

日期：2026-09-02
目的：为后续 SFT / RL-only / SFT-then-RL / OPD-FKL 通道准备多套候选训练集，
下载到本地、做完整性校验、转成仓库 verl parquet schema（SFT 与 RL 两种），
确保之后可以直接接进训练 pipeline。

## 总览

| 数据集 | 下载体量 | 状态 | 产物 |
|---|---|---|---|
| ViRL39K | ~6.4 GB | ✅ 已转换校验 | RL 26 shard / 38,348 行 |
| MMFineReason-586K | ~26 GB | ✅ 已转换校验（已过 4 类 GT 过滤） | SFT 307 shard / 460,142 行 + RL 318 shard / 476,668 行 |
| LLaVA-CoT-100K | 171.8 GB（解压后） | ✅ 已转换校验 | SFT 66 shard / 98,580 行 |
| VisualWebInstruct | ~17 GB | ✅ 已转换校验 | SFT 65 shard / 97,056 行 + RL 53 shard / 78,510 行 |
| Vision-Flan（191-task-1k） | 36.32 GB zip | ✅ 已转换校验 | SFT 125 shard / 186,102 行 + RL 29 shard / 42,023 行 |
| MAmmoTH-VL-Instruct-12M | multi-image 201.93 GB / single-image 3.4 TB | 🔄 multi_image shard 并行下载中（8 并发） | 待下载完成后转换 |

## 已转换并校验的数据集（5 / 6）

统一输出根目录：`$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/sft_rl/<name>/`

SFT 分片 schema（verl SFT 输入合约）：
`messages`(list<struct<role,content>>), `images`(list<struct<bytes,path>>),
`source`, `pass_rate`(float64), `image_hash`(sha256[:16] 逗号连接)。

RL/GRPO 分片 schema：
`data_source`, `prompt`(list<struct>), `images`, `ability`(="reasoning"),
`reward_model`(struct{style:"rule", ground_truth}),
`extra_info`(struct{question, source, gt_source, gt_type, original_answer}),
`question`, `sample_uid`。

两套 schema 均已通过 `<image>` 占位符计数 == 图片数 断言（verl 硬约束），
图片 bytes 内嵌、`image_hash` 16 字符、`pass_rate=-1.0`。

### 1. ViRL39K（RL only）

- 转换脚本：`scripts/sft_rl/convert_virl39k.py`
- 产物：`virl39k_rl_train__part_*.parquet`（26 shard，38,348 行，26 中的 26 + 1 stats）
- GT 分布：letter 12,389 / pure_number 18,037 / has_number 7,498 / yesno 424，
  other 522 条被丢弃；`missing_image=0`，placeholder 注入 850 条。
- 关键列核对：`ph=1 imgs=1 OK`。

### 2. MMFineReason-586K（SFT + RL，含 4 类 GT 前置过滤）

- 转换脚本：仓库既有 MMFineReason 转换器（`sample_mmf_rl.py` 同源 `gt_type`）。
- 产物：
  - SFT：`mmfinereason_586k/mmfinereason586k_sft__part_*.parquet`（307 shard，460,142 行）
  - RL：`mmf586k_rl/mmf586k_rl_train__part_*.parquet`（318 shard，476,668 行）
- RL 4 类 GT 过滤：pure_number 149,974 / has_number 217,351 / letter 103,542 /
  yesno 5,801，`other` 91,175 条被丢弃（与仓库 MMFineReason 口径一致）。
- `answer_null=17,901`，`placeholder_mismatch=0`。

### 3. LLaVA-CoT-100K（SFT only）

- 转换脚本：`scripts/sft_rl/convert_llavacot_100k.py`
- 产物：`llavacot100k_sft__part_*.parquet`（66 shard，98,580 行）
- 输入 `train.jsonl` 98,582 行；`missing_image=2`（zip 内确无此 2 图），
  `placeholder_injected=98,580`（LLaVA 约定隐式单图 → 注入一个 `<image>`）。
- 图片完整性：zip 1,384,013 个文件条目 = 磁盘文件数，`unzip -t` 0 errors；
  377 个 0 字节文件为 zip 内真实空文件（vg=340, ocr_vqa=37），非损坏。
- 子来源：coco 32,606 / chartqa 17,008 / ai2d 11,397 / geoqa+ 11,364 / sqa 5,650 / vg 4,767 / ocr_vqa 4,448 / docvqa 3,978 / gqa 3,529 / textvqa 1,200 / …（18 类）。
- 说明：老师 CoT 为 target，GT 自由文本 → 仅 SFT，不做 RL。

### 4. VisualWebInstruct（SFT + RL）

- 转换脚本：`scripts/sft_rl/convert_visualwebinstruct.py`
- 产物：
  - SFT：`vwinstruct_sft__part_*.parquet`（65 shard，97,056 行）
  - RL：`vwinstruct_rl_train__part_*.parquet`（53 shard，78,510 行）
- 输入 360,781 行；过滤 `no_answer` 156,509（空 / "Not found"）、`no_images` 107,216。
- RL GT 过滤：has_number 77,055 / letter 499 / pure_number 709 / yesno 247，other 18,546 丢弃。

### 5. Vision-Flan（191-task-1k）（SFT + RL）

- 转换脚本：`scripts/sft_rl/convert_visionflan.py`
- 产物：
  - SFT：`visionflan_sft__part_*.parquet`（125 shard，186,102 行）
  - RL：`visionflan_rl_train__part_*.parquet`（29 shard，42,023 行）
- 输入 `annotation_191-task_1k.json` 186,103 行，191 task；`missing_image=0`，
  `empty_answer=1`（丢弃）。
- RL 4 类 GT 过滤：has_number 28,962 / yesno 7,653 / pure_number 5,401 / letter 7，
  `other` 144,079 丢弃（Vision-Flan 答案多为自由文本/单词计数，仅少数可规则校验）。
- 校验：SFT 首行 `ph=1 imgs=1`，RL 41 字 reward_model + `<image>` 计数一致。
- **下载坑（重要）**：首次用 v1 并行下载器对 hf-mirror 做 Range 恢复下载时，
  v1 用 `curl -r -o chunk`「续传」，但 `curl -o` 会**截断**文件，静默丢掉已取
  前缀 → 块长度正确但字节偏移错位 → 36.32 GB zip 中 45,201/181,394 条目 CRC 失败。
  - 修复：清空 chunk 目录后用 `scripts/hf_parallel_download_v2.py`（全有或全无
    `.tmp` + 原子改名）重下 → `unzip -t` 0 errors，181,394 文件全量校验通过。
  - **文件名坑**：非 ASCII 文件名在 zip/磁盘上被存成 Python 风格 `#Uxxxx` 转义
    （如 `#U91d1`），而 annotation JSON 里是原生 unicode。转换器里 `decode_ufname()`
    用 `#U[0-9a-f]{4}` 正则解码，把 annotation 的 `image` 名正确映射到磁盘文件。
  - ModelScope 上无该数据集镜像（`AI-ModelScope/vision-flan_191-task_1k` 404）。
  - hf-mirror 直链已确认支持 206 分段（`content-range`）。

### 6. MAmmoTH-VL-Instruct-12M（SFT only，multi_image 下载中）

- 转换脚本：`scripts/sft_rl/convert_mammothvl.py`（ijson 流式读取
  `mammoth_ov_2M.json` 是 JSON **数组**而非 JSONL，之前按行读会炸）。
- 规模（抽 200k 抽样）：no-image 40,623 / single-image 99,637 / multi-image 59,740，
  图片顶层前缀 M4-Instruct-Data 占绝大多数（265,571 引用）。
- 分 shard tar.gz：`multi_image_data` 20 shard ≈ 201.93 GB（下载中）；
  `single_image_data` 100 shard ≈ 3.4 TB（暂缓，等磁盘决策）；`video_data` 20 shard ≈ 119.69 GB。
- 下载：`AI-ModelScope/MAmmoTH-VL-Instruct-12M`（ModelScope 无外网直连，走内网
  `modelscope.hub.snapshot_download(repo_type='dataset', allow_patterns=['multi_image_data/*.tar.gz'], local_dir=<dataset dir>, max_workers=8)`）。
  实测 ModelScope 内网吞吐 ~330–500 kB/s（201.93 GB ≈ 5–6 天，8 并发并行）。
  备选：`hf-mirror` 直链（`MAmmoTH-VL/MAmmoTH-VL-Instruct-12M`）实测 ~230 kB/s，
  略慢。`modelscope download` CLI 单文件顺序下载比 `snapshot_download` 慢一个量级，弃用。
- 当前已下载 `mammoth_ov_2M.json`（4.05 GB）+ `mammoth_si_10M.json`（21.8 GB）
  两个索引。
- 下载完成后由 `scripts/sft_rl/mammoth_postdownload.sh`（detached 后台）自动接力：
  等 20 shard 全部 `.incomplete` 清零 → 逐 shard 比对 ModelScope 登记字节数 →
  解压到 `multi_image_data/extracted/`（顶层前缀 `M4-Instruct-Data/` 占 274 万引用，
  其余 `LLaVA-OneVision-Data` 18.8 万 / `Cambrian-10M` 15.5 万 / `tinychart_train` 12.4 万 …
  共 28 个前缀；含 `LLaVA-NeXT-Data`/`SVIT_*` 等跨 single/multi 前缀）→
  删 tar.gz（解压树等价，且图片 bytes 将内嵌进 parquet，稳态磁盘 = 树 207G + parquet 207G，
  避免与 202G tar 叠加）→ 跑 `convert_mammothvl.py` → 写 `multi_image_CONVERTED.ok`。
- 磁盘观察：tar.gz 压缩比对图片几乎不压缩（shard_14 压缩 0.85G ↔ 解压 0.87G），
  所以 `extracted` 树 ≈ tar.gz 总量 ≈ 207G；parquet 因 bytes 内嵌体积 ≈ 同样量级。
- GT 自由文本 → 仅 SFT。

## 存储位置

- 数据：`$DTOPD_ROOT/dataset/<DatasetName>/`（不入仓，符合 AGENTS git 卫生要求）
- 转换产物：`$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/sft_rl/<name>/`
- 转换脚本：`scripts/sft_rl/convert_*.py`（入仓）

## 校验方法（抽查）

每个产物做三类抽查：

1. parquet 全量可读 + 行数核对（`pq.ParquetFile(...).metadata.num_rows` 求和）
2. 首 shard 首行 schema 字段 + `<image>` 计数 == 图片数 断言
3. `image_hash` 16 字符、`pass_rate=-1.0`

结果：5 个已完成数据集全部 `rows/shard/ph=imgs OK`（见上）。
Vision-Flan 待下载完成后对 zip 跑 `unzip -t`（必须 0 errors）再转换。