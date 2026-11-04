# SCOPED MOPD 数学/多图数据策略：候选登记

- **状态：Candidate / 候选方案，Not finalized / 非最终方案。**
- 登记日期：2026-11-04。
- 本文只记录下一轮数据构建与专家验证的待检验假设，**不是已确认的最终数据配比、
  不是已经生成的训练集，也不证明 C3+C4 一定优于其他专家组合。**
- 合同去污染、任务族分类、候选数据实际构建、专家下游评测和 MOPD 训练均尚未完成。
- 本文不授权训练、MOPD、teacher export、skill extraction 或新的大规模数据下载；
  这些步骤需要另行确认。

## 1. 目标与边界

训练优化预算优先投入**数学/科学 Reasoning**和**综合**，VQA 不做专项数据冲刺。
这不意味着取消 VQA 评测：合同 `TC20260430027` 的三个类别仍须各选至少两个
benchmark 测试，完整约束见 `docs/project_benchmark_vision_opd.md` 和
`configs/eval/project_vision_opd.yaml`。

“VQA 短答案必然与推理任务冲突”目前只是待检验假设，不能作为既定事实。历史评测
同时受模型、训练轮数、输出截断、答案抽取和 judge 协议影响，不足以单独证明能力冲突。

当前候选专家为 **C3 + C4**，依据是实际训练样本、目标任务结构和已有教师能力，
不是 `math`、`multiimage` 这些名称：

- **C3**：候选的单图视觉数学教师。
- **C4**：候选的多图联合理解教师，但其原始训练数据不能直接当作 ReMI 高匹配池。
- **C2**：OCR/chart/table 专项教师不进入当前主线；这不等于删除所有含图、表或
  读数的数学题，ReMI 的跨图读数仍可由相应任务族覆盖。
- **C1**：不为 VQA 专项冲刺增加教师；三类照常评测。

C3/C4 是否保留、是否需要替换，必须在专家下游诊断后决定。此处没有冻结专家数量。

## 2. 从真实样本得到的专家与数据判断

### 2.1 C3：单图数学匹配较直接，但不覆盖 ReMI 的核心难点

本地 `capability3/train/train.parquet` 共 9,500 条训练样本，来源为：

| 来源 | 数量 |
|---|---:|
| MMK12 | 5,729 |
| WeMathPro | 3,771 |

样本全部为单图，覆盖几何、函数、代数、数字规律和少量应用推理；答案以数值、
短数学表达和选项为主。这与 MathVerse 的单图几何/函数、DynaMath 的部分子域有真实
交集，因此 C3 是合理候选。

但“同属视觉数学”不代表可替代多图训练：C3 原始数据没有直接提供 ReMI 所需的多图
变量共享、信息互补和跨图联合计算监督。

### 2.2 C4：教师能力可候选复用，原始训练分布需要大幅替换

本地 `capability4/train/train.parquet` 共 9,500 条，实际来源为：

| 来源 | 数量 |
|---|---:|
| ScanNet frames | 4,561 |
| nuScenes | 1,330 |
| NLVR2 | 855 |
| `m4_instruct_data/Spot-the-Diff` | 807 |
| `mantis_instruct/spot-the-diff` | 807 |
| NextQA | 475 |
| STAR | 380 |
| Multi-VQA | 190 |
| Birds-to-Words | 95 |

近半数是 ScanNet 室内多帧，其他主要来源包括驾驶多摄像头、视频动作/因果和自由文本
多图问答。它们与 ReMI 的人工数学图、共享变量、精确对应和联合计算并不高度匹配。
因此：

- 保留 C4 的候选理由是其已经训练过的**多图输入和跨图证据整合能力**；
- 不能把 C4 原始 9.5K 直接作为 ReMI 主训练池，更不能为凑数量保留全部自然图像/视频样本；
- 原始 C4 只候选保留 NLVR2、明确需要对应/差异定位的 Spot-the-Diff，以及经样本审查
  确认必须联合多图证据的少量样本；
- NextQA、STAR、Multi-VQA、Birds-to-Words 不进入 ReMI 主池候选。

C4 对 ReMI 是否有独立增益仍是待验证项，不能由“多图”标签直接推断。

## 3. 目标 benchmark 不是同一种分布

### 3.1 MathVerse：单图几何/函数，与 C3 交集较高

本地 `MathVerse/testmini.parquet` 有 3,940 条，全部为单图；Plane Geometry 2,550、
Functions 795、Solid Geometry 595。相同原题具有不同视觉信息/文本信息版本，训练与评测
需同时考虑图像、问题变体和答案形式，而不是只匹配一个“数学”标签。

来源包含 GeoQA、Geometry3K、GEOS 等，因此不能把这些公开训练集直接视为干净补充。
它们必须经过来源、题目和图像级去污染审查。

### 3.2 ReMI：多图互补信息与联合推理，不等于自然多图/视频 QA

本地 ReMI test 共 2,600 条，13 个任务各 200 条。图像数分布为：2 图 2,092、3 图 95、
4 图 86、5 图 292、6 图 35。真实任务包括：

- 多张函数图的极限、导数、组合函数；
- 两张几何图共享角度/长度，面积图与价格表联合；
- 多组 emoji 方程、两个图结构同构、碰撞前后状态；
- 时钟与时刻表、两个钟面、双图数值变化、地图互补路径；
- 原图/目标图反推 TikZ 修改、矩阵 IQ、参考目标与候选位置。

其核心是**共享变量、互补信息、精确跨图对应和联合计算**。图像数量或“多图”标签本身
不构成匹配；仅比较自然场景差异、理解视频动作也不是 ReMI 的充分训练替代。

ReMI 与 MathVerse 因而不应被合并成同一个数据池：前者主要对应候选 C4 的跨图任务族，
后者主要对应候选 C3 的单图任务族。

### 3.3 DynaMath：强调题目变体，但验收 seed 不可变成训练数据

DynaMath 的公开样本以 seed 和 variants 组织，覆盖几何、代数、统计、图论、科学图等。
训练时可借鉴独立题目的稳健变体思想，但不能使用公开验收 seed 生成训练变体，也不能
将同源变体重新包装后声称为干净泛化。

### 3.4 历史分数只用于提出假设

历史不同训练臂中 ReMI 和 DynaMath 有较好表现，例如统一 seed43 协议中 ISO-MOPD 的
DynaMath 0.6938、ReMI 0.5365；MMF-only 历史臂 ReMI exact 0.3662。它们支持优先研究
ReMI/数学方向，但不是当前 SCOPED C3/C4 的效果证明，也不能跨协议直接排名。

ReMI 统一使用全 2,600 条为分母的 strict exact；不可采用旧的
`normalized_exact_diagnostic`。MV-MATH 等 judge 结果必须考虑输出是否完成，截断输出不能
借 judge 判对而计为有效完成。详见 `docs/remi_mv_math_protocol_20260913.md`。

## 4. 候选训练池：按任务族，而不是按数据集名称混配

以下 12K 仅是一个**小规模核心池的目标配额假设**，用于估计路由和构建工作量；不是最终
采样配方，也不是现有数据已经可以无缺口填满的承诺。实际规模应由合格样本数、去重结果
和专家诊断决定；不足时缩小规模，不以弱相关数据补齐。

| 候选任务族 | 目标配额（未冻结） | 候选路由 | 主要验证方向 |
|---|---:|---|---|
| 单图几何、函数、代数与图论 | 4,000 | C3 | MathVerse、DynaMath 对应子域 |
| 双图函数/几何/图结构联合推理 | 3,000 | C4 | ReMI FuncRead、GeomShape、Isomorphism 等 |
| 跨图符号、读数、时间/表格组合 | 2,000 | C4 | ReMI EmojiAlgebra、Charts、Schedule、Clocks 等 |
| 跨图对应、视觉矩阵、编辑与路径 | 2,000 | C4 | ReMI IQ、CodeEdit、RefCOCO、Maps 等 |
| 综合科学/多学科保持样本 | 1,000 | 主要 C3，按样本复核 | MMMU-Pro、MMBench 保持性 |
| **合计** | **12,000** | **约 C3 5K / C4 7K** | **仅为待验证初始假设** |

候选来源定位：

- **SCOPED C3**：经质量、难度和合同去重后作为单图数学锚点，不重复下载同一来源冒充新数据。
- **SCOPED C4**：只精选真正需要跨图对应或互补证据的样本，不整池进入。
- **ViRL39K**：按任务族与难度精选；其中 MV-MATH 派生内容必须识别并排除。
- **MMFineReason-586K**：只筛几何、函数、代数、图论、视觉 puzzle 和适量科学样本，
  结合 pass rate、答案可验证性和样本审查；不全量混入。
- **MMFineReason-123K**：全部是已知 4B pass-rate=0 的困难样本，不单独作为训练起点。
- **Cauldron、Vision-Flan、LLaVA-CoT、VisualWebInstruct、MAmmoTH-VL**：不进入当前核心池；
  只有发现明确、不可替代且去污染合格的任务族缺口时才另行评审。
- **Geometry3K、GeoQA、GEOS**：有 MathVerse 来源重合风险，不默认视为可用补充。

当前最明显的数据缺口是**真正需要联合图间信息的数学/科学题**，尤其是双图函数、几何、
图结构、物理状态和符号系统。后续新增数据调研应优先核查这类来源，而不是继续扩大通用
多图 VQA、自然视频或 OCR/chart 专项数据。

如考察独立合成生成器，必须与 ReMI 的题目、图像、seed 和生成逻辑隔离；不得复制、改写
或近邻生成验收题。新增来源在确认许可、原始 split、来源重合和样本质量之前只能标记为
“待调研”，不能写成已选训练集。

## 5. 所有合同 benchmark 默认 held-out

不得因为存在 train/dev/few-shot 字段就把合同数据放进训练：

- **ReMI**：仅 26 个 few-shot 示例和 2,600 test；全部禁止训练，包括那 26 条。
- **MathVista**：无 train split，明确禁止训练。
- **MathVerse、DynaMath、MV-MATH**：公开题、seed、variants 和派生近邻均不进入训练。
- **MMMU-Pro**：仅 test，且源自 MMMU 筛选/改造；MMMU validation 不能默认视为无污染训练集。
- **MMSI-Bench、MMVet**：公开评测数据不训练，MMSI 的 `thought` 也不作为 SFT 目标。
- **MMBench dev、BLINK val**：只用于开发诊断，不混入主训练池。

候选池必须做合同 benchmark ID、规范化问题、图像 SHA256、感知 hash、文本近似重复、
跨来源重复和题组变体去重。`verifiable`、不同数据集名称或不同 split 均不等于已去污染。

OPD 输入只需要图像、问题、来源与教师路由，不要求 gold CoT；RL 需要可验证答案；SFT 才
需要目标 response/CoT。旧 verl RL parquet 不能未经 schema 和语义验证直接接入 SCOPED MOPD。

## 6. 候选评测与验证门槛

三类评测保持完整。下一轮重点诊断组合暂列为：

| 类别 | 候选重点 benchmark | 目的 |
|---|---|---|
| 数学/科学 | ReMI + DynaMath，辅以 MathVerse | 区分多图联合推理、变体稳健性和单图视觉数学 |
| 综合 | MMMU-Pro + MMBench | 多学科保持与通用综合保持 |
| VQA | GQA + ViewSpatial-Bench | 合同覆盖和退化监控，不配置专项训练池 |

这张表也是候选诊断组合，不缩减合同要求，也不表示其他 benchmark 永久取消。MathVerse
不能替代 ReMI 检验多图能力，ReMI 也不能替代 MathVerse 的单图几何/函数诊断。

在冻结方案前至少完成：

1. **专家下游诊断**：分别比较 base、C3、C4 在同一 prompt/解码协议下的 ReMI 分任务、
   MathVerse 子域和综合保持表现；检查截断、答案可抽取率及错误类型。
2. **数据命中审计**：逐任务族抽样人工核验“是否必须联合多图”“是否需要数学计算”，
   记录来源、split、答案类型、图像数、难度和路由理由。
3. **合同去污染报告**：输出精确/近似/图像/变体命中及排除清单，而非只声明数据源不同。
4. **小规模数据试点**：只有去重合格的样本才进入实际池；按合格量缩减，不凑满 12K。
5. **双教师接入验证**：MOPD 目前按样本 `data_source` 静态路由。改为 C3+C4 必须同步
   修改数据列表、teacher/key/skill、distillation resource pool 和 launcher 校验，不能只删
   两份 parquet。保持上游 OPD 语义，未经另行批准不新增 loss 或训练臂。
6. **训练后同协议验收**：使用固定 backend、模型 checkpoint、seed、生成预算和评测协议，
   同时报告数学增益、综合保持和 VQA 退化，不能只展示受益项。

## 7. 当前状态

- **VERIFIED LOCALLY**：已检查 C3/C4 的真实 parquet 来源与样本形态，以及本地 ReMI、
  MathVerse、DynaMath 的结构；上述匹配与错配判断基于这些只读检查。
- **REQUIRES SERVER VERIFICATION**：C3/C4 的独立下游能力、候选数据的实际训练收益、
  双教师 MOPD 的资源与运行正确性。
- **UNKNOWN**：最终数据规模与配比、新增多图数学来源、完整合同去污染结果、是否最终采用
  C3+C4，以及训练后能否同时保持综合和 VQA。

本登记不修改 SCOPED 复现、teacher export、skill extraction 或训练暂停状态，也不把历史
训练 reward 当作下游专家有效性的证据。
