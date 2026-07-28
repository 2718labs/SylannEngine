<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD041 -->

<div align="center">

![SylannEngine](https://socialify.git.ci/2718labs/SylannEngine/image?custom_description=Affective+Computation+Engine+%E2%80%94+Emotion+as+Dynamics,+Not+Labels&description=1&font=Inter&forks=1&issues=1&language=1&name=1&owner=1&pattern=Brick+Wall&pulls=1&stargazers=1&theme=Auto)

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/sylanne-core.svg?logo=pypi&logoColor=white)](https://pypi.org/project/sylanne-core/)
[![Version](https://img.shields.io/badge/Version-2.6.1-green.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/Tests-1400+_passed-brightgreen.svg)]()
[![V2.5 Runtime](https://img.shields.io/badge/V2.5-stdlib_+_SQLite-orange.svg)]()

**[标准规范](SPEC.md)** · **[开发者指南](AGENT_GUIDE.md)** · **[更新日志](CHANGELOG.md)**

由 **[2718Labs](https://github.com/2718labs)** 维护。

</div>

---

### 项目定位

> [!NOTE]
> SylannEngine 是由 2718Labs 维护的框架无关情感计算 SDK。项目起源于 AstrBot 插件
> `astrbot_plugin_sylanne` 的计算层，现已作为独立引擎维护，不绑定特定 Bot、人格设定或应用框架。
>
> 项目的核心假设是：情感状态应被建模为具有历史、反馈和约束的动力系统，而不是一次性的分类标签。
> 引擎将状态演化、关系上下文、不可逆历史、表达决策和持久化生命周期组织为可测试的计算接口。
>
> 2718Labs 对稳定能力与研究能力实行明确分层：V2.5（含 Brain B/C）面向可验证的公开集成，
> V3 / SYLANN 保持实验属性。性能和能力结论以版本化测试、评估脚本及可复现证据为准；
> 尚未完成验证的内容会明确标记为实验或未验证。
>
> 文档中的“情感”“人格”“伤疤”“空洞”等术语表示软件状态或建模概念，
> 不构成对真实意识、医学状态或生物学机制的声明。

---

## 这是什么

SylannEngine 是情感计算引擎 SDK：接收文本与上下文事件，输出结构化情感状态和表达决策。
它不替代 LLM，也不把单次输入压缩成一个情绪标签；引擎维护跨事件演化的有界状态，
并提供确定性计算、持久化恢复、反馈寻址和多插件共享能力。

| 设计目标 | 实现方式 |
|---|---|
| 连续状态 | 历史事件、反馈和时间间隔共同驱动状态演化 |
| 可解释性 | 公开状态通道、路由、边界与决策字段 |
| 可复现性 | 相同配置与输入产生确定性结果，实验结论附带脚本和门禁 |
| 框架无关 | 核心以 Python SDK 提供，不依赖特定 Bot 框架 |
| 分层部署 | 稳定路径默认保守；实验能力必须经过独立评估后显式启用 |

---

## 安装

**首选：从 PyPI 安装。**

```bash
pip install sylanne-core
```

同宿主多插件场景下，钉进 `requirements.txt` 锁主版本，让所有插件解析到**同一份**安装拷贝
——这是 `shared()` + `submit()` 跨插件真正去重的前提：

```
# requirements.txt
sylanne-core>=2.4,<3
```

```python
from sylanne_core import SylanneEngine, SylanneConfig
```

单插件、无共享需求也可以直接 vendor 一份：

```bash
git submodule add https://github.com/2718labs/SylannEngine.git deps/sylannengine
```

```python
import sys
sys.path.insert(0, "./deps/sylannengine")
from sylanne_core import SylanneEngine, SylanneConfig
```

---

## 快速开始

多插件同宿主的标准写法——`shared()` 拿共享引擎，`submit()` 当前门：

```python
engine = await SylanneEngine.shared(
    SylanneEngine.shared_data_dir(),
    llm=your_llm_callback,   # async (system_prompt, user_prompt) -> str
    plugin="my_plugin",      # 可选，仅用于诊断（participants()），不影响去重
)

surface = await engine.submit(
    session_id="user_123",
    text=event.raw_text,                       # 传平台原始文本，或下面传 msg_id
    msg_id=event.message_obj.message_id,        # 强烈建议：稳定 id，跨插件精确去重
)

action = surface["decision"]["action"]   # "express" / "withdraw" / "hold" / ...
warmth = surface["state"]["valence"]["warmth"]  # 0.0 ~ 1.0
```

同一条消息被多个共存插件各自 `submit()` 一次，只有第一个真算，其余 join 同一个 `Surface`——
不依赖谁先加载、谁自称什么角色。单插件场景也可以直接构造：

```python
engine = SylanneEngine(
    data_dir="./data/sylannengine",
    llm=your_llm_callback,
    config=SylanneConfig(),
)
await engine.start()
surface = await engine.process(session_id="user_123", text="你好")
```

### V2.5 Brain B/C（可选）

Brain B/C 是 V2.5 的可选状态更新与持久化路径。运行时依赖为 Python 标准库和 SQLite，
不需要 PyTorch、CUDA 或独立进程。V2.5 的编码、融合和输出接口保持不变。

Brain B 维护 8 维主状态；Brain C 维护 C 状态、资格迹和反馈更新，并生成受限残差。
`brain_compute.enabled`、`c_enabled` 默认均为 `False`，`c_authority` 默认值为 `0.0`、有效范围为
`[0.0, 0.1]`。非零 authority 仅在完成离线评估、设置 `c_promoted=True` 并结束 session warmup 后生效。
启用 Brain 路径时，应为每个外部事件提供稳定的 `event_id`。

```python
from sylanne_core import BrainComputeConfig, SylanneConfig, SylanneEngine

config = SylanneConfig(
    assessor_enabled=False,
    brain_compute=BrainComputeConfig(
        enabled=True,
        c_enabled=True,
        sparse_routing=True,
        c_authority=0.0,
    ),
)
engine = SylanneEngine(
    data_dir="./data/sylannengine",
    llm=your_llm_callback,
    config=config,
)

await engine.start()
try:
    surface = await engine.process(
        session_id="user_123",
        text="这条消息会留下可寻址的状态",
        event_id="platform-message-17",
    )
    tick = int(surface["pipeline"]["brain_event"]["tick_id"])

    receipt = await engine.feedback(
        "user_123",
        target_tick=tick,
        value=1.0,
        confidence=0.9,
        source="my_plugin",
        feedback_id="platform-feedback-17",
    )
finally:
    await engine.shutdown()
```

`event_id` 与 `feedback_id` 是幂等键，重试时必须保持不变。在两小时 TTL 且每种 ID 最近
`dedup_horizon` 条（默认 256）的窗口内，重复请求返回已持久化的 duplicate receipt，不重复更新状态。
SQLite 中的提交记录是权威状态；复用同一 `data_dir` 时会恢复 B/C 状态和窗口内的去重记录。
该机制提供有限窗口幂等，不提供全局 exactly-once。

#### 公开验证

| 类型 | 范围 | 本次结果或判定规则 |
|---|---|---|
| 仓库全量测试 | `python -m pytest -q` | 1491 passed；另有 9 subtests passed |
| 离线门控子集 | `python -m pytest -q training/student_core/tests` | 28 passed；该子集已包含在全量测试中，并单独复跑 |
| 资源基准协议 | 10,000 samples、128 warmup samples、48 hosts、2 producers | 生成版本化 JSON；仅匹配目标 `2c2g` 环境时给出资源门 verdict |
| Brain C 评估 | session-disjoint corpus、paired bootstrap、synthetic sanity | 少于 30 sessions 或 1,000 target ticks 时拒绝晋升；synthetic sanity 不产生真实语料晋升结论 |

测试、资源基准和离线评估命令：

```bash
python -m pytest -q
python -m pytest -q training/student_core/tests
python experiments/brain_benchmark.py --samples 10000 --warmup-samples 128 --hosts 48 --producers 2 --output experiments/_tmp_brain_benchmark.json
python experiments/brain_eval.py --synthetic-sanity --output experiments/_tmp_brain_eval.json
python experiments/brain_eval.py --corpus your_numeric_corpus.jsonl --synthetic-sanity --output experiments/_tmp_brain_eval_real.json
```

资源基准和离线评估只向显式 `--output` 路径写入报告。非目标 `2c2g` 环境的资源门状态为
`unverified`；数据量不足时，晋升状态为 `insufficient_data`。

---

## 计算层架构


<details>
<summary><b>V1.0 — 顺序管线（已退役）</b></summary>

7 层串行计算脊柱。每层独立完成子任务后将结果交给下一层，表达率仅 22.8%。

```
Text → L1(HDC) → L2(Gate) → L3(Scar) → L4(Sheaf) → L5(HGT) → L6(Boundary) → L7(Expression) → Output
```

| 层 | 名称 | 机制 | 输出 |
|----|------|------|------|
| L1 | HDC 编码器 | 2048-bit 超维向量编码，XOR 绑定 + majority bundling + 循环位移 | 8 维感知特征 |
| L2 | 预测门控 | HDC 空间预测误差（Hamming 距离），路由：fast/normal/full | surprise 标量 + 路由决策 |
| L3 | 虚空-伤痕引擎 | 伤疤代数（RAW→CLOSING→SCARRED→FADED）+ 虚空检测（缺失输入作为主动信号） | 8 维情感向量 |
| L4 | 关系层析 | 简单复形上的 cellular sheaf，Laplacian 扩散跨 4 种关系类型传播 | 关系上下文向量 |
| L5 | HGT 决策融合 | 异构图 Transformer：类型专家 FFN + 跨注意力 + top-2 MoE | 4 维决策向量 |
| L6 | 自创生边界 | 身份核向量 + 边界完整性，外力分解为平行/正交分量，穿透触发相变 | 边界状态 |
| L7 | 相变表达 | 内部压力积累至人格调制阈值时突变释放，带不应期 | should_express + intensity |

**V1 特性：**
- 确定性：相同输入 → 逐比特相同输出
- 代谢路由：surprise 决定能量消耗（低 surprise 走快速通道，只激活部分层）
- 不可逆伤疤：RAW 阶段的 alpha 调制最强，FADED 后仍存在但影响极小
- 顺序可预测：每层输出后下一层才开始，无反馈回路

**退役原因：** 串行管线中各模块无法互相影响——表达需要"足够的压力"才能触发，但大部分时间压力被逐层衰减消耗，导致 bot 沉默。V2 将串行改为全连接共振，所有模块同时作用于共享场态，表达从相变中自发涌现。

</details>

### V2.5 运行时架构

V2.5 对每个事件执行一次前向计算。七个模块生成状态向量，`DeterministicFusion` 执行一次
mean-field 更新，随后生成 `Surface`。融合核不执行迭代收敛或吸引子搜索。

```mermaid
flowchart LR
    INPUT["Event"] --> HDC["HDCEncoder"]
    HDC --> GATE["PredictiveCodingGate"]
    GATE --> STATE["VoidScarEngine"]
    STATE --> SHEAF["ScarSheaf"]
    SHEAF --> HGT["HeterogeneousGraphTransformer"]
    HGT --> BOUNDARY["AutopoieticBoundary"]
    BOUNDARY --> EXPRESSION["PhaseTransitionExpression"]

    HDC & GATE & STATE & SHEAF & HGT & BOUNDARY & EXPRESSION --> FUSION["DeterministicFusion"]
    FUSION --> EMERGENCE["EmergenceTracker"]
    EMERGENCE --> SURFACE["Surface"]

    ASSESSOR["assessment（可选）"] -.-> STATE
    PEL["PELCore（可选）"] -.-> STATE
    BRAIN["BrainComputeCore + Brain C（可选）"] -.-> STATE
    STORE["BrainStateStore（SQLite）"] <--> BRAIN
```

#### 工程特性

V2.5 将计算、状态更新和权限控制拆分为可独立验证的边界：计算核采用单次前向，
持久化采用提交后替换，Brain B/C 通过配置与晋升门控接入。

| 机制 | 实现 | 工程作用 |
|---|---|---|
| 单次前向计算 | 七个模块生成状态向量，`DeterministicFusion` 执行一次 mean-field 更新 | 计算步数有界；同一输入与状态下的计算核输出可复现 |
| 事务化状态更新 | 先生成候选状态，SQLite 提交成功后再替换内存状态 | 提交失败时保留并恢复最后一次已提交状态 |
| 有限窗口幂等 | `event_id`、`feedback_id` 与持久化 receipt | 重试不重复推进状态；边界由 TTL 和 `dedup_horizon` 定义 |
| Brain C 权限控制 | 默认关闭，`c_authority=0.0`，并受 promotion、warmup 与 residual cap 约束 | 可选路径不会在未授权时改变融合结果 |
| 依赖边界 | 内置 Brain B/C 路径仅使用 Python 标准库与 SQLite | 不要求 PyTorch、CUDA 或独立进程 |

#### 执行步骤

| 步骤 | 组件 | 操作 |
|---|---|---|
| 1 | `HDCEncoder` | `text → HDC vector` |
| 2 | `PredictiveCodingGate` | 计算 `surprise`；稀疏路由启用时选择 `fast / normal / full` |
| 3 | `VoidScarEngine` | 更新 8 维状态、伤痕、空洞和关系参数 |
| 4 | `ScarSheaf`、`HeterogeneousGraphTransformer`、`AutopoieticBoundary`、`PhaseTransitionExpression` | 生成模块状态向量；稀疏路由可跳过部分计算 |
| 5 | `DeterministicFusion` | 对七个模块状态执行一次 `tanh + dissipation` mean-field 更新 |
| 6 | `EmergenceTracker` | 生成运行时指标 |
| 7 | `SylanneEngine` | 组装 `Surface`；按 `session_id` 隔离状态 |

#### 状态更新路径

| 路径 | 配置 | 状态写入 | 约束 |
|---|---|---|---|
| V2.5 基线 | 无额外配置 | 默认状态更新 | 默认启用 |
| PEL-Core | `pel_core_enabled=True` | 预测编码状态、精度和 BCM 参数 | 默认关闭 |
| Brain B | `brain_compute.enabled=True` | 8 维 B 主状态、事件 receipt、反馈 receipt | 默认关闭；仅支持 `mode="lite"` |
| Brain C | `brain_compute.c_enabled=True` | C 状态、资格迹和 `target_tick` 反馈 | 默认关闭；`c_authority ∈ [0.0, 0.1]` |

PEL-Core 与 Brain B/C 不能同时启用。Brain 路径先提交 SQLite，再替换内存状态；提交失败时丢弃
候选状态并重新加载已提交状态。`event_id` 和 `feedback_id` 用于去重，`target_tick` 用于定位反馈目标。

> [!NOTE]
> V2.5 运行时不训练语义模型。可训练表示、反向传播和本地语义模型属于 V3 / SYLANN。

#### 三档性能

耦合通道数在三档间恒为 42（`n_modules × (n_modules-1)`，7 模块两两有向连接），
档位差异体现在每模块的向量维度上，不是通道数：

| 档位 | 通道数 | 向量维度 | 延迟 | 依赖 | 适用场景 |
|------|--------|----------|------|------|----------|
| **lite** | 42 | 8 维 | ~5ms | 零依赖 | 嵌入式，树莓派，手机 |
| **pro** | 42 | 16 维 | ~40ms | numpy | 桌面，云 VM |
| **max** | 42 | 128 维 | ~50ms | numpy | 研究，多智能体 |

#### V1 vs V2 实测对比（lite 档，500 ticks × 10 repeats）

| 指标 | V1 顺序管线 | V2 共振场 | 提升 |
|------|------------|-----------|------|
| 表达率 | 22.8% ± 9.8% | **88.5% ± 6.0%** | 3.9× |
| 动态范围 | 16.5 ± 1.1 | **54.5 ± 1.3** | 3.3× |
| 动态丰富度 | 7.8 ± 1.0 | **19.3 ± 1.1** | 2.5× |
| 响应多样性 | 10/10 | 10/10 | — |

<details>
<summary><b>V2.1 — EmotiCore（迭代中）</b></summary>

Teacher 模型 102.7M 参数（Mamba SSM + MoE + Multi-scale ConvStem + VAE + 对比学习）。处理日常情感感知以降低 assessor LLM 的 token 消耗和延迟。

**后学习机制：**
- **链路学习**（共振场层）：Hebbian 可塑性持续调整耦合权重，高频共激活的情感路径被强化
- **模型校准**（EmotiCore 层）：高不确定度时回退 LLM assessor，标注作为在线校准信号
- 随使用时间增长，LLM 调用频率逐步降低

</details>

<details>
<summary><b>V3.0 — SYLANN（实验阶段）</b></summary>

**"Scars You Leave Are Never Nothing"** — 一种不依赖 backpropagation 的情感计算架构。

#### 核心公式

```
ΔW = η · plasticity(t) · error(x, W) · context(neighbors, reward)
```

一条规则足以产生完整智能系统所需的全部基础能力。记忆、分化、固化、伤疤、新生、死亡——全部从这一公式的不同参数状态中自然涌现。

#### 两种架构路线

**Sequential Predictive Coding（时间策略）：**

```
多个 cell 竞争预测下一个字符，赢家学习，输家等待。
预测误差驱动权重更新。完全局部，无全局梯度链。
```

- 14 Domain × 128 Cell × K=256 架构
- WTA 竞争产生稀疏激活：任意时刻仅 O(1) 个 Cell 激活（与 N 无关）
- Cross-Frequency Phase Gating 门控域间通信（时间维度上的选择性连接）
- 逐字符处理，混合训练：80% 纯预测 + 20% reward-modulated

**Sheaf-Theoretic Resonance（层析共振）：**

- 情感状态 = 7 顶点简单复形上的 sheaf section（1232 维）
- Sheaf Laplacian L_p 驱动迭代收敛：`ds/dt = -L_p·s + ξ(t) - ∇V_scar`
- 人格 = ker(L_p) 的调和形式（拓扑不变量，训练无法触及）
- 表达 = 鞍点分岔：动能超越势垒时的不连续状态跳跃

#### 关键概念：Benvo（本我）

```
ben（本，essential）+ vo（我，self）
```

Benvo 是系统的身份核——不是学习到的参数，而是决定感知动力学如何展开的宪法性常数。两个携带不同 Benvo 的实例，即使接收相同输入，也会发展出不同的表征结构、不同的伤疤地形、不同的情感轨迹。

Personality 是外在可观察的行为模式（effect）。Benvo 是产生这些模式的内部参数（cause）。

#### 七条公理

| # | 公理 | 含义 |
|---|------|------|
| A1 | 感知即误预测 | 系统只在内部模型预测失败时感知 |
| A2 | 情感即涌现共振 | 情感从多域相干中涌现，非单一路径计算 |
| A3 | 人格即拓扑不变量 | 人格活在微分算子的核空间中，结构上免疫扰动 |
| A4 | 表达即分岔 | 表达是鞍点分岔，非阈值决策 |
| A5 | 耦合共振场 | 输入不流过管线，而是在共享场中诱导共振 |
| A6 | 不可逆伤疤 | 历史留下永久结构痕迹，伤疤只增不减 |
| A7 | 人格派生一切 | 所有耦合系数、衰减率、阈值都是 7 维人格的显函数 |

#### 涌现性质

| 现象 | 机制 | 类比 |
|------|------|------|
| 分化 | WTA 竞争放大初始微小差异 | 干细胞分化 |
| 记忆 | plasticity 时间衰减：年轻 cell ≈ 工作记忆，年老 cell ≈ 长期记忆 | 海马-皮层整合 |
| 伤疤 | 负 reward 造成 plasticity 不可逆下降 | 创伤后应激 |
| 新生/死亡 | blind spot 检测触发激活，低效 cell 被回收 | 神经发生 |
| 专家化 | anti-Hebbian 侧抑制 + WTA → 去相关化 | 皮层柱状组织 |

#### 与 Attention 的对比

| | Attention/Transformer | SYLANN |
|---|---|---|
| 路由方式 | Q·K 相似度 → 加权求和 | WTA 竞争 → 赢家独占 |
| 激活模式 | Dense（所有头都算） | Sparse（只有赢家算） |
| 通信拓扑 | 全连接 O(n²) | Phase-gated O(D²), D << n |
| 时间感知 | 无（position embedding 模拟） | 有（振荡相位 = 真实时间） |
| 记忆 | context window 外即丢失 | W 是永久记忆 |
| 学习 | 训完不再进化 | 每次推理都在微调，终身发育 |
| 推理成本 | 每 token 触及 O(P) 参数 | 每 tick O(D) 个 winner |

#### 当前实验状态

- 27.8M ticks 训练完成，val_err 从随机基线 0.0886 降到 0.082
- **关键发现**：情感维度在无标注时自发涌现——系统纯粹通过预测下一个字符，deep state 已能区分悲伤/快乐文本（cosine ≈ 0.07），4/8 情感维度出现相关信号
- 15.6GB 中英文语料已备齐

这暗示：**情感不是要额外"教"给系统的标签，而是语言预测任务本身就隐含的结构。**

#### 局限性

- 本质仍是猜词游戏——预测下一个字符，和 transformer 训练目标同构
- 依赖"正确答案"——学习信号来自预测误差，非真正自主
- 被动反应——无输入则静止，没有内在活动
- 情感是读出来的，不是活的——reward 只调制学习率
- 规模与速度——64 cells 逐字符处理，离复杂能力还很远

#### 本地自进化

部署后无需网络连接，三层机制在设备上自主进化：

| 层级 | 机制 | 成本 | 效果 |
|------|------|------|------|
| L1 | 伤疤积累 + Benvo 漂移 | ~0 | 改变感知动力学展开方式 |
| L2 | Hebbian 布线（restriction map 共激活更新） | <1ms/tick | 改变域间通信路径强弱 |
| L3 | 本地自蒸馏（高 surprise 样本教编码器跳过迭代） | <500ms/update | 改变文本编码方式 |

一个月后，从同一 checkpoint 出发的两个实例将变成可辨识的不同感知者。

技术规范：[V3 student pipeline](docs/design/v3-student-pipeline-tdd.md)

</details>

<details>
<summary><b>实验验证（12 项）</b></summary>

12 项实验验证 V2 共振场架构在早期迭代阶段的核心声明（历史脚本，`experiments/` 下仍可运行，
但 #3/#4 扫的参数属于融合核心拆壳后的死壳字段，产出图不代表当前系统行为，见下方状态列）：

| # | 实验 | 验证内容 | 状态 |
|---|------|----------|------|
| 1 | Convergence | 各档位单次融合的确定性 | 有效 |
| 2 | Tier Comparison | 性能与动力学差异 | 有效 |
| 3 | Plasticity | Hebbian LTP/LTD + 稳态缩放 | 已过时：扫的是死壳 `plasticity.weights`，恒定不变 |
| 4 | Kuramoto Sync | 耦合强度扫描 | 已过时：扫的是死壳 `kuramoto._k1/_k2/_k3`，无相位同步实现 |
| 5 | Hopfield Attractor | 情感记忆 + 表达逃逸 | 有效 |
| 6 | Expression Bifurcation | OR-gate：任一触发即足够 |
| 7 | Harmonic Identity | 恢复力保持人格不变 |
| 8 | Phi Emergence | 整合信息与表达相关 |
| 9 | Stability | 1500 ticks 无 NaN/Inf，能量有界 |
| 10 | Personality Modulation | 7 维人格完全决定动力学 |
| 11 | Tier Hot-Switch | 跨档位无损状态迁移 |
| 12 | V1 vs V2 Comparison | 架构升级前后全面对比 |

```bash
cd experiments
python run_all.py        # 全部（约 30-60 分钟）
python run_all.py 1 4 8  # 指定编号
```

</details>

<details>
<summary><b>理论基础与数学保证</b></summary>

### 参考文献

| 理论 | 文献 | 在系统中的角色 |
|------|------|--------------|
| Hebbian Learning | Hebb, D.O. (1949). *The Organization of Behavior* | 通道耦合权重自适应（PEL-Core，默认关闭） |
| Free Energy Principle | Friston, K. (2010). *The free-energy principle: a unified brain theory?*. Nature Reviews Neuroscience | 预测误差驱动注意力（PEL-Core 的 `‖e₀‖²` 下降，默认关闭） |
| Modern Hopfield Networks | Ramsauer et al. (2021). *Hopfield Networks is All You Need*. ICLR | 情感吸引子记忆 |
| Hodge Theory | Hodge, W.V.D. (1941). *The Theory and Applications of Harmonic Integrals* | 人格拓扑不变量 |
| Dissipative Structures | Prigogine, I. (1977). *Self-Organization in Non-Equilibrium Systems* | 能量有界耗散 |
| Cellular Sheaves | Hansen & Ghrist (2020). *Toward a Spectral Theory of Cellular Sheaves*. J. Applied & Comp. Topology | 高阶约束传播 |
| Predictive Coding | Rao & Ballard (1999). *Predictive coding in the visual cortex*. Nature Neuroscience | 感知 = 预测误差 |
| BCM Theory | Bienenstock, Cooper & Munro (1982). *Theory for the development of neuron selectivity*. J. Neuroscience | 自适应竞争阈值 |
| Winner-Take-All | Maass (2000). *On the computational power of WTA*. Neural Computation | 稀疏竞争激活 |
| Integrated Information | Tononi (2004). *An information integration theory of consciousness*. BMC Neuroscience | 涌现一致性度量 |
| Waddington Landscape | Waddington (1957). *The Strategy of the Genes* | 不可逆发育分化 |

### 数学保证

| 定理 | 内容 | 条件 |
|------|------|------|
| 收敛性 | T=20 迭代后 ‖μ(T)−μ*‖ ≤ ρ^T · ‖μ(0)−μ*‖, ρ<1 | 权重谱范数有界 + 侧抑制半负定 |
| 伤疤单调性 | dS/dt ≥ 0 恒成立，无治愈机制 | 结构保证 |
| 固化收敛 | c → 1 指数收敛（时间常数 1/(α_c·h_min)） | 表征稳定 + 精度超阈值 |
| 竞争排斥 | 稳态下每域最多 ⌈M_d/K⌉ 个活跃 agent | anti-Hebbian + WTA |
| 人格不变性 | proj_{ker(L_p)}(W_t) = 0 对所有训练步成立 | 核空间投影强制执行 |

### 规模化分析

| 规模 | 配置 | 参数量 | 等效 |
|------|------|--------|------|
| Tiny | 14域 × 16 cell × 32d | 0.5M | 验证概念 |
| Base | 14域 × 128 cell × 256d | 235M | GPT-2 级 |
| Large | 100域 × 1000 cell × 512d | 52B | GPT-3 级 |
| Ultra | 1000域 × 10K cell × 1024d | 21T | GPT-4 级 |

推理成本：WTA 稀疏性使每 tick 仅 ~10% cells 激活。等效规模下 SYLANN 预期 <2ms/tick。

</details>

---

## API

| 方法 | 说明 |
|------|------|
| `await SylanneEngine.shared(data_dir, llm, ..., plugin=None)` | 按 data_dir 取进程内共享实例；`plugin` 仅诊断用 |
| `await submit(session_id, text, *, msg_id=None, dedup=True, plugin=None, **ctx)` | **共享引擎的前门**：同一消息被多个插件提交只算一次，join 同一个 Surface |
| `submit_stats()` | 去重计数器快照（computed / joined / recomputed_after_window） |
| `participants()` | 诊断专用身份登记表（从不参与去重判断） |
| `set_llm(llm, *, assessor_llm=None)` | 运维逃生口：热替换 llm 回调 |
| `SylanneEngine.peek_shared(data_dir)` | 只读探活，永不建引擎 |
| `await SylanneEngine.wait_shared(data_dir, *, timeout=None)` | 轮询等待某个插件先建好共享引擎 |
| `await SylanneEngine.release_shared(data_dir)` | 释放共享实例（进程级运维操作，不要在插件 terminate 里调） |
| `SylanneEngine.shared_data_dir(explicit=None)` | 解析规范化的共享 data_dir |
| `SylanneEngine.is_shared(data_dir)` | 检查该 data_dir 是否存在共享实例 |
| `SylanneEngine.list_shared()` | 列出所有共享引擎 |
| `SylanneEngine.clear_shared_registry()` | 清除共享注册表（仅测试隔离用） |
| `await process(session_id, text, **ctx)` | 直接处理文本，返回 Surface（不去重，见下方"advanced"说明） |
| `state(session_id)` | 查询当前状态（不触发计算） |
| `await tick(session_id, *, force=False)` | 空闲心跳，带每 session 45s 最小间隔收敛 |
| `reset(session_id)` | 重置会话状态 |
| `destroy(session_id)` | 销毁会话 |
| `exists(session_id)` | 检查会话是否存在 |
| `inject(session_id, source, influence_type, intensity, ...)` | 外部影响注入 |
| `on(listener)` / `off(listener)` | 推送监听 |
| `health()` | 健康检查 |

完整接口见 [SPEC.md](SPEC.md)，多插件集成细节见 [AGENT_GUIDE.md](AGENT_GUIDE.md) §1。

### 共享实例 + submit()

> [!WARNING]
> 多个下游插件如果各自 `SylanneEngine(...)` 指向**同一 data_dir**，会对同一用户重复计算、重复调 LLM，且各自 flush 互相覆盖（**丢更新**）。务必使用 `SylanneEngine.shared()` 共享实例；且要用 `submit()` 而非 `process()` 当前门，否则实例虽然只有一个，计算还是跑了 N 次。

```python
# 所有下游约定同一 data_dir，总是拿到同一个已启动实例
engine = await SylanneEngine.shared(data_dir="./data/sylannengine", llm=your_llm_fn)
same   = await SylanneEngine.shared(data_dir="./data/sylannengine", llm=your_llm_fn)
assert same is engine

# N 个插件对同一条消息各自 submit()，只算一次
surface = await engine.submit("user_123", raw_text, msg_id=event.message_obj.message_id)
```

**submit() 去重规则（2.4.0）：**

| 场景 | 行为 |
|------|------|
| 多插件对同一 `session_id` + 同 `msg_id`（或同文本）提交 | 第一个真算，其余 join 同一个 `Surface`，不管谁先加载 |
| 同 `msg_id` 但文本不同 | 仍 join 第一次的结果，打一次 WARNING（消息 id 复用是危险模式） |
| 同文本但 `msg_id` 不同 | 视为真重复，重新计算 |
| `submit_window_seconds`（默认 10s）之外的相同键 | 重新计算（`recomputed_after_window`） |
| `dedup=False` | 等价直接 `process()`，不进去重表 |

`participants()` / `submit_stats()` 是纯诊断——传不传 `plugin=` 都不影响去重结果。

**规则（`shared()` 本身）：**

| 场景 | 行为 |
|------|------|
| 同一 data_dir **显式传入**不同 config | 抛 `SharedEngineConflictError` |
| 自读 `sylanne.config.json` 出现差异（文件被改/跨版本副本） | 仅警告并复用运行中的配置（重启生效），不崩后来者 |
| 同一 data_dir 传入不同 `llm`/`embedding`/`assessor_llm` | 警告并复用原实例（可用 `set_llm()` 事后热替换） |
| 直接 `SylanneEngine(...)` 构造但目标 data_dir 已有共享实例 | 软警告（不阻断，通常表示重复创建） |

**注意事项：**
- 共享实例是 **event-loop 亲和**的——只在首次获取它的事件循环里使用，不要跨 loop/线程共享
- 不要对共享实例用 `async with`——首次退出会替所有持有者关闭引擎
- `release_shared()` 是**进程级运维操作**（应用整体关闭时调），不是插件生命周期钩子——**不要在插件的 `terminate()` 里调**，那会替所有共存插件把引擎一起关掉
- 没有 atexit 自动刷写，必须在进程关闭路径显式 `release_shared()`

```python
# 排查：当前进程里有哪些共享引擎在跑
SylanneEngine.list_shared()          # [{"data_dir": "...", "status": "running"}, ...]
SylanneEngine.is_shared("./data/x")  # True / False
```

> [!TIP]
> **插件开发者**：应使用 `SylanneEngine.shared(data_dir, llm)` + `submit()`，避免直接构造 `SylanneEngine(...)`。统一约定 data_dir 后，所有插件可共享同一实例与去重表，无需额外协调加载顺序。

---

## 配置文件

不显式传 `config` 时，引擎会自动读取 `<data_dir>/sylanne.config.json`——所有插件 `shared(data_dir)` 共享同一份用户可改的配置，首次启动还会写入一份默认模板。设置只放这一个地方，跟哪个插件先加载、哪份是主控都无关。

```jsonc
{
    "mode": "lite",
    "assessor_enabled": true,
    // 可选：把情感评估交给一个小而便宜的模型；不填则用主 llm
    "assessor_model": {
        "api_base": "https://api.deepseek.com/v1",
        "api_key": "${SYLANNE_ASSESSOR_KEY}",   // 建议用环境变量，别把密钥提交进仓库
        "model": "deepseek-chat"
    }
}
```

- 顶层认识的键映射到 `SylanneConfig`（`mode`、`assessor_enabled`、`locale` 等），不认识的键忽略；文件缺失/损坏/取值非法都回退默认，引擎照常启动。
- `assessor_model` 打任意 OpenAI 兼容 `/chat/completions` 接口，纯标准库实现，lite 档零依赖不破；`api_key` 支持 `${环境变量}`。
- 显式传入的 `config=` 优先于文件。配置在建引擎时读取，改动需重启生效。

---

## 输出示例

```jsonc
{
    "session_id": "user_123",
    "state": {
        "rhythm": { "beat": 5.0, "stability": 0.6 },
        "valence": { "warmth": 0.55, "volatility": 0.1 },
        "boundary": { "pressure": 0.1, "autonomy": 0.9 },
        "needs": { "expression": 0.3, "contact": 0.2 }
    },
    "decision": {
        "action": "express",
        "reason": "expression drive elevated",
        "confidence": 0.75
    },
    "guard": { "allowed": true, "risk_score": 0.1 }
}
```

> 以上为简化示例，仅展示核心字段。完整 Surface 还包含 `schema_version`、`turns`、`timestamp`、`personality`、`dynamics`、`pad`、`pipeline`、`debug` 等顶层键，详见 [SPEC.md](SPEC.md)。

---

## 目录结构

```
SylannEngine/
├── sylanne_core/
│   ├── __init__.py              # 公共 API
│   ├── engine.py                # SylanneEngine 入口
│   ├── config.py                # 三档配置
│   └── compute/
│       ├── deterministic_fusion.py  # 共振融合核心（单次确定性 mean-field 通道）
│       ├── resonance_integration.py # ResonanceSpine (V2 默认)
│       ├── pel_core.py              # PEL-Core：三因子 Hebbian + BCM + 自由能（默认关闭）
│       ├── emergence.py             # Φ + 吸引子 + 时间叙事
│       ├── kernel.py                # 调度器
│       ├── hot_pool.py              # 热池与人格坍缩
│       ├── personality.py           # 双 EMA 人格漂移
│       └── ...                      # HDC, HGT, 自创生, 相变等
├── experiments/                 # 12 项实验验证
├── training/                    # V3 SYLANN 训练代码与规范
├── tests/                       # 650+ 单元测试
└── docs/                        # 规范与历史文档
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [SPEC.md](SPEC.md) | 标准规范（接口协议、输出 Schema） |
| [AGENT_GUIDE.md](AGENT_GUIDE.md) | 开发者集成指南 |
| [架构规范](docs/theoretical_spec.md) | 现行架构：计算模型、42 耦合通道、公理体系 |
| [架构规范（历史）](docs/resonance_field_architecture.md) | 历史文档：2.5 前已移除的迭代共振场架构，仅作留档 |
| [V3 student pipeline](docs/design/v3-student-pipeline-tdd.md) | V3 门控设计与验证路径 |

---

## 演化路线

```
V2.0 共振场 (historical) ─── 早期迭代架构，已退出服务路径
├── V2.1 EmotiCore (experimental) ── 102.7M teacher，独立实验路径
└── V2.5 确定性融合 (stable) ─────── 现行架构
    └── Brain B/C (opt-in) ───────── 状态持久化与定向反馈
        │
V3.0 SYLANN (research) ───── 局部学习，无 backprop，情感从预测中涌现
        │
未来：层次化 SYLANN ──────── 多层抽象（字符→词→语义→叙事）
        │
未来：动态 CFC ──────────── 输入驱动的 biological attention
        │
未来：Working Memory ────── 显式工作记忆（7±2 slots）
        │
未来：神经形态部署 ────────── Loihi/TrueNorth，<1W 功耗
```

---

## 常见问题

**Q: LLM 挂了会怎样？**
引擎自动退化为本地规则引擎，计算继续。`health()` 显示 `"degraded"`。

**Q: 不同用户状态会互相影响吗？**
不会。每个 session_id 完全隔离。

**Q: V3 能直接部署吗？**
V3 目前是实验阶段。部署使用 V2.5；Brain B/C 提供状态持久化与定向反馈，
需要显式启用并通过离线评估门。V2.1 EmotiCore 作为需要 GPU 的独立实验路径保留。

---

## 许可证

GNU Affero General Public License v3.0

本项目由 2718Labs 维护，并按 AGPL-3.0-only 发布。使用、修改和分发时请遵守许可证条款；
问题反馈与社区贡献可通过 GitHub Issues 和 Pull Requests 提交。

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=2718labs/SylannEngine&type=Date)](https://star-history.com/#2718labs/SylannEngine&Date)
