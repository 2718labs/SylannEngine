# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Security

- `training/generate_data.py`：移除硬编码的 provider API 密钥，改为运行时从环境变量读取
  （`SYLANNE_GPT55_API_KEY` / `SYLANNE_MIMO_API_KEY`），未配置的 provider 自动跳过、全未配置则报错退出。
  原先提交的密钥已在 provider 端轮换作废，并从 git 历史与 PyPI 2.4.0 归档中清除。

## [2.6.1] — 2026-07-10

### 情感动力学（affect-dynamics，设计代号 v26）—— 首个带发布号的版本

> 全部功能藏在**默认 `False`** 的 `SylanneConfig` 开关后，**全关时行为与 2.5.0 逐字节一致**
> （确定性字段逐字相同、快照无新增键，已 `random.seed(0)`+`PYTHONHASHSEED=0` 实证对拍）。
> 发布号定为 2.6.1：含 v26 全量落地（PR #26/#27）+ 合并后独立机器人评审的健壮性加固（PR #28）。

在 8 维情感核（`lite` 档）上引入双速情感动力学 E 律：墙钟惰性衰减到人格均衡 Φ_eq（慢/时间
通道）+ 每轮 appraisal 饱和快更新（快/语义通道）。E 即 `ScarredState.base`（原地升级，非并行核）。
全部藏在三个默认 `False` 的 `SylanneConfig` 开关后，**全关时行为与 2.5 逐字节一致**（确定性字段
逐字相同、快照无新增键）；与设计契约 `docs/design/v26-affect-dynamics-design.md` /
`docs/design/v26-upgrade-path.md` 对应。已过独立对抗红队（10 条 finding 全修）。

#### 新增（纯函数层，零接线）

- `sylanne_core/compute/affect_dynamics.py`：E 律纯函数——墙钟衰减 `decay`（半衰期 lerp 到 Φ_eq）、
  饱和快更新 `saturating_update`、人格函数 `equilibrium`/`half_lives`/`gain_vector`、良定域断言
  `validate_gain`/`validate_scalar_params`，以及 `[0,1]↔(-1,1)` 域适配器 `to_unit_interval`/
  `from_unit_interval`（`base` 是 tanh (-1,1)，E 律以 [0,1] 立式；`decay` 仿射等变、
  `saturating_update` 非等变必须整体折进折出）。
- `sylanne_core/compute/affect_projection.py`：assessor 三标量 + 意图 → 8 维 appraisal 增量。
- `sylanne_core/compute/affect_output_contract.py`：circumplex 情绪标签 LUT + 逐维 Schmitt 迟滞。
- `sylanne_core/compute/memory_coupling.py`：情绪相似度 `emotion_match` / 传染凸混合
  `contagion_blend` / κ 人格函数 `contagion_kappa`（**κ 是人格函数，非扁平 config 标量**）。
  canonical 无逐条打分记忆库可接，故只出原语、零接线（flag-and-defer）。
- `sylanne_core/compute/slow_channel.py`：poignancy 漏桶 + 墙钟冷却反思 + 版本化回滚环。

#### 新增配置开关（默认全 `False`）

- `affect_dynamics_enabled`（**Gate A**）：E 律**只算不写**——并行"影子"E 落日志/诊断，
  绝不写 `base`、不进 `observe()`、不进 prompt。
- `affect_takeover`（**Gate B**）：E 律**夺权**写权威 `base`——衰减在 `step()` 顶部先于事件
  演化落地，饱和快更新替代 assessor 手写意图规则，异常 fail-closed 回落旧规则。需
  `affect_dynamics_enabled`，**与 `pel_core_enabled` 互斥**（`__post_init__` 校验，违背即 `ValueError`）。
- `affect_slowchannel_enabled`（**Gate C**）：慢通道 poignancy→反思→锚回弹宏漂移（不可逆记忆写，
  含原子提交 + 回滚环）。

开 Gate B/C 属**有意行为变更**（非字节一致），验收门是 warmth 行为标定——标定完成前不建议生产开启。

#### Surface / 诊断

- `diagnostics=True` 且 affect 启用时，`diagnostics` 附只读键 `affect_label_shadow`（迟滞情绪标签）——
  仅诊断、不进 prompt，与既有 `Surface.pad.label` 并存。关闭时该键不出现（诊断面板字节一致）。

#### 决策落地（2026-07-10，标定呈报 D1/D2/D3 裁决）

- `affect_full_takeover`（**Gate B-full**，默认关，需 takeover）：E 律全权——主步
  MLP/PEL 演化在 8 维核上旁路，base 只由衰减 + appraisal + 伤痕形成驱动。可观测常驻
  情绪 = Φ_eq，半衰期先验成为活杠杆（"时间治愈"可感）；伤痕形成/粘滞不变，且 scar
  粘滞首次产品可观测（受伤维度好得慢）。
- `SylanneEngine.set_relationship(session_id, r)`（D2）：host 显式供给关系相位 R∈[0,1]，
  Φ_eq 的 warmth 行随 R 上移；未调用保持 0.5（今日行为）；随快照持久化（additive，
  affect 关时无新增键）。

#### 修复（本轮附带，均 gated 在 affect 开关后以保关时字节一致）

- `ScarredState.step()`：`feedback()`（`timestamp=0.0`）不再清零静默愈合墙钟（丢下一步的静默奖励
  愈合）——仅在 `affect_dynamics_enabled` 开时生效，关时保持旧的无条件赋值。
- 冷载 hoist：`SylanneEngine._get_or_create_host` 改协程，首触快照磁盘读经 `asyncio.to_thread`
  挪出事件循环，冷会话不再卡住 loop。

#### 健壮性加固（PR #28，合并后 gemini-code-assist 评审）

- `slow_channel.observe`：appraisal 分量补 `math.isfinite` 消毒——非有限值不再灌进持久化人格漂移累积器。
- `affect_dynamics.poignancy_magnitude`：`a_k` 越界索引加长度守卫，短输入不再 IndexError。
- `ScarredState.from_dict`：学到的增益/资格迹复原包 try/except——损坏快照的非数值学习态不再崩会话加载。
- `affect_output_contract.resolve_label`：迟滞合并改 `list(raw)`，消除 `prev.key` 长度不一致时的 IndexError。

## [2.5.0] — 2026-07-06

诚实化地基版：拆净 2.5 前架构退役后遗留的死代码，全面文档校真。零 Surface 变更、
零行为变化（固定输入下所有确定性字段字节级一致，经 stash 对比 main 实测）；API 与 2.4.1 完全一致。

### 内部清理（零行为变化）

- 拆除死机制桩类 `_Kuramoto` / `_Plasticity` / `_FreeEnergy`
  （`sylanne_core/compute/deterministic_fusion.py`）：三者是 v2.5 迭代共振核心退役后留下的
  惰性占位符——`_Plasticity.update()` 函数体即 `return None`，`_Kuramoto` 只剩几个从未被
  真实动力学读取的字段，`_FreeEnergy._precision` 写入即黑洞。清点确认全仓库零消费者、
  零测试直接断言这三者的内部字段。同步删除 `ResonanceSpine.apply_personality()` /
  `feedback()`（`resonance_integration.py`）里写向这些死字段的 personality/meta-learner
  reach-in，以及 `MetaLearner`（`meta_learner.py`）里重复的 `kuramoto_k1` seed 公式/边界/
  常量表条目。`resonate()` 的 mean-field 融合本体与 `sync_order` 数值计算逐字未动——原先
  经 `_coupling.kuramoto._order` 中转的缓存直接改为存在融合器自身字段
  （`_last_sync_order`），验证 `resonate()` 输出与改动前逐字段比对完全一致。
  `_Broadcast`/`topology_gate`/`_Complex`（`active_channels`）不在本次范围内，原样保留。
  config.py 公开面未暴露过这些名字，无需 DeprecationWarning。
- 删除 `DeterministicFusion` 的四个活写死读幽灵参数（`_hopfield_strength` / `_identity_inertia`
  / `_identity_max_norm` / `_max_attractors`）：每次 `apply_personality` / `switch_tier` 都写入，
  但 `resonate` / `inject` / `observe` 零读取。删除写入线、字段本体与 `to_dict` 输出键；
  `from_dict` 对老 snapshot 里的这些键静默忽略，保证向后兼容。
- 删除 `set_criticality` 死写线：`_Coupling.set_criticality` 是显式 no-op；删除方法本体与
  `resonance_integration.py` 的两处调用。`emergence.py` 的 criticality 计算与读侧消费者完整保留。

### 文档校真（无代码逻辑改动）

- `force_backend`：更正为「被接受并校验、写入 `DimensionProfile.backend`，但该字段当前无任何
  读取者（`HGT._use_numpy` 硬编码读 `_HAS_NUMPY`），故对计算无可观测影响；参数位保留是因为
  下游插件构造时传 `force_backend="python"`」。删除此前「max 档静默降级为 numpy」「仅影响 HGT
  numpy 加速决策」等不实表述（`config.py` / `SPEC.md` / `docs/theoretical_spec.md` 三处同步）。
- `docs/theoretical_spec.md` / `SPEC.md`：耦合通道数校真为全档恒 42（= n_modules×(n_modules−1)，
  n_modules=7；档位差异体现在向量维度 8/16/128，非拓扑复杂度）；§11.3 移除 max 档 GPU / <5ms
  表述（torch 路径已随 `29b402a` 删除，全档 CPU）；新增 Kuramoto 墓志铭一节；Φ 明确标注为
  two-component 近似（空间相关×时间连贯），非划分近似的严格 IIT，不蹭 IIT 合法性。
- `computation_spine.py` 模块 docstring：max 档「GPU 加速」→ numpy（GPU 路径已移除），
  并标注 `ComputationSpine` 自 v2.5 起冻结。
- 四个 2.5 前的 `docs/resonance_field_*` / `docs/max_tier_workflow.md` 加历史横幅（引 `29b402a`），
  标明描述的是已移除的迭代共振场架构、仅作历史留档；`README.md` 文档索引同步修正指向现行 spec。

## [2.4.1] — 2026-07-06

纯打包/文档补丁版，无任何运行时代码改动；API 与 2.4.0 完全一致。

### 打包

- `pyproject.toml` 新增 `[tool.hatch.build.targets.sdist]` 白名单（`only-include`），
  源码分发包（sdist）从 ~21.7MB 收敛到 ~0.4MB。hatchling 默认按 VCS 跟踪文件全收，
  会把 `presentation/`（官网静态站）、`.codegraph/`（二进制索引库）、`docs/` 论文 PDF 等
  一并塞进 sdist；2.4.0 首发因发布流程 checkout 的 tag 早于本修复，其 sdist 未受益，
  故补发 2.4.1。wheel 一直是干净的（由 `packages = ["sylanne_core"]` 控制），不受影响。
- 新增 `.github/workflows/publish.yml`：`push tags v*` + `workflow_dispatch` 触发，
  发布前校验 tag / `pyproject` / `__version__` 三处版本号一致，经 PyPI Trusted Publishing
  （OIDC，免 API token）上架。

### 文档

- README.md / AGENT_GUIDE.md 的安装指引在钉版本示例（`sylanne-core>=2.4,<3`）旁补充了
  PyPI 上架前的等价 GitHub 安装命令（`pip install "sylanne-core @ git+..."`）。

## [2.4.0] — 2026-07-02

### 💥 撤回 driver/observer 角色层（发布前，从未进入任何 tag），改用 single-fire submit()

`SylanneEngine.acquire` / `AcquireResult` / `ObserverView` / `SylanneEngine.role` /
`_sharing.shared_role` / `as_observer` **全部删除**——不是弃用垫片，是彻底移除。
所有旧调用点会立刻 `AttributeError`/`ImportError`，不留一个行为相似的假亲戚。该层只存在于
发布前的开发分支（见下方「开发期记录」），从未打 tag、从未合入 main、没有任何下游见过——
因此本版本沿用 2.4.0 号：对一切已发布版本（≤2.3.2）而言这里全部是纯新增，semver 恰为 minor。

**为什么砍掉一个开发期刚做出来的机制**：driver/observer 把"角色"锚定在 SDK 物理拷贝目录上
（`my_id == builder_id`）。AstrBot 的默认部署是所有插件共用一份 site-packages 拷贝——于是这个判断
对每个插件恒真，N 个插件全拿到 `role="driver"` 和完整引擎，N 路 `process()` 全接上，N 倍 LLM
账单，机制在它唯一要解决的默认部署下是彻底 no-op，且过去无任何告警。以拷贝为身份单位这条路，
在共享 venv 部署下从根上就错了。

**新模型，五行说完**：
1. 每个插件都用 `shared()` 拿同一个完整引擎——不再区分 driver / observer。
2. 前门从 `process()` 换成 `await engine.submit(session_id, text, msg_id=...)`。
3. N 个插件对同一条消息各自 `submit()`，第一个 miss 真算，其余 join 同一个 `Surface`——**不依赖谁先加载、谁自称什么身份**。
4. 幂等表挂在引擎实例上（不在 rendezvous cell 里），双索引（`msg_id` + 文本哈希），10 秒默认窗口。
5. 插件死亡是非事件：没有 driver 就没有孤儿，幸存者 `submit()` 照算；唯一残留是建者的 `llm` 闭包，`health()` 可见降级，`set_llm()` 可运维换绑。

### ✨ 新增

- `await engine.submit(session_id, text, *, msg_id=None, confidence=None, flags=None, now=None, values=None, dedup=True, plugin=None) -> Surface`——
  共享引擎的新前门。双索引 join 规则：无 `msg_id` 的查询可 join 哈希命中；带 `msg_id` 的查询仅当命中条目未记录
  `msg_id` 时才 join 哈希命中（记录了不同 `msg_id` = 真重复 → 重新计算）；直接 `msg_id` 命中恒 join（文本分歧则
  WARNING 一次）。真计算跑在 detached task 里，所有 awaiter（含首提交者）`asyncio.shield()` 等它——一个 awaiter
  自己的取消永远杀不掉共享计算。`dedup=False` 等价直接 `process()`，逃生口不锁死。
- `engine.submit_stats() -> {"computed", "joined", "recomputed_after_window", ["by_plugin"]}`——去重计数器快照。
- `engine.participants() -> list[dict]`——诊断专用的身份登记表（`shared(..., plugin=...)` / `submit(..., plugin=...)`
  可选传入）。**硬规则：这份数据永不参与任何行为判断**，只喂日志/统计——身份负责观测，幂等负责保证，这正是
  2.4.0 式"以身份定角色"翻车之后要守住的边界。
- `engine.set_llm(llm, *, assessor_llm=None) -> None`——运维逃生口：热替换一个已死插件留下的 `llm` 闭包，无自动魔法，
  没人替你调用它，纯粹给运维脚本/健康检查用。替换时打一条 INFO。
- `engine.tick(session_id, flags=None, *, force=False) -> Surface`——tick 现在带**每 session 绝对最小间隔**
  `config.tick_min_interval_seconds`（默认 45s）：间隔内的调用直接返回缓存 `Surface`、不推进状态。几个共存插件
  各跑自己 ~60s 心跳循环，打到同一个共享引擎上会被收敛到约每 45s 一次真实 tick，而不是 N 次。`force=True` 绕过
  收敛器——这是测试/运维逃生口，**不是心跳所有权的暗号**，谁传了 force 都不会因此获得特殊地位。
- `classmethod SylanneEngine.peek_shared(data_dir) -> SylanneEngine | None`——只读探活，永不建引擎；给没有自己
  `llm`、绝不该成为建引擎那份拷贝的纯监听插件用。
- `classmethod async SylanneEngine.wait_shared(data_dir, *, timeout=None, interval=0.5) -> SylanneEngine | None`——
  轮询版 `peek_shared`，等某个有 `llm` 的插件先把引擎建起来。0.5s 轮询，不引入任何跨拷贝唤醒协议（曾评估
  parked-future 唤醒方案，会在不同模块名的 vendored 拷贝间丢唤醒/泄漏，弃用）。超时打一条 INFO 后返回 `None`。
- `shared(..., plugin: str | None = None)` / `submit(..., plugin: str | None = None)`：可选调用方身份字符串，
  写入 `participants()`，首次 attach 打 INFO。
- `config.py` 新增 `submit_window_seconds: float = 10.0`、`submit_max_entries: int = 1024`、
  `tick_min_interval_seconds: float = 45.0`。

### 🗑️ 移除

- `SylanneEngine.acquire()` / `AcquireResult` / `ObserverView` / `SylanneEngine.role()` / `_sharing.shared_role()` /
  `as_observer` 参数——开发分支引入，发布前整层撤回，从未出现在任何已发布版本里。`__init__.py` 的 `__all__` 同步移除
  `ObserverView`/`AcquireResult` 导出。
- `tests/test_shared_engine.py::TestRole`/`TestAcquire` 一并删除。

### ⚠️ 已知诚实限度（不是 bug，是文档化的取舍）

- `process()` 旁路：文档不锁死直连 `process()`，绕过 `submit()` 的调用方依旧各付各的 LLM 账单——契约不是牢房；
  共享引擎上直连 `process()` 现会打一次性 nudge log 引导去 `submit()`。
- 1.0.0 孤岛：结构性够不到 rendezvous cell 的旧内嵌拷贝（如未迁移的 emotion_spirit）唯一出路是升级，SDK 侧只能
  在首次 `get_shared_engine` 时扫描 `sys.modules` 里的 pre-2.0 拷贝并 WARNING 点名，无法机械收敛。
- 跨进程双 flush：维持既定取舍，一个 data_dir 一个进程，不加 PID 锁。
- 无 `msg_id` 时，窗口内真·重复消息会被合并成一次（状态不推进）——启发式键的诚实代价，`msg_id` 可根除，
  `dedup=False` 可逃生。
- 混布过渡窗（部分拷贝已 3.0、部分还在 2.x）：加载顺序决定谁先建引擎，是一次性的"抽签"而非机制性收敛——
  但从静默变成响亮：附着到无 `submit()` 的旧引擎会打一条命名建者版本的 WARNING。

### 版本

`pyproject.toml` 与 `sylanne_core/__init__.py` 的 `__version__` 同步到 `2.4.0`（此前发生过两处失步，这次
在同一 commit 里核对一致）。

### 开发期记录（2026-06-30 · 本节的角色层 API 已于发布前撤回，Surface 护栏等其余内容随 2.4.0 发布）

### 🔀 Driver/Observer 角色层 + Surface 兼容性护栏

多插件同进程共存时，只有一个插件跑引擎（driver），其余转纯监听（observer）。

- `SylanneEngine.acquire(data_dir, llm, ..., as_observer=False) -> AcquireResult`：按角色取共享引擎
- `ObserverView`：只读句柄（`on/off/state/exists/health`），不绑定 `process/tick/inject`——
  合作式护栏（防误触发第二条计算流），非安全边界
- `SylanneEngine.shared_data_dir(explicit=None) -> Path`：解析 host 级共享目录
- `SylanneEngine.role(data_dir) -> str`：合作式角色标签
- `AcquireResult`：`role` / `engine` / `observer` / `handle` / `is_driver`
- `tests/test_surface_compat.py`：Surface 只加不删不改（additive-only），CI 强制
- `SHARING_INTEGRATION.md` 内容合并入 `AGENT_GUIDE.md`，原文件删除

发布前对抗性自审修订（同版本内）：

- 修 `__version__` 漏改：`sylanne_core.__version__` 同步到 `2.4.0`（之前仍报 `2.3.2`）
- Surface 运行时护栏由 6 处抽查改为递归全树校验；新增 TypedDict 边接线锁 +
  `HealthStatus` 字段锁——adapter 漏发任意叶子、改接更薄类型都会红 CI
- `state()` 补 `_ensure_started()`：已释放的共享引擎读 `state()` 现按复活守卫报错，
  不再静默重建 host、绕过守卫
- `release_shared_engine` 释放时一并清 `builders[key]`，不留陈旧 driver id
- 清理删档/改名遗留断链（`SHARING_INTEGRATION.md`、`docs/SPEC.md`→`docs/theoretical_spec.md`）：
  `test_surface_compat`/`test_axiom_conformance`/`resonance_integration`/interchange schema
- `README` `inject` 形参名修正 `type` → `influence_type`

### 🔗 多插件共享机制重设计（2.3.2）

多个（可能各自 vendored 一份 SDK 的）插件指向同一 `data_dir` 时，让"一个 data_dir 一个引擎 + 一份用户配置"真正
成立。弃掉早先一版"版本选举 / lease / 热交接 / HMAC"设计——8-agent 红队判其对单进程单作者部署过度设计且核心不
成立（自报 `__version__` 可被任意 copy 冒充夺全场、热交接因 `_assess` 在 session 锁外丢更新、Windows `os.kill(pid,0)`
杀主控、热切安全闸依赖不存在的整数 state-schema-major）。公共 API 向后兼容，新增可选 `assessor_llm` 参数。

- **rendezvous cell**（`_rendezvous.py`）：进程级 cell 挂在 `builtins` 一个固定 sentinel key 上，各 vendored copy
  （不同模块名）汇合到同一注册表、真去重——修掉"两份 namespaced copy 各自 `_REGISTRY` 抢同一 data_dir、flush
  互覆盖丢更新"。`_REGISTRY`/`_LOCK` 别名指过去，init-Future / tombstone / loop 亲和不变。first-builder-wins，
  进程内主控不切换，靠重启升级。
- **写一次持久 copy 身份**（`_identity.py`）：每份 copy 一个 `<pkg>/_identity.json` 里的 uuid（O_EXCL 跨进程安全、
  损坏自愈、只读安装回退路径 hash）。纯诊断 label，非选举非安全；据此在版本不一致时发串味告警。
- **共享 config 文件**（`_config_store.py`）：无 `config` 时引擎自读 `<data_dir>/sylanne.config.json`，所有插件
  `shared(data_dir)` 共享同一份用户可改配置（首启写默认模板）。不认识的键忽略；缺失/损坏回退默认。
- **可选小模型 assessor**（`_assessor_llm.py`）：config 里 `assessor_model` 块变成零依赖 OpenAI 兼容回调
  （urllib + `asyncio.to_thread`；`api_key` 支持 `${ENV}`），把情感评估交给小而便宜的模型；不配则回落主 llm。
- **审计加固**：完备性审计（对真码 + 真两-copy 实测）查出的 2 blocker + 4 major 全修，每条带复现测试——released
  引擎不再自我复活成双引擎双写（`_ensure_started` 守卫）；自读 config 差异 warn+复用而非硬抛崩无辜后来者；
  `shutdown` flush 失败打日志且保 `degraded` 不被 `closed` 盖；畸形 assessor timeout 兜底不崩构造；跨 copy 守护函数
  改 duck-type 判活（`clear` 不再误删别 copy 活引擎）；`shared()` docstring 改 per-process 诚实化。

### 🛡️ 鲁棒性硬化（2.3.1）

外部 LLM / 调用方返回畸形结构时的防御缺口：插件 PR #45 的 gemini 自动审查转交三条（逐条对 canonical 真码核实，
非照抄建议），落地评审（对抗性多视角复核）又补全两条同类完成项。触发面同源——非 dict 的 JSON / 容器、显式 `null`
字段、跨档位或 ragged 旧快照；正常路径不触发，故非 blocker。公共导出面（43 符号）与结果契约零改动，re-vendor 后
插件无需改代码。

- **assessor 非 dict JSON 不再穿透崩溃**（`assessor.py`）：`_parse_response` 的 except 补 `AttributeError`。合法但
  非 dict 的 JSON（`[]` / `"text"` / `null` / `42`）`json.loads` 成功后 `data.get` 抛 `AttributeError`，原 except
  只兜 `(JSONDecodeError, ValueError, TypeError)` 漏接；现统一回退 `_neutral`，与该函数"畸形 JSON 一律走中性读"的
  既有契约一致。
- **DeterministicFusion 跨档位 / 旧快照恢复对齐维度**（`compute/deterministic_fusion.py`）：`from_dict` 载入 `states`
  时按本实例 `state_dim` 走 `_resize`（沿用 `switch_tier` 同款写法）。别档位（pro=16 / max=128）或 legacy
  ResonanceField 快照载入后，`resonate()` 以 `range(state_dim)` 索引 module_states——小快照进大档位会 `IndexError`，
  反向则维度静默不一致；resize 恢复 `len(state)==state_dim` 不变量。评审补全：同一 `from_dict` 块再把模块行数
  pad/截断到 `n_modules`，否则 ragged 快照（`states` 行数 ≠ 7）在 `resonate()` 的 `range(n_modules)` 索引处仍越界
  （低可达：所有实例化均 `n_modules=7`、legacy 场亦 7，仅损坏/手改快照触发）。
- **assessor 标量 None 字段不再掀翻注入管线**（`compute/resonance_integration.py`）：`_apply_assessment_to_engine`
  的 5 处 `float(assessment.get(...))`（wound_risk / valence / arousal / confidence ×2）改用共享 `_coerce_float`。
  公共入口 `process(assessment=...)` / `host.on_request(assessment=...)` 收到调用方原始结构 `{"wound_risk": null}`
  时落回默认值并 clamp，而非 `float(None)` 抛 `TypeError`（canonical 自走的 `assess_text` 路径字段已是 float，
  本修针对外部直传 assessment 的公开边界）。新增 leaf 模块 `sylanne_core/_numeric.py` 容纳 `_coerce_float`，
  assessor 与 compute 两侧共用，无 import 环、无层级倒挂。评审补全（gemini/sourcery）：`_coerce_float` 再兜
  `OverflowError`（400 位巨整数 `float()` 会抛）与非有限值（`NaN`/`±inf`——`float()` 接受、`json` 还从字面量
  `NaN`/`Infinity` 解析；裸 clamp 会把 `NaN` wound_risk 静默映成 `1.0` 满信号），统一落 default。
- **assessor 畸形容器（非 dict assessment）不再崩公开入口**（`compute/resonance_integration.py` + `compute/kernel.py`
  + `compute/computation_spine.py`）：评审补全。上一条硬化了 assessment 里的畸形「字段」，但「容器」本身若是非 dict
  （`process(text, assessment=[...])` / `="angry"` / `=42`——与 Fix 1 同源的 LLM 畸形形状）仍会 `assessment.get` 抛
  `AttributeError`：`ResonanceSpine.process`、`AlphaKernel._tick_inner`（其 `_update_affect_debt` 的 except 只兜
  `TypeError/ValueError`）、fallback `ComputationSpine` 三处公开/半公开边界。三个公开入口（`ResonanceSpine.process`、
  `AlphaKernel._tick_inner`、`ComputationSpine.process`）各在最顶部做一次容器归一（非 dict→None，等同「无
  assessment」）；`_capture_telemetry` 早已裹 `except Exception` 故无需改。评审补全（gemini）：`ComputationSpine`
  改在 `process()` 入口归一，而非仅守 `apply_assessment`——否则更早的缓存签名 `assessment.items()` 会先崩。

### ♻️ v2.5 类脑引擎重设计 —— 实测落地(2.3.0)

按"实现 → 测量 → 按证据诚实处置"走完,不 force-ship。全文 `docs/design/v25-neuromorphic-redesign.md`。

- **删除死的 simplicial resonance-field 栈(~4.5k LOC)**:`resonance_field`/`_numpy`/`_torch` +
  `coupling_dynamics` + `topology_gate`(serving 路径自 v2.5 起已是 DeterministicFusion + PEL-Core,这些零活
  引用、仅互引 + 自身测试)。行为中性;活测试迁入 `tests/test_spine_integration.py`。这是"连贯类脑引擎"的真交付。
- **修 torch footgun**:`config.build_profile()` 的 lite/pro 改用 `importlib.util.find_spec` 定 backend,部署
  路径不再 eager `import torch`(实测 RSS −458MB);backend 字符串不变;加守卫测试 `tests/test_config_backend.py`。
- **B 根因修复(x_t 改预测 HDC + assessor 作精度加权先验 e2 + 塌双写)实测否决,CUT 到默认 off**:双红线全挂
  (精度 0.50→0.29 更差、assessor→z ~200× 衰减),独立审查复现确认,连 salvage 变体也被更脑 v2 严格支配。
  `SEMANTIC_PRIOR=False` 默认 = 已验证的更脑 v2;e2 机件留作有界、off 时无行为变更、可消融选项(收缩界已证)。
- 擂台招牌(神经调质总线 / criticality / k-WTA / 字面 STP / 多余 Turrigiano)经审查判 theater/冗余/relabel,
  **不建**(设计记录留存)。诚实结论:更脑 v2 已交付类脑核;本轮净增 = footgun + 删死码 + e2 选项 + 否决发现。

### 💥 Breaking Changes

- 移除 AstrBot 前置插件形态。SylannEngine 现在是纯 SDK，直接 `SylanneEngine(...)` 实例化并传入自己的 LLM 回调
- 删除 `main.py`、`metadata.yaml`（AstrBot 插件入口与元数据）
- 移除 `sylanne_core.get_engine()` 与共享实例 `_shared_engine`——插件版专用的共享引擎获取方式不再提供
- 删除 `sdk` 镜像分支与 `sync-sdk.yml` 工作流：`main` 分支本身即 SDK

### ✨ Added

- **PEL-Core（v2.5 预测编码情绪核，default-off）**：新增两层预测编码微电路
  `sylanne_core/compute/pel_core.py`——演化 8 维潜信念 `μ` 与情绪读出 `z`（下游写入
  `scar_state.base`），纯 Python（`math` + list-of-lists，无 numpy，lite 可用）、全 mypy-strict。
  K=2 自由能下降 + 三因子 surprise 门控的 `W_gen` 在线 Hebbian 可塑 + 在线精度 + 谱钳 ≤0.9；
  有界性与收缩为结构性保证（`μ,z∈[−1,1]^8` 前向不变，`‖J_μ‖₂≤1−αδ`）。
  - 新增 `SylanneConfig.pel_core_enabled` 开关，**默认 `False`**——关时引擎走遗留 `_evolve_base`，
    行为与此前逐字节一致（既有套件全绿、未改）；开时才跑 PEL，由其自有测试置位
  - **加性 `free_energy` 键（D-1）**：PEL 开启时 `result["resonance"]` 多一个有限的 `free_energy`
    诊断键；关闭时结果形状完全不变（无契约改动）
  - **非语义 `assessor_advisable` 门信号（D-10）**：经 `diagnostics()["pel"]` 暴露
    （低 surprise 且无创伤 => False，否则 True），连同 surprise/精度一并提供。SDK 只产出信号，
    不接任何下游 call-skip
  - 快照两条路径（`ResonanceSpine` / `ComputationSpine`）均往返 PEL 子键；旧档缺键时从人格重初始化
  - P3 消融扫测试（`tests/test_pel_ablation.py`）：`Pi_top→0` / `eta_W→0` / `ρ_p→0` 各产生可测变化，
    证明无机制是 no-op
  - **更脑 v2（在真流量上让脑机件活起来，仍 default-off）**：三个在线可塑机制，全在 `pel_core_enabled`
    之后、v2 默认 on-path、各自独立可消融，schema v1→v2（加 `pi0`/`theta`/`s_bar`，带 v1 回退）：
    - **M1 除法归一化精度**（Heeger 1992）：固定预算的竞争再分配（均值 1.0），解掉真 spine 上精度
      钉死 `[5.0]×8`（跨维 std≈0）的注意力死饱和——实测跨维 spread 0→~0.46。`PRECISION_DIVISIVE=False`
      与 committed 精度更新**逐位一致**；eta_w ×5 复原设计均值
    - **M2 BCM 式滑动阈元可塑增益**（Bienenstock+1982）：`m_i∈[0,2]` 按各维误差超阈与否调 Hebbian
      速率（不改方向），`θ_i=EMA(e0²)` ~100 拍自调；门见证用 m-spread + path-length
    - **M3 锚定 allostatic π**（Sterling 2012）：朝 ⟨z⟩ 漂的同时拉回冻结 trait 先验 `π0`，止住身份侵蚀
      （渐近保留 ~80% π0），凸更新前向不变
    - **生产见证**：`pel_diagnostics()` 暴露跨维精度 spread / 乘积 spread / `precision_live` / `pi_anchor_drift`，
      真流量上可窗口化告警——把"语料上活"变成"生产上死了能被发现"
    - 收缩/有界性闭式可证且红队实测复核（最坏 `‖J_μ‖₂=0.977<0.985`）；新增 merge-blocking 门 #13–#19
      （T-DIV / T-DIV-OFF / T-BCM / T-PROD / T-ANCHOR / T-SCHEMA / proof 守卫）+ surprise-gate 覆盖；
      `pel_core_enabled=False` 仍逐字节一致；快照恢复按 host flag gate（关时不偷开 PEL）
- `SylanneEngine.shared(data_dir, llm, ...)`：进程内按解析后的 data_dir 去重的共享实例机制，替代已删除的插件版 `get_engine()`。同一目录只由一个引擎拥有，避免状态分裂与 flush 丢更新；返回已 start 的引擎
- `SylanneEngine.release_shared(data_dir)`：关闭并释放共享实例（应用关闭时调用，会 flush 落盘）
- `SylanneEngine.is_shared(data_dir)` / `SylanneEngine.list_shared()`：内省接口，查当前进程里有哪些共享引擎在跑
- 冗余软护栏：直接 `SylanneEngine(...)` 构造时若目标 data_dir 已有活跃共享实例，记 warning 提示重复创建（不阻断）
- `SharedEngineConflictError`：同一 data_dir 以不同 config 获取共享实例时抛出
- **ResonanceSpine 接入 embodiment 人格漂移（canonical）**：`ResonanceSpine.process()` 现在
  在 return 前调用 `_drift_embodiment(result)`，与 `ComputationSpine` 同位语义。此前 ResonanceSpine
  （运行时默认 spine）虽已具备全部漂移基建字段，却从不触发漂移——基建就绪、接线缺失。补齐后
  AGENT_GUIDE「每次 process() 自动漂移」对默认 spine 成立。
- **对话质量自我进化（CP8-P4「越聊越校准」）端到端贯通**：agent 把回复质量自评经
  `process(..., values={"dialogue_quality": q})`（或 spine 层 `process(..., dialogue_quality=q)`）
  喂回，引擎据此漂移人格——质量高强化表达欲+拉近关系引力，质量低收敛表达欲。全程走 canonical
  自动漂移通道（`engine → kernel → spine.process → result → DriftSignalExtractor → _drift_embodiment`），
  无「第四写动词」后门。具体三层改动：
  - `DRIFT_SIGNALS` 新增 `dialogue_quality_high` / `dialogue_quality_low` 两条映射
  - `DriftSignalExtractor.extract()` 认 `result["dialogue_quality"]` → 产高/低信号
    （阈值 `_DIALOGUE_QUALITY_HIGH=0.7` / `_LOW=0.3`，中间区不触发）
  - 两个 spine 的 `process()` 加可选 `dialogue_quality` 入参；kernel 从 `values` 通道透传
  - 质量分是滞后反馈：对第 N 轮回复的评分在第 N+1 轮传入。不传时行为完全不变（默认 None）

### 🔧 Changed

- 文档（README / SPEC / AGENT_GUIDE / 架构文档）改为单一 SDK 安装与接入方式
- `release.yml` 不再打包 AstrBot 插件 zip，仅创建 GitHub Release
- Issue / PR 模板移除 AstrBot 插件专属字段

### 🔧 表达策略：硬闸人格显函数化（axiom A7）+ credit assignment 修正

`ExpressionPolicy` 的表达硬闸从死常数升级为人格显函数，并修正 contextual bandit 的
off-policy 信用分配。**默认行为逐 tick 不变、存档无损**——所有新参数均可选，缺省即旧行为。

- **T1 硬闸人格显函数化**：`_DRIVE_FORCE_EXPRESS=0.95`/`_DRIVE_FORCE_HOLD=0.1` 两处死常数
  （`expression_policy.decide()` 与 `resonance_integration` spine 覆盖点）改为实例字段
  `_force_express`/`_force_hold`，由 `set_personality(openness, expression_drive_trait,
  sovereignty_guard)` 派生：
  - `force_express = 1.05 − 0.20·expression_drive_trait`（中性 0.5 → 0.95）
  - `force_hold = 0.02 + 0.16·sovereignty_guard`（中性 0.5 → 0.10）
  - 单调性：表达欲越强越早强制开口；主权越强"懒得说"区越大。`force_express>1.0` 合法
    （该人格永不被强制开口）。spine 覆盖点改读 `force_express_threshold`/
    `force_hold_threshold` 属性，消灭第二份重复常数。
- **T2 credit assignment 接真实 action**：`update_from_feedback(..., actual_action=None)`
  与 `ResonanceSpine.feedback(..., actual_expressed=None)`——当表达决策由上层仲裁器拥有时，
  可告知真实执行的行动，让 bandit 为真正做过的选择领赏受罚（`None` 缺省=旧行为）。
  反馈总线其余五家消费者一律不动（纯透传）。
- **T3 强制决策不训练（可选）**：`update_from_feedback(..., skip_forced=False)`——置 True 时
  硬闸强制的样本只记 diagnostics、跳过梯度步，避免 policy 被自己没选的行动污染。缺省
  False 保留旧行为。新增 `last_decision_forced` 属性暴露上次决策是否走硬闸。
- **持久化**：`to_dict`/`from_dict` 新增 `force_express`/`force_hold` 可选键；旧档缺键回退
  legacy 常数（0.95/0.1），新档被旧版读取时多余键被忽略——三向往返不炸。
- **诊断**：`diagnostics()` 新增 `force_express`/`force_hold` 可观测。
- 测试：`tests/test_expression_policy_saddle.py`（23 例，覆盖中性锚定/单调性/真实 action
  梯度/强制不训练/存档往返/spine 接线）；存量全绿。

## [v2.0.0] - 2026-06-01

SylannEngine V2 — 共振场架构正式版。完全重写计算核心，从顺序 7 层管线升级为单纯形共振场。

### 🏗️ 架构：共振场

完全架构重设计——顺序 7 层管线替换为全连接单纯形共振场：

- **441 条有向耦合通道**（完全 6-单纯形 Δ⁶）三档分配：lite=42, pro=287, max=441
- **Hebbian 可塑性**：通道用进废退（LTP + LTD + 稳态缩放 + 神经达尔文主义剪枝）
- **高阶 Kuramoto 同步**：两体 + 3 体（Millán 2020）+ 4 体相位耦合，爆炸性同步转变
- **Hopfield 吸引子景观**：情感记忆作为能量极小值，表达作为分岔（逃离吸引子）
- **谐波身份（"灵魂"）**：Hodge Laplacian 零空间提取，指数移动平均，恢复力跨扰动保持人格
- **回声状态储备池**：泄漏积分器时间记忆，过去输入的衰减历史
- **自由能最小化**（Friston 2010）：精度加权预测误差驱动信念更新
- **全局工作空间广播**（Baars 1988）：赢者通吃竞争 + 点火
- **临界性反馈环**：涌现指标在相变附近放大耦合（自组织临界性）
- **表达作为 OR 门分岔**：max(惊讶, 新颖性, 点火, 原始驱动) × Φ——任一强信号触发表达
- **无损档位热切换**：升级线性插值，降级平均池化，可塑性模式继承

### ✨ 新功能

- `ResonanceSpine`：`ComputationSpine` 的直接替代，API 完全兼容（process/feedback/express/to_dict/from_dict）
- `switch_tier("pro")`：lite/pro/max 间热切换不丢失状态
- 完整人格→参数映射：7 个人格维度控制所有耦合/可塑性/场/表达参数
- 涌现追踪：Φ（整合信息）、χ（临界性）、吸引子计数、时间叙事、记忆深度
- 改进的 Φ 计算：空间相关 × 时间连贯（替代朴素方差比）

### 📝 文档

- `docs/resonance_field_spec.md`：形式化数学规范（21 篇文献引用）
- `docs/resonance_field_architecture.md`：完整架构规范 + 开发者集成指南（英文）
- `docs/resonance_field_architecture_zh.md`：同上（中文）
- `docs/max_tier_workflow.md`：MAX 档 Mermaid 计算流程图
- `docs/theoretical_spec.md`：更新共振模型、平台要求、人格映射表
- `docs/resonance_field_paper_en.tex`：学术论文（英文）
- `docs/resonance_field_paper_zh.tex`：学术论文（中文）
- README：重写，含共振场说明、Mermaid 图、档位对比

### 🧪 测试

- 66 个专用共振场测试（拓扑、可塑性、Kuramoto、Hopfield、储备池、身份、分岔、档位切换）
- 434 个测试全部通过，零回归

### 🔧 实验验证

- 11 项实验协议验证核心声明（收敛性、可塑性、同步、吸引子、表达、稳定性等）
- 每实验 1000+ tick × 10 次重复，统计检验
- 实验代码：`experiments/`

### ⚡ 性能

- lite 档：~5ms/tick，50+ 并发会话/核
- pro 档：~40ms/tick（numpy 加速）
- max 档：~50ms/tick CPU，<5ms GPU（torch 路径已结构化但未实现）

### 🔧 配置

- lite 的 `concurrency_target` 从 5 提升到 50
- 插件版锁定 lite 档（pro/max 仅 SDK）
- 单代码库策略：插件版与 SDK 版仅 main.py 存在与否 + 依赖声明不同

---

## [v1.0.0a4] - 2026-05-31

(Previously labeled v1.0.0rc3)

### 💥 Breaking Changes

- 移除三层记忆系统（L1/L2/L3），Surface 输出不再包含 `memory` 字段
- 移除 `SylanneConfig.memory_capacity` 配置项

### 🐛 Bug Fixes

- 修复 `adapter._map_guard()` 读取 `constraints` 但 kernel 返回 `flags`
- 修复 `metadata.yaml` license 标识与 pyproject.toml 不一致
- 修复 `metadata.yaml` personality_impact 三项全部标错为 false

### 🔒 Robustness

- 添加 `terminate()` 方法：插件卸载/热重载时正确关闭引擎
- `initialize()` 添加防御性检查

## [v1.0.0a3] - 2026-05-31

(Previously labeled v1.0.0rc2)

### ⚡ Performance

- HDC similarity 从 O(n) 字节循环改为 O(1) `int.bit_count()`
- LLM assess 移出 session lock

### 🐛 Bug Fixes

- 修复 `confidence=0.0` 被 falsy 逻辑覆盖
- 修复 NaN/Inf 可穿透 kernel 事件解析
- 修复 `save_buffer()` 静默吞掉 OSError

### ✨ API Improvements

- 添加 `async with engine:` 上下文管理器
- 添加 `engine.exists(session_id)` 方法

## [v1.0.0a2] - 2026-05-31

(Previously labeled v1.0.0rc1)

### 🔒 Type Safety

- 全模块 mypy 零错误
- 修复 190 处类型标注缺失
- Release 打包改为 AstrBot 插件格式

## [v1.0.0a1] - 2026-05-31

### ✨ Initial Alpha

- 完整的 7 层情感计算管线
- 29 维身体状态向量，8 个子系统
- 双 EMA 人格漂移系统
- Void-Scar Engine / HDC / HGT 底层计算引擎
- 异步 `SylanneEngine` 高层 API
- 140 个单元测试
- GitHub Actions CI

## [v0.1.0-preview] - 2026-05-29

### Initial Preview Release

- 7 层情感计算管线原型
- 8 子系统情感状态模型
- 双层人格系统
- 三层记忆系统（后移除）
- LLM 语义评估器

