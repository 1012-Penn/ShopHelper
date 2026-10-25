# CH12 受控 Agentic RAG · 评测与验证报告(2026-10-25)

分支 `codex/ch12-agentic-rag`(基线 origin/main 0751de1)。设计稿:
`docs/superpowers/specs/2026-10-25-ch12-controlled-agentic-rag-design.md`(ADR-1~5)。
三组定义:baseline=CH11 单轮 hybrid_rerank;decompose_once=一轮拆解+补检索+终判;
controlled_loop=最多 3 轮「蕴含再判→缺口分析→子查询去重→补检索→max-score 合并」。
三组同 frozen-v1(90 题)/同语料/同索引快照(build_id `ebfd4947…`)/同回答模型
(deepseek-flash,温度 0)/同 `EVAL_ANSWER_PROMPT` 与判分;契约差异仅 `rag.*` 与
`harness.*` 块(验收脚本第 6 项机械校验)。

**证据分级声明(ADR-5)**:本报告区分三级证据——① 离线替身验证(fakes);
② 缓存重放(真实模型历史输出);③ 真实付费模型结果。**本轮未执行任何付费调用,
三组质量对比指标全部未实测**;下文逐节标注证据等级。

## 1. 离线替身验证(证据等级①)

- 全量测试 `python -m pytest tests/`:**704 passed / 2 skipped / 1 xfailed**,连续两轮全绿,
  零回归(默认 `retrieval_route=hybrid_rerank` 下既有测试全绿 = 默认路径零变化的实证)。
- 新增 45 项 ch12 测试:
  - `tests/test_ch12_rag_loop.py`(18):首轮即停零额外调用、decompose_once 语义、
    轮数/token/时间熔断、子查询规范化去重与 no_new_queries、蕴含判断器说「没回答」
    高分也不放行(闸未放宽)、clarify/refuse/escalate 收口、检索/分析器故障谨慎收敛、
    BudgetExceeded 穿透、审计回调逐轮触发且失败不阻断;
  - `tests/test_ch12_graph.py`(10):默认路由无 agentic_retrieve 节点、开启后注册并
    重定向、initial_state 每回合重置 rag 字段、充分→agent/不充分→log 全图走线、
    审计行数=轮数且末行回填停止原因;
  - `tests/test_ch12_idempotency.py`(8):同键双击/超时重试同一工单、同内容不同键
    两单、无键旧路径不变、**确定性复现「双请求先查无再插入」交错下唯一约束兜底**、
    线程竞争下「同键至多一单」不变量、预览帧每次下发新键;
  - `tests/test_ch12_harness_modes.py`(9):三组检索段分派、rag 轨迹字段、
    GuardedGapAnalyzer 缓存命中不计费/未命中先守卫后记账/守卫拦截零调用、
    三组契约 diff 仅 rag/harness 块、BudgetExceeded 穿透。
- `--mode offline`:frozen-v1 与 regression-v1 数据/泄漏校验通过
  (frozen canonical_sha256 与冻结 manifest 一致;leakage blocking=0)。
- 验收脚本 `scripts.acceptance_ch12`:**9/9 通过**(冻结不动/默认路由零变化/循环界限/
  审计表仅可审计/幂等三场景/三组契约/缓存重放零新增/台账零 ch12 扣减/证据闸源码与
  origin/main 零差异)。

## 2. 缓存重放验证(证据等级②,baseline 组)

命令:`--split frozen-v1 --mode full --allow-paid --rag-mode baseline --snapshot …ebfd4947….db
--ledger <全局台账> --out reports/ch12`(单次运行上限默认 $0:任何缓存未命中会在守卫处
安全中止,故本运行结构上不可能产生付费调用)。

结果(reports/ch12/frozen-v1/,90 题逐题 trace 落盘):

| 指标 | 本次重放 | CH11 M1 报告 | 一致 |
|---|---|---|---|
| L1 R@3 / R@5 / R@10 | 0.6154 / 0.7538 / 0.9077 | 0.6154 / 0.7538 / 0.9077 | ✓ |
| MRR@10 / group_recall | 0.5116 / 0.9538 | 0.5116 / 0.9538 | ✓ |
| multi_hop R@3 / policy_conflict R@3 | 0.1875 / 0.25 | 0.188 / 0.250 | ✓ |
| L2 gate false accept / false block | 0 / 3 | 0 / 3 | ✓ |
| 真实新增 LLM 调用 | **0** | 0(同 contract 复跑) | ✓ |
| 等价成本 cost_estimate | $0.0636 | $0.0636 | ✓ |
| 全局台账 | 385 条,$0.4644/$10,零新增 | $0.4644 | ✓ |

结论:harness 的 ch12 改动未扰动 baseline 路径(指标与 CH11 M1 逐位一致);
baseline 组零付费即可完整复现。重放轨迹另给出循环入口统计:**90 题中 23 题首轮
证据闸未过**(hard_negative 10、ood 6、clarify 3、multi_hop 2、multi_constraint 1、
entity_mismatch 1)——其中期望行为为 refuse 的 16 题(循环不应放行,闸未放宽),
真正可能靠补检索翻盘的约 7 题;这也是 CH11「gate_false_block=3 + 拒答桶误放 0」
口径的交集,与「只有值得补查的问题进入循环」的设计假设一致。

## 3. 三组质量对比(证据等级③ —— **未实测**)

| 维度 | baseline | decompose_once | controlled_loop |
|---|---|---|---|
| L1 检索(group_recall/R@3/分桶) | 见 §2 重放值 | 未实测 | 未实测 |
| L2 闸误放/误拦 | 0 / 3 | 未实测 | 未实测 |
| L3 strict/partial accuracy、unsupported_claim_rate | 未实测(重放可算,留待付费后随三组一并出) | 未实测 | 未实测 |
| L4 合理拒答/澄清 | 未实测 | 未实测 | 未实测 |
| L5 延迟 P50/P95、每题调用数、成本 | 未实测 | 未实测 | 未实测 |
| multi_hop 16 题逐题得失 | 未实测 | 未实测 | 未实测 |
| policy_conflict 8 题逐题得失 | 未实测 | 未实测 | 未实测 |
| CH11 已知 citation_unsupported 题得失 | 未实测 | 未实测 | 未实测 |

说明:回答层为评测专用 prompt(与 CH11 同口径,非线上 system prompt),检索循环为
线上同一 `app/rag_loop.ControlledRagLoop` 核心——本报告不冒充线上完整链路;线上
路径的行为由 §1 图级测试覆盖。付费实测获批后,重跑三组 `--rag-mode` 即可填表,
契约与判分无需任何改动。

## 4. 付费预估(未执行;等统一安排)

依据:重放轨迹的循环入口统计(23 题进循环)+ pricing.yaml(deepseek-flash
$0.30/M 输入、$1.20/M 输出;嵌入/重排 $0.01/M,verified=true)。
单价估算:缺口分析 ≈1.5K tok 入/200 出 ≈ $0.00069/次;蕴含判断 ≈1.5K/128 ≈
$0.00062/次;回答 ≈1.8K/400 ≈ $0.0010/次;子查询重排 ≈ $0.0001/次。

| 组别 | 新增付费调用构成(估) | 调用数(估) | 费用(估) |
|---|---|---|---|
| baseline | 0(全缓存命中,§2 已证) | 0 | $0 |
| decompose_once | 23 缺口分析 + 23 合并终判 + ~46 子查询(嵌入+重排) + ~6 回答(证据集变化致缓存 miss) | ≈100 | $0.04–0.08 |
| controlled_loop(≤3 轮) | ~35 缺口分析(假设进循环题均轮 1.5 轮)+ ~58 蕴含判断 + ~70 子查询 + ~8 回答 | ≈170 | $0.07–0.15 |

- 合计新增 **$0.11–0.23**(±50%:循环深度与 token 用量依赖真实分析器输出,
  付费前无法精确);建议实测时 `--max-cost-usd 0.5`。
- 全局台账余额 $9.5356($10 − $0.4644),充足。
- **并发语义(ADR-4)**:台账 flock 只互斥记录写入,不保证 guard→调用→add 整段
  互斥;若与其他 worktree(如 CH13)并跑付费,须先加整轮互斥或原子预算预留。

## 5. 结论与下一步

1. 实现与验证完成度:循环核心/图接入/审计表/幂等/三组 harness 全部落地并离线验证;
   baseline 组缓存重放与 CH11 M1 逐位一致、零新增付费、台账零扣减。
2. 按总纲「没有净收益时保留基线路由」:`retrieval_route` 默认仍为 `hybrid_rerank`,
   新路由未切默认、未宣称生产可用(校准 `calibration_status=stale`,ADR-5)。
3. 待统一安排后执行付费实测(预估 $0.11–0.23):跑三组 `--rag-mode` 填 §3 表,
   逐题报告 multi_hop/policy_conflict/citation_unsupported 得失;证据闸不动。
