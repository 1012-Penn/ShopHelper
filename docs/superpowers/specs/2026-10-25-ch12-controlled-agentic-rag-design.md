# CH12 受控 Agentic RAG 设计（2026-10-25）

上游总纲:`docs/superpowers/specs/2026-10-24-ch11-ch14-agent-production-roadmap-design.md` §5;
评测契约约束:CH11 设计稿 §7(`docs/superpowers/specs/2026-10-24-ch11-evaluation-baseline-design.md` L540-548)。
工作分支 `codex/ch12-agentic-rag`(自 origin/main 0751de1),worktree `.worktrees/ch12-agentic-rag`。

## 1. 问题与目标

现状:商品咨询一次 `hybrid_rerank` 检索 + 一次证据闸判断,缺证据直接拒答。
CH11 基线(frozen-v1, 90 题)显示 multi_hop R@3=0.188、policy_conflict R@3=0.250,
是检索层主要瓶颈;直接拒答可能错过补一轮检索就能答的题。

目标(总纲 §5):首次检索 → 判断已覆盖事实与缺口 → 选择澄清/子问题/补检索 →
合并去重并检查冲突 → 证据充分则回答,否则拒答或转人工。界限明确(轮数/token/时间),
过程可审计,写操作幂等;与 CH11 基线做同题配对消融,无净收益则保留基线路由。

## 2. 范围与非目标

做:
- 受控循环核心 `app/rag_loop.py`(线上图与评测 harness 共用同一实现);
- 图接入:`retrieval_route` 开关,默认 `hybrid_rerank` 图结构零变化;
- `rag_loop_steps` 审计表(仅可审计,见 ADR-1);
- 工单确认幂等(ADR-3);
- 评测 harness 三组模式(baseline / decompose_once / controlled_loop)+ 契约 + 逐题轨迹;
- 离线与缓存验证;付费预估(不执行付费调用)。

不做(边界):
- 图检索/知识索引服务(ch13);生产路由最终决策、鉴权/部署/压测(ch14);
- 不修改 frozen-v1/regression 题集、金标、manifest 与任何旧报告;
- 不放宽证据闸(分数闸阈值与蕴含判断器语义与 CH11 完全一致);
- 不把新路由切为默认、不宣称生产可用(ADR-5);
- 不做跨进程断点续跑/回放(ADR-1)。

## 3. 架构

### 3.1 受控循环(单回合闭环)

```
首轮检索(原查询, hybrid_rerank, 与基线完全同一调用)
  └─ 通过证据闸(分数闸+蕴含判断器) → stop=first_pass_sufficient → 回答(零额外 LLM 调用)
  └─ 未通过 → 循环 r = 1..max_rounds:
       ├─ 预算检查(轮数/token/时间) → 超限即谨慎收敛
       ├─ 蕴含判断器再判合并证据(同 EvidenceJudge、同 prompt、同缓存)
       │    answered=true → stop=sufficient → 回答
       ├─ 缺口分析器(GapReport, 温度0) → action:
       │    retrieve_more → 子查询(规范化去重,同查询绝不重复检索) → 补检索 → max-score 合并去重 → 下一轮
       │    clarify      → stop=clarify_needed → 澄清回复结束本回合
       │    refuse       → stop=refuse_no_evidence → 拒答+落池+转人工动作
       │    escalate     → stop=escalate_conflict → 说明冲突+转人工动作
       └─ 分析器故障 → stop=analyzer_error → 按已知证据分数闸谨慎收敛或拒答
```

关键语义:
- **充分(放行)判定与基线同源**:分数闸用同一 `evidence_confidence`+同一 calibration;
  语义层用同一 `EvidenceJudge`(同一 prompt、温度 0、评测侧同一缓存键)。合并证据的
  蕴含判断是把基线单轮的判定在合并结果上重跑,**不是新标准,也未放宽**。
- 缺口分析器的 `sufficient/action=answer` 仅为参考,**不作为放行依据**;放行只认证据闸。
- 首轮即充分(单跳多数情形)不产生任何额外 LLM 调用,行为与基线一致。
- 子查询检索 `use_rewrite=False`(分析器输出已是规范问法,同 refund/policy_retrieve 惯例);
  低置信子查询结果只进诊断快照,不进证据合并(同 refund.py 语义)。
- `decompose_once` = `controlled_loop` 的 `max_rounds=1` 特例(一轮拆解+合并+终判)。

### 3.2 界限与参数(`app/config.py` ch12 块)

| 参数 | 默认 | 说明 |
|---|---|---|
| `retrieval_route` | `hybrid_rerank` | 三值枚举 hybrid_rerank / decompose_once / controlled_loop(ADR-2) |
| `max_rag_rounds` | 3 | controlled_loop 最大补检索轮数(不含首轮) |
| `rag_token_budget` | 3000 | 循环内缺口分析输出 token 预算(estimate_tokens 口径,仿 agent_token_budget) |
| `rag_time_budget_s` | 45.0 | 循环墙钟预算(秒) |
| `rag_max_sub_queries` | 3 | 每轮最多子查询数 |

停止原因枚举:`first_pass_sufficient / sufficient / rounds_exhausted / time_budget /
token_budget / no_new_queries / clarify_needed / refuse_no_evidence / escalate_conflict /
retrieval_error / analyzer_error`。全部进 trace、rag_loop_steps 与评测逐题轨迹。

查询去重:规范化(折叠空白+小写)后与已执行查询集合比对,重复子查询跳过;
全部子查询重复时 stop=no_new_queries。

### 3.3 图接入

- `ChatState` 增 `rag_rounds / rag_stop_reason / rag_steps`;`initial_state` 同步补默认值
  (防 checkpointer 跨回合残留,同现有字段纪律)。
- 新节点 `agentic_retrieve`(app/graph/agentic.py):整个循环在 `asyncio.to_thread` 内执行;
  出口条件边 `gate_passed → agent`,否则 `→ log`(澄清/拒答/转人工在本节点内收口,
  不使用 interrupt()——py3.10 下被官方守卫禁用,澄清以普通回复结束本回合,下回合重走图)。
- `builder.py`:仅当 `retrieval_route ∈ {decompose_once, controlled_loop}` 时注册节点并把
  intent 条件边映射中的 `"retrieve"` 重定向到 `agentic_retrieve`;默认值下图结构与
  hybrid_rerank 路径逐字节不变。`route()` 纯函数不动。
- citations 帧为覆盖语义:循环结束后一次性全量发送最终合并证据(agent 轮内工具证据再按
  n_offset 续编,遵守 agent.py 既有规则)。
- 拒答路径复用 `REFUSAL_MARKER` + `pool.insert("retrieval_low_conf", ...)`(与基线 fallback
  同入口),另附 `suggested_actions=["transfer_human","create_ticket"]`。

### 3.4 审计表 `rag_loop_steps`(仅可审计)

每轮一行:conversation_id、turn_user_msg_id(本轮 user 消息 id,initial_state 生成)、
round_index、query(原查询)、queries(本轮实际执行查询 JSON)、sub_queries、evidence_ids、
merged_evidence_ids、judge_answered、analyzer_action、conflicts、stop_reason(仅末行)、
latency_ms、created_at。写入失败仅 warning,绝不阻断回合(审计不拦执行,同 tool_audit 惯例)。

## 4. ADR(架构决策记录;1/3/4/5 含用户 2026-10-25 修正)

### ADR-1 执行状态持久化 = 单回合闭环,`rag_loop_steps` 仅声称「可审计」
多轮检索在一次图执行内完成(复刻 agent.py:111 步数/token 双闸+熔断收敛范式);
checkpointer 维持 InMemorySaver。`rag_loop_steps` 的语义边界:**只提供事后审计,
不提供断点续跑、回放或跨进程恢复**。若将来要声称恢复能力,必须补:稳定 run/turn ID、
持久化中间状态(已执行查询、合并结果、剩余预算)、恢复入口,并测试每个关键步骤后
杀进程再恢复——本轮不做、代码与文档均不做此声明。故障恢复语义 = 按已知证据谨慎
收敛(能过分数闸则答,否则拒答/转人工),停止原因如实记录。

### ADR-2 线上路由切换 = `retrieval_route` 三值枚举
`hybrid_rerank | decompose_once | controlled_loop`,默认 `hybrid_rerank`(现状零变化);
与评测三组一一对应,支持线上单独灰度「仅拆解一次」中间档;回滚 = 改回默认值重启。
不采用布尔开关(表达不了中间档);不加请求级覆盖(避免多一个生产入口)。

### ADR-3 工单确认幂等 = 「一次用户确认」稳定请求键 + DB 唯一约束
- 确认键(confirm_key)代表**一次用户确认动作**:预览卡下发时服务端生成并随
  `ticket_preview` 帧带给前端,确认时原样回传;`/api/tickets` 按钮路径由前端为
  「一次点击」生成并复用该键(重试同键、再点新键)。
- `tickets` 表增 `confirm_key VARCHAR(64) NULL UNIQUE`;create_ticket 工具持键插入,
  唯一约束冲突时查出已建工单并返回同一 ticket_no(响应标记 `deduplicated: true`)。
  覆盖 `/api/tickets` 与 `/api/tickets/confirm` 两个端点。
- **不**依赖 `tool_call_id`(图重放/前端重建时可能变化或缺失);
  **不**用参数指纹合并两次合法提交(同内容两次独立确认 = 两个键 = 两张工单)。
- 测试:并发双击同键只建一单;超时后同键重试返回同一工单;同内容两次独立确认
  (不同键)建两单;无键旧路径行为不变(向后兼容,文档注明无幂等保护)。

### ADR-4 预算与并发的真实语义(修正:台账 flock ≠ 整段互斥)
现有 `BudgetLedger.record()` 的 `flock` 只保证**台账记录写入**的互斥;
`CostTracker.guard → 付费调用 → add` 整段之间无互斥——两个进程交错执行时,
各自 guard 都可能在对方 add 前通过,理论可合计超出台账上限(超限量 ≤ 一段交错窗口
内的调用总额)。本轮 **不执行任何付费调用**;`.eval.lock` 只互斥本 worktree 的评测
进程,不跨 worktree。将来付费并跑前必须先做整轮互斥(全局锁包住整次评测)或原子
预算预留(预扣额度、结束退回),届时另行评审——本 ADR 只如实声明现状边界。

### ADR-5 校准状态与生产可用性边界(修正:stale 校准、三级证据)
CH11 契约已标注 `calibration_status=stale`(旧校准产物不覆盖新语料)。因此:
- 新路由(`decompose_once`/`controlled_loop`)在新语料上重新校准并验证之前,
  **不得切为默认值、不得宣称生产可用**;本设计与报告仅称「实验路径」。
- **禁止用 frozen-v1 调阈值/校准**(冻结集只用于对照评测,防自证)。
- 评测报告必须区分三级证据:**离线替身验证**(fakes,证明逻辑与界限)、
  **缓存重放**(真实模型历史输出,证明链路与缓存兼容)、**真实付费模型结果**
  (本轮未运行)。未运行的三组质量指标一律标「未实测」,不填占位数值。
- 证据闸阈值未做任何放宽;若对照显示新路由无净收益,保留 hybrid_rerank 基线路由。

## 5. 评测设计

三组同一 frozen-v1(90 题)、同语料/索引快照(build_id ebfd4947…)、同回答模型
(deepseek-flash,温度 0)、同 `EVAL_ANSWER_PROMPT` 与证据块格式、同判分
(`check_case_deterministic` 不改);唯一变量 = 证据如何取得(baseline 单轮 /
decompose_once / controlled_loop)。`contract_diff()` 机械验证三组契约差异仅
`rag.*` 与 `harness.*` 块。

- harness:`--rag-mode` 三值;decompose/loop 组检索段调 `app/rag_loop.ControlledRagLoop`
  (与线上同一核心;检索服务/蕴含判断器/缺口分析器均注入评测的 Guarded+Cache 包装)。
- 缺口分析器评测侧包装 `GuardedGapAnalyzer`:ResponseCache(key=model+prompt_sha+温度+
  query+evidence_ids)+ CostTracker guard/add,与 GuardedEvidenceJudge 同模式。
- 逐题轨迹新增:`rag_steps[]`(每轮查询/子查询/证据ID/judge 判定/action/延迟)、
  `rag_stop_reason`、`rag_rounds`、analyzer/judge 调用数与缓存命中数;
  启用既有未用的 `timeout` 标签(stop_reason=time_budget)。
- 输出 `reports/ch12/<split>/`(gitignored),聚合结论写 `docs/eval/ch12-agentic-rag-report.md`;
  回答层为评测专用 prompt(与 CH11 同口径,**非线上 system prompt**),报告明示该边界,
  不冒充线上完整链路;线上路径另有图级单测覆盖。

### 5.1 本轮执行的验证(全部零付费)
1. 全量离线单测(fakes);
2. `--mode offline` 数据/泄漏校验;
3. baseline 组缓存重放(0 新增真实调用、全局台账零新增记录);
4. 三组契约一致性校验(contract_diff 仅 rag/harness 块差异);
5. 付费预估:按缓存命中分布估算新增调用量 × pricing.yaml 单价,逐组明细,等统一安排。

## 6. 验收标准(acceptance_ch12.py)

1. 冻结文件 sha 与 CH11 验收一致(题集/金标/旧报告未被改动,acceptance_ch11 的
   OLD_SHA 校验继续通过);
2. 默认 `retrieval_route=hybrid_rerank` 下既有测试全绿(默认路径零变化);
3. 界限测试:轮数/token/时间熔断、查询去重(同查询不重复检索)、故障谨慎收敛、
   停止原因记录齐全;
4. `rag_loop_steps` 逐轮落库且含 turn 标识;文档与代码仅声称「可审计」(ADR-1);
5. 工单幂等三场景(并发双击/超时重试/同内容两次独立确认)+ 无键旧路径回归;
6. 三组 contract 除 `rag.*`/`harness.*` 外 diff 为空;
7. baseline 缓存重放 0 新增付费调用、全局台账零新增记录(ADR-4 边界如实声明);
8. 报告区分三级证据;未实测质量指标标「未实测」;
9. 新路由未切默认、未宣称生产可用;证据闸阈值与蕴含判定语义未放宽(ADR-5)。

## 7. 风险

- 缺口分析器是新 prompt、无历史缓存:付费实测时该部分全部为新增调用,预估与实测
  可能有 ±30% 偏差;
- CH11 已知最大问题是回答层引用支持(unsupported_claim_rate 0.5267)与条件合规,
  补检索可能提升 L1 但不改善甚至不改变 L3——按总纲「没有净收益时保留基线路由」
  如实报告,不以 Agentic 名称代替结果;
- 合并证据跨查询 max-score 融合,分数来自不同子查询的重排,可比性弱于单查询内排序
  (refund.py 已有同语义先例,风险接受并记录);
- Milvus Lite 快照为本 worktree 私有副本(只读),与 CH13 无写冲突;台账经 `--ledger`
  指向全局文件,只读校验 + 本轮零付费,无并发扣减窗口(ADR-4)。
