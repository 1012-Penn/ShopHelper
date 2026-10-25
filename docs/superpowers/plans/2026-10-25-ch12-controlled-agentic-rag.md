# CH12 受控 Agentic RAG 实施计划（2026-10-25）

> For agentic workers: REQUIRED SUB-SKILL `executing-plans`。
> 设计稿:`docs/superpowers/specs/2026-10-25-ch12-controlled-agentic-rag-design.md`(含 ADR-1~5,均已用户确认)。
> 分支 `codex/ch12-agentic-rag`,基线 origin/main 0751de1。禁止 push、禁止合并 main、禁止改冻结题/金标/旧报告。

## Goal

按设计稿实现受控 Agentic RAG:循环核心 + 图接入(`retrieval_route` 开关,默认零变化)+
审计表 + 工单幂等 + 评测三组模式;完成离线/缓存验证与付费预估;全程零付费调用。

## Global Constraints

- 证据闸阈值与蕴含判定语义不得放宽;frozen-v1/manifest/旧报告 sha 不动。
- 默认 `retrieval_route=hybrid_rerank` 下全部既有测试必须保持绿色。
- 台账不复制到 worktree;评测用 `--ledger` 指向全局台账,本轮零扣减(ADR-4)。
- `rag_loop_steps` 相关文档/代码只说「可审计」,不说恢复/回放(ADR-1)。
- 未实测的质量指标一律标「未实测」(ADR-5)。

## File Structure

Create:
- `app/rag_loop.py`(循环核心:GapReport/RagLoopLimits/ControlledRagLoop)
- `app/graph/agentic.py`(agentic_retrieve 节点)
- `app/rag_steps.py`(RagLoopStepStore)
- `tests/test_ch12_rag_loop.py`、`tests/test_ch12_graph.py`、`tests/test_ch12_idempotency.py`、`tests/test_ch12_harness_modes.py`
- `scripts/acceptance_ch12.py`、`docs/eval/ch12-agentic-rag-report.md`、`dev-notes/ch12.md`

Modify:
- `app/config.py`(ch12 参数块)、`app/graph/state.py`(3 字段)、`app/graph/builder.py`(条件边重定向+initial_state)
- `app/prompts.py`(GAP_ANALYZER_PROMPT + GapReport)
- `app/models.py`(RagLoopStep;Ticket.confirm_key)、`db/init.sql`(两表 DDL)
- `app/tools/builtin.py`(create_ticket 幂等)、`app/schemas.py`(confirm_key)、`app/routers/tickets.py`(透传)、`app/graph/agent.py`(预览帧带 confirm_key)
- `scripts/ch11_harness.py`(rag 模式检索段+轨迹)、`scripts/eval_ch11.py`(--rag-mode/--ledger/契约 rag 块)、`scripts/eval_cost.py`(GuardedGapAnalyzer)、`scripts/eval_prompts.py`(JsonGapAnalyzer)
- `.env.example`、`README.md`(ch12 节)

Not changing: `app/retrieval.py` 闸逻辑、`app/graph/entry.py route()`、冻结评测数据、旧报告、CH11 harness 判分。

## Tasks

### Task 1 循环核心(app/rag_loop.py + prompts)
- [ ] `GapReport`(sufficient/action/missing_aspects/conflicts/sub_queries/clarify_question/reason)+ `GAP_ANALYZER_PROMPT`(app/prompts.py)
- [ ] `RagLoopLimits`(max_rounds/token_budget/time_budget_s/max_sub_queries/final_top_k)
- [ ] `ControlledRagLoop.run(query, prepared=None) -> RagLoopResult`:首轮检索→闸通过即停;否则蕴含再判→缺口分析→子查询去重→补检索→max-score 合并;预算/故障熔断;停止原因全集;`on_step` 回调逐轮外抛
- [ ] BudgetExceeded 一律向上抛;其余故障谨慎收敛
- [ ] 单测:首轮即停零额外调用、轮数/token/时间熔断、子查询去重与 no_new_queries、judge 短路(merging 后 answered=true 即 sufficient)、analyzer 故障收敛、retrieval 故障收敛、decompose_once=max_rounds(1)

### Task 2 图接入与审计表
- [ ] config.py ch12 块;state.py 3 字段;builder.py 条件边重定向(默认不变)+ initial_state 默认值
- [ ] `app/graph/agentic.py`:节点包循环(to_thread)、rag_loop_steps 逐轮落库(失败仅 warning)、澄清/拒答/转人工收口、citations 全量单发
- [ ] `db/init.sql` rag_loop_steps + tickets.confirm_key DDL;models.py 对应 ORM;`app/rag_steps.py` store
- [ ] 单测:默认路由零变化冒烟、decompose_once/controlled_loop 走 agentic_retrieve、initial_state 无跨回合残留、拒绝/澄清出口、审计行数=轮数

### Task 3 工单确认幂等(ADR-3)
- [ ] schemas 两请求体加 confirm_key(8-64 字符);tickets 两端点透传;agent.py 预览帧生成并携带 confirm_key(uuid4().hex)
- [ ] CreateTicketInput 加 confirm_key;create_ticket 持键插入,唯一冲突回查返回同一 ticket_no(deduplicated=true)
- [ ] 单测:并发双击同键→1 单;同键重试→同一工单;同内容不同键→2 单;无键→旧行为

### Task 4 评测三组模式
- [ ] HarnessConfig 增 rag_* 字段;`run_rag_retrieval_case`(复用 ControlledRagLoop,产出与基线同 schema 的 trace + rag_steps/rag_stop_reason/rag_rounds)
- [ ] eval_ch11.py:--rag-mode/--ledger;契约 `rag` 块(三组同键不同值)+ harness.version 区分;gap analyzer 装配(Guarded+Cache)
- [ ] eval_cost.py GuardedGapAnalyzer、eval_prompts.py JsonGapAnalyzer
- [ ] 单测:三组模式分派、缓存命中不扣费、guard 抛错零调用、rag 轨迹字段、failure_labels 合并

### Task 5 离线与缓存验证(零付费)
- [ ] WSL 全量 pytest 绿;`--mode offline` 两 split 通过
- [ ] baseline 组 full 缓存重放:0 新增真实调用、cost_actual=0、全局台账记录数不变
- [ ] 三组 contract_diff 仅 rag./harness. 差异(offline 构造校验)

### Task 6 付费预估与文档留痕
- [ ] 预估脚本/干跑统计(不调用模型):逐组新增调用量×单价,写入报告与 dev-notes
- [ ] `docs/eval/ch12-agentic-rag-report.md`(三级证据分节,未实测标「未实测」)
- [ ] `scripts/acceptance_ch12.py` 九项验收;README/.env.example;`dev-notes/ch12.md`
- [ ] WSL 内 `ch12:` 前缀原子提交,不 push

## Review Focus

- 默认路径是否真零变化(builder 映射、既有测试全绿为准);
- 证据闸语义是否与基线严格同源(合并后再判用的是同一 judge/prompt/阈值);
- 幂等是否覆盖两端点且不依赖 tool_call_id/参数指纹;
- 台账/快照是否零写(本轮零付费,--ledger 只读);
- 报告是否区分三级证据、未实测是否留空。
