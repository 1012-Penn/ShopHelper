# ShopHelper 简历证据候选

日期：2026-10-27

集成分支：`codex/interview-integration-review-20261027`

这里只列本集成版本可回查到代码、测试或逐题轨迹的表述。样本均为模拟客服场景，不能写成真实
客户 ROI、生产 SLA 或普遍模型优越性。

## 推荐优先级

简历建议放 2–3 项：

1. 同会话可靠执行：工程门槛高、证据强、无需模型效果假设。
2. 多 Agent 的成本/任务分桶取舍：能讲 Agent 设计、评测和成本意识。
3. 三态 RAG 评分审计：能讲“先修尺子、再报结果”和负结果，不把技术存在当收益。

## 候选句与证据

### 1. 持久 run、SSE 重放与租约

可写：

> 为客服 Agent 增加持久 run、可重放 SSE 事件和数据库会话租约，覆盖同会话竞争、客户端断连、
> 子进程崩溃与未知写恢复；集成全量回归 893 passed，故障注入验收 0 模型调用。

证据：

- 集成 commit `fe8c18c`；
- `tests/test_interview_reliability.py`；
- `scripts/acceptance_interview_reliability.py`；
- 本次隔离 SQLite 32/32 run 成功，同会话竞争 1 created / 1 busy；
- 临时 MySQL 8.0.46 独立 socket 完成 schema、租约、事件和确认键唯一约束 smoke。

边界：

- 这是 run 级只读恢复，不是任意 LangGraph 节点 checkpoint；
- 32/32 与 30.32 run/s 只来自隔离 SQLite 固定负载，不是生产吞吐；
- 不承诺任意第三方写 exactly-once。

### 2. 多 Agent 与单 Agent 的同题配对

可写：

> 实现 Supervisor/Planner、检索 Agent 与行动 Agent 的受控协作链路，使用独立工具白名单和结构化
> 依赖；20 道 held-out 模拟题上多 Agent 17/20、单 Agent 15/20，配对 4 胜 2 负 14 平。

更强的任务分桶句：

> 在 20 道 held-out 中，复杂桶多 Agent 6/8、单 Agent 3/8；简单桶为 11/12 对 12/12，
> 因此保留单 Agent 为默认并探索选择性路由。

成本句：

> held-out 每成功任务成本约 $0.00255 对 $0.00040，P95 约 16.4s 对 3.55s；多 Agent 仅作复杂
> 任务的可选路径。

证据：

- `tests/eval/multi_agent/runs/frozen/summary.json` 与压缩逐题轨迹；
- 本次从 60 条保存轨迹独立重算，复现 17/20、15/20、4/2/14 与分桶结果；
- `MA-FRZ-022` 等失败仍保留在分母。

数据条件：

- `deepseek-flash`，温度 0，零重试；
- 10 dev + 20 held-out 模拟题；
- 两侧共享模型、工具、证据和输出上限；
- 多 Agent 金标不进入模型输入。

禁止外推：

- 不说“多 Agent 提升 10 个百分点”；10/20 的二分差值不是稳定总体估计；
- 不把 27/30 对 23/30 当作 held-out 泛化数字；
- 不说所有复杂任务都更强，也不说多 Agent 已适合默认生产；
- 不把内部角色 handoff 与“人工转接为 0”的 N/A 混为一谈。

### 3. 三态 RAG 评分与负结果

可写：

> 构建 RAG 三态引用评测与版本化重评，对 14 组、1050 条历史响应零调用统一重算；区分确定支持、
> 确定冲突和待复核，修复标点后引用、单位/日期、背景事实与 220 条声明级计量错误。

如果要突出研究诚实性：

> 在固定 10 题真实模型小样本中发现答案复核虽能拦截订单串用，但事实正确由 1/9 降至 0/9、
> 误拒由 1 增至 2，因此保持功能默认关闭并保留负结果。

证据：

- `tests/eval/overlays/rag-quality-v3.1.json`；
- `tests/eval/rag_quality/v3_1/aggregate.json`；
- `tests/eval/rag_quality/v3_1/verified/dev/aggregate.json`；
- v3 产物和原始回答未覆盖。

禁止外推：

- `uncertain` 不是正确，也不能隐藏；
- 1050 条是历史响应评分，不是 1050 次新模型实验；
- 固定 dev 的三个模式共享历史回答/缓存，不是三次独立复现；
- 历史 prompt 含 `expected_behavior`，不能写成无提示线上准确率；
- 61.992s 是复核调用尝试 P95，不是端到端客服延迟。

### 4. 上下文预算

可写：

> 将累计摘要作为可丢弃的投影计入实际滑窗预算，优先保留最近用户改口；覆盖多段摘要、过期工具
> 结果和超预算边界，不改原始消息或摘要存储。

证据：

- `build_layered` 的 `total_injected_tokens`、`summary_truncated`、`budget_exceeded`；
- `tests/test_interview_reliability.py` 中的摘要投影测试。

边界：

- token 估算不是供应商 tokenizer；
- 单轮原文自身超预算时是 best-effort，不是硬保证。

## 不建议写进简历

- “多 Agent 显著提升”“生产级高可用”“任意节点故障恢复”；
- “答案复核提升质量”；
- “结构缺口触发提升 RAG”；
- “真实客服 ROI”“线上准确率”“生产 SLA”；
- 仅用 `893 passed` 证明模型能力；它只证明软件回归通过。
