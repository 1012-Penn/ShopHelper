# Typed Evidence Handoff R2 Design

日期：2026-10-27

## 目标

在现有可选多 Agent runtime 中增加一个默认关闭的 typed handoff 模式。默认 `raw`
模式保持现有构造方式、prompt 投影和 API 响应不变；显式选择 `typed` 后，角色只接收
任务依赖内、主体与作用域匹配、仍有效的工具观察。该模式同时提供同一 run 内只读查询
去重，但不改变写工具的确认、权限和幂等边界。

这不是模型事实抽取实验。模型 draft 和 Supervisor 摘要不产生已证实事实；结构化事实
只来自工具原始返回，原始返回保留用于审计。

## 契约

`app/evidence_contracts` 提供可 JSON 序列化的 `EvidenceClaim`、`TaskHandoff` 和
`EvidenceLedger`。

`EvidenceClaim` 包含：

- claim、run、产生任务和依赖任务；
- 主体 ID、predicate、value、unit；
- scope、policy version、知识截止/观察/有效时间；
- 原始工具、原始参数、原始返回、chunk/order 来源；
- `verified | unverified | unknown` certainty；
- superseded、过期状态。

缺失和无法解析的字段使用显式 `"unknown"` 或空列表，不根据自然语言 draft 猜测。
完全无工具来源的结构化 claim 会被拒绝。

`TaskHandoff` 区分：

- `role_handoff`：Supervisor 给内部角色的任务；
- `human_handoff`：人工转接事件；
- `tool_call`：普通工具调用事件。

内部 handoff 不携带可扩权字段。工具白名单仍只来自不可变 `ROLE_SPECS`。

## 投影与生命周期

- 无依赖任务只接收自己的 objective 与从 objective/调用参数确定的 subject，不继承其他
  任务的 facts。
- 有依赖任务只接收 `depends_on` 中已完成任务的 claims；主体、scope 和有效期也须匹配。
- 新观察对同一 subject/predicate/scope 形成更高 observation sequence 时，旧 claim 标记
  superseded。
- 动态订单/物流状态按可配置 TTL 过期；政策按返回的有效期和知识截止过滤。过期状态必须
  重查，不进入角色上下文。
- Supervisor 的 typed 输入只包含已完成任务的有效 claims，并保留来源与 unknown 标记。

## 只读去重

run 级 `ToolResultCache` 的 key 为工具名、规范化参数、主体 ID、知识/时效 scope。并发
等价调用合并到同一个 in-flight future。只有注册表明确标记为 `read` 的工具可进入缓存；
写工具、未知工具和动态过期读取不命中缓存。

统计始终保留 attempted、executed、deduplicated、error 调用数，以及每个角色实际构造的
prompt token 估算。去重返回仍计入模型可见工具结果 token。

## 兼容与接线

- `MultiAgentRunner(..., handoff_mode="raw")` 是默认值，老构造不变。
- `AgentOutcome` 和 `MultiAgentState` 只追加有默认值字段；现有 `facts`/`evidence` 输出形状
  不变。
- 不改 `app/main.py`、`app/config.py`、共享 RAG 图、评分脚本或 API。
- typed 诊断放在独立 trace/handoff 记录，不混入写预览。

## 评测

离线测试覆盖序列化、来源拒绝、依赖隔离、subject/scope、改口、过期、并发去重、写不缓存、
raw 默认兼容与实际角色 prompt。

真实模型对照使用同 `deepseek-flash`、温度、工具 fixture、Planner 条件和单次运行预算，
固定 10 dev + 20 held-out。数据按业务事实/源模板先分组再切分，报告数据、评分和配置 hash。
只运行一次完整对照；基础设施故障按预先规则记录，不据 held-out 改策略。

指标：任务成功、实体/条件错误、来源缺失、handoff 关键事实遗漏、重复工具调用、输入 token、
P50/P95、每成功任务成本、配对得失。任何漏关键事实和额外解析失败都留在分母。

## 停止规则

- 本轮独立持久账本保护上限 `$2.00`，调用前原子预留，按 provider usage 结算。
- 最大 2 次有界重试；429 才退避；工具、裁判、修订和失败费用全部记账。
- 超过保护上限或凭证/基础设施不可用时保留已完成结果，报告未测原因，不扩额。
