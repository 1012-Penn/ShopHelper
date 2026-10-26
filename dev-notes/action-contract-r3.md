# Action Contract R3 开发记录

## 观察

第二轮真实轨迹中出现三类动作失败：

1. 多 Agent 最终文本写“预览已生成”，但没有 `ticket_preview` 对象；
2. typed 实验生成的工单类型不满足生产 `CreateTicketInput` 的三值枚举；
3. 纯取消可行性问题反而生成预览。

这些问题不能靠增加“来源字段”修复。根因分别是：文本和动作没有同一边界、模型参数没有
从生产 registry 契约投影、以及没有区分问题、预览、确认和取消四种用户意图。

## 定位

旧 runtime 在 `on_write_intercept` 里无条件构造 `ticket_preview`，并对缺失类型使用
`"售后"` 默认值。它也没有完成检查，因此：

- 文本可被误当成成功；
- 模型给错会话号时可能保留错误值；
- 同一轮可重复进入写工具；
- 参数校验失败会直接终止，缺少一次受控修复；
- 取消没有会话级状态，后续可能重建预览。

## 设计取舍

新增 `app/action_contracts/`，不改 `builtin.py`：

- `ActionToolContext` 从 `ToolRegistryV2` 的实际 JSON Schema 读取参数和 enum；
- `ActionArgumentDecision` 保留原始参数、规范参数和理由；
- `ActionArtifact` 明确 `preview/awaiting_confirmation/confirmed/cancelled/failed`；
- `ActionSessionLedger` 保存每个会话至多一个 pending 动作；
- `MultiAgentRunner` 只做 opt-in 接线，默认关闭时现有路径和输出字段不变。

规范化只处理明确同义词。比如“退款/维修”归“售后”，但“物流处理/其他”必须澄清。
`conversation_id` 来自当前 run 的可信作用域；模型填另一个会话号会记录
`trusted_run_scope` 后被覆盖。`confirm_key` 由预览卡生成，模型值被移除；
即使模型尝试 `confirmed=True`，也会因不属于 schema 而被拒绝。

动作成功只由 artifact 判断。模型先声称有预览时，只追加一次明确的 schema/缺口反馈，
让 action Agent 生成合法 tool call；再失败则 artifact 为 `failed/missing_preview`，
最终回复被 guard 成“未能生成”，不会保留“已生成”的假成功。

取消/可行性问题可以在 planner 层就不安排 action；即使 planner 误安排了 action，
runtime 也短路而不调用模型。取消会写结构化 artifact；没有 pending 时返回
`cancelled/no_pending_action`。描述被修改时旧 pending 变为
`cancelled/superseded_by_revision`，新预览使用新确认键。

## TDD 与测试

先写失败测试，再补实现。定向覆盖：

- registry schema 的 required/enum；
- 退款、维修规范化和原文/理由留档；
- 模糊类型、缺描述；
- 模型提供另一会话 ID；
- 模型提供 `confirm_key` / `confirmed=True`；
- 只声称预览，一次反馈后仍失败；
- 一次合法工具调用完成修复；
- 未确认写不进 fake sink；
- 同键二次提交只生成一行且响应丢失重放成功；
- 确认后才把 artifact 标 `confirmed`；
- 已有预览取消、无 pending 取消、描述改口；
- 纯取消可行性不建预览；
- 默认关闭时旧 runtime 输出兼容。

离线 fixture 用来证明接线和安全不变量，不能代替模型效果。

## 真实实验结论

冻结全量运行保留了首轮失败：运行中 12,000-byte 请求包上限和每臂 180 次 generation
上限触发了大量基础设施错误，held-out 不能用于质量结论，未重跑。

修正预算检查后的 dev 诊断得到：

- baseline `5/8`，契约臂 `5/8`，配对 `2胜/2负/4平`；
- 额外预览 `1→0`；
- 创建调用尝试 `8→2`；
- 两臂未确认写均 0；
- 费用 `$0.0131205→$0.0115809`，P50 `6.263s→4.643s`。

这个结果说明动作约束减少了明显错误动作，但没有证明更会完成任务。合法预览和取消在
契约臂反而各少 1 个；不能只展示有利指标。诊断之后又修复了“只预览、不要提交”被误判
取消和 planner 未安排 action task 时取消 artifact 漏记的问题，未再跑付费验证。

## 适用边界

- 当前只绑定 `create_ticket`，其他写工具需各自定义业务规范化和完成条件；
- 会话 ledger 为进程内状态，不支持跨进程恢复；
- 最终 HTTP 仍需统一集成确认路由和响应模型；
- 模拟题小、模型为单一 provider，不能外推真实客服成功率或 ROI。
