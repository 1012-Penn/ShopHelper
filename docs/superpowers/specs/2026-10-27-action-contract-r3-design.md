# Action Contract R3 短 Spec

日期：2026-10-27

## 目标

把“模型说已经预览/创建”改为可验证的动作 artifact，并区分读取、工单预览、客户端确认、
取消和失败。默认不改变现有 runtime。

## 契约

- 工具参数和 enum 从活动 registry 的 JSON Schema 读取。
- `conversation_id` 由当前可信 run 注入；模型不能选择另一会话。
- `confirm_key` / `confirmed=True` 只能来自客户端确认路径。
- 只格式化无歧义类型：退款、退货、维修、返修、换货归售后。
- 模糊类型、非法 enum 或缺少业务描述要求澄清。
- `ActionArtifact` 状态为 `preview`、`awaiting_confirmation`、`confirmed`、
  `cancelled`、`failed`。
- 普通查询/取消可行性不建预览；取消 pending 后不重建。
- 模型声称有预览但没有结构化 tool call 时，最多一次明确反馈，仍失败则如实失败。

## 状态流

```text
none ──explicit ticket request──> awaiting_confirmation
awaiting_confirmation ──client confirm + ToolEngine write──> confirmed
awaiting_confirmation ──cancel/revise──────────────────────> cancelled
explicit request ──schema/gap repair exhausted─────────────> failed
query/cancel feasibility ──────────────────────────────────> no write
```

## 兼容

`action_contracts_enabled=False` 为默认。关闭时旧调用参数、工具引擎、确认接口和返回
结构保持兼容。新 artifact 字段只在开启时进入 graph state。

## 验证

- 单测覆盖 schema、规范化、澄清、跨会话号、伪预览、一次修复、取消、改口、幂等确认；
- 新题集 8 dev + 20 held-out，按模板组隔离，gold 只在 scorer；
- 同模型/工具/预算配对；
- 首轮 held-out 保留基础设施失败，不重跑；
- 修正 harness 后只做 dev 诊断，明确其非独立验证属性。
