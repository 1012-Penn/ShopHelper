# Action Contract R3 集成说明

本分支只拥有 `app/action_contracts/`、`app/multi_agent/runtime.py`、
`roles.py`、`models.py` 和本任务测试。没有修改 `app/main.py`、`config.py`、
`graph/builder.py`、`graph/agent.py`、`db/init.sql` 或
`app/tools/builtin.py`。

## 默认兼容

`MultiAgentRunner` 与 `build_multi_agent_runtime` 新增参数均有默认值：

```python
action_contracts_enabled: bool = False
action_ledger: ActionSessionLedger | None = None
action_policy: ActionContractPolicy | None = None
```

关闭时：

- 不推断动作意图、不写 action artifact；
- 不加入动作契约 prompt；
- 保持原来的模型绑定、一次拦截和输出字段；
- 不改变 `ToolEngine`、确认接口或 `tickets.confirm_key` 幂等边界。

## 建议配置补丁

由验收会话在 `app/config.py` 增加兼容默认：

```python
multi_agent_action_contracts_enabled: bool = False
```

由验收会话在 `app/main.py` 的 runtime 构造处接线：

```python
from app.action_contracts import ActionSessionLedger

app.state.action_ledger = ActionSessionLedger()
app.state.multi_agent = build_multi_agent_runtime(
    engine=app.state.engine,
    model=app.state.chat_model,
    limits=None,
    planner_structured=make_structured_extractor(settings, AgentPlan),
    handoff_mode=settings.multi_agent_handoff_mode,
    dynamic_status_ttl_seconds=settings.multi_agent_dynamic_status_ttl_seconds,
    action_contracts_enabled=settings.multi_agent_action_contracts_enabled,
    action_ledger=app.state.action_ledger,
)
```

建议在 `app/multi_agent/api.py` 的响应模型增加可选字段：

```python
action_artifacts: list[dict] = Field(default_factory=list)
```

并从 graph state 原样透传。现有 `write_previews` 保留，避免旧客户端回滚。

## 确认/取消联动

模型永远不调用 `confirmed=True`。客户端仍调用既有 `/api/tickets/confirm`。
集成层在确认成功后，若 `confirm_key` 命中 `app.state.action_ledger` 的 pending，
调用：

```python
app.state.action_ledger.mark_confirmed(
    conversation_id=conversation_id,
    confirm_key=confirm_key,
    ticket=result,
)
```

该调用必须在真实 ToolEngine 写成功之后发生。不得把模型文本确认为落库依据。

取消预览由 runtime 处理；前端取消卡片若走独立端点，可同样调用
`ActionSessionLedger.cancel(conversation_id)`，但必须保持“未确认动作不写业务库”。

## 事件与状态

- `AgentOutcome.action_artifacts` 可选，默认空列表；
- `MultiAgentState.action_artifacts` 由 runtime 产生；
- 预览键仍为 `confirm_key`，同键幂等由现有 `tickets.confirm_key` 保证；
- 模型只说不做时，stop reason 为 `action_contract_missing_preview`；
- 非法参数/枚举为 `action_contract_clarification`；
- 非用户要求的写动作/重复预览分别有明确 stop reason；
- 取消后不会再次自动建预览。

## 回滚

把 `multi_agent_action_contracts_enabled` 设为 `false` 即可回到旧 runtime。
不要删除 `ActionSessionLedger` 或 artifact 字段；它们默认不参与旧路径，可作为兼容
观测面保留。已有预览如需撤销，先确认没有业务写发生。

## 冲突风险

- `app/main.py` 和 `config.py` 由集成会话接线，避免本分支与其他两支争抢共享文件；
- `multi_agent/api.py` 当前未透传 artifact，集成时要做最小响应扩展；
- `ActionSessionLedger` 是进程内状态；多 worker 或跨重启需要持久化，但本轮未实现；
- 如果集成时仍使用旧结构化输出分支，必须继续走 `make_structured_extractor`，
  不要用强制 `tool_choice` 的 `with_structured_output`；
- 若确认接口与多 Agent runner 不在同一进程，需先设计跨进程动作状态，否则契约
  只能识别当前进程内的 pending。
