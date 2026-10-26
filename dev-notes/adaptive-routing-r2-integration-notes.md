# Adaptive Routing R2 Integration Notes

## 默认行为与回滚

本分支没有修改 `app/main.py`、`app/config.py`、图入口、数据库或共享验收脚本。
默认生产行为保持原样，router 完全没有自动启用。

回滚只需不挂载 `create_adaptive_routing_router(...)`，或移除对应 include；
没有迁移、持久表或后台任务需要清理。

## 最小接线

生产适配器需要把现有会话 ID 和工具上下文绑定到两个 executor，再交给
`RouteDispatcher.dispatch()`。最小形状：

```python
service = ExistingSessionAdaptiveService(
    single_executor=lambda routing_input, session_id: existing_single_service(
        routing_input.question, routing_input.history, session_id=session_id
    ),
    multi_executor=lambda routing_input, session_id: existing_multi_service(
        routing_input.question, routing_input.history, session_id=session_id
    ),
)
app.include_router(create_adaptive_routing_router(service))
```

`ExistingSessionAdaptiveService` 应由应用层实现：它复用同会话、`ToolEngine`、
预览 `confirm_key` 和用户确认入口。路由模块不导入 `app.main` 或
`app.multi_agent.runtime`，因此不会形成循环依赖。

建议只在新显式入口或功能开关后挂载，不要把本模块直接插到默认 `/api/chat` 前；
统一验收会话应在确认 SSE、恢复和 usage 语义后再决定入口。

## 两个现有执行器的差异

| 维度 | 现有单 Agent | 现有多 Agent |
|---|---|---|
| 入口 | `POST /api/chat` | `POST /api/multi-agent/chat` |
| 输出 | 支持 SSE 帧与最终消息 | 当前返回一次性 JSON |
| 会话持久化 | 有 run、event、lease 与恢复 | 进程内 graph，当前不等价 |
| usage | 有 request usage 持久化 | 当前 tokens 为上下文估算，非 billing |
| 工具权限 | 既有工具注册与确认边界 | retrieval/action 白名单 |
| 写预览 | 现有前端确认流 | 返回 `write_previews`，可交给现有确认接口 |

`app/adaptive_routing/api.py` 是可选 JSON API 适配器，不声称两路已经具备相同 SSE、
断点恢复或 usage 语义。生产接入若要求流式响应，需要在应用层分别适配原入口，
不能在 router 中伪造统一事件流。

## 预算与观测

- `RoutingDecision.routing_latency_ms` 和 `routing_cost_usd` 必须进入应用日志/指标。
- 若启用可选 judge，其 token/费用必须写入同一会话预算；当前默认 rules-only。
- executor 的 provider billing usage 仍由现有账本负责；不要把 `NormalizedResponse.tokens`
  当作供应商实际 billable usage。
- 同一请求只允许一个 executor 被调用；不得在异常后自动补跑另一条路径。

## 冲突与文件所有权

新增文件仅位于：

- `app/adaptive_routing/`
- `scripts/eval_adaptive_routing_r2.py`
- `scripts/audit_adaptive_routing_r2.py`
- `scripts/benchmark_adaptive_routing_r2.py`
- `tests/eval/adaptive_routing_r2/`
- 本主题的 docs/dev-notes

没有修改结构化交接模块、claim grounding 模块、共享生产入口或旧多 Agent 评测脚本。
统一验收会话如需接线，可只挑选上述适配器并保留默认关闭。
