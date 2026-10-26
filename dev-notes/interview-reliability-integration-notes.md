# 同会话可靠性：接入与回滚

## 合并前置

1. 先应用 [`db/ch11-reliability-execution.sql`](../db/ch11-reliability-execution.sql)。本分支遵守文件所有权，
   没有直接修改共享 `db/init.sql`。
2. 集成时将上述 DDL 追加到 `db/init.sql`，建议放在 `rag_loop_steps` 之后：

```sql
-- 内容与 db/ch11-reliability-execution.sql 完全一致
-- chat_runs / chat_run_events / chat_session_leases / chat_tool_attempts
```

3. 应用启动后确认 `app.state.execution_store`、`execution_service`、`rag_steps_store` 存在。

## 必要接线

`app/main.py` 已完成：

```python
rag_steps_store = RagLoopStepStore(session_factory)
compiled_graph = build_graph(
    ..., rag_steps_store=rag_steps_store,
)
```

生产工具引擎使用包装层：

```python
base_engine = ToolEngine(...)
app.state.engine = ExecutionToolEngine(base_engine, app.state.execution_store)
```

若另一会话在 `main.py` 有冲突，保留以下顺序：

1. 注册工具与构造 base `ToolEngine`。
2. 构造 `ExecutionToolEngine`。
3. 用包装后的 `app.state.engine` 组图。
4. 创建 `RagLoopStepStore(session_factory)` 并传 `rag_steps_store`。
5. 最后用 `ObservabilityService` 绑定已编译图。

不得把 `RagLoopStepStore` 改成独立数据库 factory，否则会破坏“应用图与审计表同库”的验证前提。

## 接口契约

- `POST /api/chat` 新增可选 `client_request_id`；旧字段不变。
- `session` / `done` 事件增加 `run_id`；所有 SSE 增加 `id:` 行。
- 状态：`GET /api/chat/runs/{run_id}`。
- 重放：`GET /api/chat/runs/{run_id}/events`，接受 `Last-Event-ID` 或 `?cursor=`。
- 恢复：`POST /api/chat/runs/{run_id}/resume`。
- 取消：`POST /api/chat/runs/{run_id}/cancel`。

## 协作边界

- 不修改 RAG 会话拥有的 `config.py`、`graph/builder.py`、`graph/agent.py`、`graph/agentic.py`、
  `rag_loop.py`、`prompts.py`。
- 本分支新增 `app/execution/`、`app/orm.py`，并只改 `models.py` 的 Base 导入和 ORM 导出。
- `app/context.py` 只改注入投影，不改数据库摘要/消息结构。
- 前端若未带 `client_request_id`，旧路径仍可运行；重连客户端应保存 `X-Run-ID` 或 `session` 事件中的 `run_id`。

## 回滚

代码回滚：整体回退本分支 commit 即可；没有切换默认 RAG 路由，也没有修改共享模型配置。

数据保留：新表均为追加式，不改变既有业务表列。回滚应用代码后可保留四张执行表供审计；若确认不保留，
按依赖顺序删除 `chat_run_events → chat_tool_attempts → chat_session_leases → chat_runs`。
不要删除 `tickets` 中的确认键工单；它们仍是已发生的业务事实。

## 集成后验证

```bash
python -m pytest tests/test_interview_reliability.py -q
python scripts/acceptance_interview_reliability.py
```

连接真实 MySQL 后再做一次只读 schema/insert/replay smoke；本分支没有迁移或写入共享 MySQL。
