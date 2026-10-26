# CH11 同会话可靠执行与面试证据：短规格

日期：2026-10-27
基线：`main=392e058d195bf731deff9ef6b4c2113da2a36a8b`
分支：`codex/interview-reliability-20261027`
范围：只使用隔离的 SQLite 测试库与模拟业务数据，不调用模型，不连接共享 MySQL/Milvus。

## 用户可观察行为

1. `POST /api/chat` 保持原请求兼容，并接受可选 `client_request_id`。
2. 同一 `(conversation_id, client_request_id)` 只创建一个 run；同键异载荷返回 409。
3. 同会话在租约有效期内只允许一个执行者；第二个不同请求返回 409。不同会话可并行。
4. 每轮持久化 run 状态、usage、耗时和带递增游标的 SSE 事件。
5. `GET /api/chat/runs/{run_id}` 返回状态；事件接口支持 `Last-Event-ID` 或 `cursor` 续读。
6. 客户端断开后图任务在受控最长存活期内继续；重连只重放事件，不重复执行图或确认写工具。
7. 进程重启可重放已完成 run。未完成且未进入写工具的 run 可显式 resume；写结果未知时拒绝
   自动恢复，要求调用方按确认幂等键核对。
8. 取消是进程内 best-effort；一旦写工具已开始，取消返回 409，避免把未知副作用报告为已取消。

## 持久化与并发边界

- 使用现有 SQLAlchemy/MySQL 体系；SQLite 仅测试。新增 `chat_runs`、`chat_run_events`、
  `chat_session_leases`、`chat_tool_attempts`。
- `(conversation_id, client_request_id)` 唯一约束负责请求幂等；会话租约行条件更新负责
  跨连接互斥。租约带过期时间与 fencing owner，不把进程内锁包装成多进程安全。
- 事件序号是同一 run 内递增游标；事件表另有全局自增 id 作为 SSE 的 `Last-Event-ID`。
- 不承诺任意外部副作用 exactly-once。`create_ticket` 等已确认写仍依赖业务确认键幂等；
  写调用“开始/成功/未知”状态持久化，未知状态禁止自动 resume。

## 上下文预算

- 累积摘要以“可丢弃的注入投影”进入 `sliding` 总预算，限制的是本轮真正拼给模型的摘要；
  原始消息、`conversation_summaries` 全部分段和历史摘要都不删改。
- 最近原文优先保留；摘要投影有份额上限，层 2 从旧端裁剪。单轮原文已经超过预算时仍至少保留
  最后一轮，并显式报告 best-effort 超限，不伪造硬保证。
- 测试覆盖用户改口、过期工具结果和多次压缩；报告写明 CJK/ASCII token 估算与硬截断边界。

## 可观测性与 RAG 接线

- 正常、失败、取消与超时都写 run 的终态、usage 和耗时；Langfuse SDK/网络失败继续降级。
- `create_app` 创建 `RagLoopStepStore(session_factory)`，传入 `build_graph(...,
  rag_steps_store=...)`，并用测试应用的同一 SQLite factory 验证真实落表。
- 不修改其他会话拥有的 RAG 核心图文件。

## TDD/验收顺序

1. 先测请求幂等、会话租约、run/事件持久化与 SSE 游标。
2. 再测同会话冲突、跨会话并行、断连后台继续、完成/失败状态与 usage。
3. 用文件 SQLite + 独立 engine/session factory 测崩溃窗口；用真实子进程验收重启回放和 stale 租约恢复。
4. 测写调用开始后未知状态拒绝自动 resume；复用确认键证明响应丢失后不重复建单。
5. 测摘要投影总预算、用户改口、工具过期与多段摘要，不修改原始存储。
6. 最后跑本分支定向测试和一次完整回归。性能表与可靠性表分开报告。
