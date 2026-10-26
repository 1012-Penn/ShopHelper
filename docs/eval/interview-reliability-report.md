# CH11 同会话可靠执行与可靠性证据

日期：2026-10-27  
分支：`codex/interview-reliability-20261027`  
基线：`main=392e058d195bf731deff9ef6b4c2113da2a36a8b`  
状态：完成工程接线、故障注入与完整回归；未调用模型，未写共享 MySQL/Milvus。

## 结论

- `/api/chat` 保持原请求兼容，并新增可选 `client_request_id`、`run_id`、持久事件游标与状态查询。
- 同一会话由数据库租约保证单执行者；不同会话可并行。第二个并发请求显式返回 409，不排队。
- SSE 只消费已持久事件，支持 `Last-Event-ID` / `cursor` 续读。客户端断开后后台任务继续到 120 秒
  最长存活期；重连不会重新提交图或已确认写工具。
- 真实子进程验收覆盖了完成回放、事件持久化前/后崩溃、只读 run 二次恢复、写工具开始后崩溃。
  写结果未知时 `write_state=unknown`，自动 resume 被拒绝。
- 摘要投影已计入 `sliding` 总窗口；原始消息、摘要分段和历史摘要不改写。最近原文优先于旧摘要。
- 正常、失败、取消轮均记录 run 状态、usage 和耗时；Langfuse 故障仍只降级观测。
- 受控 RAG 的 `RagLoopStepStore(session_factory)` 已传入应用图，并通过真实应用工厂落表测试。

## 持久状态与接口

新增表：

- `chat_runs`：请求指纹、状态、attempt、结果、usage、耗时、`write_state`。
- `chat_run_events`：run 内 `seq` 单调递增；全局自增 `id` 作为 SSE 游标。
- `chat_session_leases`：以 `conversation_id` 为主键的租约，带 owner 与过期时间。
- `chat_tool_attempts`：工具开始、成功、失败、拦截、未知状态及可选确认幂等键。

新增接口：

- `GET /api/chat/runs/{run_id}`：状态、结果、usage、耗时与恢复信息。
- `GET /api/chat/runs/{run_id}/events`：从 `Last-Event-ID` 或 `cursor` 重放。
- `POST /api/chat/runs/{run_id}/resume`：仅恢复安全的未完成 run。
- `POST /api/chat/runs/{run_id}/cancel`：本进程内 best-effort 取消；写已开始返回 409。

请求幂等键为 `(conversation_id, client_request_id)`。同键同载荷返回同一 run；同键异载荷返回 409。
匿名请求若不携带 `session_id`，会先新建会话，因此其幂等范围也只在该新会话内。

## 执行与重放边界

### 正常执行

1. 解析会话并原子预留 run/租约。
2. 先持久 `session` 事件，再启动后台图任务。
3. 节点帧进入队列，由独立 pump 逐条持久化；SSE 只读持久事件。
4. 图完成后先持久 `done`，再提交 `completed` 与 usage；客户端可随时按游标重放。

后台任务最长存活期为 120 秒，租约 45 秒、每 15 秒续租。客户端断开只停止事件生成器，不取消后台
图任务；显式 `/cancel` 才请求取消。租约丢失后，工具执行、消息落库、事件/终态写入均做 owner fencing。

### 崩溃恢复

- 已完成 run：任何新进程只读 `chat_runs + chat_run_events` 即可重放原 `session/token/done` 结果，
  不执行图。
- 未完成且从未启动写工具：租约过期后标 `interrupted`；显式 resume 从 run 边界重新执行只读路径。
  这是 run 级安全重启，不宣称任意节点内断点续跑。
- 写工具已开始但未写回结果：标为 `interrupted + write_state=unknown`，resume 返回 409。调用方必须
  用业务确认幂等键核对是否已写入。
- 已确认工单写入继续使用 `tickets.confirm_key` 唯一约束。确认响应丢失后的同键重试返回同一工单。

不承诺任意外部副作用 exactly-once。当前可证明的是：聊天 run 不重复执行、已确认单工具不重复建单，
以及未知写入不会被系统静默重放。

## 故障注入证据

机器结果：[`interview-reliability-acceptance.json`](interview-reliability-acceptance.json)。  
复跑入口：[`scripts/acceptance_interview_reliability.py`](../../scripts/acceptance_interview_reliability.py)。

| 场景 | 证据 | 结果 |
|---|---|---|
| 同会话双请求竞争 | 两个独立连接/线程同时 reserve | 1 `created`、1 `busy`，两线程均结束 |
| 重复 client_request_id | 同载荷两次、异载荷一次 | 同 run、第二请求未创建；异载荷冲突 |
| API 同键重试 | 两次 `/api/chat` 同键 | 模型 `prompts` 只增加 1 次 |
| 同会话异请求 | run 活动时发第二个请求 | HTTP 409，首 run 正常完成 |
| SSE 断连重连 | 取消客户端请求、等待后台完成、按 run 重放 | 模型只执行 1 次，历史只有 1 组问答 |
| 游标续读 | 全量事件后设置 `Last-Event-ID` | 仅返回游标后的事件，顺序一致 |
| 崩溃于事件持久化前 | 子进程预留后 `os._exit(23)` | 父进程恢复为 `interrupted`；第二次子进程 resume 后 `completed`，attempt=2 |
| 崩溃于事件持久化后 | 子进程写 `session` 后 `os._exit(24)` | 重放为 `session → interrupted` |
| 崩溃于写工具开始后 | 子进程写前状态后 `os._exit(25)` | `write_state=unknown`，自动 resume 被拒绝 |
| DB 瞬时故障 | factory 注入一次异常 | 调用失败且不污染序列；恢复后首个事件仍为 `seq=1` |
| 确认响应丢失 | 同 `confirm_key` 重提两次 | 同一 `ticket_no`、`deduplicated=true`、数据库 1 张工单 |
| SSE 重放写预览 | 重放两次聊天事件 | 工单一律为 0；用户确认后才执行 |

真实子进程验收由父进程使用新的 `engine/sessionmaker` 读取数据库；不是同进程内休眠或纯脚本布尔断言。

## 并发与重启证据不等于性能

可靠性证据回答“是否重复执行、能否恢复、未知写是否被拦住”，不回答“系统更快”。

固定负载说明：

- 后端：隔离文件 SQLite，独立 session/连接；不连接共享 MySQL。
- 负载：8 个逻辑 worker，每个顺序执行 4 个 run；每个 run 写 `session/token/done` 并提交终态。
- 硬件：16 个逻辑 CPU。
- 模型：0 次；因此不含生成 token、检索或网络延迟。

本次固定运行结果：

| 指标 | 结果 |
|---|---:|
| 成功 | 32/32（100%） |
| 墙钟 | 1.212 秒 |
| 吞吐 | 26.39 run/s |
| run P50 / P95 | 128.06 / 828.05 ms |
| 首个持久事件 P50 / P95 | 32.97 / 666.01 ms |
| 平均 run | 208.33 ms |

“首个持久事件”是事件提交到隔离 SQLite 的耗时，不是 HTTP 首字节。高 P95 来自并发 SQLite 写锁竞争。
在未与旧实现同硬件做重复对照前，不能宣称固定百分比性能提升。

## 上下文预算修复

旧实现只把层 1 + 层 2 计入 `sliding`，随后又把完整摘要作为背景块注入，因此实际投影可超过窗口。
当前规则：

1. 层 1 最近原文优先。
2. 层 2 从最老端裁剪；最新用户改口优先保留。
3. 摘要只使用剩余预算，超出部分生成本轮投影，不修改原始摘要。
4. 单轮层 1 本身已超过窗口时仍至少保留当前原文，并记录 `budget_exceeded=true`，属于 best-effort。

`build_layered` 额外输出 `tokens_summary/full`、`total_injected_tokens`、`summary_truncated` 和
`budget_exceeded`。测试覆盖用户改口、过期工具结果和多段摘要累积。

口径边界：

- `estimate_tokens` 对 CJK/全角字符按 1 token，其他字符按 `ceil(chars/4)`；不是供应商 tokenizer。
- 公式不含消息角色/工具 schema/JSON 封装等全部协议开销；这些由既有固定预算项额外预留。
- 摘要投影是在字符边界截断后加省略号，保证估算不超预算，但不保证语义完整句。

## 观测与受控 RAG 接线

- 正常轮由 `RequestTrace.finish` 写 `request_usage`；失败/取消由 `RequestTrace.fail` 写 usage，
  run 表与事件同时保留状态、错误和耗时。
- Langfuse observation 创建、更新、退出或 flush 失败均只记 warning；现有观测测试 11/11 通过。
- `main.create_app` 创建 `RagLoopStepStore(session_factory)` 并传给
  `build_graph(..., rag_steps_store=...)`。
- 测试在 `retrieval_route=controlled_loop` 下走真实应用图、纯替身检索/模型，确认
  `rag_loop_steps` 使用同一 SQLite factory 落表并写停止原因。

## 复用既有冻结评测

本轮没有新增模型调用，也没有生成新质量数字。引用基线必须保持原口径：

- CH15 事后修订规则：Constrained Agent 语义任务成功 28/30，Fixed refund flow 20/30。
- CH16：RRF+rerank frozen-v1 R@5 为 50/65（76.9%），这是检索覆盖，不是答案正确率；旧答案评分
  仍保留已知偏差。

## 验证命令与结果

```bash
/root/workplace/ShopHelper/.venv/bin/python -m pytest tests/test_interview_reliability.py -q
# 25 passed

/root/workplace/ShopHelper/.venv/bin/python -m pytest -q
# 839 passed, 2 skipped, 1 xfailed, 4 warnings in 138.81s

/root/workplace/ShopHelper/.venv/bin/python scripts/acceptance_interview_reliability.py \
  --json-out docs/eval/interview-reliability-acceptance.json
# model_calls=0, paid_calls=0
```

完整回归比协调基线多 25 个通过测试。4 条 warning 是既有测试函数错误使用 asyncio 标记；pytest
退出后还有 Milvus Lite 关闭时日志流已关闭的清理噪声，不影响测试结果。

## 已知边界

- MySQL DDL 已独立提供，但未在共享 MySQL 上执行；实际运行前必须应用
  `db/ch11-reliability-execution.sql`。
- 性能只测隔离 SQLite 的持久/重放路径，不代表 MySQL 或真实模型端到端性能。
- 不提供任意节点内 checkpoint/断点续跑；只提供 run 级只读重试。
- 不提供跨进程主动取消；其他 worker 持有任务时 cancel 返回 409，等待租约过期恢复。
- `write_state` fencing 不能证明任意第三方工具 exactly-once；未知写入仍要求业务核对。
- 未对共享 MySQL、共享 Milvus 或外部 MCP 做写操作；未使用真实凭据。

## 可用于简历的候选句

- “为客服 Agent 增加持久 run、可重放 SSE 事件和数据库租约互斥，覆盖同会话冲突、跨会话并发、
  客户端断连、进程崩溃与未知写恢复；真实子进程故障验收全部通过，完整回归 839 passed。”
- “将累计摘要纳入实际上下文注入预算，保留原始消息与摘要存储，优先保留最近改口；用用户更正、
  过期工具结果和多段摘要测试验证投影边界。”

以上只描述工程实现与故障证据，不把可靠性工作写成模型效果或固定性能提升。
