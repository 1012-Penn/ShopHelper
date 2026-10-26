# 同会话可靠性开发记录

## 观察到的问题

- `/api/chat` 原实现把图任务生命周期绑在 SSE 生成器上，断连即取消，且没有跨请求的 run 身份或回放。
- 同会话没有数据库级单执行者约束；进程内集合不能解释多 worker 行为。
- 写工具开始后若进程崩溃，原状态无法区分“没执行”和“结果未知”。
- 上下文只给层 1/层 2 算预算，却另注入完整累计摘要，实际投影可以超过滑窗。

## 选择与实现

- 新增 `app/execution`：持久 run/event/lease/tool-attempt，唯一约束负责请求幂等，租约行锁负责同会话互斥。
- SSE 改为持久事件轮询；断连不影响后台任务。`Last-Event-ID` 使用事件全局 id，run 内另保留 seq。
- 工具经包装层记录开始/完成；写前状态落库失败时写工具 fail closed。未知写禁止自动 resume。
- `current_lease_guard` 同时约束工具与消息落库，旧 owner 失去租约后不能补写。
- 后台任务 120 秒上限、45 秒租约、15 秒心跳；显式取消为进程内 best-effort。
- `build_layered` 改为层 1 → 最新层 2 → 摘要投影的预算顺序，原始存储不变。
- `RequestTrace.fail` 收口失败轮 usage；Langfuse 仍保留全部降级边界。
- `RagLoopStepStore` 由应用 factory 创建并传入图。

## 验证

- 新增 25 项可靠执行测试，覆盖 API、SSE、上下文和受控 RAG。
- 真实子进程验收：完成回放、事件前后崩溃、只读 resume、写未知阻断、DB 瞬时故障和确认响应丢失。
- 完整回归：839 passed、2 skipped、1 xfailed。
- 验收脚本与机器结果均在报告中；模型调用和付费调用均为 0。

## 成本

本轮 `model_calls=0`、`paid_calls=0`、模型费用 `$0`。未混用任何旧全局费用账本。

## 边界

- SQLite 性能只能说明测试后端的持久/重放路径，不比作 MySQL 生产吞吐。
- 恢复是 run 级只读重试，不是任意节点 checkpoint。
- 不宣称任意外部副作用 exactly-once；未知写必须按业务幂等键核对。
