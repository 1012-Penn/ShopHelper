# Task 1 实现报告：可观测性核心、配置与 token 用量底座

## 改动文件

- `app/observability.py`：可选 Langfuse 服务、根 observation、no-op 降级、请求独立 `UsageCallback`；所有 metadata 值在单一出口统一转成字符串并截断到 128 字符。
- `app/config.py`、`pyproject.toml`：Langfuse 配置及 SDK 依赖声明。
- `app/main.py`、`app/routers/chat.py`：已编译 Graph 默认 handler 绑定；每轮 invocation 传递短字符串 metadata 和独立 token callback，SSE sink 保持不变。
- `app/models.py`、`app/store.py`、`db/init.sql`、`db/ch09.sql`：`request_usage` ORM、读写/按意图聚合与 MySQL DDL。
- `tests/test_observability.py`：no-op、单次 handler、metadata 全值规范化、两种 token metadata、请求隔离、LangChain callback 协议、聚合回归。
- `dev-notes/ch09.md`：任务过程留痕。

## TDD 与测试证据

先写 `tests/test_observability.py`，首次运行：

```text
ERROR tests/test_observability.py
ModuleNotFoundError: No module named 'app.observability'
1 error in 0.38s
```

纠正 metadata/request 隔离后，新增回归先失败：

```text
2 failed, 4 passed in 0.53s
conversation_id: 42 != "42"
RequestTrace object has no attribute usage_callback
```

相关聊天回归随后定位 LangChain callback 协议问题：

```text
'UsageCallback' object has no attribute 'run_inline'
19 failed, 34 passed, 2 warnings in 12.24s
```

本次接手时 `HEAD` 已包含上一轮主体提交 `2baf60e`。先跑专项建立基线：

```text
/root/workplace/ShopHelper/.venv/bin/pytest tests/test_observability.py -q
7 passed in 0.30s
```

随后补 metadata 全值契约测试。生产代码变异目标是“直接透传非字符串或长文本”；新测试首次运行真实失败：

```text
/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_observability.py::test_request_metadata_normalizes_every_value_to_short_string -q
1 failed in 0.39s
AssertionError: 长 intent 未截断到 128 字符
```

最小实现把字符串化/截断放到 `RequestTrace.metadata()` 的统一返回出口。GREEN：

```text
/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_observability.py::test_request_metadata_normalizes_every_value_to_short_string -q
1 passed in 0.03s

/root/workplace/ShopHelper/.venv/bin/pytest tests/test_observability.py -q
8 passed in 0.29s
```

相关回归命令与真实输出摘要：

```text
/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_observability.py tests/test_chat_api.py tests/test_ch05_chat_api.py \
  tests/test_ch06_chat_api.py tests/test_ch07_pipeline.py tests/test_ch08_chat_api.py -q
46 passed, 1 xfailed, 2 warnings in 15.21s

/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_config.py tests/test_db.py tests/test_models.py tests/test_store.py -q
14 passed in 0.25s
```

## 实现期纠偏与风险

- 所有 Langfuse metadata 值均由统一出口转换为字符串并限制为 128 字符；`session_id`、`conversation_id`、`source`、`intent`、`entry_route` 不再包含整数或长文本。
- `UsageCallback` 下沉到 `RequestTrace`，每次 `start_request()` 都创建独立计数器，避免 token 跨请求污染；它继承 `BaseCallbackHandler`，可安全传给 LangGraph/LangChain invocation。
- 缺失 Langfuse key、SDK、服务或 flush 失败均记录 warning 并降级，不阻断聊天。
- 未对真实 Langfuse 服务做联机验证：当前测试环境无 key/服务；SDK 依赖已写入 `pyproject.toml`，运行时仍保持 no-op。
- 相关聊天回归有 1 个既有 xfail；`tests/test_ch07_pipeline.py` 有 2 个既有 warning，原因是同步测试带 `asyncio` 标记，均非 Task 1 引入。
- `UsageStore.record()` 通过后台 task 落库，进程被强制终止时仍存在未完成写入的窗口；本任务按既定“观测失败不阻断聊天”边界保留该权衡。
- 工作树存在大量非本任务 `.agents` 与 CRLF 行尾噪声；提交仅显式暂存 Task 1 路径，并在提交前核对 staged diff，不纳入环境噪声。

## Fix round：评审 findings

本轮仅修改 `app/observability.py`、`tests/test_observability.py` 与本报告。

- `RequestTrace.activate()` 不再在异常分支二次 `yield`。Langfuse context 的 `__enter__`/`__exit__` 失败均记录 warning 并降级；聊天主体自身抛出的异常仍保留原异常语义。
- `RequestTrace.metadata()` 对每个值统一执行安全字符串化，并截断到 128 字符。
- `ObservabilityService.bind_graph()` 按源 compiled graph 的对象身份缓存首次绑定结果；重复传入同一 graph 时直接返回缓存 wrapper，不再重复 `with_config()` 或叠加 callback。绑定失败不缓存，允许后续重试。

新增测试均先在旧实现上观察到预期失败：

```text
/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_observability.py::test_request_trace_suppresses_langfuse_context_exit_failure -q
1 failed in 0.32s
RuntimeError: generator didn't stop

/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_observability.py::test_bind_graph_configures_same_compiled_graph_only_once -q
1 failed in 0.35s
assert second is first
```

对应 GREEN 与专项结果：

```text
/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_observability.py::test_request_trace_suppresses_langfuse_context_exit_failure -q
1 passed in 0.02s

/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_observability.py::test_bind_graph_configures_same_compiled_graph_only_once -q
1 passed in 0.03s

/root/workplace/ShopHelper/.venv/bin/pytest tests/test_observability.py -q
10 passed in 0.26s
```

提交前最终相关回归：

```text
/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_observability.py tests/test_chat_api.py tests/test_ch05_chat_api.py \
  tests/test_ch06_chat_api.py tests/test_ch07_pipeline.py tests/test_ch08_chat_api.py -q
48 passed, 1 xfailed, 2 warnings in 17.04s

/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_config.py tests/test_db.py tests/test_models.py tests/test_store.py -q
14 passed in 0.27s
```
