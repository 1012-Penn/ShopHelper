# Task 1 实现报告：可观测性核心、配置与 token 用量底座

## 改动文件

- `app/observability.py`：可选 Langfuse 服务、根 observation、no-op 降级、请求独立 `UsageCallback`。
- `app/config.py`、`pyproject.toml`：Langfuse 配置及 SDK 依赖声明。
- `app/main.py`、`app/routers/chat.py`：已编译 Graph 默认 handler 绑定；每轮 invocation 传递短字符串 metadata 和独立 token callback，SSE sink 保持不变。
- `app/models.py`、`app/store.py`、`db/init.sql`、`db/ch09.sql`：`request_usage` ORM、读写/按意图聚合与 MySQL DDL。
- `tests/test_observability.py`：no-op、单次 handler、metadata、两种 token metadata、请求隔离、LangChain callback 协议、聚合回归。
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

最终验证命令与真实输出摘要：

```text
/root/workplace/ShopHelper/.venv/bin/pytest tests/test_observability.py -q
7 passed in 0.24s

/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_observability.py tests/test_chat_api.py tests/test_ch05_chat_api.py \
  tests/test_ch06_chat_api.py tests/test_ch07_pipeline.py tests/test_ch08_chat_api.py -q
45 passed, 1 xfailed, 2 warnings in 14.94s

/root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_config.py tests/test_db.py tests/test_models.py tests/test_store.py -q
14 passed in 0.22s
```

## 实现期纠偏与风险

- 所有 Langfuse metadata 值均为短字符串；`session_id`、`conversation_id`、`source`、`intent`、`entry_route` 不再包含整数或长文本。
- `UsageCallback` 下沉到 `RequestTrace`，每次 `start_request()` 都创建独立计数器，避免 token 跨请求污染；它继承 `BaseCallbackHandler`，可安全传给 LangGraph/LangChain invocation。
- 缺失 Langfuse key、SDK、服务或 flush 失败均记录 warning 并降级，不阻断聊天。
- 未对真实 Langfuse 服务做联机验证：当前测试环境无 key/服务；SDK 依赖已写入 `pyproject.toml`，运行时仍保持 no-op。
- 工作树存在大量非本任务 `.agents` 与行尾噪声，提交使用显式路径暂存，未纳入这些改动。
