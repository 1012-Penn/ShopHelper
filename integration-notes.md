# Integration Notes: 受控 RAG 质量与答案复核

## 合并顺序

1. 先合入本分支的 `app/config.py`、`app/prompts.py`、`app/rag_loop.py`、
   `scripts/eval_checks.py` 和新增 `app/answer_verifier.py`。
2. 再合入 `app/graph/agent.py`、`app/graph/agentic.py`、`app/graph/builder.py`。
3. 最后合并评测资产与文档。

## 接口契约

`build_graph(...)` 新增仅关键字参数：

```python
answer_verifier=None
```

不传时行为不变。`settings.answer_verification_enabled=True` 时 builder 自动构造
`AnswerVerifier` 并传给 `make_agent_node`；测试可注入替身。

`make_agent_node(..., *, answer_verifier=None)` 保持原调用兼容。关闭开关时 token
逐片发送；开启时只缓冲最终答案 token，`tool_status`、`ticket_preview` 与 `citations`
仍按原顺序流式发送，验证结束后发送一个完整最终 token。

`ControlledRagLoop` 新增关键字 `gap_trigger=None`。`RagLoopStepStore.insert_step`
无需改表和字段；新增的 `gap_trigger`、`judge_reused` 只是写入既有 JSON step 的审计字段。

## 配置

```dotenv
ANSWER_VERIFICATION_ENABLED=false
ANSWER_VERIFICATION_MAX_REVISIONS=1
RAG_GAP_TRIGGER_ENABLED=false
```

- 前两项作用于最终 Agent 回答。
- `RAG_GAP_TRIGGER_ENABLED` 只在 `RETRIEVAL_ROUTE=controlled_loop` 时生效。
- 本轮证据不支持开启任何一项；默认值必须保持 `false`。

## DDL / 存储

本分支没有新增 DDL，也不需要修改 `init.sql`。可靠性会话已有/计划中的
`RagLoopStepStore` 接线继续使用原接口。

## 回滚

保持三个默认开关为 `false`，即可回到旧流式契约、`hybrid_rerank + baseline`、原
受控循环停止行为。评分 overlay 与 `tests/eval/rag_quality/` 可独立删除，不影响旧
CH11/CH16 产物。若要完全移除代码，再回退 `scripts/eval_checks.py` 各新增字段的消费方；
旧 `passed` 字段仍保留。

## 验证

- 全量：`/root/workplace/ShopHelper/.venv/bin/python -m pytest -q`
- 结果：`843 passed, 2 skipped, 1 xfailed`
- 评测：`python -m scripts.eval_rag_quality --offline-rescore`
- 真实复核/触发入口见
  [`docs/eval/agentic-rag-quality-report.md`](docs/eval/agentic-rag-quality-report.md)。
