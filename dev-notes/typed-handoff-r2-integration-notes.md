# Typed Handoff R2 Integration Notes

## 最小接线

现有接线保持不动，默认行为即为 raw。启用 typed 时只改 runtime 构造：

```python
runtime = build_multi_agent_runtime(
    engine=engine,
    model=model,
    handoff_mode="typed",
    dynamic_status_ttl_seconds=300,
)
```

不需要修改 `app/main.py`、`app/config.py`、API response model、数据库或共享工具注册。

## 兼容性

- `MultiAgentRunner` 原构造参数不变；`handoff_mode` 默认 `"raw"`。
- `AgentOutcome` 与 `MultiAgentState` 只追加有默认值的内部字段。
- `facts`、`evidence`、`write_previews`、trace 的既有 API 字段不变。
- `handoff_mode="typed"` 时 action 的初始 prompt 与 raw 不同；这是显式实验变量。
- 工具权限仍来自 `ROLE_SPECS`；`TaskHandoff` 没有授权字段。
- 写工具仍经过原 `ToolEngine` 拦截、确认和 `execute_confirmed` 幂等路径。

## 冲突面

本分支拥有：

- `app/evidence_contracts/`
- `app/multi_agent/models.py`
- `app/multi_agent/runtime.py`
- `tests/test_typed_handoff_contracts.py`
- `tests/eval/typed_handoff_r2/`
- `scripts/typed_handoff_r2_eval.py`
- `scripts/typed_handoff_r2_rescore.py`

其它并行分支若改同一 runtime/models，需要以 `446fbb7` 为共同基线人工合并。不要把
本分支的 typed prompt 视为其它路由/grounding 会话的依赖。

## 回滚

代码回滚可直接不传 `handoff_mode`，或回退本分支提交；没有数据库迁移和外部资源需要清理。

运行 `scripts/typed_handoff_r2_eval.py` 会拒绝覆盖已有结果和账本。不要删除
`runs/live-20261027` 下的原始 trace、v1/v2 评分或 SQLite ledger。

## 验证入口

```bash
/root/workplace/ShopHelper/.venv/bin/python -m pytest -q \
  tests/test_typed_handoff_contracts.py \
  tests/eval/typed_handoff_r2/test_scoring.py \
  tests/test_multi_agent_contracts.py \
  tests/test_multi_agent_runtime.py \
  tests/test_multi_agent_integration.py
```

评分重放不调用模型：

```bash
/root/workplace/ShopHelper/.venv/bin/python scripts/typed_handoff_r2_rescore.py
```

该脚本要求原始 `results.jsonl` 与 `summary-v1.json` 存在，并拒绝覆盖已有 `v3-final/`。
