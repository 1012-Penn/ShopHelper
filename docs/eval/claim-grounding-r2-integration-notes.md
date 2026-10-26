# Claim Grounding R2 接线说明

## 新增边界

本分支只新增：

- `app/claim_grounding/`
- `scripts/eval_claim_grounding_r2.py`
- `tests/eval/claim_grounding_r2/`
- 本报告、dev notes、spec/plan

未修改 `app/main.py`、`app/config.py`、`app/graph/`、
`app/answer_verifier.py`、`scripts/eval_checks.py`、旧多 Agent 评测、
数据库 schema 或共享入口。

## 最小接线

只在显式实验路径中构造 backend：

```python
from app.answer_verifier import AnswerVerifier
from app.claim_grounding import ClaimGrounder, ClaimGroundingAnswerSemantic
from app.claim_grounding.backends import StructuredGroundingBackend
from app.claim_grounding.budget import ClaimBudgetLedger

ledger = ClaimBudgetLedger(
    run_dir / "claim-grounding-budget.sqlite",
    total_limit_usd=2.0,
)
backend = StructuredGroundingBackend(
    caller=caller,
    ledger=ledger,
    pricing=deepseek_flash_pricing,
    model="deepseek-flash",
    max_retries=2,
)
semantic = ClaimGroundingAnswerSemantic(
    ClaimGrounder(backend=backend, only_uncertain=True)
)
verifier = AnswerVerifier(semantic, max_revisions=0)
```

实际接线需要接受的参数：

- `openai_api_key` 与 `openai_base_url`：只在进程内读取；
- `claim_grounding_enabled`：默认 `false`；
- `claim_grounding_frontier_unsupported`：预留给“是否也复核 v3.1
  unsupported”的新实验，本轮必须保持关闭；
- budget path：每次实验新路径，不复用旧 pending。

由于本分支所有权不包含 `app/config.py`，以上 config 字段只作为验收会话
的最小补丁建议，本分支没有写入。

## 兼容性

- `ClaimGrounder(backend=None, only_uncertain=True)` 是默认安全形态；
- `AnswerVerifier` 不传新 adapter 时行为完全不变；
- `ClaimGroundingAnswerSemantic` 只依赖现有 `semantic.invoke(prompt)`
  接口，返回 `AnswerVerdict`；
- `AnswerVerifier` 的确定性冲突仍在其调用 semantic 前执行；
- 不调用 backend 时不产生 provider usage。

## 回滚

移除 `AnswerVerifier` 构造参数中的 claim adapter，恢复原有 semantic 或
`None` 即可。新包与评测数据可以保留，不影响默认聊天。

## 冲突与集成风险

- 当前结论不支持默认启用；验收若接入，应作为显式实验开关。
- `v3.1 unsupported` 会短路语义 backend。单位等价和条件判断若被
  v3.1 误判，selective grounding 无法修正它。
- 模型可能把“所有/任何”式绝对 claim 与条件规则判为矛盾；现有 gold
  待人工复核，不能把这两个标签当行业真值。
- API 输出曾因 300 token 上限截断；当前按要求返回 `uncertain`。
  验收若提高输出上限，应冻结新 prompt/config 和新 held-out，不能拿现有
  结果作独立复现。
- 未验证下游答案拒答/修订；默认保持关闭。

## 评测命令与产物

当前结果目录：

- `tests/eval/claim_grounding_r2/results/rules-dev-1/`
- `tests/eval/claim_grounding_r2/results/selective-dev-1/`
- `tests/eval/claim_grounding_r2/results/selective-heldout-1/`
- `tests/eval/claim_grounding_r2/results/selective-heldout-2-bugfix/`

每轮保存逐 claim 输入、gold、规则状态、最终状态、理由、证据 ID、
原始 prompt/response、usage、费用、延迟和预算记录。不要覆盖已有 run-id；
修复后必须使用新 run-id。
