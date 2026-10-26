# E2E Quality R2 Integration Notes

## 默认兼容性

本轮没有修改 `app/main.py`、`app/config.py`、`app/graph/`、`app/answer_verifier.py`、
`scripts/eval_checks.py`、`db/init.sql` 或其他会话文件。新 runner、评分器、数据、fixture 和测试
均为默认关闭的独立路径，无需生产接线即可回滚。

## 必须由验收会话决定的生产修复

### 1. DeepSeek thinking 模式结构化输出

复现：

```text
POST /api/chat
{"message":"SH-E300 现在卖多少钱？"}
→ resolve/intent/evidence judge 中的 tool_choice 结构化调用返回 HTTP 400
→ Error: Thinking mode does not support this tool_choice
```

默认路径仍可由业务 fallback 产出部分回答，但每轮出现上游技术失败，不能把 0/28 解释为 RAG 能力。

最小修复形态：

- 在 `app/llm.py` 集中提供 `make_structured_extractor(settings, schema)`；
- 对 thinking 模型使用“prompt + JSON Schema → JSON 文本 → `model_validate`”的 Runnable；
- 替换 `build_graph` resolver/expander、`app/evidence.py`、`app/rewrite.py` 及需要结构化输出的调用点；
- 保留原 `with_structured_output(method="function_calling")` 给支持它的模型；
- 对 JSON 解析失败走现有安全降级，不无限重试。

本分支的诊断实现位于
`scripts/e2e_quality_r2/harness.py:BudgetedJsonStructured`，复现了该形态；验收接入生产时应放在
`app/llm.py`，不要在 evaluator 内当作线上行为。

### 2. 固定退款/售后路由

复现：`DEV-POLICY-V2`、`HELD-CUSTOM-RETURN` 等 7 个通用政策题被 intent=`退款退货/售后` 直接送入
`prepare_order`，在用户未给订单号时返回空回答或索要订单号，未回答通用规则。若业务期望“先用政策回答，
只有需要订单判定时才澄清”，应调整路由前置条件。该修改涉及 `app/graph/`，本会话未做。

### 3. 订单工具与知识库不一致

当前 `app.orders.py`：

- 1002：机械键盘 / 399 / 已发货；
- 1003：硅胶手机壳 / 19.9 / 待发货。

当前《订单与支付》：

- `ORD-1002`：SH-E300 + SH-C60 / 328 / 已支付未发货；
- `ORD-1003`：SH-K68 / 149 / 已发货。

这会产生实体相同、事实冲突的工具/知识回答。验收会话需要确定唯一业务源，再改 fixture 和解析格式，
并用新的数据版本重跑。不能只修改评测金标让数字变好。

### 4. 随机商品工具

`query_product` 使用 `random.uniform/randint`。真实运行里 SH-E300 被它在 baseline 返回 195.1 元，
SH-S10 在兼容臂返回 411.55 元。即使最终答案未采用，模型可见的冲突事实会放大串价风险。若保留该工具，
应由确定性商品目录提供结果为普通 read tool，而非随机生成。

## 回滚

删除新增 `scripts/e2e_quality_r2/`、`scripts/eval_e2e_quality_r2.py`、对应测试/数据/报告即可；
它没有改变默认 API、数据库、缓存或共享评测入口。生产若应用上述修复，应以独立分支补真实模型回归，
不要复用本轮 held-out 做调参。
