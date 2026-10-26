# Evidence-Bound Answer Plan R3

日期：2026-10-27

共同基线：`59e233f5145298846f8870b74e351ac3db98f5e1`

分支：`codex/opt-answer-plan-r3-20261027`

实现与结果快照：`05266eb2566d50c495924d72c6ed15c67b232539`

## 结论

结构化证据计划能稳定保留适用条件，但没有在这组新题上改善严格成功，
还引入了明确误拒和更高成本。默认生产路径保持关闭；不把本轮接入现有
AnswerVerifier 或默认 graph。

held-out 唯一一次运行的主结果是自由文本 `16/20`，计划臂 `13/20`。
之后发现冻结评分器有三处确定性缺陷，新增 `score-v2` 对全部保存答案做
非独立审计重评，结果为 `18/20` 与 `16/20`，仍然是负结果。

## 机制

- `app/answer_plan/models.py`：`AnswerPlan`、`PlanClaim`、
  `EvidenceBinding`、`EvidenceCandidate`、`PolicyScope`。
- `planner.py`：只把用户问题和候选证据交给 structured extractor。
- `verification.py`：检查 evidence ID、quote/span、主体、数值/单位、
  政策日期/版本、必要条件和整段复制。
- `pipeline.py`：失败/不确定时最多修正一次并重新验证。
- `renderer.py`：仅把已核验 claim 渲染为 `[n]`，保留条件、限制和来源。
- `graph_node.py`：先发 planning/validated 进度，再一次性输出最终回答。
- `runtime.py`：显式启用时才构建；缺少独立账本直接失败。

`uncertain` 不算正确，也不把词面重合当成语义支持。

## 数据与公平性

- 固定证据诊断：8 dev + 20 held-out，新题，按 source group 和 evidence
  template 隔离。
- 两臂使用同一 `deepseek-flash`、temperature 0、同一题、同一候选证据和
  1,200 output-token 上限。
- 模型输入只包含当前问题和证据可见字段；`expected_behavior`、facts、
  forbidden claims、clarifying slots 和评分规则都不进入 prompt。
- 公共配置哈希：`acc6f66ca8a5131f6ba9adbde2e65041372793bee8d5750003a88c2b1f6edc05`。
- held-out 在代码/配置/数据冻结后只运行一次，声明见
  `runs/heldout-once.json`。

这不是生产全 graph 实验，也没有真实检索；它只回答“同一候选证据下，计划
结构能否改善最终回答”。

## 主结果：冻结 score-v1

| Arm | 严格成功 | 事实 | 条件 | 误拒 | 调用 | 费用 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| free text | 16/20 | 18/20 | 19/20 | 0 | 20 | $0.00530490 | 2,274.082 ms |
| answer plan | 13/20 | 14/20 | **20/20** | 4 | 27 | $0.04426290 | 22,390.696 ms |

配对：计划臂对自由文本 `2胜 / 5负 / 13平`。

| Arm | supported | unsupported | uncertain | not required |
|---|---:|---:|---:|---:|
| free text | 17 | 0 | 1 | 2 |
| answer plan | 13 | 4 | 1 | 2 |

全部 40 个 held-out arm-case 无技术失败。计划臂共 7 次修正；这不等于 7 次
修正都成功。

## 审计复评：score-v2，非独立

首跑后逐题检查发现：

1. 两个 fact 共享一个实体时，只要实体出现就把两个 fact 都视为 covered；
2. 同一谓词出现多次时只检查第一处上下文，导致“95 折活动中……不享 95 折”
   被判错；
3. 答案和证据分别使用同一条件的不同声明别名时，没有连接两边。

`score-v2` 修复上述确定性错误，并复评全部两臂、所有题，不选择性改样本。
它发生在 held-out 已被查看之后，不能称为独立验证。

| Arm | 严格成功 | 事实 | 条件 | 误拒 | supported / unsupported / uncertain |
|---|---:|---:|---:|---:|---|
| free text | 18/20 | 19/20 | 19/20 | 0 | 17 / 0 / 1 |
| answer plan | 16/20 | 16/20 | **20/20** | 4 | 14 / 3 / 1 |

配对变为 `2胜 / 4负 / 14平`。评分修正后仍是负结果；此外 `score-v2`
仍保留少量语言覆盖面限制，因此结论以“两版都为负”为准，不取较高数字
包装提升。

## dev 调试

第一轮 dev 的计划臂是 `0/8`。这不是直接归因于模型：账本 committed
漏算已成功记录，中文数值、条件别名、引用句号和分号切句均有实现错误。
全部用失败测试修复。

第二轮 dev 原始生成为自由文本 `4/8`、计划臂 `6/8`；其中若干差异只需
在当前代码中重放已保存计划和评分，不增加 provider 调用。确定性重算后：

| Arm | 严格成功 | 事实 | 条件 | 误拒 | 调用 | 费用 |
|---|---:|---:|---:|---:|---:|---:|
| free text | 7/8 | 8/8 | 8/8 | 0 | 8 | $0.00243690 |
| answer plan | 8/8 | 8/8 | 8/8 | 0 | 10 | $0.01739557 |

这不是 held-out 效果，只说明 dev 上的实现和评分链路已对齐。

## 失败分析

计划臂四个主要严格失败都在引用/锚定层：

- `AP3-HLD-E120-ANC`：计划写出“不支持主动降噪”，引用整段正确证据，但
  `value="不支持"` 未与 quote 中的能力短语锚定，最终安全收口。
- `AP3-HLD-W30-CALL`：正确写出“支持独立通话”，但自由文本 value 和整段
  quote 无法通过当前 anchor 规则。
- `AP3-HLD-T50-WARRANTY`：正确写出“半年质保”，但数值单位嵌入 free-text
  value，验证器没有把它当成可锚定断言。
- `AP3-HLD-S10-FEATURES`：续航 claim 已核验，防水 claim 未锚定，输出只保留
  一半事实并判为有限帮助。

因此本轮不是证明“所有 claim 都需要结构化”或“自由文本更好”，而是证明：
当前结构化字段到原文锚点的映射过窄，遇到能力/否定/多事实时安全拒答成本高。

## 成本与账本

- held-out 总计 47 次 provider 调用：20 free text + 27 answer plan；
- held-out 费用 `$0.04956780`；
- dev 两次生成加审计前总计 `$0.05130637`；
- 全部独立账本费用 `$0.100874172`，共 85 条真实调用记录；
- 上限 `$2.00`，pending `$0.00`，所有成功记录都已 usage 结算；
- 价格快照 `2026-10-26`：input $0.30/M、cache hit $0.006/M、
  output $1.20/M；
- 无缓存重放；表中 P95 为端到端真实请求耗时。

## 复现

```bash
/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_answer_plan_r3 \
  --split dev --allow-paid --max-cost-usd 2.0

/root/workplace/ShopHelper/.venv/bin/python -m scripts.rescore_answer_plan_r3 \
  --results tests/eval/answer_plan_r3/runs/heldout-20261027T024707Z/results.jsonl \
  --output /tmp/answer-plan-r3-heldout-v2/results.jsonl \
  --scoring-version v2
```

held-out 已有一次性 marker，重复命令会 fail closed。逐题结果保留在：

- `runs/dev-20261027T023958Z/`：第一轮全失败调试；
- `runs/dev-20261027T024340Z/`：第二轮原始生成；
- `runs/dev-v2-audit/`：dev 确定性重算；
- `runs/heldout-20261027T024707Z/`：唯一 held-out 首跑；
- `runs/heldout-v2-audit/`：held-out 评分审计。

## 验证

- answer-plan 定向契约：`43 passed`。
- 分支全量：`1054 passed, 2 skipped, 1 xfailed`，退出码 0。
- `compileall` 与 `git diff --check` 通过。
- 全量退出后 Milvus Lite 仍打印 teardown logging error，但 pytest 已成功退出；
  属于测试服务清理噪声，不影响结果。
- 环境没有 `ruff` 可执行文件，本轮记录为未运行，不以其他静态检查替代。
- 首次全量收集遇到两个目录的同名测试模块冲突；本目录增加包标识后重跑通过。

## 未验证与风险

- 没有接入生产 graph 的真实检索运行，收益不能外推到 RAG 全链路。
- 没有工具/状态轨迹，因为本实验固定候选证据；tool-only 订单题不在本集。
- 20 题 held-out 只能描述当前模拟领域，不构成通用准确率。
- `score-v2` 是事后审计，不是独立 held-out。
- `score-v2` 仍有少量语言覆盖局限；所有 uncertain 均未算成功。
- `app/main.py`、`app/config.py`、现有 graph/agent 不修改，统一验收按
  integration notes 决定是否仍保留默认关闭的接线补丁。
- 新 path 带来的误拒与约 8.34 倍 held-out 费用使当前版本默认关闭。
