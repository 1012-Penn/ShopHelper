# 检索补齐是否转化为答案质量

日期：2026-10-27。分支：`codex/answer-quality-depth-20261027`。

## 结论

结论是有条件的小幅正向，不是“检索覆盖提升 28.9 个百分点，答案也提升 28.9 个百分点”。

- 126 道证据答案题：baseline **94/126（74.6%）**，证据面 slate **102/126（81.0%）**，净增 8 道（+6.35pp）。
- 全部 170 题：baseline **137/170（80.6%）**，slate **145/170（85.3%）**；两臂各有 3 道不确定，均保留在分母外。
- 同题配对：slate **9 胜、1 负、160 平**。唯一确认退化是 `FRZ-ENTITY-02`：已有明示证据支持心率监测，slate 回答却把另一张表未重复该字段误判成冲突。
- 17 道“baseline 未找齐、slate 找齐”的题中，最终状态为 `{"correct->correct": 8, "incorrect->correct": 2, "incorrect->incorrect": 5, "incorrect->uncertain": 1}`。覆盖补齐直接贡献 2 道净增益；另有 5 道净增益发生在两臂都覆盖完整的题上，体现的是上下文组成/排序对生成的二次影响。
- 追问/拒答整体没有因新检索变好或变差：澄清两臂各 16/17 确定正确、各 1 道题目标歧义保留不确定；应拒答题两臂均 27/27 正确。

因此，这轮结果支持“证据面选择值得 opt-in”，不支持“已证明普遍提升答案正确率”。由于 holdout 仅 20 题且只净增 1 题，也不足以证明统计稳定的泛化收益。

## 同题生成结果

两个 arm 使用相同模型、提示、温度、最大输出预算和 10 条证据上限；唯一差异是证据集合/顺序。证据置信度闸门在两臂中统一旁路，避免把“是否让生成器看到证据”混成第二个变量。每条回答匿名交给同一 `deepseek-flash` 的非 thinking 裁判；裁判输入只有问题、对话上下文、该 arm 自己的证据和回答，没有 gold、bucket 或 arm 名。非 thinking 开关依据 [DeepSeek Chat Completions 文档](https://api-docs.deepseek.com/api/create-chat-completion/)。

完整证据组覆盖按同一个 Top-10 上下文计算：

| 范围 | Top-5 baseline | Top-5 slate | Top-10 baseline | Top-10 slate |
|---|---:|---:|---:|---:|
| 128 道需证据题 | 88/128 | 125/128 | 111/128 | 128/128 |

同题生成答案：

| 范围 | 证据答案 baseline | 证据答案 slate | 配对 W/L/T | McNemar p |
|---|---:|---:|---:|---:|
| dev | 27/42 (64.3%) | 30/42 (71.4%) | 3 / 0 / 57 | 0.2500 |
| frozen-v1 回归 | 49/64 (76.6%) | 53/64 (82.8%) | 5 / 1 / 84 | 0.2188 |
| retrieval-slate holdout | 18/20 (90.0%) | 19/20 (95.0%) | 1 / 0 / 19 | 1.0000 |
| 合计 | 94/126 (74.6%) | 102/126 (81.0%) | 9 / 1 / 160 | 0.0215 |

`strict_accuracy` 把不确定保留为独立类别；本表分子只含确定性正确。frozen 的每臂有 2 道不确定（证据答案和歧义澄清各 1），holdout 的每臂有 1 道证据答案不确定。`p` 值仅是题内配对的描述性检验；dev/frozen 有已看题和重复模板，不能当作独立泛化显著性。

## 事实、条件、引用与行为

- 事实/条件层先于裁判：数值、单位、主体、日期和明确禁用断言由确定性规则检查；裁判只能在硬事实通过后解析语义引用差异。
- 证据答案中，裁判判 ground supported 的条数为 baseline 122/126、slate 121/126；判 partly supported 的条数为 4 与 5。
- 裁判列出无证据断言的条数为 baseline 0 条、slate 3 条。只列短断言并由人工回看，不把裁判主观置信度写成概率。
- 确定性引用检查结果：baseline supported/unsupported/uncertain = 1/58/67，slate = 0/66/60。这组结果明显受同义改写影响，只作硬冲突筛查；语义支持率采用上述匿名裁判结论，并保留人工复核。
- 澄清：baseline 16/17，slate 16/17；其余为题目标歧义，保留 `uncertain`。
- 拒答：baseline 27/27，slate 27/27。

## 证据覆盖到答案的归因

在相同 Top-10 上下文预算下：

| 覆盖关系 | 题数 | 可评答案题 | 最终状态转移 |
|---|---:|---:|---|
| baseline 漏、slate 齐 | 17 | 16 | `{"correct->correct": 8, "incorrect->correct": 2, "incorrect->incorrect": 5, "incorrect->uncertain": 1}` |
| 两臂均齐 | 111 | 110 | `{"correct->correct": 85, "correct->incorrect": 1, "incorrect->correct": 6, "incorrect->incorrect": 16, "uncertain->correct": 1, "uncertain->uncertain": 1}` |
| baseline 齐、slate 漏 | 0 | 0 | `{}` |

旧检索报告中的 88/128 是 Top-5 baseline；本实验为了与实际生成的 Top-10 上下文预算一致，答案生成使用 Top-10，并把旧 Top-5 覆盖单列，避免用不同预算直接比较。

按题型：

| 类型 | baseline 正确/总题 | slate 正确/总题 |
|---|---:|---:|
| single-hop | 33/33 | 33/33 |
| multi-hop | 24/37 | 29/37 |
| multi-constraint | 16/27 | 17/27 |
| entity mismatch | 16/18 | 16/18 |
| policy conflict | 9/15 | 11/15 |
| hard negative（拒答） | 17/17 | 17/17 |
| clarify（澄清） | 12/13 | 12/13 |
| OOD（拒答） | 10/10 | 10/10 |

按证据状态：

| evidence_state | baseline 正确 | slate 正确 |
|---|---:|---:|
| sufficient | 84/107 | 90/107 |
| conflicting | 11/17 | 13/17 |
| partial | 1/4 | 1/4 |
| insufficient | 41/42 | 41/42 |

事实/条件/引用分项：

| 指标 | baseline | slate |
|---|---:|---:|
| 有事实金标且事实通过 | 94/105 | 97/105 |
| 有条件金标且条件通过 | 42/59 | 47/59 |
| 证据题行为通过 | 126/126 | 126/126 |

| 生成+裁判分段 | baseline P50 / P95 | slate P50 / P95 |
|---|---:|---:|
| 证据答案 | 3839.861 / 13820.864 ms | 3795.714 / 14370.15 ms |
| 澄清 | 4470.967 / 16443.714 ms | 3591.366 / 13843.394 ms |
| 拒答 | 3884.03 / 5200.277 ms | 3987.984 / 5891.474 ms |

这里的延迟是单题“生成 + 匿名裁判”的顺序耗时；不包含证据预计算、检索重放和并发排队。

## 题尺修正与裁判可靠性

评分版本为 `answer-quality-depth-score-v7`，在 v3.1 基础上修正：

1. 句号后的 `[n]` 归属前一句；
2. Markdown 粗体不切断数值、单位或条件；
3. 半年/6 个月、一年/12 个月等单位归一；
4. 背景金标不强制回答；
5. 澄清按是否问齐必要槽位判断，拒答按是否明确承认证据不足判断；
6. 完整答案末尾的“需要我继续…”不被判成纯澄清；
7. `FRZ-ENTITY-02` 将错误的 AMOLED 金标改为问题真正询问的心率监测，并逐题记录依据；
8. 题目标分歧或裁判报告关键条件缺失时保留 `uncertain`，不凑进正确率。

所有版本化修订及逐题依据位于 `tests/eval/answer_quality_depth/scoring-overlay-v1.json`。这些修订是在观察到响应后进行的评分复核，不是独立的预先冻结结果；生成阶段的原始答案和原评分文件均保留。

`tests/eval/answer_quality_depth/frozen-config.json` 保留了首次 dev 后冻结的生成/裁判身份；`analysis-config-v7.json` 另外记录最终评分审计完成时的代码与题尺哈希。两者分开，避免把事后评分修订伪装成预先冻结。

## 失败与人工复核

- dev 首轮出现裁判 `content` 为空/格式浮动；根据 DeepSeek 官方 Chat Completions 参数改用 `thinking.type=disabled` 重新裁判，全部 dev 裁判均为非 thinking 模式。
- 340 条初版生成中有 25 条 `finish_reason=length`（其中含 15 条空正文）；第一次修复后又发现 17 条正文虽非空但被截断。合计 42 条按相同提示和模型重生成，失败响应保存在 `generation_history`。最终 340 条均非空且无 length 截断。
- 17 条关键不确定/退化/行为/纠正样本已逐题写入 `tests/eval/answer_quality_depth/manual-review.jsonl`；其中确认 `FRZ-ENTITY-02 slate` 是真实退化，`DEV-ENTITY-06` 的裁判否定断言属误判并已版本化纠正，`FRZ-CLARIFY-04` 保留题目标歧义。
- 最终 340 条无技术失败。首轮失败和重试没有隐藏，也没有被计成模型答错。
- dev/frozen 的改写来自冻结缓存，holdout 使用恒等改写；本轮没有新增改写模型调用。生成和裁判的失败重试都进入同一个 `$2.00` 独立 SQLite 台账。

## 成本、延迟与复现

最终每条结果保留 340 次生成和 340 次裁判；整轮独立台账记录了 1075 个调用/重试 reservation。逐题结果中保留的有效生成/裁判链路成本合计 **$0.499860**；整轮独立保守台账（含接口调试、裁判格式失败、截断重跑和未结算 reservation 的预留额）为 **$0.866871 / $2.00**。费用按 2026-10-26 冻结价格表计算，属于估算而非供应商发票。

逐题原始答案、证据、裁判原文、确定性检查与成本在：

- `tests/eval/answer_quality_depth/runs/dev-final3/results-rescored.jsonl`
- `tests/eval/answer_quality_depth/runs/frozen-v1-final3/results-rescored.jsonl`
- `tests/eval/answer_quality_depth/runs/retrieval-slate-holdout-v1-final3/results-rescored.jsonl`

复现：

```bash
python3 -m scripts.eval_answer_quality_depth prepare-evidence   --split dev   --output tests/eval/answer_quality_depth/evidence/dev.json   --credentials /path/to/read-only/.env
python3 -m scripts.eval_answer_quality_depth estimate   --split dev   --evidence tests/eval/answer_quality_depth/evidence/dev.json
python3 -m scripts.eval_answer_quality_depth run   --split dev   --evidence tests/eval/answer_quality_depth/evidence/dev.json   --run-dir tests/eval/answer_quality_depth/runs/dev-repro   --credentials /path/to/read-only/.env   --ledger tests/eval/answer_quality_depth/repro-ledger.sqlite3   --allow-paid --max-cost-usd 2
python3 -m scripts.eval_answer_quality_depth report
```

## 适用边界

- 这是 Top-10 证据上下文上的小型真实模型实验，不是线上流量 A/B，也不验证真实订单读取/写入。
- dev、frozen 都不是独立泛化集；只有 20 题 retrieval holdout 有更强的新颖性，但仅净增 1 题，不能宣称稳定显著。
- 规则依赖当前商品/订单/政策文档结构；换语料、换检索器或换模型后应重新校准。
- 不应把覆盖率、事实通过率或裁判 `correct` 混写成生产客服最终解决率。

## 简历候选

> 在 ShopHelper 用同题同模型、同 Top-10 上下文的 170 题真实生成+匿名 LLM 裁判对照验证证据面重排：126 道证据答案由 94 道正确提升到 102 道，独立 20 题由 18 道提升到 19 道；同题配对 9 胜 1 负，并定位 1 个“明示事实被冲突判断覆盖”的退化。

不要把这句写成“答案正确率提升 28.9pp”；28.9pp 是旧 Top-5 证据覆盖指标，不是本轮最终答案正确率。
