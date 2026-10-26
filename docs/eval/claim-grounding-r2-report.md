# 选择性 Claim Grounding R2 实验报告

日期：2026-10-27
分支：`codex/opt-claim-grounding-r2-20261027`
基线：`446fbb7f9972875b7649555cb9b4b19e402cea29`

## 结论

本轮跑通了“v3.1 规则判为 `uncertain` 后，只对单条 claim + 候选证据调用
DeepSeek Flash 三态判定”的完整链路。真实调用没有把任何 gold
`unsupported` 升为 `supported`，且首轮 held-out 分类正确数从规则基线
14/32 提升到 22/32。该结果仍不足以默认启用：

- `supported` 只召回 2/8；第二轮 bugfix 后为 3/8。主要原因是 v3.1 把
  “两年=24 个月、1.5 小时=90 分钟、20Ah=20000mAh、1200g=1.2kg”
  错判成 `unsupported`，而 selective 开关按规定不会再调用语义层。
- 首轮 `unsupported` precision 为 16/24，第二轮为 17/23；模型把两个
  gold 为 `uncertain` 的绝对化条件 claim 判为矛盾，说明“条件缺失默认
  uncertain”没有被稳定执行。
- gold 本身仍是 `authored_pending_domain_review`，不是人类双标结果。

因此该模块适合离线复核队列或诊断，不应接入默认答案拒答链路；默认路径
保持不变。

## 数据与冻结

数据位于 `tests/eval/claim_grounding_r2/data/`：

- dev 12 条：从历史 v3.1 `uncertain` trace 抽取 claim，保留 case/trace/
  chunk 来源。
- held-out 32 条：新写对抗 claim，覆盖同义支持、条件缺失、否定、单位、
  日期边界、跨订单串用和证据过期。
- 12 个 dev group 与 32 个 held-out group 不相交；同一业务事实/模板只在
  一个集合中出现。
- 冻结记录：`tests/eval/claim_grounding_r2/frozen-config.json`。

冻结 hash：

```text
dev data    a6f66cf2500abfd255eb00642ce91ddc06aeaed903e70ec9793bbea4e88f48a2
heldout data 70632498218bcab72102dc31d3db0b3cc9d44a0b33c15deba4573bfbe505939e
prompt       5d4f4506bc0ebd0a4128975258903cc417fe77b7004cb1347eb3bcd5505644bc
pricing      99261018b4e931a54f6b55f81eed1d413c38eacc8188d7f0723ac7c8e728e478
```

Gold 只进入独立评分器。对三份真实运行 prompt 做了自动扫描，未发现
`gold_reason` 或 gold 状态值进入请求。

## 技术选择

官方模型卡比较：

- [BGE-M3 zero-shot v2.0](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0)
  多语、CPU/GPU 可运行、MIT，但主权重约 1.14 GB，超过本轮 600 MB
  下载上限；其二分类的 not-entailment 也不能直接解释为矛盾。
- [multilingual MiniLMv2-L6 NLI](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli)
  权重 428 MB、0.1B、MIT，XNLI 中文准确率 0.721，三分类且有 neutral。
  它是可行的本地候选，但本轮只选择一个 backend 做真实对照。
- 最终选择现有 DeepSeek Flash 单 claim 结构化判定。它支持三态、证据 ID
  和矛盾码约束，无需下载模型。本报告不把该路径称为“已部署本地 NLI”。

最终 backend 为 `deepseek-flash`。价格使用冻结的
`pricing-snapshot-2026-10-26.yaml`：输入 $0.30/M、输出 $1.20/M、
缓存命中输入 $0.006/M。

## 判定机制

1. 硬确定性层先检查订单、型号、数值、单位、政策日期边界、明确否定和
   支付渠道冲突；命中后返回 `unsupported`，backend 无覆盖权。
2. 未命中时运行现有 `rag-quality-v3.1` claim 规则。
3. 仅当 v3.1 为 `uncertain` 时调用一次语义 backend。
4. API 超时、解析失败、上下文外引用、缺证据的 supported 或缺矛盾码的
   unsupported 全部降级为 `uncertain`。
5. 没有候选证据时不调用 backend。

## Dev 结果

| 指标 | v3.1 规则 | selective grounding |
|---|---:|---:|
| 分类正确 | 3/12 | **10/12** |
| supported | 0/5 | 5/5 |
| supported precision | 0/0（N/A） | 5/6 = 83.3% |
| unsupported recall | 1/5 = 20.0% | 4/5 = 80.0% |
| uncertain 率 | 11/12 = 91.7% | 2/12 = 16.7% |
| backend 调用比例 | 0/12 | 8/12 = 66.7% |
| token | 0 | 3,848 in + 1,507 out |
| 费用 | $0 | $0.002699376 |
| 在线 P50 / P95 | N/A | 1,459.87 / 2,493.19 ms |

错误：一个 gold `uncertain` 被升为 `supported`（历史“当前仍在验货”
证据过旧）；一个 gold `unsupported` 保持 `uncertain`。这两条限制
`uncertain` 结论的可信度。

## 首次 Held-out（冻结后首轮）

运行：`tests/eval/claim_grounding_r2/results/selective-heldout-1/`。

| 指标 | v3.1 规则 | selective grounding |
|---|---:|---:|
| 分类正确 | 14/32 = 43.75% | **22/32 = 68.75%** |
| supported precision | 0/0（N/A） | 2/2 = 100% |
| supported recall | 0/8 | 2/8 = 25% |
| unsupported precision | 7/11 = 63.6% | 16/24 = 66.7% |
| unsupported recall | 7/17 = 41.2% | 16/17 = 94.1% |
| uncertain precision | 7/21 = 33.3% | 4/6 = 66.7% |
| uncertain recall | 7/7 = 100% | 4/7 = 57.1% |
| macro-F1 | 0.5000 | 0.5986 |
| uncertain 率 | 21/32 = 65.6% | 6/32 = 18.75% |
| backend 调用比例 | 0/32 | 16/32 = 50% |
| token | 0 | 6,121 in + 2,999 out |
| 费用 | $0 | $0.004682460 |
| 在线 P50 / P95 | N/A | 1,427.25 / 2,367.46 ms |

关键失败：

- 4 个 gold-supported 单位等价 claim 被 v3.1 直接判为 `unsupported`，
  没有进入语义层。
- 一个 gold-supported 政策边界 claim 被旧硬编码日期规则误判。
- 一个“今日送达”claim 的 `as_of_date` 被错误当成历史订单日期，
  导致 stale evidence 被误判为矛盾。
- 条件缺失样本中有 2 条绝对化 claim 被模型判为 `unsupported`。
- 3 个语义请求因 JSON 为空/截断解析失败，全部保留为 `uncertain`。

## 事后 Bugfix 版本

首轮 read-only 结果保留在 `selective-heldout-1/`。发现上述两个确定性
解析错误后，仅修复：

1. `as_of_date` 只作为实验上下文，不再自动等同 claim 内断言日期；
2. 政策版本边界从证据片段/区间推导，不再硬编码 `2026-06-01` 的
   v1/v2 二分。

修复后运行：`tests/eval/claim_grounding_r2/results/selective-heldout-2-bugfix/`。
这是冻结后 bugfix，不是新的独立验证。

| 指标 | 值 |
|---|---:|
| 分类正确 | 25/32 = 78.125% |
| supported precision | 3/3 = 100% |
| supported recall | 3/8 = 37.5% |
| unsupported precision | 17/23 = 73.9% |
| unsupported recall | 17/17 = 100% |
| uncertain precision | 5/6 = 83.3% |
| uncertain recall | 5/7 = 71.4% |
| macro-F1 | 0.7216 |
| uncertain 率 | 6/32 = 18.75% |
| backend 候选比例 | 17/32 = 53.1% |
| 其中 exact-cache 命中 | 15；新 provider 请求 2 |
| 新请求 token | 861 in + 505 out |
| 新请求费用 | $0.000751404 |
| 新请求在线 P50 / P95 | 1,872.83 / 2,963.57 ms |

第二轮仍有 4 个单位等价 gold-supported 被 v3.1 短路；这是“只在
uncertain 上调用”的结构限制，不是 backend 能力结论。

## 成本与延迟口径

- 三组真实运行共 26 次 provider 请求、10,830 input + 5,011 output token，
  实际费用 **$0.00813324**，低于本轮独立 $2.00 保护上限。
- 26 次在线请求 P50/P95 为 **1,459.87 / 2,493.19 ms**。
- dev 与第二轮 held-out 的 exact-cache replay P95 分别为
  **1.023 ms** 和 **1.035 ms**；这只是本地缓存重放，不是在线 P95。
- 上述延迟是单 claim provider request 墙钟，不是 `/api/chat` 端到端延迟。
- 所有调用走独立 SQLite 账本，先原子预留、后按 provider usage 结算；
  26 次全部 settled，无 pending 遗留。

## 下游答案修改

本轮没有启用答案改写、拒答或修订。由于 claim 层尚未证明可默认启用，
下游分数变化无法把质量问题归因于 grounding，不能宣称模型回答变好。

## 简历候选句

1. “在 32 条人工待审对抗 claim 上实现 selective three-state grounding，
   将分类正确数从 v3.1 规则的 14/32 提高到首轮 held-out 22/32，并把
   17 条确定矛盾中的 16 条判为 unsupported、0 条误升为 supported。”
2. “用独立 SQLite 预算账本完成 26 次真实 DeepSeek Flash 调用，
   总费用 $0.00813；单 claim 请求 P95 2.49s，exact-cache 重放
   P95 约 1.03ms，二者分开报告。”

限制：样本是模拟电商数据且 gold 仍待领域人工复核；单位等价支持召回仅
3/8，存在条件缺失误判，不能表述为线上准确率或“默认启用后回答更好”。

## 可复跑入口

```bash
/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_claim_grounding_r2 freeze
/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_claim_grounding_r2 \
  run --mode rules --split dev --run-id rules-dev-1
/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_claim_grounding_r2 \
  run --mode selective --split dev --run-id selective-dev-1 \
  --allow-paid --credential-env /root/workplace/ShopHelper/.env \
  --max-cost-usd 2 --measure-cache-replay
/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_claim_grounding_r2 \
  rescore --run-id selective-heldout-1
```

`--allow-paid` 才会真实调用；凭据只从进程内读取，不写入结果或 manifest。
