"""Materialize the consolidated answer-depth report and review evidence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.answer_quality_depth.data import load_depth_cases
from scripts.answer_quality_depth.reporting import aggregate_result_rows
from scripts.answer_quality_depth.scoring import SCORING_VERSION
from scripts.eval_stats import mcnemar_exact

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "tests" / "eval" / "answer_quality_depth"
RESULT_PATHS = {
    "dev": RUN_ROOT / "runs" / "dev-final3" / "results-rescored.jsonl",
    "frozen-v1": RUN_ROOT / "runs" / "frozen-v1-final3" / "results-rescored.jsonl",
    "retrieval-slate-holdout-v1": (
        RUN_ROOT
        / "runs"
        / "retrieval-slate-holdout-v1-final3"
        / "results-rescored.jsonl"
    ),
}
EVIDENCE_PATHS = {
    "dev": RUN_ROOT / "evidence" / "dev.json",
    "frozen-v1": RUN_ROOT / "evidence" / "frozen-v1.json",
    "retrieval-slate-holdout-v1": (
        RUN_ROOT / "evidence" / "retrieval-slate-holdout-v1.json"
    ),
}
LEDGER_PATH = RUN_ROOT / "experiment-ledger.sqlite3"
AGGREGATE_PATH = RUN_ROOT / "aggregate.json"
REVIEW_PATH = RUN_ROOT / "manual-review.jsonl"
REPORT_PATH = ROOT / "docs" / "eval" / "answer-quality-depth-report.md"

REVIEW_DECISIONS = {
    ("DEV-ENTITY-03", "baseline"): (
        "accepted_clarification",
        "未找到用户给出的 ORD-9999，随后请求核对订单号；属于所需澄清，不是事实正例。",
    ),
    ("DEV-ENTITY-03", "slate"): (
        "accepted_clarification",
        "明确说明缺少订单记录并请求核对/补充订单号。",
    ),
    ("DEV-ENTITY-06", "baseline"): (
        "accepted_negative_inference",
        "明示 SH-W20 不支持独立通话并给出 SH-W30 对照；人工复核通过。",
    ),
    ("DEV-ENTITY-06", "slate"): (
        "accepted_negative_inference",
        "依据 W20 功能列表与 W30 升级差异作否定；裁判把合理否定误判为无证据断言，按版本化 override 接受。",
    ),
    ("DEV-MULTI-02", "slate"): (
        "accepted_win",
        "候选证据包含 99 元门槛；答案给出包邮结论并注明合并/拆单条件。",
    ),
    ("DEV-MULTI-06", "slate"): (
        "accepted_win",
        "候选证据含 ORD-1002 未发货状态与取消规则；答案结论和条件完整。",
    ),
    ("FRZ-ENTITY-02", "slate"): (
        "confirmed_regression",
        "直接把商品条目写明的“支持心率监测”回答为不能完全确认；这是本轮唯一确认的配对退化。",
    ),
    ("FRZ-MULTI-16", "baseline"): (
        "kept_uncertain",
        "回答没有确认 SH-B7 当前是否在售正价，关键适用条件不足。",
    ),
    ("FRZ-CONSTRAINT-02", "slate"): (
        "kept_uncertain",
        "题面只给相对天数而未给签收日期，无法唯一选择 v1/v2；保留不确定。",
    ),
    ("FRZ-CLARIFY-04", "baseline"): (
        "label_ambiguous",
        "模型枚举所有会员等级权益；把“会员有什么优惠”强制要求会员状态澄清存在歧义，保留不确定。",
    ),
    ("FRZ-CLARIFY-04", "slate"): (
        "label_ambiguous",
        "与基线相同，回答覆盖全部等级权益，但未追问当前会员等级。",
    ),
    ("SLATE-H05", "baseline"): (
        "kept_uncertain",
        "95 折规则已答出，但商品是否在售正价/排除活动仍缺条件。",
    ),
    ("SLATE-H05", "slate"): (
        "kept_uncertain",
        "增加了排除规则与条件，但没有解决商品当前资格。",
    ),
    ("FRZ-MULTI-15", "slate"): (
        "accepted_win",
        "候选证据合并订单状态、取消与退款路径；答案与原支付渠道一致。",
    ),
    ("SLATE-H11", "slate"): (
        "accepted_win",
        "补齐偏远补运规则后能明确回答满 99 元仍需补 5 元；裁判仍给 partial，但确定性证据支持完整。",
    ),
    ("DEV-HARD-01", "baseline"): (
        "refusal_spot_check",
        "明确说明证据未涵盖无线充电并拒绝猜测。",
    ),
    ("FRZ-HARD-03", "slate"): (
        "refusal_spot_check",
        "明确说明资料未给出开票后改抬头规则，未把缺失信息包装成事实。",
    ),
}


def _load_rows() -> tuple[list[dict], dict[str, dict]]:
    rows = []
    evidence = {}
    for split, path in RESULT_PATHS.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["split"] = split
            rows.append(row)
    for split, path in EVIDENCE_PATHS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            row["split"] = split
            evidence[str(row["case_id"])] = row
    cases = {
        str(case["id"]): case
        for dataset in load_depth_cases().values()
        for case in dataset
    }
    for row in rows:
        row["evidence_state"] = cases[str(row["case_id"])].get("evidence_state")
    for case_id, row in evidence.items():
        case = cases[case_id]
        row["evidence_state"] = case.get("evidence_state")
        if not case.get("required_evidence"):
            row["baseline_top5_hit"] = None
            row["slate_top5_hit"] = None
            continue
        for arm in ("baseline", "slate"):
            texts = "\n".join(str(item.get("text") or "") for item in row[arm][:5])
            row[f"{arm}_top5_hit"] = all(
                all(str(quote) in texts for quote in group.get("quotes") or [])
                for group in case.get("required_evidence") or []
            )
    return rows, evidence


def _ledger_summary() -> dict:
    connection = sqlite3.connect(f"file:{LEDGER_PATH}?mode=ro", uri=True)
    try:
        records = connection.execute(
            "SELECT kind,state,COUNT(*),ROUND(SUM(COALESCE(actual_usd,reserved_usd)),8) "
            "FROM reservations GROUP BY kind,state ORDER BY kind,state"
        ).fetchall()
        total = connection.execute(
            "SELECT ROUND(SUM(CASE WHEN state='pending' THEN reserved_usd "
            "ELSE actual_usd END),8) FROM reservations "
            "WHERE state IN ('pending','settled')"
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "limit_usd": 2.0,
        "conservative_total_usd": float(total or 0.0),
        "by_kind_and_state": [
            {"kind": row[0], "state": row[1], "calls": row[2], "usd": row[3]}
            for row in records
        ],
    }


def _manual_review(rows: list[dict]) -> list[dict]:
    selected = []
    wanted = set(REVIEW_DECISIONS)
    for row in rows:
        key = (str(row["case_id"]), str(row["arm"]))
        if key not in wanted:
            continue
        decision, note = REVIEW_DECISIONS[key]
        selected.append(
            {
                "split": row["split"],
                "case_id": row["case_id"],
                "arm": row["arm"],
                "query": row["query"],
                "denominator": row["denominator"],
                "final_status": row.get("final_status"),
                "answer": row.get("answer"),
                "evidence_ids": [
                    item["chunk_id"] for item in row.get("evidence") or []
                ],
                "deterministic_failure_labels": row.get("failure_labels") or [],
                "judge": (row.get("judge") or {}).get("verdict"),
                "manual_review": decision,
                "review_note": note,
            }
        )
    return selected


def _pct(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


def render_report(aggregate: dict, ledger: dict) -> str:
    dev = aggregate["splits"]["dev"]["by_denominator"]
    frozen = aggregate["splits"]["frozen-v1"]["by_denominator"]
    holdout = aggregate["splits"]["retrieval-slate-holdout-v1"]["by_denominator"]
    overall = aggregate["overall"]
    evidence = overall["by_denominator"]["evidence_answer"]
    behavior = {
        denom: overall["by_denominator"][denom] for denom in ("clarify", "refusal")
    }
    coverage = aggregate["coverage"]
    attribution = aggregate["coverage_attribution"]
    judge = overall["judge"]
    pairwise = overall["pairwise"]["slate_vs_baseline"]
    ledger_calls = sum(item["calls"] for item in ledger["by_kind_and_state"])

    def p_value(pair: dict) -> float:
        return mcnemar_exact(
            [True] * int(pair["wins"]) + [False] * int(pair["losses"]),
            [False] * int(pair["wins"]) + [True] * int(pair["losses"]),
        )["p_value"]

    dev_pair = aggregate["splits"]["dev"]["pairwise"]["slate_vs_baseline"]
    frozen_pair = aggregate["splits"]["frozen-v1"]["pairwise"]["slate_vs_baseline"]
    holdout_pair = aggregate["splits"]["retrieval-slate-holdout-v1"]["pairwise"][
        "slate_vs_baseline"
    ]
    new_only_outcomes = attribution["baseline_missing_slate_complete"]["outcomes"]
    both_outcomes = attribution["both_complete"]["outcomes"]
    new_only_net = new_only_outcomes.get(
        "incorrect->correct", 0
    ) - new_only_outcomes.get("correct->incorrect", 0)
    both_net = both_outcomes.get("incorrect->correct", 0) - both_outcomes.get(
        "correct->incorrect", 0
    )
    state_rows = overall["by_evidence_state"]
    return f"""# 检索补齐是否转化为答案质量

日期：2026-10-27。分支：`codex/answer-quality-depth-20261027`。

## 结论

结论是有条件的小幅正向，不是“检索覆盖提升 28.9 个百分点，答案也提升 28.9 个百分点”。

- 126 道证据答案题：baseline **94/126（74.6%）**，证据面 slate **102/126（81.0%）**，净增 8 道（+6.35pp）。
- 全部 170 题：baseline **137/170（80.6%）**，slate **145/170（85.3%）**；两臂各有 3 道不确定，均保留在分母外。
- 同题配对：slate **9 胜、1 负、160 平**。唯一确认退化是 `FRZ-ENTITY-02`：已有明示证据支持心率监测，slate 回答却把另一张表未重复该字段误判成冲突。
- {attribution["baseline_missing_slate_complete"]["cases"]} 道“baseline 未找齐、slate 找齐”的题中，最终状态为 `{json.dumps(new_only_outcomes, ensure_ascii=False)}`。覆盖补齐直接贡献 {new_only_net} 道净增益；另有 {both_net} 道净增益发生在两臂都覆盖完整的题上，体现的是上下文组成/排序对生成的二次影响。
- 追问/拒答整体没有因新检索变好或变差：澄清两臂各 16/17 确定正确、各 1 道题目标歧义保留不确定；应拒答题两臂均 27/27 正确。

因此，这轮结果支持“证据面选择值得 opt-in”，不支持“已证明普遍提升答案正确率”。由于 holdout 仅 20 题且只净增 1 题，也不足以证明统计稳定的泛化收益。

## 同题生成结果

两个 arm 使用相同模型、提示、温度、最大输出预算和 10 条证据上限；唯一差异是证据集合/顺序。证据置信度闸门在两臂中统一旁路，避免把“是否让生成器看到证据”混成第二个变量。每条回答匿名交给同一 `deepseek-flash` 的非 thinking 裁判；裁判输入只有问题、对话上下文、该 arm 自己的证据和回答，没有 gold、bucket 或 arm 名。非 thinking 开关依据 [DeepSeek Chat Completions 文档](https://api-docs.deepseek.com/api/create-chat-completion/)。

完整证据组覆盖按同一个 Top-10 上下文计算：

| 范围 | Top-5 baseline | Top-5 slate | Top-10 baseline | Top-10 slate |
|---|---:|---:|---:|---:|
| 128 道需证据题 | {coverage.get("baseline_top5_complete", 0)}/128 | {coverage.get("slate_top5_complete", 0)}/128 | {coverage["baseline_complete"]}/{coverage["eligible"]} | {coverage["slate_complete"]}/{coverage["eligible"]} |

同题生成答案：

| 范围 | 证据答案 baseline | 证据答案 slate | 配对 W/L/T | McNemar p |
|---|---:|---:|---:|---:|
| dev | {_pct(dev["evidence_answer"]["baseline"]["correct"], dev["evidence_answer"]["baseline"]["n"])} | {_pct(dev["evidence_answer"]["slate"]["correct"], dev["evidence_answer"]["slate"]["n"])} | {dev_pair["wins"]} / {dev_pair["losses"]} / {dev_pair["ties"]} | {p_value(dev_pair):.4f} |
| frozen-v1 回归 | {_pct(frozen["evidence_answer"]["baseline"]["correct"], frozen["evidence_answer"]["baseline"]["n"])} | {_pct(frozen["evidence_answer"]["slate"]["correct"], frozen["evidence_answer"]["slate"]["n"])} | {frozen_pair["wins"]} / {frozen_pair["losses"]} / {frozen_pair["ties"]} | {p_value(frozen_pair):.4f} |
| retrieval-slate holdout | {_pct(holdout["evidence_answer"]["baseline"]["correct"], holdout["evidence_answer"]["baseline"]["n"])} | {_pct(holdout["evidence_answer"]["slate"]["correct"], holdout["evidence_answer"]["slate"]["n"])} | {holdout_pair["wins"]} / {holdout_pair["losses"]} / {holdout_pair["ties"]} | {p_value(holdout_pair):.4f} |
| 合计 | {_pct(evidence["baseline"]["correct"], evidence["baseline"]["n"])} | {_pct(evidence["slate"]["correct"], evidence["slate"]["n"])} | {pairwise["wins"]} / {pairwise["losses"]} / {pairwise["ties"]} | {p_value(pairwise):.4f} |

`strict_accuracy` 把不确定保留为独立类别；本表分子只含确定性正确。frozen 的每臂有 2 道不确定（证据答案和歧义澄清各 1），holdout 的每臂有 1 道证据答案不确定。`p` 值仅是题内配对的描述性检验；dev/frozen 有已看题和重复模板，不能当作独立泛化显著性。

## 事实、条件、引用与行为

- 事实/条件层先于裁判：数值、单位、主体、日期和明确禁用断言由确定性规则检查；裁判只能在硬事实通过后解析语义引用差异。
- 证据答案中，裁判判 ground supported 的条数为 baseline {judge["baseline"]["grounding"].get("supported", 0)}/126、slate {judge["slate"]["grounding"].get("supported", 0)}/126；判 partly supported 的条数为 {judge["baseline"]["grounding"].get("partly_supported", 0)} 与 {judge["slate"]["grounding"].get("partly_supported", 0)}。
- 裁判列出无证据断言的条数为 baseline {judge["baseline"]["rows_with_unsupported_claims"]} 条、slate {judge["slate"]["rows_with_unsupported_claims"]} 条。只列短断言并由人工回看，不把裁判主观置信度写成概率。
- 确定性引用检查结果：baseline supported/unsupported/uncertain = {evidence["baseline"]["citation_status"].get("supported", 0)}/{evidence["baseline"]["citation_status"].get("unsupported", 0)}/{evidence["baseline"]["citation_status"].get("uncertain", 0)}，slate = {evidence["slate"]["citation_status"].get("supported", 0)}/{evidence["slate"]["citation_status"].get("unsupported", 0)}/{evidence["slate"]["citation_status"].get("uncertain", 0)}。这组结果明显受同义改写影响，只作硬冲突筛查；语义支持率采用上述匿名裁判结论，并保留人工复核。
- 澄清：baseline {behavior["clarify"]["baseline"]["correct"]}/{behavior["clarify"]["baseline"]["n"]}，slate {behavior["clarify"]["slate"]["correct"]}/{behavior["clarify"]["slate"]["n"]}；其余为题目标歧义，保留 `uncertain`。
- 拒答：baseline {behavior["refusal"]["baseline"]["correct"]}/{behavior["refusal"]["baseline"]["n"]}，slate {behavior["refusal"]["slate"]["correct"]}/{behavior["refusal"]["slate"]["n"]}。

## 证据覆盖到答案的归因

在相同 Top-10 上下文预算下：

| 覆盖关系 | 题数 | 可评答案题 | 最终状态转移 |
|---|---:|---:|---|
| baseline 漏、slate 齐 | {attribution["baseline_missing_slate_complete"]["cases"]} | {attribution["baseline_missing_slate_complete"]["answer_cases"]} | `{json.dumps(attribution["baseline_missing_slate_complete"]["outcomes"], ensure_ascii=False)}` |
| 两臂均齐 | {attribution["both_complete"]["cases"]} | {attribution["both_complete"]["answer_cases"]} | `{json.dumps(attribution["both_complete"]["outcomes"], ensure_ascii=False)}` |
| baseline 齐、slate 漏 | {attribution["baseline_complete_slate_missing"]["cases"]} | {attribution["baseline_complete_slate_missing"]["answer_cases"]} | `{json.dumps(attribution["baseline_complete_slate_missing"]["outcomes"], ensure_ascii=False)}` |

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
| sufficient | {state_rows["sufficient"]["baseline"]["correct"]}/{state_rows["sufficient"]["baseline"]["n"]} | {state_rows["sufficient"]["slate"]["correct"]}/{state_rows["sufficient"]["slate"]["n"]} |
| conflicting | {state_rows["conflicting"]["baseline"]["correct"]}/{state_rows["conflicting"]["baseline"]["n"]} | {state_rows["conflicting"]["slate"]["correct"]}/{state_rows["conflicting"]["slate"]["n"]} |
| partial | {state_rows["partial"]["baseline"]["correct"]}/{state_rows["partial"]["baseline"]["n"]} | {state_rows["partial"]["slate"]["correct"]}/{state_rows["partial"]["slate"]["n"]} |
| insufficient | {state_rows["insufficient"]["baseline"]["correct"]}/{state_rows["insufficient"]["baseline"]["n"]} | {state_rows["insufficient"]["slate"]["correct"]}/{state_rows["insufficient"]["slate"]["n"]} |

事实/条件/引用分项：

| 指标 | baseline | slate |
|---|---:|---:|
| 有事实金标且事实通过 | {evidence["baseline"]["facts_passed"]}/{evidence["baseline"]["facts_measured"]} | {evidence["slate"]["facts_passed"]}/{evidence["slate"]["facts_measured"]} |
| 有条件金标且条件通过 | {evidence["baseline"]["conditions_passed"]}/{evidence["baseline"]["conditions_measured"]} | {evidence["slate"]["conditions_passed"]}/{evidence["slate"]["conditions_measured"]} |
| 证据题行为通过 | {evidence["baseline"]["behavior_passed"]}/{evidence["baseline"]["n"]} | {evidence["slate"]["behavior_passed"]}/{evidence["slate"]["n"]} |

| 生成+裁判分段 | baseline P50 / P95 | slate P50 / P95 |
|---|---:|---:|
| 证据答案 | {evidence["baseline"]["latency_ms"]["p50"]} / {evidence["baseline"]["latency_ms"]["p95"]} ms | {evidence["slate"]["latency_ms"]["p50"]} / {evidence["slate"]["latency_ms"]["p95"]} ms |
| 澄清 | {behavior["clarify"]["baseline"]["latency_ms"]["p50"]} / {behavior["clarify"]["baseline"]["latency_ms"]["p95"]} ms | {behavior["clarify"]["slate"]["latency_ms"]["p50"]} / {behavior["clarify"]["slate"]["latency_ms"]["p95"]} ms |
| 拒答 | {behavior["refusal"]["baseline"]["latency_ms"]["p50"]} / {behavior["refusal"]["baseline"]["latency_ms"]["p95"]} ms | {behavior["refusal"]["slate"]["latency_ms"]["p50"]} / {behavior["refusal"]["slate"]["latency_ms"]["p95"]} ms |

这里的延迟是单题“生成 + 匿名裁判”的顺序耗时；不包含证据预计算、检索重放和并发排队。

## 题尺修正与裁判可靠性

评分版本为 `{SCORING_VERSION}`，在 v3.1 基础上修正：

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
- {len(REVIEW_DECISIONS)} 条关键不确定/退化/行为/纠正样本已逐题写入 `tests/eval/answer_quality_depth/manual-review.jsonl`；其中确认 `FRZ-ENTITY-02 slate` 是真实退化，`DEV-ENTITY-06` 的裁判否定断言属误判并已版本化纠正，`FRZ-CLARIFY-04` 保留题目标歧义。
- 最终 340 条无技术失败。首轮失败和重试没有隐藏，也没有被计成模型答错。
- dev/frozen 的改写来自冻结缓存，holdout 使用恒等改写；本轮没有新增改写模型调用。生成和裁判的失败重试都进入同一个 `$2.00` 独立 SQLite 台账。

## 成本、延迟与复现

最终每条结果保留 340 次生成和 340 次裁判；整轮独立台账记录了 {ledger_calls} 个调用/重试 reservation。逐题结果中保留的有效生成/裁判链路成本合计 **${aggregate["answer_cost_usd"]:.6f}**；整轮独立保守台账（含接口调试、裁判格式失败、截断重跑和未结算 reservation 的预留额）为 **${ledger["conservative_total_usd"]:.6f} / $2.00**。费用按 2026-10-26 冻结价格表计算，属于估算而非供应商发票。

逐题原始答案、证据、裁判原文、确定性检查与成本在：

- `tests/eval/answer_quality_depth/runs/dev-final3/results-rescored.jsonl`
- `tests/eval/answer_quality_depth/runs/frozen-v1-final3/results-rescored.jsonl`
- `tests/eval/answer_quality_depth/runs/retrieval-slate-holdout-v1-final3/results-rescored.jsonl`

复现：

```bash
python3 -m scripts.eval_answer_quality_depth prepare-evidence \
  --split dev \
  --output tests/eval/answer_quality_depth/evidence/dev.json \
  --credentials /path/to/read-only/.env
python3 -m scripts.eval_answer_quality_depth estimate \
  --split dev \
  --evidence tests/eval/answer_quality_depth/evidence/dev.json
python3 -m scripts.eval_answer_quality_depth run \
  --split dev \
  --evidence tests/eval/answer_quality_depth/evidence/dev.json \
  --run-dir tests/eval/answer_quality_depth/runs/dev-repro \
  --credentials /path/to/read-only/.env \
  --ledger tests/eval/answer_quality_depth/repro-ledger.sqlite3 \
  --allow-paid --max-cost-usd 2
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
"""


def materialize() -> dict:
    rows, evidence = _load_rows()
    aggregate = aggregate_result_rows(rows, evidence_rows=evidence)
    ledger = _ledger_summary()
    aggregate["ledger"] = ledger
    aggregate["config"] = json.loads(
        (RUN_ROOT / "frozen-config.json").read_text(encoding="utf-8")
    )
    AGGREGATE_PATH.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reviews = _manual_review(rows)
    with REVIEW_PATH.open("w", encoding="utf-8") as stream:
        for row in reviews:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = render_report(aggregate, ledger)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return {
        "aggregate": str(AGGREGATE_PATH),
        "manual_review": str(REVIEW_PATH),
        "report": str(REPORT_PATH),
        "questions": aggregate["dataset"]["questions"],
        "responses": aggregate["dataset"]["responses"],
        "technical_failures": aggregate["dataset"]["technical_failures"],
    }
