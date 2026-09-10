"""ch10 · 留出测试集评测:每类目 P/R/F1 + 每类目混淆矩阵 + 错例清单 + 多标签验收样例。

评测类产出,除脚本逻辑外按「评估集真跑」验证:
  .venv/bin/python -m scripts.eval_topic_classifier
产物:reports/ch10-topic-classification-report.md / reports/ch10-error-samples.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.topic_taxonomy import LABEL_COUNT, TOPIC_LABELS
from scripts.train_topic_classifier import load_jsonl, make_collate

ACCEPTANCE_SAMPLE = "买大了想退"  # 验收 3:须同时命中多个类目


def predict_texts(model, tokenizer, texts: list[str], max_len: int, threshold: float,
                  batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tokenizer(chunk, truncation=True, max_length=max_len)
            batch = make_collate(tokenizer)([{"multi_hot": [0.0] * LABEL_COUNT, **e} for e in enc])
            del batch["labels"]
            logits = model(**batch).logits
            all_probs.append(torch.sigmoid(logits).numpy())
    probs = np.concatenate(all_probs)
    return probs, (probs >= threshold).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="ch10 主题分类器测试集评测")
    parser.add_argument("--data-dir", default="tests/eval/topic_dataset")
    parser.add_argument("--model-dir", default="models/topic_classifier")
    parser.add_argument("--report", default="reports/ch10-topic-classification-report.md")
    parser.add_argument("--errors", default="reports/ch10-error-samples.md")
    parser.add_argument("--error-sample-cap", type=int, default=40,
                        help="错例报告最多人工复核条数")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / "label_config.json").read_text(encoding="utf-8"))
    threshold = float(config.get("threshold", 0.5))
    max_len = int(config.get("max_len", 96))
    assert list(config["labels"]) == list(TOPIC_LABELS), "模型标签序与权威术语表不一致"

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    test_rows = load_jsonl(Path(args.data_dir) / "test.jsonl")
    probs, preds = predict_texts(model, tokenizer, [r["question"] for r in test_rows],
                                 max_len, threshold)
    gold = np.array([[1 if label in row["labels"] else 0 for label in TOPIC_LABELS]
                     for row in test_rows])

    per_class = precision_recall_fscore_support(gold, preds, average=None, zero_division=0)
    micro = precision_recall_fscore_support(gold, preds, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(gold, preds, average="macro", zero_division=0)
    exact = float((preds == gold).all(axis=1).mean())

    lines = ["# ch10 · 主题分类器评测报告(留出测试集)", "",
             f"- 模型:`{model_dir}`(RoBERTa-wwm-ext 全参微调,17 类多标签,阈值 {threshold})",
             f"- 测试集:{len(test_rows)} 条;完全匹配(exact match)率:{exact:.3f}",
             f"- 微平均 P/R/F1:{micro[0]:.3f} / {micro[1]:.3f} / {micro[2]:.3f}",
             f"- 宏平均 P/R/F1:{macro[0]:.3f} / {macro[1]:.3f} / {macro[2]:.3f}",
             "", "## 各类目指标与混淆矩阵", "",
             "| 类目 | 支持数 | 精确率 | 召回率 | F1 | TP | FP | FN | TN |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    confusion_rows = []
    for i, label in enumerate(TOPIC_LABELS):
        p, r, f1, support = per_class[0][i], per_class[1][i], per_class[2][i], per_class[3][i]
        tn, fp, fn, tp = confusion_matrix(gold[:, i], preds[:, i], labels=[0, 1]).ravel()
        confusion_rows.append((label, tn, fp, fn, tp))
        lines.append(f"| {label} | {int(support)} | {p:.3f} | {r:.3f} | {f1:.3f} "
                     f"| {tp} | {fp} | {fn} | {tn} |")

    # 验收 3:「买大了想退」多诉求句须同时命中多个类目
    acc_probs, acc_preds = predict_texts(model, tokenizer, [ACCEPTANCE_SAMPLE], max_len, threshold)
    acc_labels = [label for i, label in enumerate(TOPIC_LABELS) if acc_preds[0][i]]
    lines += ["", "## 多标签验收样例", "",
              f"- 样例:「{ACCEPTANCE_SAMPLE}」→ 命中 {len(acc_labels)} 类:{'、'.join(acc_labels) or '无'}"
              f"(概率:{', '.join(f'{label}={acc_probs[0][TOPIC_LABELS.index(label)]:.2f}' for label in acc_labels) or '—'})",
              f"- 判定:{'✅ 同时命中多个类目' if len(acc_labels) >= 2 else '❌ 未同时命中多个类目'}"]

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] → {report}")

    # 错例清单(预测集合 ≠ 金标集合),供人工抽判复核
    error_lines = ["# ch10 · 错例清单(人工抽判复核用)", "",
                   f"- 测试集 {len(test_rows)} 条,错例 {int((preds != gold).any(axis=1).sum())} 条;"
                   f"下面按「多/漏标签 → 误标」排序,抽前 {args.error_sample_cap} 条。",
                   ""]
    errors = []
    for idx, row in enumerate(test_rows):
        if (preds[idx] == gold[idx]).all():
            continue
        gold_labels = [label for i, label in enumerate(TOPIC_LABELS) if gold[idx][i]]
        pred_labels = [label for i, label in enumerate(TOPIC_LABELS) if preds[idx][i]]
        missed = [l for l in gold_labels if l not in pred_labels]
        spurious = [l for l in pred_labels if l not in gold_labels]
        errors.append((len(spurious) * 2 + len(missed), idx, row, gold_labels, pred_labels,
                       missed, spurious))
    errors.sort(key=lambda e: -e[0])
    for _, idx, row, gold_labels, pred_labels, missed, spurious in errors[:args.error_sample_cap]:
        prob_str = ", ".join(f"{l}={probs[idx][TOPIC_LABELS.index(l)]:.2f}" for l in pred_labels)
        error_lines.append(f"- {row['question']}\n"
                           f"  - 金标:{'、'.join(gold_labels)};预测:{'、'.join(pred_labels) or '无'}({prob_str})\n"
                           f"  - 漏:{'、'.join(missed) or '—'};误:{'、'.join(spurious) or '—'}"
                           f" [origin={row.get('origin', '?')}]")
    errors_path = Path(args.errors)
    errors_path.write_text("\n".join(error_lines) + "\n", encoding="utf-8")
    print(f"[errors] 错例 {len(errors)} 条 → {errors_path}")

    passed_multi = len(acc_labels) >= 2
    print(f"[accept-3] 「{ACCEPTANCE_SAMPLE}」命中 {len(acc_labels)} 类:{'、'.join(acc_labels)} "
          f"→ {'PASS' if passed_multi else 'FAIL'}")
    raise SystemExit(0 if passed_multi else 3)


if __name__ == "__main__":
    main()
