"""ch10 · RoBERTa-wwm-ext 全参微调(多标签 17 类)。训练类产出,验证方式:真跑 + 留出评估集。

不用 LoRA;正则 = weight_decay + warmup,过拟合防线 = 验证集微 F1 早停。
产物:models/topic_classifier/(checkpoint + tokenizer + label_config.json);
     训练曲线 reports/ch10-training-metrics.json(models/ 不进 git,指标进 git)。

用法:
  .venv/bin/python -m scripts.train_topic_classifier
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

from app.topic_taxonomy import LABEL_COUNT, TOPIC_LABELS

BASE_MODEL = "hfl/chinese-roberta-wwm-ext"
DEFAULT_MODEL_DIR = Path("models/topic_classifier")
DEFAULT_DATA_DIR = Path("tests/eval/topic_dataset")
DEFAULT_THRESHOLD = 0.5


class TopicDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_len: int) -> None:
        self.texts = [row["question"] for row in rows]
        self.encodings = tokenizer(self.texts, truncation=True, max_length=max_len)
        self.multi_hot = [[1.0 if label in row["labels"] else 0.0 for label in TOPIC_LABELS]
                          for row in rows]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.encodings[idx])
        item["multi_hot"] = self.multi_hot[idx]
        return item


def make_collate(tokenizer):
    def collate(rows: list[dict]) -> dict:
        model_inputs = [{k: v for k, v in row.items() if k != "multi_hot"} for row in rows]
        batch = tokenizer.pad(model_inputs, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([row["multi_hot"] for row in rows], dtype=torch.float)
        return batch
    return collate


def compute_metrics(eval_pred):
    """阈值 0.5 的 sigmoid 多标签:微 F1 给早停,宏 F1 看长尾类目。"""
    logits, gold = eval_pred
    probs = 1.0 / (1.0 + torch.exp(-torch.tensor(logits)))
    preds = (probs >= DEFAULT_THRESHOLD).numpy().astype(int)
    return {"f1": f1_score(gold, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(gold, preds, average="macro", zero_division=0)}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="ch10 主题分类器全参微调")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    data_dir, model_dir = Path(args.data_dir), Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=LABEL_COUNT, problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(TOPIC_LABELS)},
        label2id={label: i for i, label in enumerate(TOPIC_LABELS)})

    train_set = TopicDataset(load_jsonl(data_dir / "train.jsonl"), tokenizer, args.max_len)
    val_set = TopicDataset(load_jsonl(data_dir / "val.jsonl"), tokenizer, args.max_len)
    # transformers 5.x 只有 warmup_steps(没有 warmup_ratio),按 10% 步数折算
    steps_per_epoch = -(-len(train_set) // args.batch_size)
    warmup_steps = max(1, round(0.1 * args.epochs * steps_per_epoch))
    print(f"[data] train={len(train_set)} val={len(val_set)} max_len={args.max_len} "
          f"steps/epoch={steps_per_epoch} warmup={warmup_steps}")

    hub_args = TrainingArguments(
        output_dir=str(model_dir / "runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,            # 正则:全参微调防过拟合第一道闸
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        save_total_limit=2,
        seed=42,
        dataloader_num_workers=0,     # CPU 训练 + 内存紧,不开多进程
        report_to=[],
    )
    trainer = Trainer(model=model, args=hub_args, train_dataset=train_set, eval_dataset=val_set,
                      data_collator=make_collate(tokenizer), compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)])
    trainer.train()

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    (model_dir / "label_config.json").write_text(json.dumps({
        "labels": list(TOPIC_LABELS), "threshold": DEFAULT_THRESHOLD,
        "max_len": args.max_len, "base_model": args.base_model,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path = Path("reports/ch10-training-metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    best = [h for h in trainer.state.log_history if "eval_f1" in h]
    print(f"[done] 模型 → {model_dir};最佳验证微 F1 = {max(h['eval_f1'] for h in best):.4f}")


if __name__ == "__main__":
    main()
