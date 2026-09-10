"""ch10 · 把微调好的模型导出 ONNX,并做 torch/ORT 数值对齐校验(通过才落盘)。

前置:scripts/train_topic_classifier.py 已产出 models/topic_classifier/。
用法:.venv/bin/python -m scripts.export_topic_onnx
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="导出主题分类器 ONNX")
    parser.add_argument("--model-dir", default="models/topic_classifier")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "label_config.json").is_file():
        raise SystemExit(f"{model_dir} 缺 label_config.json,先训练")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.eval()

    sample = tokenizer(["这个能开发票吗", "买大了想退"], padding=True, return_tensors="pt")
    onnx_path = model_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "attention_mask": {0: "batch", 1: "seq"},
                          "logits": {0: "batch"}},
            opset_version=args.opset,
        )

    # 对齐校验:随机文本上 torch 与 ORT 的 logits 必须一致,不一致就拒交
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    probe = tokenizer(["麻烦帮我催一下快递,太慢了", "七天内无理由退货要运费吗",
                       "会员积分快过期了怎么办"], padding=True, return_tensors="np")
    feed = {name: probe[name] for name in (i.name for i in session.get_inputs())}
    ort_logits = session.run(None, feed)[0]
    with torch.no_grad():
        torch_logits = model(input_ids=torch.from_numpy(probe["input_ids"]),
                             attention_mask=torch.from_numpy(probe["attention_mask"])).logits.numpy()
    max_diff = float(np.abs(ort_logits - torch_logits).max())
    if max_diff > 1e-3:
        onnx_path.unlink(missing_ok=True)
        raise SystemExit(f"ONNX 与 torch 输出偏差 {max_diff:.2e} 超限,已拒绝落盘")
    print(f"[done] {onnx_path}(opset {args.opset},对齐偏差 {max_diff:.2e})")


if __name__ == "__main__":
    main()
