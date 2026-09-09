"""用 ch04 标注样例校准正式证据置信度闸。

输入 signals JSONL 的每行至少包含 ``id``、``relevant``、``top1_relevance``，
可选 ``effective_count``、``effective_score_floor``、``top1_top2_margin``。
检索评估脚本或人工标注完成后运行本脚本，把产物作为线上配置输入。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import calibrate_evidence_thresholds

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True, help="带证据分数与相关性标注的 JSONL")
    parser.add_argument("--eval-set", default="tests/eval/ch04_eval_set.jsonl")
    parser.add_argument("--min-recall", type=float, default=1.0)
    parser.add_argument("--output", default="reports/ch09-evidence-calibration.json")
    args = parser.parse_args()

    eval_path = ROOT / args.eval_set
    eval_ids = {
        json.loads(line)["id"] for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    signal_path = ROOT / args.signals
    samples = [json.loads(line) for line in signal_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    unknown = sorted({sample.get("id") for sample in samples} - eval_ids)
    if unknown:
        raise SystemExit(f"signals 含不在 ch04 评估集中的 id:{unknown}")
    calibration = calibrate_evidence_thresholds(samples, min_recall=args.min_recall)
    payload = {
        "source": calibration.source,
        "min_recall": args.min_recall,
        "top1_floor": calibration.top1_floor,
        "effective_score_floor": calibration.effective_score_floor,
        "min_effective_count": calibration.min_effective_count,
        "margin_floor": calibration.margin_floor,
        "combined_floor": calibration.combined_floor,
        "sample_count": len(samples),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"校准产物已写入 {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
