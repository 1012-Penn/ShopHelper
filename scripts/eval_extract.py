# scripts/eval_extract.py
"""抽取质量评估:跑真实 DeepSeek,报告逐字段准确率。需要 .env 有真实 key。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.llm import make_extract_model
from app.schemas import AfterSaleExtraction

SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "eval" / "extract_samples.jsonl"


def main() -> int:
    model = make_extract_model(Settings()).with_structured_output(
        AfterSaleExtraction, method="function_calling"
    )
    samples = [
        json.loads(line)
        for line in SAMPLES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    order_ok = type_ok = 0
    for i, s in enumerate(samples, 1):
        got: AfterSaleExtraction = model.invoke(s["text"])
        exp = s["expected"]
        o_ok = got.order_no == exp["order_no"]
        t_ok = got.issue_type == exp["issue_type"]
        order_ok += o_ok
        type_ok += t_ok
        print(f"[{'PASS' if o_ok and t_ok else 'FAIL'}] #{i} {s['text'][:24]}…")
        print(f"    order_no   got={got.order_no!r} want={exp['order_no']!r}")
        print(f"    issue_type got={got.issue_type!r} want={exp['issue_type']!r}")
        print(f"    resolution got={got.expected_resolution!r} want(语义)~{exp['expected_resolution']!r}")

    n = len(samples)
    print(f"\norder_no:   {order_ok}/{n}")
    print(f"issue_type: {type_ok}/{n}")
    print("expected_resolution 请对照上方 got/want 逐条人工复核")
    passed = order_ok == n and type_ok == n
    print("RESULT:", "达标" if passed else "不达标——迭代抽取字段描述后重跑")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
