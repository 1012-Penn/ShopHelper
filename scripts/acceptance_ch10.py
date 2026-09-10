"""ch10 真机验收:微调主题分类器三条验收标准。

前提:
  docker compose up -d                                   # MySQL(已 apply db/ch10-topic-classifier.sql)
  .venv/bin/python -m scripts.build_topic_dataset        # 语料已构建
  .venv/bin/python -m scripts.train_topic_classifier     # 已训练
  .venv/bin/python -m scripts.export_topic_onnx          # 已导出 ONNX
  .venv/bin/uvicorn app.main:app --port 8000             # 主服务
  .venv/bin/python scripts/acceptance_ch10.py --scenario all

场景:
  report     验收 1:测试集各类目 F1 + 混淆矩阵报告存在且数字齐(reports/ch10-*.md)
  classify   验收 2:演示问题灌入池子 → 旁路批量归类 → /admin/topics 分布页有数
  multilabel 验收 3:「买大了想退」等多诉求句同时命中多个类目(真 ONNX 推理)
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = os.environ.get("ACCEPT_BASE", "http://127.0.0.1:8000")
REPORT = ROOT / "reports/ch10-topic-classification-report.md"
DEMO_POOL = ROOT / "tests/eval/topic_dataset/demo_pool.jsonl"

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))


def scenario_report() -> None:
    """验收 1:报告能跑出来,数字齐、口径对。"""
    if not REPORT.is_file():
        record("验收1 分类评测报告存在", False, f"缺 {REPORT},先跑 eval_topic_classifier")
        return
    text = REPORT.read_text(encoding="utf-8")
    checks = {
        "微平均行": "微平均 P/R/F1" in text,
        "宏平均行": "宏平均 P/R/F1" in text,
        "17 类目指标表": all(f"| {label} |" in text for label in
                            ["退换货", "物流", "尺码", "发票", "价保", "保修维修", "其他"]),
        "多标签验收样例": "买大了想退" in text,
    }
    micro = re.search(r"微平均 P/R/F1:([\d.]+) / ([\d.]+) / ([\d.]+)", text)
    if micro:
        checks[f"微 F1={micro.group(3)} ≥ 0.6"] = float(micro.group(3)) >= 0.6
    for name, ok in checks.items():
        record(f"验收1 {name}", ok)
    errors = ROOT / "reports/ch10-error-samples.md"
    record("验收1 错例清单可复核", errors.is_file(), str(errors.relative_to(ROOT)))


def seed_demo_pool() -> int:
    """演示问题灌入真实问题池(幂等:池里已有的原话跳过),返回新灌条数。

    池内待归类为 0 时(上一轮验收已把演示问题归类过),现造一批新演示问题,
    保证验收永远能演示「攒够一批归一次」的产量。
    """
    from sqlalchemy import select

    from app.config import Settings
    from app.db import make_engine, make_session_factory
    from app.models import LowConfidenceQuestion, TopicClassification
    factory = make_session_factory(make_engine(Settings()))
    questions = [json.loads(line)["question"] for line in
                 DEMO_POOL.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if DEMO_POOL.is_file() else []
    with factory() as session:
        existing = set(session.scalars(select(LowConfidenceQuestion.raw_question)).all())
        pending = session.query(LowConfidenceQuestion.id).outerjoin(
            TopicClassification,
            LowConfidenceQuestion.id == TopicClassification.question_id
        ).filter(TopicClassification.id.is_(None)).count()
    if pending == 0:
        from app.llm import make_extract_model
        from scripts.build_topic_dataset import BareQuestions, generate_demo_questions
        model = make_extract_model(Settings()).with_structured_output(
            BareQuestions, method="function_calling")
        questions += [q for q in generate_demo_questions(model, 40)
                      if q not in existing]
    seeded = 0
    with factory() as session:
        existing = set(session.scalars(select(LowConfidenceQuestion.raw_question)).all())
        for q in questions:
            if q in existing:
                continue
            session.add(LowConfidenceQuestion(
                raw_question=q, source="retrieval_low_conf",
                reason="ch10 验收:模拟日常积压灌入的演示问题"))
            seeded += 1
        session.commit()
    return seeded


def scenario_classify() -> None:
    """验收 2:攒下的问题跑批量归类,后台分布页出统计。"""
    if not DEMO_POOL.is_file():
        record("验收2 演示问题文件存在", False, "先跑 build_topic_dataset(--demo-only)")
        return
    seeded = seed_demo_pool()
    record("验收2 演示问题灌入问题池", True, f"新灌 {seeded} 条(重复跳过)")

    with httpx.Client(timeout=300) as client:
        resp = client.post(f"{BASE}/api/topics/classify-batch", json={"limit": 200})
        if resp.status_code == 503:
            record("验收2 旁路批量归类", False, "模型未就绪:先 train + export 再起服务")
            return
        body = resp.json()
        record("验收2 旁路批量归类执行", resp.is_success,
               f"本批归类 {body.get('classified')} 条")
        record("验收2 本批确有产量", body.get("classified", 0) > 0)

        dist = client.get(f"{BASE}/api/topics/distribution").json()
        nonzero = [i for i in dist["items"] if i["count"] > 0]
        record("验收2 分布 API 返回 17 类目", len(dist["items"]) == 17)
        record("验收2 已归类条数 > 0", dist["classified"] > 0,
               f"已归类 {dist['classified']},待归类 {dist['unclassified']}")
        record("验收2 覆盖 ≥5 个类目", len(nonzero) >= 5,
               "、".join(f"{i['label']}×{i['count']}" for i in
                         sorted(nonzero, key=lambda x: -x["count"])[:6]))

        page = client.get(f"{BASE}/admin/topics")
        record("验收2 后台主题分布页可访问", page.status_code == 200
               and "主题分布" in page.text, f"{BASE}/admin/topics")


def scenario_multilabel() -> None:
    """验收 3:多诉求句同时命中多个类目(进程内真 ONNX 推理,不经聊天主链路)。"""
    from app.config import Settings
    from app.topic_classifier import TopicClassifierService

    service = TopicClassifierService(Settings())
    if not service.available():
        record("验收3 ONNX 模型可用", False, "缺 model.onnx,先跑 export_topic_onnx")
        return
    record("验收3 ONNX 模型可用", True)

    cases = {
        "买大了想退": {"退换货", "尺码"},
        "尺码拍大了想退掉,运费谁出": {"尺码", "退换货", "运费"},
        "退了重新买个小一码的,能快点发货吗": {"退换货", "尺码", "物流"},
    }
    predictions = service.predict(list(cases))
    for (text, expected), labels in zip(cases.items(), predictions):
        hit = set(labels)
        record(f"验收3 「{text}」命中 ≥2 类", len(labels) >= 2,
               f"命中:{'、'.join(labels) or '无'}")
        missed = expected - hit
        record(f"验收3 「{text}」字面诉求不漏", not missed,
               f"漏:{'、'.join(missed)}" if missed else "字面提到的类目全中")


def main() -> None:
    parser = argparse.ArgumentParser(description="ch10 真机验收")
    parser.add_argument("--scenario", default="all",
                        choices=["all", "report", "classify", "multilabel"])
    args = parser.parse_args()
    if args.scenario in ("all", "report"):
        print("== 验收 1:评测报告 ==")
        scenario_report()
    if args.scenario in ("all", "classify"):
        print("\n== 验收 2:批量归类 + 主题分布页 ==")
        scenario_classify()
    if args.scenario in ("all", "multilabel"):
        print("\n== 验收 3:多标签 ==")
        scenario_multilabel()
    print("\n== 验收结果 ==")
    for name, ok, detail in RESULTS:
        print(f"{'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过" + (f";未过:{failed}" if failed else ""))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
