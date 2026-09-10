"""ch10 · 旁路批量归类:低置信度问题攒够一批归一次,结果落 topic_classifications。

与 POST /api/topics/classify-batch 共用 TopicClassifierService;实时对话主链路不调它。
用法:.venv/bin/python -m scripts.classify_topics [--limit 200]
"""
import argparse
from collections import Counter

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.store import TopicStore
from app.topic_classifier import TopicClassifierService


def main() -> None:
    parser = argparse.ArgumentParser(description="ch10 旁路批量主题归类")
    parser.add_argument("--limit", type=int, default=None, help="本批最多归类多少条")
    args = parser.parse_args()

    settings = Settings()
    store = TopicStore(make_session_factory(make_engine(settings)))
    service = TopicClassifierService(settings)
    if not service.available():
        raise SystemExit("模型未就绪:先跑 scripts/train_topic_classifier.py 与 "
                         "scripts/export_topic_onnx.py")

    pending = store.pending_questions(limit=args.limit or settings.topic_classify_batch_limit)
    if not pending:
        print("[done] 池子里没有待归类的问题")
        return
    classified = service.classify_batch(store, limit=len(pending))
    distribution = store.distribution()
    top = Counter({item["label"]: item["count"] for item in distribution["items"]
                   if item["count"] > 0}).most_common(5)
    print(f"[done] 本批归类 {classified} 条;池累计已归类 {distribution['classified']} 条,"
          f"待归类 {distribution['unclassified']} 条")
    print(f"[top] 当前分布 Top5:{'、'.join(f'{label}×{count}' for label, count in top)}")


if __name__ == "__main__":
    main()
