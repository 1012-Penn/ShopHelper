"""ch10 · 微调主题分类器的旁路批量归类服务。

ONNX Runtime 惰性加载;模型文件缺失时 available()=False,接口 503,不影响聊天主链路。
批量入口只有两个:scripts/classify_topics.py 与 POST /api/topics/classify-batch。
"""
import json
from pathlib import Path

import numpy as np

from app.config import Settings
from app.store import TopicStore
from app.topic_data import clean_question
from app.topic_taxonomy import LABEL_COUNT, TOPIC_LABELS, validate_labels


class TopicClassifierService:
    """加载 models/topic_classifier/ 下的 ONNX 模型做批量多标签推理。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session = None
        self._tokenizer = None
        self._threshold = settings.topic_threshold
        self._max_len = settings.topic_max_len

    def _model_dir(self) -> Path:
        return Path(self._settings.topic_model_dir)

    def available(self) -> bool:
        directory = self._model_dir()
        return (directory / "model.onnx").is_file() and (directory / "label_config.json").is_file()

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        if not self.available():
            raise FileNotFoundError(
                f"主题分类模型未就绪:先跑 scripts/train_topic_classifier.py 与 "
                f"scripts/export_topic_onnx.py,期望目录 {self._model_dir()}")
        import onnxruntime as ort
        from transformers import AutoTokenizer

        config = json.loads((self._model_dir() / "label_config.json").read_text(encoding="utf-8"))
        if list(config["labels"]) != list(TOPIC_LABELS):
            raise ValueError("模型标签序与权威术语表不一致,拒绝服务")
        self._threshold = float(config.get("threshold", self._threshold))
        self._max_len = int(config.get("max_len", self._max_len))
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir()))
        self._session = ort.InferenceSession(str(self._model_dir() / "model.onnx"))

    def predict(self, questions: list[str]) -> list[list[str]]:
        """批量多标签推理:清洗 → 编码 → sigmoid → 阈值;全类低于阈值回落 Top-1。"""
        if not questions:
            return []
        self._ensure_loaded()
        cleaned = [clean_question(q) for q in questions]
        encodings = self._tokenizer(cleaned, truncation=True, max_length=self._max_len)
        batch = self._tokenizer.pad(encodings, padding=True, return_tensors="np")
        feed = {name: batch[name] for name in (i.name for i in self._session.get_inputs())}
        logits = self._session.run(None, feed)[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        results = []
        for row in probs:
            labels = [label for i, label in enumerate(TOPIC_LABELS) if row[i] >= self._threshold]
            if not labels:  # 一条问题至少归一类,否则后台分布少一块拼图
                labels = [TOPIC_LABELS[int(np.argmax(row))]]
            results.append(validate_labels(labels))
        return results

    def classify_batch(self, store: TopicStore, limit: int) -> int:
        """攒够一批归一次:捞未归类 → 批量推理 → upsert 落 topic_classifications。"""
        pending = store.pending_questions(limit=limit)
        if not pending:
            return 0
        predictions = self.predict([row["question"] for row in pending])
        results = [(row["id"], labels) for row, labels in zip(pending, predictions)]
        return store.save_results(results)


class StubTopicClassifier(TopicClassifierService):
    """测试替身:按文本→标签映射直接返回,绕过 ONNX;其余行为与真服务一致。"""

    def __init__(self, mapping: dict[str, list[str]], top1_fallback: bool = False) -> None:
        super().__init__(Settings(topic_model_dir="models/stub"))
        self._mapping = mapping
        self.top1_fallback = top1_fallback

    def available(self) -> bool:
        return True

    def predict(self, questions: list[str]) -> list[list[str]]:
        results = []
        for question in questions:
            cleaned = clean_question(question)
            labels = validate_labels(self._mapping.get(cleaned, []))
            if not labels and self.top1_fallback:
                labels = ["其他"]
            results.append(labels)
        return results
