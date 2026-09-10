"""ch10 · 构建主题分类语料(数据类产出,验证方式:标注样例抽查,见 dev-notes)。

四步:①从低置信度问题池捞真实问题(LLM 纠错+预标)→ ②池量不够照术语表 LLM 造模拟问题
(生成即标注)→ ③多标签分层抽样 80/10/10 → ④同义词/句式增强只扩训练集。
另造一批 demo 问题(不进训练,验收 2 用,走同一条 LLM 预标链路)。

用法:
  .venv/bin/python -m scripts.build_topic_dataset            # 全量重建(LLM 调用,约 5 分钟)
  .venv/bin/python -m scripts.build_topic_dataset --demo-only # 只补 demo_pool(验收前刷新用)
"""
import argparse
import json
import random
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db import make_engine, make_session_factory
from app.llm import make_extract_model
from app.models import LowConfidenceQuestion
from app.topic_data import (augment_train, clean_question, stratified_multilabel_split,
                            validate_sample)
from app.topic_taxonomy import TOPIC_LABELS, TOPIC_TAXONOMY

DEFAULT_OUT_DIR = Path("tests/eval/topic_dataset")
SEED = 42

# 每类首要标签的目标条数:热门类多配,兜底「其他」略少
CLASS_TARGETS: dict[str, int] = {
    "退换货": 140, "物流": 140, "尺码": 130, "质量问题": 120, "运费": 120,
    "优惠活动": 120, "价保": 110, "发票": 120, "支付": 110, "订单修改": 110,
    "库存补货": 100, "商品信息": 120, "保修维修": 110, "账号": 100, "会员积分": 100,
    "评价": 100, "其他": 90,
}
GEN_BATCH_SIZE = 25
DEMO_COUNT = 60


class GeneratedQuestion(BaseModel):
    question: str = Field(description="用户口语原话")
    labels: list[str] = Field(description="命中的主题类目,首要类目在前")


class GeneratedBatch(BaseModel):
    items: list[GeneratedQuestion]


class CleanedLabeled(BaseModel):
    corrected_question: str = Field(description="修过错别字和格式后的问题,不改语义不加戏")
    labels: list[str] = Field(description="命中的主题类目,按字面诉求一个不多一个不少")


class BareQuestions(BaseModel):
    items: list[str] = Field(description="问题列表")


def _taxonomy_prompt_line(label: str) -> str:
    return f"{label}:{TOPIC_TAXONOMY[label]}"


def fetch_pool_questions(factory: sessionmaker) -> list[str]:
    """池里真实问题全量捞取,只做确定性清洗;纠错交给 LLM 预标一步。"""
    with factory() as session:
        rows = session.scalars(
            select(LowConfidenceQuestion.raw_question).order_by(LowConfidenceQuestion.id)
        ).all()
    return [clean_question(r or "") for r in rows if (r or "").strip()]


def correct_and_prelabel(model, raw: str, retry: int = 2) -> tuple[str, list[str]] | None:
    """池内真实问题的折中标注路线:LLM 照术语表纠错+预标,非法标签由闸丢弃。"""
    taxonomy = "\n".join(_taxonomy_prompt_line(l) for l in TOPIC_LABELS)
    prompt = (f"你是电商客服问题标注员。修正如下一句里的错别字和格式(不改语义、不加戏、不回答问题),"
              f"再按字面提到的诉求打多标签:提到几个诉求就打几个,一个不多一个不少,拿不准宁可不打。\n"
              f"类目表(含边界):\n{taxonomy}\n\n用户原话:{raw}")
    for _ in range(retry + 1):
        try:
            result = model.invoke(prompt)
            labels = [l for l in dict.fromkeys(result.labels) if l in TOPIC_LABELS]
            if result.corrected_question.strip() and labels:
                return result.corrected_question.strip(), labels
        except Exception:
            continue
    return None


def generate_class_batch(model, label: str, count: int) -> list[dict]:
    """照术语表给单个类目造模拟问题,生成即标注;首要类目不符或标签非法的直接丢。"""
    taxonomy = "\n".join(_taxonomy_prompt_line(l) for l in TOPIC_LABELS)
    others = ", ".join(l for l in TOPIC_LABELS if l != label)
    prompt = (f"你是电商平台的客服数据标注员。模拟生成 {count} 条用户向客服提问的口语原话。\n"
              f"要求:\n"
              f"1. 每条 6-40 字,用户口吻,允许少量错别字/标点随意,但诉求清晰;\n"
              f"2. 每条的首要诉求都是「{label}」。该类目边界:{TOPIC_TAXONOMY[label]}\n"
              f"3. 约 20% 的条目要自然地同时提到第二个诉求(第二个诉求从这些里选:{others}),"
              f"labels 首要类目在前;其余条目只打一个标签;\n"
              f"4. 不得出现真实手机号/邮箱/订单号,不得与彼此重复,尽量覆盖该类目不同侧面;\n"
              f"5. 全部类目表(供参考边界):\n{taxonomy}\n只返回结构化结果。")
    try:
        result = model.invoke(prompt)
    except Exception:
        return []
    out = []
    for item in result.items:
        labels = [l for l in dict.fromkeys(item.labels) if l in TOPIC_LABELS]
        question = item.question.strip()
        if not question or not labels or labels[0] != label:
            continue
        sample = {"question": question, "labels": labels, "origin": "synthetic"}
        if validate_sample(sample):
            out.append(sample)
    return out


def generate_demo_questions(model, count: int) -> list[str]:
    """造一批验收用演示问题(不带标签生成,靠预标链路打标,保证验收诚实)。"""
    per_batch = 20
    taxonomy = "\n".join(_taxonomy_prompt_line(l) for l in TOPIC_LABELS)
    questions: list[str] = []
    while len(questions) < count:
        n = min(per_batch, count - len(questions))
        prompt = (f"模拟生成 {n} 条电商平台用户向客服提问的口语原话,像真实用户随手打字:"
                  f"覆盖各种主题(退换货/物流/尺码/发票/质量/运费/优惠/价保/支付/订单修改/"
                  f"库存/商品咨询/保修/账号/积分/评价,以及套不进这些类目的杂题),"
                  f"每条 6-40 字,允许错别字,不得出现真实手机号/邮箱/订单号,彼此不重复。\n"
                  f"类目表(知道边界即可,不用你打标):\n{taxonomy}\n只返回问题列表。")
        try:
            questions.extend(q.strip() for q in model.invoke(prompt).items if q.strip())
        except Exception:
            break
    # 去重保持顺序
    return list(dict.fromkeys(questions))[:count]


def dedup_samples(samples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for item in samples:
        key = item["question"].strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 ch10 主题分类语料")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--demo-only", action="store_true", help="只重建 demo_pool.jsonl")
    parser.add_argument("--demo-count", type=int, default=DEMO_COUNT)
    args = parser.parse_args()

    settings = Settings()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_extract_model(settings).with_structured_output(
        GeneratedBatch if not args.demo_only else BareQuestions, method="function_calling")
    prelabel_model = make_extract_model(settings).with_structured_output(
        CleanedLabeled, method="function_calling")
    rng = random.Random(SEED)

    if not args.demo_only:
        # ① 池内真实问题:纠错 + 预标
        pool = fetch_pool_questions(make_session_factory(make_engine(settings)))
        pool_samples = []
        for raw in pool:
            fixed = correct_and_prelabel(prelabel_model, raw)
            if fixed:
                pool_samples.append({"question": fixed[0], "labels": fixed[1], "origin": "pool"})
        print(f"[pool] 池内 {len(pool)} 条,纠错预标成功 {len(pool_samples)} 条")

        # ② 模拟补足:逐类目分批生成
        synthetic: list[dict] = []
        for label, target in CLASS_TARGETS.items():
            need = target
            for _ in range(6):  # 每类最多 6 批,够不齐就认了
                if need <= 0:
                    break
                got = generate_class_batch(model, label, min(GEN_BATCH_SIZE, need))
                synthetic.extend(got)
                need = target - sum(1 for s in synthetic if s["labels"][0] == label)
            have = sum(1 for s in synthetic if s["labels"][0] == label)
            print(f"[gen] {label}: 目标 {target},实收 {have}")
        synthetic = dedup_samples(synthetic)

        # ③ 分层抽样 + ④ 增强(只扩训练集)
        corpus = dedup_samples(pool_samples + synthetic)
        rng.shuffle(corpus)
        train, val, test = stratified_multilabel_split(corpus, seed=SEED)
        train_aug = augment_train(train, ratio=0.2, seed=SEED)
        for split, rows in (("train", train_aug), ("val", val), ("test", test)):
            for row in rows:
                row["split"] = split
        write_jsonl(out_dir / "train.jsonl", train_aug)
        write_jsonl(out_dir / "val.jsonl", val)
        write_jsonl(out_dir / "test.jsonl", test)
        write_jsonl(out_dir / "all.jsonl", train_aug + val + test)
        stats = {
            "pool_raw": len(pool), "pool_kept": len(pool_samples),
            "synthetic": len(synthetic), "corpus": len(corpus),
            "train": len(train_aug), "val": len(val), "test": len(test),
            "augmented": len(train_aug) - len(train),
        }
        (out_dir / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] 训练语料 → {out_dir}(train={len(train_aug)} val={len(val)} test={len(test)})")

    # 验收演示问题:不进训练,单独落文件
    demo_model = model if args.demo_only else make_extract_model(settings).with_structured_output(
        BareQuestions, method="function_calling")
    demo_questions = generate_demo_questions(demo_model, args.demo_count)
    write_jsonl(out_dir / "demo_pool.jsonl",
                [{"question": q, "origin": "demo"} for q in demo_questions])
    print(f"[demo] 演示问题 {len(demo_questions)} 条 → {out_dir / 'demo_pool.jsonl'}")


if __name__ == "__main__":
    main()
