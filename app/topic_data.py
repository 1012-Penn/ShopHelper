"""ch10 · 语料纯函数层:清洗(脱敏/格式)、多标签分层抽样、数据增强(只扩训练集)。

全部为无副作用纯函数,供 scripts/build_topic_dataset.py 与旁路批量归类共用;
LLM 造数/纠错/预标在脚本层,不进这里。
"""
import random
import re

from app.topic_taxonomy import LABEL_COUNT, TOPIC_LABELS, validate_labels

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ORDER_RE = re.compile(r"(?<!\d)\d{8,}(?!\d)")
_FULLWIDTH_RE = re.compile(r"[Ａ-Ｚａ-ｚ０-９]")

# 同义词表:增强用的小词表,口语等价替换;键在句中出现才替换
SYNONYMS: dict[str, list[str]] = {
    "退货": ["退掉", "退了"],
    "退款": ["退钱", "退钱款"],
    "换货": ["换一件", "换个新的"],
    "多少钱": ["什么价格", "价格多少"],
    "快递": ["物流", "包裹"],
    "发货": ["寄出", "安排发货"],
    "尺码": ["尺寸", "码数"],
    "质量": ["品质"],
    "优惠券": ["券"],
    "开票": ["开发票"],
    "客服": ["人工客服"],
    "什么时候到": ["多久能到", "几天能到"],
}


def desensitize(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _ORDER_RE.sub("[ORDER]", text)
    return text


def _to_halfwidth(match: re.Match) -> str:
    return chr(ord(match.group(0)) - 0xFEE0)


def normalize_format(text: str) -> str:
    """全角字母数字转半角、空白折叠;相邻汉字之间的空格是排版噪声,直接去掉。"""
    text = _FULLWIDTH_RE.sub(_to_halfwidth, text)
    text = " ".join(text.split())
    return re.sub(r"(?<=[\u4e00-\u9fff]) +(?=[\u4e00-\u9fff])", "", text)


def clean_question(text: str) -> str:
    """清洗入口:先格式归一(全角数字转半角才能被手机号正则命中),再脱敏。"""
    return desensitize(normalize_format(text or ""))


def synonym_replace(text: str, rng: random.Random | None = None) -> str:
    """同义词替换:每个键只替换第一次出现,选第一个候选,保证确定性可测。"""
    out = text
    for word, alternatives in SYNONYMS.items():
        if word in out:
            choice = alternatives[0] if rng is None else rng.choice(alternatives)
            out = out.replace(word, choice, 1)
    return out


_STYLE_PREFIXES = ("请问", "你好,请问", "想问下", "麻烦问下")
_STYLE_SUFFIXES = (",谢谢", ",急", ",在线等")


def stylize(text: str, rng: random.Random | None = None) -> str:
    """句式微调:随机加礼貌前缀或急切后缀。"""
    pick = rng.choice if rng is not None else (lambda seq: seq[0])
    if rng is None or rng.random() < 0.5:
        return pick(_STYLE_PREFIXES) + text
    return text + pick(_STYLE_SUFFIXES)


def augment_once(text: str, rng: random.Random) -> str:
    return stylize(synonym_replace(text, rng), rng)


def augment_train(train_items: list[dict], ratio: float = 0.2, seed: int = 42) -> list[dict]:
    """数据增强只扩训练集:不改输入、不碰 val/test,增强样本标 origin=augmented。

    抽样按比例选 k 条(至少 1 条)确定性增强,避免随机数把小训练集整个跳过。
    """
    out = list(train_items)
    if not train_items:
        return out
    rng = random.Random(seed)
    k = max(1, round(len(train_items) * ratio))
    chosen = rng.sample(range(len(train_items)), min(k, len(train_items)))
    for idx in chosen:
        item = train_items[idx]
        augmented = augment_once(item["question"], rng)
        if augmented != item["question"]:
            out.append({"question": augmented, "labels": list(item["labels"]),
                        "origin": "augmented"})
    return out


def stratified_multilabel_split(items: list[dict], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                                seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """多标签分层抽样:按首要标签(第一个标签)分组,组内按比例切 80/10/10。

    组内不足 3 条时全部进训练集——小类目硬凑三份只会让 val/test 噪声化。
    """
    groups: dict[str, list[dict]] = {}
    for item in items:
        primary = item["labels"][0]
        groups.setdefault(primary, []).append(item)
    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for label in TOPIC_LABELS:
        group = groups.pop(label, [])
        rng.shuffle(group)
        n = len(group)
        if n < 3:
            train.extend(group)
            continue
        n_val = max(1, round(n * ratios[1]))
        n_test = max(1, round(n * ratios[2]))
        n_val, n_test = min(n_val, n - 1), min(n_test, n - 1 - n_val)
        train.extend(group[:n - n_val - n_test])
        val.extend(group[n - n_val - n_test:n - n_test])
        test.extend(group[n - n_test:])
    for leftover in groups.values():  # 未知首要标签不该出现,兜底进训练集
        train.extend(leftover)
    return train, val, test


def validate_sample(sample: dict) -> bool:
    """语料闸:问题非空、标签 1..17 个、全部合法且不重复。"""
    question = (sample.get("question") or "").strip()
    labels = sample.get("labels") or []
    if not question:
        return False
    if not 1 <= len(labels) <= LABEL_COUNT:
        return False
    if len(set(labels)) != len(labels):
        return False
    return validate_labels(labels) == list(labels)
