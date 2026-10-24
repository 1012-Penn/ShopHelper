"""严格证据判定纯函数：评估脚本与证据闸共用，不依赖数据库/模型。

口径：
- 一组 evidence = (doc, section?, quotes)；quotes 必须同时出现在同一个 chunk；
- 多组 evidence 之间是 AND，coverage_rank 取各组首次命中名次的最大值；
- 不再使用「文档名 + 任一关键词」的宽松判定。
"""
import unicodedata
from dataclasses import dataclass


class EvalDataError(ValueError):
    """评估集/语料漂移：某组 evidence 找不到任何 chunk。"""


@dataclass(frozen=True)
class EvidenceNeed:
    doc: str
    quotes: tuple[str, ...]
    section: str | None = None


def normalize_span(text: str) -> str:
    """NFKC + 连续空白折叠；不改标点、不做同义词替换。"""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def needs_from_case(case: dict) -> list[EvidenceNeed]:
    return [
        EvidenceNeed(doc=e["doc"], quotes=tuple(e["quotes"]), section=e.get("section"))
        for e in case["required_evidence"]
    ]


def chunk_covers(chunk, need: EvidenceNeed) -> bool:
    if not chunk.section_path.startswith(need.doc + " >"):
        return False
    if need.section and need.section not in chunk.section_path:
        return False
    blob = normalize_span(chunk.questions + "\n" + chunk.answer)
    return all(normalize_span(quote) in blob for quote in need.quotes)


def resolve_gold_chunks(case: dict, rows_by_id: dict) -> list[frozenset[int]]:
    """每组 evidence 对应的 gold chunk 集合；任一组为空 → EvalDataError。"""
    gold: list[frozenset[int]] = []
    for need in needs_from_case(case):
        ids = frozenset(
            cid for cid, row in rows_by_id.items() if chunk_covers(row, need)
        )
        if not ids:
            raise EvalDataError(
                f"{case.get('id')}: evidence 无法映射到任何 chunk "
                f"(doc={need.doc}, section={need.section}, quotes={need.quotes})"
            )
        gold.append(ids)
    return gold


def first_rank(ranked_ids: list[int], gold: frozenset[int], cap: int = 10) -> int | None:
    for pos, cid in enumerate(ranked_ids[:cap], start=1):
        if cid in gold:
            return pos
    return None


def coverage_rank(ranked_ids: list[int], gold_sets: list[frozenset[int]],
                  cap: int = 10) -> int | None:
    """所有 evidence 组首次命中的最大名次；任一组未命中 → None。"""
    ranks = [first_rank(ranked_ids, gold, cap) for gold in gold_sets]
    if any(rank is None for rank in ranks):
        return None
    return max(ranks) if ranks else None


def single_chunk_rank(ranked_ids: list[int], case: dict, rows_by_id: dict,
                      cap: int = 10) -> int | None:
    """Top-K 内第一个同时覆盖全部 evidence 组的 chunk 名次。"""
    needs = needs_from_case(case)
    for pos, cid in enumerate(ranked_ids[:cap], start=1):
        row = rows_by_id.get(cid)
        if row is not None and all(chunk_covers(row, need) for need in needs):
            return pos
    return None


def evidence_group_recall(ranked_ids: list[int], gold_sets: list[frozenset[int]],
                          cap: int = 10) -> float:
    """命中的 evidence 组数 / 总组数。"""
    if not gold_sets:
        return 0.0
    hits = sum(first_rank(ranked_ids, gold, cap) is not None for gold in gold_sets)
    return hits / len(gold_sets)


def group_recall_at(ranked_ids: list[int], gold_sets: list[frozenset[int]],
                    k: int = 10) -> float:
    """evidence_group_recall 的显式 Top-K 别名。"""
    return evidence_group_recall(ranked_ids, gold_sets, cap=k)


def mrr_at(ranked_ids: list[int], gold_sets: list[frozenset[int]],
           k: int = 10) -> float | None:
    """首次覆盖全部 evidence 的名次倒数;未命中返回 None。"""
    rank = coverage_rank(ranked_ids, gold_sets, cap=k)
    return None if rank is None else 1.0 / rank


def evidence_precision(ranked_ids: list[int], support_ids: set[int],
                       cap: int = 10) -> float | None:
    """Top-K 中属于支持 chunk 的比例;Top-K 为空返回 None。"""
    top = list(ranked_ids[:cap])
    if not top:
        return None
    return sum(1 for cid in top if cid in support_ids) / len(top)


def hard_negative_hit_rate(ranked_ids: list[int], k: int = 10) -> float:
    """负例题是否检索到任何 chunk;单题返回 0/1,聚合时取平均。"""
    return 1.0 if ranked_ids[:k] else 0.0


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def macro_average(by_bucket: dict[str, float | None]) -> float | None:
    """只对非 None 的桶取宏平均;无可用桶返回 None。"""
    values = [value for value in by_bucket.values() if value is not None]
    return safe_div(sum(values), float(len(values))) if values else None
