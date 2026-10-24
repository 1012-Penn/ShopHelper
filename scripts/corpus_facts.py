"""结构化事实与证据引用:模型、加载与引用完整性校验。

真源:`corpus/sources.yaml`(来源 + 证据单元)与 `corpus/facts/*.yaml`(事实)。
文档真源是 `knowledge/*.md`,必须先在 `corpus/manifest.yaml` 登记;quote 必须逐字
出现在文档中,且同一证据单元的全部 quote 要落在同一个 `chunk_markdown` 块里。
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.chunking import Chunk, chunk_markdown
from scripts.corpus_manifest import (
    DOCS_DIR,
    CorpusManifest,
    doc_types,
    load_manifest,
    registered_docs,
)
from scripts.eval_metrics import normalize_span

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"

FactValue: TypeAlias = int | float | str | bool | list[str]
EntityType = Literal["sku", "order", "policy", "promotion", "after_sales", "region"]


class CorpusDataError(ValueError):
    """语料文件缺失、YAML 非法、字段不合 schema 或 id 重复。"""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceUnit(_StrictModel):
    evidence_id: str = Field(pattern=r"^EV-[A-Z0-9-]+$")
    fact_ids: list[str] = Field(min_length=1)
    doc: str
    section: str
    quotes: list[str] = Field(min_length=1)


class Fact(_StrictModel):
    fact_id: str = Field(pattern=r"^[A-Z]+-[A-Z0-9-]+$")
    entity_type: EntityType
    entity_id: str
    key: str
    value: FactValue
    unit: str = ""
    conditions: list[str] = Field(default_factory=list)
    effective_from: date | None = None
    effective_to: date | None = None
    supersedes: list[str] = Field(default_factory=list)
    source_ref: str
    evidence_ids: list[str] = Field(min_length=1)
    conflict_group: str = ""
    notes: str = ""


class Source(_StrictModel):
    source_id: str
    type: Literal["internal_rule", "public_reference"]
    title: str
    rule_no: str = ""
    uri: str = ""
    retrieved_at: date | None = None
    license: str
    used_for: Literal["fact", "wording_only"] = "fact"
    author: str = ""
    reviewer: str = ""
    confidence: Literal["high", "medium", "low"] = "high"


@dataclass(frozen=True)
class FactBook:
    facts: dict[str, Fact]
    evidence: dict[str, EvidenceUnit]
    sources: dict[str, Source]

    def fact(self, fact_id: str) -> Fact:
        return self.facts[fact_id]

    def facts_for_entity(self, entity_type: str, entity_id: str) -> list[Fact]:
        return [
            self.facts[fact_id]
            for fact_id in sorted(self.facts)
            if self.facts[fact_id].entity_type == entity_type
            and self.facts[fact_id].entity_id == entity_id
        ]

    def evidence_for_fact(self, fact_id: str) -> list[EvidenceUnit]:
        return [
            self.evidence[evidence_id]
            for evidence_id in sorted(self.evidence)
            if fact_id in self.evidence[evidence_id].fact_ids
        ]

    def sha256(self) -> str:
        payload = {
            "sources": [self.sources[key].model_dump(mode="json") for key in sorted(self.sources)],
            "facts": [self.facts[key].model_dump(mode="json") for key in sorted(self.facts)],
            "evidence": [self.evidence[key].model_dump(mode="json") for key in sorted(self.evidence)],
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def count(self) -> int:
        return len(self.facts)


def _read_yaml(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CorpusDataError(f"无法读取语料文件 {path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise CorpusDataError(f"{path.name} 不是合法的 YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise CorpusDataError(f"{path.name} 顶层必须是映射")
    return data


def _list_field(data: dict, key: str, path: Path) -> list:
    value = data.get(key)
    if not isinstance(value, list):
        raise CorpusDataError(f"{path.name} 顶层缺少 {key} 列表")
    return value


def _parse_models(model: type[BaseModel], items: list, path: Path) -> list:
    parsed = []
    for index, item in enumerate(items):
        try:
            parsed.append(model.model_validate(item))
        except ValidationError as exc:
            raise CorpusDataError(
                f"{path.name} 第 {index + 1} 条 {model.__name__} 字段校验失败: {exc}"
            ) from exc
    return parsed


def _index_unique(items: list, attr: str, kind: str) -> dict:
    indexed = {}
    for item in items:
        key = getattr(item, attr)
        if key in indexed:
            raise CorpusDataError(f"{kind} id 重复: {key}")
        indexed[key] = item
    return indexed


def load_factbook(root: Path = CORPUS_DIR) -> FactBook:
    root = Path(root)
    sources_path = root / "sources.yaml"
    facts_dir = root / "facts"
    if not sources_path.is_file():
        raise CorpusDataError(f"语料来源文件不存在: {sources_path}")
    if not facts_dir.is_dir():
        raise CorpusDataError(f"事实目录不存在: {facts_dir}")

    sources_data = _read_yaml(sources_path)
    sources = _parse_models(
        Source, _list_field(sources_data, "sources", sources_path), sources_path
    )
    evidence = _parse_models(
        EvidenceUnit,
        _list_field(sources_data, "evidence_units", sources_path),
        sources_path,
    )

    facts: list[Fact] = []
    for path in sorted(facts_dir.glob("*.yaml")):
        data = _read_yaml(path)
        facts.extend(_parse_models(Fact, _list_field(data, "facts", path), path))

    return FactBook(
        facts=_index_unique(facts, "fact_id", "fact"),
        evidence=_index_unique(evidence, "evidence_id", "evidence"),
        sources=_index_unique(sources, "source_id", "source"),
    )


def _value_key(value: FactValue) -> tuple:
    """值比较口径:整数与等值浮点视为同一数值;list 逐项比较。"""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("number", float(value))
    if isinstance(value, list):
        return ("list", tuple(value))
    return ("str", value)


def _ranges_overlap(left: Fact, right: Fact) -> bool:
    """缺失的起止边界按 date.min/date.max 视为无界,再按闭区间判断重叠。"""
    left_from = left.effective_from if left.effective_from is not None else date.min
    left_to = left.effective_to if left.effective_to is not None else date.max
    right_from = right.effective_from if right.effective_from is not None else date.min
    right_to = right.effective_to if right.effective_to is not None else date.max
    return left_from <= right_to and right_from <= left_to


def _supersedes_cycles(facts: dict[str, Fact]) -> list[str]:
    cycles: list[str] = []
    color: dict[str, int] = {}
    stack: list[str] = []

    def visit(fact_id: str) -> None:
        color[fact_id] = 1
        stack.append(fact_id)
        for target in facts[fact_id].supersedes:
            if target not in facts:
                continue
            if color.get(target) == 1:
                start = stack.index(target)
                cycles.append(" -> ".join([*stack[start:], target]))
            elif color.get(target, 0) == 0:
                visit(target)
        stack.pop()
        color[fact_id] = 2

    for fact_id in sorted(facts):
        if color.get(fact_id, 0) == 0:
            visit(fact_id)
    return cycles


def validate_factbook(
    book: FactBook, docs_dir: Path = DOCS_DIR, *, manifest: CorpusManifest | None = None
) -> list[str]:
    """返回人类可读的问题列表;空列表表示通过。

    事实/证据类校验问题一律以字符串返回,不抛校验异常;若 manifest 自身缺失或非法,
    ``load_manifest()`` 仍会抛 ``CorpusManifestError``(那是语料注册表损坏,不是事实问题)。
    """
    docs_dir = Path(docs_dir)
    manifest = manifest or load_manifest()
    types = doc_types(manifest)
    registered = set(registered_docs(manifest))
    issues: list[str] = []

    for fact_id in sorted(book.facts):
        fact = book.facts[fact_id]
        if fact.source_ref not in book.sources:
            issues.append(f"fact {fact_id} 的 source_ref 未引用任何 source: {fact.source_ref}")
        for evidence_id in fact.evidence_ids:
            if evidence_id not in book.evidence:
                issues.append(f"fact {fact_id} 引用了不存在的 evidence: {evidence_id}")
            elif fact_id not in book.evidence[evidence_id].fact_ids:
                issues.append(
                    f"fact {fact_id} 与 evidence {evidence_id} 引用不互惠:"
                    f" {evidence_id}.fact_ids 未包含该 fact"
                )
        if (
            fact.effective_from is not None
            and fact.effective_to is not None
            and fact.effective_to < fact.effective_from
        ):
            issues.append(
                f"fact {fact_id} 的 effective_to 早于 effective_from:"
                f" {fact.effective_to} < {fact.effective_from}"
            )
        for target in fact.supersedes:
            if target not in book.facts:
                issues.append(f"fact {fact_id} 的 supersedes 目标不存在: {target}")

    for cycle in _supersedes_cycles(book.facts):
        issues.append(f"supersedes 存在环: {cycle}")

    chunk_cache: dict[str, list[Chunk]] = {}
    for evidence_id in sorted(book.evidence):
        unit = book.evidence[evidence_id]
        for fact_id in unit.fact_ids:
            if fact_id not in book.facts:
                issues.append(f"evidence {evidence_id} 的 fact_id 不存在: {fact_id}")
            elif evidence_id not in book.facts[fact_id].evidence_ids:
                issues.append(
                    f"evidence {evidence_id} 与 fact {fact_id} 引用不互惠:"
                    f" fact.evidence_ids 未回引该 evidence"
                )

        if unit.doc not in registered:
            issues.append(f"evidence {evidence_id} 的 doc 未在 manifest 登记: {unit.doc}")
            continue
        doc_path = docs_dir / unit.doc
        if not doc_path.is_file():
            issues.append(f"evidence {evidence_id} 的 doc 不存在于 {docs_dir}: {unit.doc}")
            continue
        try:
            text = doc_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            issues.append(f"evidence {evidence_id} 的 doc 无法读取: {unit.doc} ({exc})")
            continue

        normalized_text = normalize_span(text)
        if normalize_span(unit.section) not in normalized_text:
            issues.append(
                f"evidence {evidence_id} 的 section 未出现在文档 {unit.doc} 中: {unit.section}"
            )

        quotes_ok = True
        for quote in unit.quotes:
            if not quote.strip() or normalize_span(quote) not in normalized_text:
                issues.append(
                    f"evidence {evidence_id} 的 quote 未逐字出现在 {unit.doc} 中: {quote!r}"
                )
                quotes_ok = False
        if not quotes_ok:
            continue

        chunks = chunk_cache.get(unit.doc)
        if chunks is None:
            chunks = chunk_markdown(text, unit.doc, types[unit.doc])
            chunk_cache[unit.doc] = chunks
        normalized_quotes = [normalize_span(quote) for quote in unit.quotes]
        covering = [
            chunk
            for chunk in chunks
            if all(quote in normalize_span(chunk.answer) for quote in normalized_quotes)
        ]
        if not covering:
            issues.append(
                f"evidence {evidence_id} 的 quotes 无法落在 {unit.doc} 的同一 chunk 中"
            )
        elif not any(
            normalize_span(unit.section) in normalize_span(chunk.section_path)
            for chunk in covering
        ):
            issues.append(
                f"evidence {evidence_id} 的 section 不在覆盖全部 quotes 的 chunk 的"
                f" section_path 中: {unit.section}"
            )

    grouped: dict[tuple[str, str, str], list[Fact]] = defaultdict(list)
    for fact_id in sorted(book.facts):
        fact = book.facts[fact_id]
        grouped[(fact.entity_type, fact.entity_id, fact.key)].append(fact)
    for (entity_type, entity_id, key), items in sorted(grouped.items()):
        for left_index, left in enumerate(items):
            for right in items[left_index + 1 :]:
                if _value_key(left.value) == _value_key(right.value):
                    continue
                if not _ranges_overlap(left, right):
                    continue
                if left.conflict_group and left.conflict_group == right.conflict_group:
                    continue
                issues.append(
                    f"{entity_type}/{entity_id}/{key} 的值冲突且生效区间重叠:"
                    f" {left.fact_id} vs {right.fact_id}"
                )

    return issues
