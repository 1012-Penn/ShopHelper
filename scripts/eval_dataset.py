"""评测数据集 v2:严格 schema、split registry、overlay、稳定哈希与校验。

设计口径:
- 真源是 `tests/eval/splits/registry.json`;旧 ch04/ch09 文件一个字节都不改。
- `regression-v1` 是 composed split:加载旧 JSONL 后补 v2 默认字段。
- 稳定哈希排除标注/复核元数据,review-only 编辑不改变数据集哈希。
- 数据问题统一抛 `EvalDataError`(ValueError 子类),一次报告全部问题。
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.chunking import Chunk, chunk_markdown
from scripts.corpus_facts import FactBook, FactValue
from scripts.corpus_manifest import (
    DOCS_DIR,
    doc_types,
    load_manifest,
    registered_docs,
)
from scripts.eval_metrics import EvalDataError, normalize_span

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_REL = Path("tests") / "eval" / "splits" / "registry.json"

SCHEMA_VERSION = 2
BUCKETS = (
    "single_hop",
    "multi_hop",
    "multi_constraint",
    "entity_mismatch",
    "policy_conflict",
    "hard_negative",
    "clarify",
    "ood",
)
EXPECTED_FROZEN = {
    "single_hop": 18,
    "multi_hop": 16,
    "multi_constraint": 14,
    "entity_mismatch": 10,
    "policy_conflict": 8,
    "hard_negative": 10,
    "clarify": 8,
    "ood": 6,
}
NEW_SPLITS = ("dev", "frozen-v1")
ANSWER_BEHAVIORS = ("answer", "answer_with_conditions", "verify_then_answer")
REVIEW_METADATA = ("annotator", "reviewer", "review_status")


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc: str = Field(min_length=1)
    section: str | None = None
    quotes: list[str] = Field(min_length=1)
    fact_ids: list[str] = Field(default_factory=list)


class GoldFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(min_length=1)
    key: str = ""
    value: FactValue
    unit: str = ""
    conditions: list[str] = Field(default_factory=list)
    tolerance: float = 0.0
    effective_from: date | None = None
    effective_to: date | None = None


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class ToolFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict = Field(default_factory=dict)
    result: dict


class EvalCase(BaseModel):
    # 旧回归题带有 expect_doc / topical_docs 等字段,必须原样保留,因此允许额外字段。
    model_config = ConfigDict(extra="allow")

    schema_version: Literal[2] = 2
    id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    bucket: str = Field(min_length=1)
    label_version: str = "v2"
    query: str = Field(min_length=1)
    conversation_context: list[ConversationTurn] = Field(default_factory=list)
    expected_behavior: Literal[
        "answer", "answer_with_conditions", "verify_then_answer", "clarify", "refuse"
    ]
    answerable: bool
    evidence_state: Literal["sufficient", "partial", "insufficient", "conflicting"]
    required_slots: list[str] = Field(default_factory=list)
    clarifying_slots: list[str] = Field(default_factory=list)
    gold_facts: list[GoldFact] = Field(default_factory=list)
    required_evidence: list[EvidenceRef] = Field(default_factory=list)
    required_conditions: list[str] = Field(default_factory=list)
    condition_aliases: dict[str, list[str]] = Field(default_factory=dict)
    slot_aliases: dict[str, list[str]] = Field(default_factory=dict)
    verification_markers: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    expected_refusal_reason: str = ""
    as_of_date: date | None = None
    requires_tools: list[str] = Field(default_factory=list)
    tool_fixtures: list[ToolFixture] = Field(default_factory=list)
    source: str = "synthetic"
    source_ref: str = ""
    annotator: str = ""
    reviewer: str = ""
    review_status: Literal["draft", "reviewed", "frozen"] = "draft"
    notes: str = ""


def _resolve(root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else Path(root) / path


def load_registry(root: Path = ROOT) -> dict:
    path = _resolve(Path(root), REGISTRY_REL)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalDataError(f"无法读取 split registry {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvalDataError(f"split registry 不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvalDataError(f"split registry 顶层必须是对象: {path}")
    return data


def split_entries(registry: dict) -> dict:
    """兼容 `{"splits": {...}}` 包装与直接映射两种 registry 形态。"""
    wrapped = registry.get("splits")
    if isinstance(wrapped, dict):
        return wrapped
    return {key: value for key, value in registry.items() if isinstance(value, dict)}


def load_cases_from_path(path: Path) -> list[dict]:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalDataError(f"无法读取评测文件 {path}: {exc}") from exc
    cases: list[dict] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalDataError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise EvalDataError(f"{path}:{lineno} 必须是 JSON 对象")
        cases.append(item)
    return cases


def load_overlay(path: Path) -> dict:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalDataError(f"无法读取 overlay {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvalDataError(f"overlay 不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvalDataError(f"overlay 顶层必须是对象: {path}")
    return data


def apply_overlay(cases: list[dict], overlay: dict | None) -> list[dict]:
    """深拷贝后合并:defaults 只补缺字段,per-case 条目优先级最高。"""
    merged_cases = copy.deepcopy(cases)
    if overlay is None:
        return merged_cases
    if not isinstance(overlay, dict):
        raise EvalDataError("overlay 必须是对象")
    unknown_top = sorted(set(overlay) - {"defaults", "cases"})
    if unknown_top:
        raise EvalDataError(f"overlay 含未知顶层字段: {', '.join(unknown_top)}")
    defaults = overlay.get("defaults") or {}
    entries = overlay.get("cases") or {}
    if not isinstance(defaults, dict):
        raise EvalDataError("overlay.defaults 必须是对象")
    if not isinstance(entries, dict):
        raise EvalDataError("overlay.cases 必须是对象")

    known_ids = {str(case.get("id", "")) for case in merged_cases}
    unknown_ids = sorted(set(entries) - known_ids)
    if unknown_ids:
        raise EvalDataError(f"overlay 引用了不存在的用例 id: {', '.join(unknown_ids)}")

    result: list[dict] = []
    for case in merged_cases:
        merged = copy.deepcopy(case)
        for key, value in copy.deepcopy(defaults).items():
            merged.setdefault(key, value)
        entry = entries.get(str(case.get("id", "")))
        if entry is not None:
            if not isinstance(entry, dict):
                raise EvalDataError(f"overlay 用例条目必须是对象: {case.get('id')}")
            merged.update(copy.deepcopy(entry))
        result.append(merged)
    return result


def _default_case(case: dict, *, split: str, label_version: str) -> dict:
    result = copy.deepcopy(case)
    result.setdefault("schema_version", SCHEMA_VERSION)
    result.setdefault("split", split)
    result.setdefault("label_version", label_version)
    answerable = bool(result.get("answerable", False))
    result.setdefault("expected_behavior", "answer" if answerable else "refuse")
    result.setdefault("evidence_state", "sufficient" if answerable else "insufficient")
    result.setdefault("gold_facts", [])
    if not answerable:
        if not str(result.get("expected_refusal_reason") or "").strip():
            result["expected_refusal_reason"] = (
                str(result.get("unanswerable_because") or "").strip()
                or str(result.get("notes") or "").strip()
                or "知识库无依据"
            )
    else:
        result.setdefault("expected_refusal_reason", "")
    return result


def load_split(split: str, root: Path = ROOT) -> list[dict]:
    root = Path(root)
    entries = split_entries(load_registry(root))
    if split not in entries:
        raise EvalDataError(f"split 未注册: {split}")
    entry = entries[split]
    if not isinstance(entry, dict):
        raise EvalDataError(f"split 条目必须是对象: {split}")
    label_version = str(entry.get("label_version") or "v2")
    kind = entry.get("kind")

    cases: list[dict] = []
    if kind == "composed":
        for item in entry.get("cases") or []:
            if not isinstance(item, dict) or "path" not in item:
                raise EvalDataError(f"composed split {split} 的 cases 条目缺少 path")
            overlay_path = item.get("overlay")
            overlay = load_overlay(_resolve(root, overlay_path)) if overlay_path else None
            cases.extend(apply_overlay(load_cases_from_path(_resolve(root, item["path"])), overlay))
    elif kind == "file":
        if not entry.get("path"):
            raise EvalDataError(f"file split {split} 缺少 path")
        overlay_path = entry.get("overlay")
        overlay = load_overlay(_resolve(root, overlay_path)) if overlay_path else None
        cases = apply_overlay(load_cases_from_path(_resolve(root, entry["path"])), overlay)
    else:
        raise EvalDataError(f"split {split} 的 kind 非法: {kind!r}")

    cases = [_default_case(case, split=split, label_version=label_version) for case in cases]
    expected_n = entry.get("n")
    if expected_n is not None and len(cases) != expected_n:
        raise EvalDataError(
            f"split {split} 题数不符: registry 声明 {expected_n}, 实际 {len(cases)}"
        )
    return cases


def _hashable_case(case: dict) -> dict:
    return {
        key: value for key, value in case.items() if key not in REVIEW_METADATA
    }


def split_sha256(cases: list[dict]) -> str:
    """按 id 排序后的 canonical JSON 哈希;排除标注/复核元数据。"""
    payload = [_hashable_case(case) for case in cases]
    payload.sort(key=lambda case: str(case.get("id", "")))
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataset_sha256(path: Path) -> str:
    """原始文件字节哈希(不解析、不规范化)。"""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvalDataError(f"无法读取评测文件 {path}: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def bucket_counts(cases: list[dict]) -> dict[str, int]:
    return dict(Counter(str(case.get("bucket", "")) for case in cases))


def behavior_counts(cases: list[dict]) -> dict[str, int]:
    return dict(Counter(str(case.get("expected_behavior", "")) for case in cases))


def cross_split_issues(cases_by_split: dict[str, list[dict]]) -> list[str]:
    """dev 与 frozen-v1 同时存在时,id 不得冲突(题面/证据可复用,编号必须唯一)。"""
    issues: list[str] = []
    if "dev" in cases_by_split and "frozen-v1" in cases_by_split:
        dev_ids = {str(case.get("id", "")) for case in cases_by_split["dev"]}
        frozen_ids = {str(case.get("id", "")) for case in cases_by_split["frozen-v1"]}
        duplicated = sorted(dev_ids & frozen_ids)
        if duplicated:
            issues.append(f"dev/frozen-v1 id 冲突: {', '.join(duplicated)}")
        dev_queries = {
            str(case.get("query", "")).strip()
            for case in cases_by_split["dev"] if case.get("query")
        }
        frozen_queries = {
            str(case.get("query", "")).strip()
            for case in cases_by_split["frozen-v1"] if case.get("query")
        }
        duplicate_queries = sorted(dev_queries & frozen_queries)
        if duplicate_queries:
            issues.append(f"dev/frozen-v1 query 冲突: {', '.join(duplicate_queries)}")
    return issues


def _as_date(value) -> date | None:
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _value_matches(gold_value, fact_value, tolerance: float) -> bool:
    if isinstance(gold_value, bool) or isinstance(fact_value, bool):
        return gold_value is fact_value
    if isinstance(gold_value, (int, float)) and isinstance(fact_value, (int, float)):
        return abs(float(gold_value) - float(fact_value)) <= float(tolerance)
    if isinstance(gold_value, list) and isinstance(fact_value, list):
        if len(gold_value) != len(fact_value):
            return False
        return all(
            _value_matches(left, right, tolerance)
            for left, right in zip(gold_value, fact_value)
        )
    return gold_value == fact_value


def _schema_issues(case: dict, position: int) -> list[str]:
    try:
        EvalCase.model_validate(case)
    except ValidationError as exc:
        details = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
            details.append(f"{location}: {error.get('msg')}")
        case_id = case.get("id") or f"#{position}"
        return [f"{case_id}: schema 校验失败: {'; '.join(details)}"]
    return []


def _evidence_issues(
    case: dict,
    *,
    registered: set[str],
    types: dict[str, str],
    docs_dir: Path,
    doc_cache: dict[str, tuple[str, list[Chunk]] | None],
) -> list[str]:
    case_id = case.get("id") or "<unknown>"
    issues: list[str] = []
    for evidence in case.get("required_evidence") or []:
        doc = evidence.get("doc")
        section = evidence.get("section")
        quotes = evidence.get("quotes") or []
        if doc not in registered:
            issues.append(f"{case_id}: required_evidence doc 未登记: {doc}")
            continue
        if not quotes:
            issues.append(f"{case_id}: required_evidence 缺少 quotes: {doc}")
            continue
        if doc not in doc_cache:
            try:
                text = (docs_dir / doc).read_text(encoding="utf-8")
                doc_cache[doc] = (text, chunk_markdown(text, doc, types[doc]))
            except (OSError, UnicodeDecodeError) as exc:
                issues.append(f"{case_id}: 无法读取文档 {doc}: {exc}")
                doc_cache[doc] = None
        cached = doc_cache[doc]
        if cached is None:
            continue
        text, chunks = cached
        normalized_text = normalize_span(text)
        if section and normalize_span(str(section)) not in normalized_text:
            issues.append(f"{case_id}: section 未出现在 {doc} 中: {section}")

        quotes_ok = True
        for quote in quotes:
            if not str(quote).strip() or normalize_span(str(quote)) not in normalized_text:
                issues.append(f"{case_id}: quote 未逐字出现在 {doc} 中: {quote!r}")
                quotes_ok = False
        if not quotes_ok:
            continue

        normalized_quotes = [normalize_span(str(quote)) for quote in quotes]
        covering = [
            chunk
            for chunk in chunks
            if all(
                quote in normalize_span(chunk.questions + "\n" + chunk.answer)
                for quote in normalized_quotes
            )
        ]
        if not covering:
            issues.append(
                f"{case_id}: required_evidence 无法落在 {doc} 的同一 chunk 中: {quotes}"
            )
        elif section and not any(
            normalize_span(str(section)) in normalize_span(chunk.section_path)
            for chunk in covering
        ):
            issues.append(
                f"{case_id}: section 不在覆盖全部 quotes 的 chunk 的 section_path 中:"
                f" {doc} / {section}"
            )
    return issues


def _fact_issues(case: dict, factbook: FactBook) -> list[str]:
    case_id = case.get("id") or "<unknown>"
    issues: list[str] = []
    as_of_date = _as_date(case.get("as_of_date"))
    for gold in case.get("gold_facts") or []:
        fact_id = gold.get("fact_id")
        if fact_id not in factbook.facts:
            issues.append(f"{case_id}: gold_fact 不存在于 FactBook: {fact_id}")
            continue
        fact = factbook.fact(fact_id)
        key = gold.get("key") or ""
        if key and key != fact.key:
            issues.append(
                f"{case_id}: gold_fact {fact_id} 的 key 与 FactBook 不一致:"
                f" {key!r} != {fact.key!r}"
            )
        if not _value_matches(gold.get("value"), fact.value, float(gold.get("tolerance") or 0.0)):
            issues.append(
                f"{case_id}: gold_fact {fact_id} 的 value 与 FactBook 不一致:"
                f" {gold.get('value')!r} != {fact.value!r}"
            )
        unit = gold.get("unit") or ""
        if unit and unit != fact.unit:
            issues.append(
                f"{case_id}: gold_fact {fact_id} 的 unit 与 FactBook 不一致:"
                f" {unit!r} != {fact.unit!r}"
            )
        if as_of_date is not None and (
            fact.effective_from is not None or fact.effective_to is not None
        ):
            lower = fact.effective_from or date.min
            upper = fact.effective_to or date.max
            if not lower <= as_of_date <= upper:
                issues.append(
                    f"{case_id}: as_of_date {as_of_date} 不在 gold_fact {fact_id}"
                    f" 的生效区间 [{lower}, {upper}] 内"
                )
    for evidence in case.get("required_evidence") or []:
        for fact_id in evidence.get("fact_ids") or []:
            if fact_id not in factbook.facts:
                issues.append(
                    f"{case_id}: required_evidence 引用了不存在的 fact: {fact_id}"
                )
    return issues


def validate_cases(
    cases: list[dict],
    *,
    split: str,
    bucket_counts: dict[str, int] | None = None,
    docs_dir: Path | None = None,
    factbook: FactBook | None = None,
) -> None:
    """校验一个 split 的全部用例;有问题时把全部问题用换行拼接后抛出。"""
    issues: list[str] = []
    schema_failed: set[int] = set()

    seen_ids: dict[str, int] = {}
    seen_queries: dict[str, int] = {}
    for position, case in enumerate(cases, start=1):
        schema_issues = _schema_issues(case, position)
        if schema_issues:
            # schema 不合法时后续语义检查没有可靠输入,只报告 schema 问题。
            issues.extend(schema_issues)
            schema_failed.add(position)
            continue
        case_id = str(case.get("id") or f"#{position}")

        if case_id in seen_ids:
            issues.append(f"id 重复: {case_id}")
        else:
            seen_ids[case_id] = position
        query = str(case.get("query") or "").strip()
        if query:
            if query in seen_queries:
                issues.append(f"query 重复: {query}")
            else:
                seen_queries[query] = position

        bucket = case.get("bucket")
        behavior = case.get("expected_behavior")
        answerable = case.get("answerable")
        state = case.get("evidence_state")

        if split in NEW_SPLITS and bucket not in BUCKETS:
            issues.append(f"{case_id}: bucket 不在 v2 分类中: {bucket!r}")
        if split in NEW_SPLITS:
            unknown = sorted(set(case) - set(EvalCase.model_fields))
            if unknown:
                issues.append(f"{case_id}: 新 split 含未知字段: {', '.join(unknown)}")
        if split in NEW_SPLITS and behavior in ANSWER_BEHAVIORS and not (
            case.get("gold_facts") or []
        ):
            issues.append(f"{case_id}: 新 split 的 answer* 题缺少 gold_facts")
        if split in NEW_SPLITS and behavior in ANSWER_BEHAVIORS and not (
            case.get("required_evidence") or []
        ):
            issues.append(f"{case_id}: 新 split 的 answer* 题缺少 required_evidence")
        if behavior != "refuse" and answerable is not True:
            issues.append(f"{case_id}: 非 refuse 题必须 answerable=true")

        if behavior == "refuse":
            if answerable is not False:
                issues.append(f"{case_id}: refuse 必须 answerable=false")
            if not str(case.get("expected_refusal_reason") or "").strip():
                issues.append(f"{case_id}: refuse 缺少 expected_refusal_reason")
            if state != "insufficient":
                issues.append(f"{case_id}: refuse 的 evidence_state 必须为 insufficient")
        elif behavior == "clarify":
            if answerable is not True:
                issues.append(f"{case_id}: clarify 必须 answerable=true")
            if not (case.get("clarifying_slots") or []):
                issues.append(f"{case_id}: clarify 缺少 clarifying_slots")
        elif behavior == "answer_with_conditions":
            if not (case.get("required_conditions") or []):
                issues.append(
                    f"{case_id}: answer_with_conditions 缺少 required_conditions"
                )
        elif behavior == "verify_then_answer":
            if not (case.get("required_conditions") or []):
                issues.append(f"{case_id}: verify_then_answer 缺少 required_conditions")
            if not (case.get("forbidden_claims") or []):
                issues.append(f"{case_id}: verify_then_answer 缺少 forbidden_claims")
        elif behavior == "answer":
            # policy_conflict 的证据本身就是冲突状态,但答案可以给出确定结论。
            if state != "sufficient" and bucket != "policy_conflict":
                issues.append(f"{case_id}: answer 题的 evidence_state 必须为 sufficient")

        if bucket == "policy_conflict":
            if state != "conflicting":
                issues.append(f"{case_id}: policy_conflict 的 evidence_state 必须为 conflicting")
            if case.get("as_of_date") is None:
                issues.append(f"{case_id}: policy_conflict 缺少 as_of_date")

        if bucket in {"hard_negative", "ood"}:
            if answerable is not False:
                issues.append(f"{case_id}: {bucket} 必须 answerable=false")
            if behavior != "refuse":
                issues.append(f"{case_id}: {bucket} 必须 expected_behavior=refuse")
            if case.get("required_evidence"):
                issues.append(f"{case_id}: {bucket} 不应有 required_evidence")

        fixture_tools = {
            str(fixture.get("tool"))
            for fixture in (case.get("tool_fixtures") or [])
        }
        for tool in case.get("requires_tools") or []:
            if tool not in fixture_tools:
                issues.append(f"{case_id}: requires_tools {tool!r} 缺少 tool_fixtures")

    if docs_dir is not None:
        docs_dir = Path(docs_dir)
        manifest = load_manifest()
        types = doc_types(manifest)
        registered = set(registered_docs(manifest))
        doc_cache: dict[str, tuple[str, list[Chunk]] | None] = {}
        for position, case in enumerate(cases, start=1):
            if position in schema_failed:
                continue
            issues.extend(
                _evidence_issues(
                    case,
                    registered=registered,
                    types=types,
                    docs_dir=docs_dir,
                    doc_cache=doc_cache,
                )
            )

    if factbook is not None:
        for position, case in enumerate(cases, start=1):
            if position in schema_failed:
                continue
            issues.extend(_fact_issues(case, factbook))

    if bucket_counts is not None:
        actual = dict(Counter(str(case.get("bucket", "")) for case in cases))
        expected = dict(bucket_counts)
        if actual != expected:
            issues.append(f"bucket 分布不符: 期望 {expected}, 实际 {actual}")

    if issues:
        raise EvalDataError("\n".join(issues))
