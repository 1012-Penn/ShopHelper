"""Order identity adapter for legacy demo records and corpus-v1 facts.

Bare numeric ids are an explicit legacy-demo contract. Full ``ORD-*`` ids are
corpus subjects. The same suffix never maps across these namespaces.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.business_facts import corpus_order_ids, get_corpus_order

ORDER_CATALOG: dict[str, dict] = {
    "1001": {"order_id": "1001", "product": "无线耳机", "amount": 299.0, "status": "已签收", "created_at": "2026-08-30"},
    "1002": {"order_id": "1002", "product": "机械键盘", "amount": 399.0, "status": "已发货", "created_at": "2026-09-03"},
    "1003": {"order_id": "1003", "product": "硅胶手机壳", "amount": 19.9, "status": "待发货", "created_at": "2026-09-07"},
}

LEGACY_NAMESPACE = "legacy_demo"
CORPUS_NAMESPACE = "corpus"
_CORPUS_ID = re.compile(r"(?<![A-Z0-9])ORD-\d{4,6}(?![0-9])", re.IGNORECASE)
_BARE_ID = re.compile(r"(?<!\d)(\d{4,6})(?!\d)")
_DATE_YEAR = re.compile(r"(?:19|20)\d{2}(?=[-/\.年]\s*\d{1,2})")
_SUFFIX_ID = re.compile(
    r"(?:尾号|后四位|末四位|后\s*4\s*位|末\s*4\s*位)\s*[:：]?\s*(\d{3,6})",
    re.IGNORECASE,
)
_CONTEXTUAL_ID = re.compile(
    r"(?:订单|order)\s*号?\s*[:：]?\s*(ORD-\d{4,6}|\d{3,6})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OrderReference:
    requested: str
    order_id: str | None
    namespace: str | None
    canonical_id: str | None
    ambiguous: bool = False
    reason: str = ""


def _subject(namespace: str, order_id: str) -> dict:
    return {
        "namespace": namespace,
        "entity_id": order_id,
        "canonical_id": f"{namespace}:{order_id}",
    }


def _legacy_view(order: dict) -> dict:
    return {
        **order,
        "subject": _subject(LEGACY_NAMESPACE, order["order_id"]),
        "source": "legacy_demo:ORDER_CATALOG",
        "version": "legacy-demo-v1",
        "as_of": "unknown",
        "missing_fields": [],
    }


def _unknown_view(requested: str, namespace: str | None) -> dict:
    view = {
        "order_id": requested,
        "error": "未找到该订单",
        "source": "unresolved",
        "version": "unknown",
        "as_of": "unknown",
        "missing_fields": [],
    }
    if namespace:
        view["subject"] = _subject(namespace, requested)
    return view


def _normalized_id(value: str) -> str:
    stripped = value.strip()
    return stripped.upper() if stripped.upper().startswith("ORD-") else stripped


def get_order(order_id: str) -> dict:
    normalized = _normalized_id(str(order_id))
    if normalized.upper().startswith("ORD-"):
        corpus = get_corpus_order(normalized)
        return corpus or _unknown_view(normalized, CORPUS_NAMESPACE)
    order = ORDER_CATALOG.get(normalized)
    return _legacy_view(dict(order)) if order is not None else _unknown_view(
        normalized, LEGACY_NAMESPACE
    )


def list_orders(namespace: str = "legacy") -> list[dict]:
    if namespace == "legacy":
        return [_legacy_view(dict(order)) for order in ORDER_CATALOG.values()]
    if namespace == "corpus":
        return [get_order(order_id) for order_id in corpus_order_ids()]
    if namespace == "all":
        return list_orders("legacy") + list_orders("corpus")
    raise ValueError(f"未知订单命名空间:{namespace}")


def _reference(
    raw: str,
    *,
    ambiguous: bool = False,
    reason: str = "",
) -> OrderReference:
    requested = raw.strip()
    if ambiguous:
        return OrderReference(requested, None, None, None, True, reason)
    if requested.upper().startswith("ORD-"):
        order_id = requested.upper()
        return OrderReference(
            requested,
            order_id,
            CORPUS_NAMESPACE,
            f"{CORPUS_NAMESPACE}:{order_id}",
        )
    return OrderReference(
        requested,
        requested,
        LEGACY_NAMESPACE,
        f"{LEGACY_NAMESPACE}:{requested}",
    )


def extract_order_reference(text: str) -> OrderReference:
    text = text or ""
    suffix = _SUFFIX_ID.search(text)
    if suffix:
        value = suffix.group(1)
        has_legacy = value in ORDER_CATALOG
        has_corpus = f"ORD-{value}" in corpus_order_ids()
        if has_legacy and has_corpus:
            return _reference(
                value,
                ambiguous=True,
                reason="同一尾号同时存在于 legacy_demo 与 corpus;请提供完整订单号",
            )

    corpus_ids = list(dict.fromkeys(m.upper() for m in _CORPUS_ID.findall(text)))
    if len(corpus_ids) > 1:
        return _reference(
            ",".join(corpus_ids),
            ambiguous=True,
            reason="文本包含多个 corpus 订单;请指定唯一主体",
        )
    if corpus_ids:
        return _reference(corpus_ids[0])

    contextual = _CONTEXTUAL_ID.search(text)
    if contextual:
        return _reference(contextual.group(1))
    for bare in _BARE_ID.finditer(text):
        if _DATE_YEAR.match(text, bare.start()):
            continue
        return _reference(bare.group(1))
    return OrderReference(text, None, None, None)


def extract_order_id(text: str, require_known: bool = False) -> str | None:
    """Preserve bare-id extraction; ambiguous suffixes intentionally return None."""
    reference = extract_order_reference(text)
    if reference.ambiguous or reference.order_id is None:
        return None
    contextual = _CONTEXTUAL_ID.search(text or "")
    if contextual:
        return _normalized_id(contextual.group(1))
    if (
        require_known
        and reference.namespace == LEGACY_NAMESPACE
        and reference.order_id not in ORDER_CATALOG
    ):
        return None
    return reference.order_id


def order_reference_dict(text: str) -> dict:
    return asdict(extract_order_reference(text))
