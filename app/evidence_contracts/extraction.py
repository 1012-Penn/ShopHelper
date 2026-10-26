"""Conservative tool-result to claim mapping."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.evidence_contracts.models import UNKNOWN, EvidenceClaim, EvidenceSource


_SUBJECT_ARGUMENT_KEYS = (
    "order_id", "order_ids", "product_id", "sku", "model", "subject_id",
)
_SCALAR_PREDICATES = (
    "status", "product", "carrier", "tracking_no", "amount", "currency",
    "estimated_delivery", "cancelled_at", "refunded_at", "logistics_status",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, int)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, list):
        return [
            str(item).strip() for item in value
            if isinstance(item, (str, int)) and str(item).strip()
        ]
    return []


def subjects_from_arguments(arguments: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for key in _SUBJECT_ARGUMENT_KEYS:
        for item in _string_list(arguments.get(key)):
            if item not in found:
                found.append(item)
    return found


def subjects_from_text(text: str) -> list[str]:
    found = [
        match.group(1)
        for match in re.finditer(r"(?:订单|order)\s*号?\s*(\d{3,8})", text or "",
                                 flags=re.IGNORECASE)
    ]
    for pattern in (
        r"\b[A-Z]{2,8}-?[A-Z0-9]{2,}-V\d+\b",
        r"\bSKU[-_:][A-Z0-9_-]+\b",
    ):
        found.extend(match.group(0) for match in re.finditer(pattern, text or "",
                                                             flags=re.IGNORECASE))
    return list(dict.fromkeys(found))


def _claim_id(source: EvidenceSource, predicate: str, subject_ids: list[str],
              scope: str, policy_version: str) -> str:
    payload = "\x1f".join([
        source.tool_name, source.tool_call_id, source.chunk_id,
        predicate, ",".join(sorted(subject_ids)), scope, policy_version,
    ])
    return "c:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _decoded_payload(raw_result: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_result)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _source(tool_name: str, arguments: dict[str, Any], raw_result: str,
            tool_call_id: str, chunk_id: str = UNKNOWN) -> EvidenceSource:
    return EvidenceSource(
        tool_name=tool_name, arguments=dict(arguments), raw_result=raw_result,
        tool_call_id=tool_call_id or UNKNOWN, chunk_id=str(chunk_id or UNKNOWN),
    )


def _claim(run_id: str, task_id: str, subject_ids: list[str], predicate: str,
           value: Any, source: EvidenceSource, observed_at: str,
           sequence: int, *, scope: str = UNKNOWN, policy_version: str = UNKNOWN,
           knowledge_as_of: str = UNKNOWN, valid_from: str = UNKNOWN,
           valid_until: str = UNKNOWN, unit: str = UNKNOWN,
           certainty: str = "verified") -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=_claim_id(source, predicate, subject_ids, scope, policy_version),
        run_id=run_id, task_id=task_id, subject_ids=subject_ids,
        predicate=predicate, value=value, unit=unit, scope=scope,
        policy_version=policy_version, knowledge_as_of=knowledge_as_of,
        observed_at=observed_at, valid_from=valid_from, valid_until=valid_until,
        observed_sequence=sequence, certainty=certainty, source=source,
    )


def claims_from_tool_result(tool_name: str, arguments: dict[str, Any],
                            raw_result: str, *, run_id: str, task_id: str,
                            tool_call_id: str = UNKNOWN,
                            observed_at: str | None = None,
                            dynamic_ttl_seconds: int = 300) -> list[EvidenceClaim]:
    """Map a tool return without using model output as evidence.

    Unknown schemas still retain one raw observation claim with explicit unknown
    fields. They do not create guessed domain facts.
    """

    observed = observed_at or _now_iso()
    payload = _decoded_payload(raw_result)
    argument_subjects = subjects_from_arguments(arguments)
    if payload is None:
        source = _source(tool_name, arguments, raw_result, tool_call_id)
        return [_claim(
            run_id, task_id, argument_subjects or ["unknown_subject"],
            "tool_result_text", raw_result, source, observed, 1,
            certainty="unknown",
        )]

    claims: list[EvidenceClaim] = []
    items = payload.get("items")
    if isinstance(items, list):
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or f"item-{index}")
            subjects = subjects_from_arguments(item) or argument_subjects
            subjects = subjects or _string_list(item.get("subject_ids"))
            if not subjects:
                subjects = ["global_policy"]
            scope = str(item.get("scope") or "global")
            policy_version = str(item.get("policy_version") or UNKNOWN)
            source = _source(tool_name, arguments, raw_result, tool_call_id, chunk_id)
            claims.append(_claim(
                run_id, task_id, subjects, "policy_content",
                item.get("answer", item.get("content", UNKNOWN)), source,
                observed, index + 1, scope=scope,
                policy_version=policy_version,
                knowledge_as_of=str(item.get("knowledge_as_of") or UNKNOWN),
                valid_from=str(item.get("valid_from") or UNKNOWN),
                valid_until=str(item.get("valid_until") or UNKNOWN),
            ))

    subject_ids = (
        subjects_from_arguments(payload) or argument_subjects
        or _string_list(payload.get("subject_ids"))
    )
    if not subject_ids:
        subject_ids = ["unknown_subject"]
    order_scope = str(payload.get("scope") or "order")
    valid_from = str(payload.get("valid_from") or UNKNOWN)
    valid_until = str(payload.get("valid_until") or UNKNOWN)
    if tool_name == "query_order" and valid_until == UNKNOWN:
        valid_until = (
            datetime.fromisoformat(observed.replace("Z", "+00:00"))
            + timedelta(seconds=max(0, dynamic_ttl_seconds))
        ).isoformat()
    claim_sequence = 1
    for predicate in _SCALAR_PREDICATES:
        if predicate not in payload or payload[predicate] in (None, ""):
            continue
        source = _source(tool_name, arguments, raw_result, tool_call_id)
        claims.append(_claim(
            run_id, task_id, subject_ids, predicate, payload[predicate], source,
            str(payload.get(f"{predicate}_observed_at") or observed),
            claim_sequence, scope=order_scope,
            policy_version=str(payload.get("policy_version") or UNKNOWN),
            knowledge_as_of=str(payload.get("knowledge_as_of") or UNKNOWN),
            valid_from=str(payload.get(f"{predicate}_valid_from") or valid_from),
            valid_until=str(payload.get(f"{predicate}_valid_until") or valid_until),
            unit=str(payload.get(f"{predicate}_unit") or UNKNOWN),
        ))
        claim_sequence += 1

    if not claims:
        source = _source(tool_name, arguments, raw_result, tool_call_id)
        claims.append(_claim(
            run_id, task_id, subject_ids, "tool_result", payload, source,
            observed, claim_sequence, scope=order_scope,
            valid_until=valid_until, certainty="unknown",
        ))
    return claims
