"""Run-local typed evidence lifecycle and role projection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from app.evidence_contracts.models import (
    UNKNOWN,
    EvidenceClaim,
    TaskHandoff,
)


def _parse_time(value: str) -> datetime | None:
    if not value or value == UNKNOWN:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if len(value) == 10:
        parsed = parsed.replace(hour=23, minute=59, second=59)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class EvidenceLedger:
    """Keeps source-backed claims and computes the smallest role projection."""

    def __init__(self, *, now: Callable[[], datetime] | None = None) -> None:
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.claims: dict[str, EvidenceClaim] = {}

    def _timestamp(self) -> str:
        return self._now().astimezone(timezone.utc).isoformat()

    @staticmethod
    def _lifecycle_key(claim: EvidenceClaim) -> tuple:
        return (
            claim.run_id, tuple(sorted(claim.subject_ids)), claim.predicate,
            claim.scope, claim.policy_version,
        )

    def add_many(self, claims: list[EvidenceClaim]) -> None:
        for claim in claims:
            key = self._lifecycle_key(claim)
            replacement = next((
                existing for existing in self.claims.values()
                if self._lifecycle_key(existing) == key
                and existing.superseded_by is None
                and existing.observed_sequence < claim.observed_sequence
            ), None)
            if replacement is not None:
                replacement.superseded_by = claim.claim_id
                replacement.superseded_at = self._timestamp()
            self.claims[claim.claim_id] = claim

    def is_active(self, claim: EvidenceClaim) -> bool:
        if claim.superseded_by is not None or claim.certainty != "verified":
            return False
        now = self._now()
        start = _parse_time(claim.valid_from)
        end = _parse_time(claim.valid_until)
        if start is not None and now < start:
            return False
        if end is not None and now > end:
            return False
        knowledge = _parse_time(claim.knowledge_as_of)
        if knowledge is not None and knowledge > now:
            return False
        return True

    @staticmethod
    def _subject_matches(claim: EvidenceClaim, handoff: TaskHandoff,
                         task) -> bool:
        task_subjects = set(handoff.subject_ids)
        claim_subjects = set(claim.subject_ids)
        if "global_policy" in claim_subjects or claim.scope == "global":
            return claim.predicate == "policy_content"
        if not task_subjects:
            return True
        return bool(task_subjects & claim_subjects)

    def project(self, *, task, handoff: TaskHandoff) -> list[EvidenceClaim]:
        dependencies = set(task.depends_on)
        if not dependencies:
            return []
        visible = [
            claim for claim in self.claims.values()
            if claim.task_id in dependencies
            and self.is_active(claim)
            and self._subject_matches(claim, handoff, task)
        ]
        visible.sort(key=lambda item: (item.task_id, item.observed_sequence,
                                       item.claim_id))
        return visible

    def project_supervisor(self, *, run_id: str,
                           task_ids: list[str],
                           subject_ids: list[str]) -> list[EvidenceClaim]:
        allowed = set(task_ids)
        subjects = set(subject_ids)
        visible = []
        for claim in self.claims.values():
            if claim.run_id != run_id or claim.task_id not in allowed:
                continue
            if not self.is_active(claim):
                continue
            if claim.scope != "global" and subjects and not (
                    subjects & set(claim.subject_ids)):
                continue
            visible.append(claim)
        visible.sort(key=lambda item: (item.task_id, item.observed_sequence,
                                       item.claim_id))
        return visible
