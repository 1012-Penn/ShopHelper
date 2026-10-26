"""Frozen case loading and integrity checks for E2E quality R2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CASE_FILES = {
    "dev": "cases-dev.jsonl",
    "heldout": "cases-heldout.jsonl",
}
BEHAVIORS = {"answer", "answer_with_conditions", "verify_then_answer", "clarify", "refuse"}


class DatasetContractError(RuntimeError):
    """Frozen cases or their manifest no longer match."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    return {
        name: file_sha256(root / filename)
        for name, filename in CASE_FILES.items()
        if (root / filename).exists()
    }


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetContractError(f"cannot read frozen cases: {path}: {exc}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetContractError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
        if not isinstance(payload, dict):
            raise DatasetContractError(f"case must be an object at {path}:{line_no}")
        rows.append(payload)
    return rows


def _validate_cases(cases: dict[str, list[dict]], manifest: dict) -> None:
    expected_counts = manifest.get("case_counts") or {}
    seen_ids: set[str] = set()
    groups: dict[str, set[str]] = {}
    for split, rows in cases.items():
        if len(rows) != int(expected_counts.get(split, -1)):
            raise DatasetContractError(
                f"case count mismatch for {split}: {len(rows)} != {expected_counts.get(split)}"
            )
        groups[split] = set()
        for row in rows:
            case_id = str(row.get("id") or "")
            if not case_id or case_id in seen_ids:
                raise DatasetContractError(f"missing or duplicate case id: {case_id!r}")
            seen_ids.add(case_id)
            if row.get("split") != split:
                raise DatasetContractError(f"{case_id}: split field does not match file")
            source_group = str(row.get("source_group") or "")
            if not source_group:
                raise DatasetContractError(f"{case_id}: missing source_group")
            groups[split].add(source_group)
            turns = row.get("turns")
            if not isinstance(turns, list) or not turns:
                raise DatasetContractError(f"{case_id}: turns must be a non-empty list")
            if any(turn.get("role") != "user" or not str(turn.get("content") or "").strip()
                   for turn in turns):
                raise DatasetContractError(f"{case_id}: only non-empty user turns are supported")
            if row.get("expected_behavior") not in BEHAVIORS:
                raise DatasetContractError(f"{case_id}: invalid expected_behavior")
            if not isinstance(row.get("facts", []), list):
                raise DatasetContractError(f"{case_id}: facts must be a list")
    overlap = groups.get("dev", set()) & groups.get("heldout", set())
    if overlap:
        raise DatasetContractError(
            f"source groups leak across dev/heldout: {sorted(overlap)}"
        )
    declared = manifest.get("source_groups") or {}
    if set(declared.get("dev") or []) != groups.get("dev", set()):
        raise DatasetContractError("manifest dev source_groups mismatch")
    if set(declared.get("heldout") or []) != groups.get("heldout", set()):
        raise DatasetContractError("manifest heldout source_groups mismatch")


def load_frozen_cases(root: Path) -> dict[str, list[dict]]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"cannot read manifest: {manifest_path}: {exc}") from exc
    actual_hashes = dataset_hashes(root)
    if manifest.get("files") != actual_hashes:
        raise DatasetContractError(
            f"case file hash mismatch: expected {manifest.get('files')}, got {actual_hashes}"
        )
    cases = {
        split: _read_jsonl(root / filename)
        for split, filename in CASE_FILES.items()
    }
    _validate_cases(cases, manifest)
    return cases
