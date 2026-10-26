from __future__ import annotations

import hashlib
import json
from pathlib import Path


class AnswerPlanDatasetError(RuntimeError):
    pass


FILES = {
    "dev": "cases-dev.jsonl",
    "heldout": "cases-heldout.jsonl",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    return {
        name: file_sha256(root / name)
        for name in (*FILES.values(), "evidence.json", "config.json")
        if (root / name).exists()
    }


def _read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnswerPlanDatasetError(f"cannot read {path}: {exc}") from exc
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnswerPlanDatasetError(
                f"invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        rows.append(row)
    return rows


def load_frozen_dataset(root: Path) -> dict[str, list[dict]]:
    root = Path(root)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnswerPlanDatasetError(f"cannot read manifest: {exc}") from exc
    actual = dataset_hashes(root)
    if manifest.get("files") != actual:
        raise AnswerPlanDatasetError(
            f"hash mismatch: expected {manifest.get('files')}, got {actual}"
        )
    cases = {
        split: _read_jsonl(root / filename)
        for split, filename in FILES.items()
    }
    for split, rows in cases.items():
        if len(rows) != int((manifest.get("case_counts") or {}).get(split, -1)):
            raise AnswerPlanDatasetError(f"case count mismatch for {split}")
        for row in rows:
            if row.get("split") != split:
                raise AnswerPlanDatasetError(f"{row.get('id')}: split mismatch")
    dev_groups = {row["source_group"] for row in cases["dev"]}
    held_groups = {row["source_group"] for row in cases["heldout"]}
    dev_templates = {row["evidence_template"] for row in cases["dev"]}
    held_templates = {row["evidence_template"] for row in cases["heldout"]}
    if dev_groups & held_groups or dev_templates & held_templates:
        raise AnswerPlanDatasetError("source groups leak across dev/heldout")
    return cases


def load_evidence(root: Path) -> dict[str, dict]:
    path = Path(root) / "evidence.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnswerPlanDatasetError(f"cannot read evidence snapshot: {exc}") from exc
    rows = {
        str(row["evidence_id"]): row
        for row in payload.get("evidence") or []
    }
    return rows


def model_projection(case: dict, evidence: dict[str, dict]) -> dict:
    turns = case.get("turns") or []
    query = str((turns[-1] if turns else {}).get("content") or "")
    visible = []
    for evidence_id in case.get("evidence_ids") or []:
        row = evidence[evidence_id]
        visible.append({
            key: row[key]
            for key in (
                "evidence_id",
                "text",
                "source_id",
                "source_title",
                "entities",
                "required_conditions",
                "condition_aliases",
                "policy_scope",
            )
            if key in row
        })
    return {"query": query, "evidence": visible}
