"""Namespace-qualified order facts backed by the immutable corpus factbook."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import yaml

from scripts.corpus_facts import FactBook, load_factbook

ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = ROOT / "corpus"
DEFAULT_MANIFEST = Path(__file__).with_name("manifests") / "corpus-orders-v1.yaml"
REQUIRED_ORDER_FIELDS = ("product", "amount", "status", "created_at", "signed_at")


@dataclass(frozen=True)
class CorpusOrderSnapshot:
    book: FactBook
    manifest: dict
    manifest_path: Path
    factbook_root: Path

    @property
    def version(self) -> str:
        return f"{self.manifest['version']}+{self.book.sha256()[:12]}"

    def _raw_record(self, order_id: str) -> dict:
        record = {
            fact.key: fact.value
            for fact in self.book.facts_for_entity("order", order_id)
        }
        overlay = (self.manifest.get("field_overlay") or {}).get(order_id) or {}
        return {**record, **overlay}

    def order_view(self, order_id: str) -> dict | None:
        facts = self.book.facts_for_entity("order", order_id)
        if not facts:
            return None
        raw = self._raw_record(order_id)
        field_map = self.manifest.get("field_map") or {}
        view = {"order_id": order_id}
        for output_key, source_key in field_map.items():
            view[output_key] = raw.get(source_key)
        for key in ("created_at", "signed_at"):
            view[key] = raw.get(key)

        view.update(
            {
                "subject": {
                    "namespace": "corpus",
                    "entity_id": order_id,
                    "canonical_id": f"corpus:{order_id}",
                },
                "source": self.manifest["source"],
                "source_uri": self.manifest["source_uri"],
                "version": self.version,
                "as_of": str(self.manifest["as_of"]),
                "missing_fields": [
                    field for field in REQUIRED_ORDER_FIELDS if view.get(field) is None
                ],
            }
        )
        return view

    def ids(self) -> list[str]:
        return sorted(
            {
                fact.entity_id
                for fact in self.book.facts.values()
                if fact.entity_type == "order"
            }
        )


@lru_cache(maxsize=8)
def load_corpus_order_snapshot(
    factbook_root: str | Path = CORPUS_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> CorpusOrderSnapshot:
    factbook_root = Path(factbook_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取订单适配 manifest: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("namespace") != "corpus":
        raise ValueError("订单适配 manifest 必须声明 namespace=corpus")
    return CorpusOrderSnapshot(
        book=load_factbook(factbook_root),
        manifest=manifest,
        manifest_path=manifest_path,
        factbook_root=factbook_root,
    )


def get_corpus_order(order_id: str) -> dict | None:
    return load_corpus_order_snapshot().order_view(str(order_id).strip().upper())


def corpus_order_ids() -> list[str]:
    return load_corpus_order_snapshot().ids()


def corpus_policy_windows():
    """Return retrieval policy windows without making the fact adapter depend on RAG."""
    from app.retrieval import PolicyWindow

    book = load_corpus_order_snapshot().book
    windows = [
        PolicyWindow(
            version=fact.entity_id.removesuffix("_returns"),
            effective_from=fact.effective_from,
            effective_to=fact.effective_to,
        )
        for fact in book.facts.values()
        if (
            fact.entity_type == "policy"
            and fact.key == "return_window_days"
            and fact.effective_from is not None
        )
    ]
    return tuple(sorted(set(windows), key=lambda item: item.version))


def _date_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace(" ", "T", 1)).date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return None


def policy_version_for_date(reference_date: str | date | None) -> str | None:
    """Resolve a return-policy version only from explicit factbook windows."""
    if reference_date is None:
        return None
    if isinstance(reference_date, str):
        try:
            selected = date.fromisoformat(reference_date[:10])
        except ValueError:
            return None
    else:
        selected = reference_date
    matches: list[str] = []
    book = load_corpus_order_snapshot().book
    for fact in book.facts.values():
        if (
            fact.entity_type != "policy"
            or fact.key != "return_window_days"
            or fact.effective_from is None
        ):
            continue
        lower = fact.effective_from
        upper = fact.effective_to
        if lower <= selected and (upper is None or selected <= upper):
            matches.append(fact.entity_id.removesuffix("_returns"))
    unique = sorted(set(matches))
    return unique[0] if len(unique) == 1 else None
