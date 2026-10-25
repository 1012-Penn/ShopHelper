"""语料 manifest:文档注册、content_type、规模上下限与稳定语料哈希。

唯一真源是 `corpus/manifest.yaml`;`knowledge/*.md` 只作运行时文档源。
未在 manifest 登记的 `.md` 会被拒绝,避免新增文档静默落到 `manual`。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "corpus" / "manifest.yaml"
DOCS_DIR = ROOT / "knowledge"


class CorpusManifestError(ValueError):
    """manifest 缺失、字段非法或与磁盘文档不一致。"""


class UnregisteredDocError(CorpusManifestError):
    """knowledge/ 下出现未登记文档。"""


@dataclass(frozen=True)
class DocSpec:
    file: str
    stem: str
    content_type: str
    role: str


@dataclass(frozen=True)
class CorpusManifest:
    corpus_version: str
    docs: tuple[DocSpec, ...]
    expected_docs: int | None = None
    expected_chunks_min: int | None = None
    expected_chunks_max: int | None = None
    expected_facts_min: int | None = None
    expected_facts_max: int | None = None


def _optional_int(data: dict, key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CorpusManifestError(f"manifest 字段 {key} 必须是整数或 null,实际为 {value!r}")
    return value


def load_manifest(path: Path = MANIFEST_PATH) -> CorpusManifest:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise CorpusManifestError(f"语料 manifest 不存在: {path}") from exc
    if not isinstance(data, dict):
        raise CorpusManifestError(f"语料 manifest 顶层必须是映射: {path}")

    corpus_version = data.get("corpus_version")
    if not isinstance(corpus_version, str) or not corpus_version.strip():
        raise CorpusManifestError("manifest 缺少非空 corpus_version")

    raw_docs = data.get("docs")
    if not isinstance(raw_docs, list) or not raw_docs:
        raise CorpusManifestError("manifest 的 docs 必须是非空列表")

    specs: list[DocSpec] = []
    seen: set[str] = set()
    for item in raw_docs:
        if not isinstance(item, dict):
            raise CorpusManifestError(f"manifest 文档条目必须是映射: {item!r}")
        file = item.get("file")
        content_type = item.get("content_type")
        if not isinstance(file, str) or not file:
            raise CorpusManifestError(f"文档条目缺少 file: {item!r}")
        if not isinstance(content_type, str) or not content_type:
            raise CorpusManifestError(f"文档 {file} 缺少 content_type")
        if file in seen:
            raise CorpusManifestError(f"manifest 重复登记文档: {file}")
        seen.add(file)
        specs.append(
            DocSpec(
                file=file,
                stem=Path(file).stem,
                content_type=content_type,
                role=str(item.get("role") or ""),
            )
        )
    specs.sort(key=lambda spec: spec.file)

    manifest = CorpusManifest(
        corpus_version=corpus_version,
        docs=tuple(specs),
        expected_docs=_optional_int(data, "expected_docs"),
        expected_chunks_min=_optional_int(data, "expected_chunks_min"),
        expected_chunks_max=_optional_int(data, "expected_chunks_max"),
        expected_facts_min=_optional_int(data, "expected_facts_min"),
        expected_facts_max=_optional_int(data, "expected_facts_max"),
    )
    if manifest.expected_docs is not None and manifest.expected_docs != len(manifest.docs):
        raise CorpusManifestError(
            f"expected_docs={manifest.expected_docs} 与 docs 条数 {len(manifest.docs)} 不一致"
        )
    if (
        manifest.expected_chunks_min is not None
        and manifest.expected_chunks_max is not None
        and manifest.expected_chunks_min > manifest.expected_chunks_max
    ):
        raise CorpusManifestError("expected_chunks_min 不得大于 expected_chunks_max")
    if (
        manifest.expected_facts_min is not None
        and manifest.expected_facts_max is not None
        and manifest.expected_facts_min > manifest.expected_facts_max
    ):
        raise CorpusManifestError("expected_facts_min 不得大于 expected_facts_max")
    return manifest


def doc_types(manifest: CorpusManifest | None = None) -> dict[str, str]:
    """文档文件名 → content_type(与旧 DOC_TYPES 的键口径一致:带 `.md`)。"""
    manifest = manifest or load_manifest()
    return {spec.file: spec.content_type for spec in manifest.docs}


def registered_docs(manifest: CorpusManifest | None = None) -> tuple[str, ...]:
    manifest = manifest or load_manifest()
    return tuple(sorted(spec.file for spec in manifest.docs))


def expected_docs(manifest: CorpusManifest | None = None) -> int:
    manifest = manifest or load_manifest()
    return manifest.expected_docs if manifest.expected_docs is not None else len(manifest.docs)


def resolve_docs(
    docs_dir: Path = DOCS_DIR, manifest: CorpusManifest | None = None
) -> tuple[Path, ...]:
    """校验磁盘与 manifest 的文档集合完全一致,返回按文件名排序的路径。"""
    manifest = manifest or load_manifest()
    docs_dir = Path(docs_dir)
    registered = set(registered_docs(manifest))
    actual = {path.name for path in docs_dir.glob("*.md")}

    unregistered = sorted(actual - registered)
    if unregistered:
        raise UnregisteredDocError(
            f"{docs_dir} 下存在未登记文档: {', '.join(unregistered)};"
            "请先登记到 corpus/manifest.yaml"
        )
    missing = sorted(registered - actual)
    if missing:
        raise CorpusManifestError(f"manifest 登记但缺失的文档: {', '.join(missing)}")
    return tuple(docs_dir / name for name in sorted(registered))


def corpus_sha256(docs_dir: Path = DOCS_DIR, manifest: CorpusManifest | None = None) -> str:
    """语料哈希:按文件名排序,逐份哈希 `filename + "\\0" + 原始字节`,与 glob 顺序无关。"""
    digest = hashlib.sha256()
    for path in resolve_docs(docs_dir, manifest):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
