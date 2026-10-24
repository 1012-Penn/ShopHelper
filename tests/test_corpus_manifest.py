"""ch11 语料 manifest:文档注册、规模校验、稳定哈希与 build_kb dry-run 一致性。"""
import hashlib
from pathlib import Path

import pytest

from app.chunking import chunk_markdown
from scripts.corpus_manifest import (
    CorpusManifest,
    DocSpec,
    corpus_sha256,
    doc_types,
    expected_docs,
    load_manifest,
    registered_docs,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "knowledge"


def _manifest(*files: str, expected_docs: int | None = None) -> CorpusManifest:
    specs = tuple(
        DocSpec(file=f, stem=Path(f).stem, content_type="manual", role="test") for f in files
    )
    return CorpusManifest(
        corpus_version="test-v1",
        docs=specs,
        expected_docs=expected_docs if expected_docs is not None else len(specs),
    )


def test_manifest_doc_types_match_registered_files():
    manifest = load_manifest()
    types = doc_types()

    assert manifest.corpus_version
    assert expected_docs() == 3 == len(registered_docs())
    assert registered_docs() == tuple(sorted(types))
    assert set(types) == {p.name for p in DOCS_DIR.glob("*.md")}
    assert types == {spec.file: spec.content_type for spec in manifest.docs}
    assert types["退货政策.md"] == "policy"
    assert types["商品FAQ.md"] == "faq"
    assert types["售后手册.md"] == "manual"


def test_unregistered_doc_rejected(tmp_path):
    from scripts.build_kb import chunk_corpus

    (tmp_path / "已登记.md").write_text("# 标题\n\n正文。", encoding="utf-8")
    (tmp_path / "未登记.md").write_text("# 标题\n\n正文。", encoding="utf-8")

    with pytest.raises(ValueError, match="未登记"):
        chunk_corpus(tmp_path, manifest=_manifest("已登记.md"))


def test_corpus_sha256_stable_across_order(tmp_path):
    # 创建顺序与登记顺序都刻意打乱,哈希只认「文件名 + \0 + 原始字节」的排序结果。
    (tmp_path / "b.md").write_text("BB", encoding="utf-8")
    (tmp_path / "a.md").write_text("AA", encoding="utf-8")
    expected = hashlib.sha256(b"a.md\0AA" + b"b.md\0BB").hexdigest()

    assert corpus_sha256(tmp_path, manifest=_manifest("a.md", "b.md")) == expected
    assert corpus_sha256(tmp_path, manifest=_manifest("b.md", "a.md")) == expected

    (tmp_path / "b.md").write_text("BC", encoding="utf-8")
    assert corpus_sha256(tmp_path, manifest=_manifest("a.md", "b.md")) != expected


def test_build_dry_run_matches_chunker():
    from scripts.build_kb import dry_run

    manifest = load_manifest()
    stats = dry_run(manifest=manifest)

    expected_counts = {
        spec.file: len(
            chunk_markdown(
                (DOCS_DIR / spec.file).read_text(encoding="utf-8"),
                spec.file,
                spec.content_type,
            )
        )
        for spec in manifest.docs
    }
    assert stats["docs"] == expected_docs() == 3
    assert stats["doc_chunk_counts"] == expected_counts
    assert stats["chunks"] == sum(expected_counts.values()) == 58
    assert stats["corpus_sha256"] == corpus_sha256(DOCS_DIR, manifest=manifest)


def test_dry_run_rejects_scale_out_of_manifest_bounds():
    from scripts.build_kb import dry_run

    with pytest.raises(ValueError, match="chunk"):
        dry_run(max_chars=100, overlap_chars=20)


def test_ch04_eval_doc_types_come_from_manifest():
    from tests.test_ch04_eval_set import DOC_TYPES as eval_doc_types

    assert eval_doc_types == doc_types()
