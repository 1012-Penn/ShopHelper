"""评测集泄漏扫描:题目、答案禁用断言、fact_id 或评测标记不得进入知识文档。

阻断项(exact_query / case_id / fact_id / forbidden_claim / eval_marker /
eval_file_in_knowledge)会让冻结失败;8-gram 相似度只作预警,>=0.9 升阻断。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.corpus_facts import FactBook, load_factbook
from scripts.corpus_manifest import DOCS_DIR, registered_docs
from scripts.eval_dataset import load_registry, load_split, split_entries
from scripts.eval_metrics import normalize_span

ROOT = Path(__file__).resolve().parent.parent
EVAL_MARKERS = ("FROZEN-", "DEV-", "fact_id", "required_evidence", "本题应回答", "标准答案")


@dataclass(frozen=True)
class LeakFinding:
    split: str
    case_id: str
    kind: str
    detail: str
    severity: str  # blocking | warning
    score: float = 0.0


def _shingles(text: str, n: int) -> set[str]:
    normalized = normalize_span(text)
    return {normalized[i : i + n] for i in range(max(0, len(normalized) - n + 1))}


def _doc_texts(docs_dir: Path = DOCS_DIR) -> dict[str, str]:
    docs: dict[str, str] = {}
    for name in registered_docs():
        path = Path(docs_dir) / name
        if path.exists():
            docs[name] = path.read_text(encoding="utf-8")
    return docs


def scan_cases(
    cases: list[dict],
    docs: dict[str, str],
    *,
    factbook: FactBook | None = None,
    ngram: int = 8,
    warn_threshold: float = 0.6,
    block_threshold: float = 0.9,
    split: str = "",
) -> list[LeakFinding]:
    findings: list[LeakFinding] = []
    normalized_docs = {name: normalize_span(text) for name, text in docs.items()}

    for case in cases:
        case_id = str(case.get("id") or "")
        query = str(case.get("query") or "")
        query_norm = normalize_span(query)
        if query_norm:
            for doc_name, text in normalized_docs.items():
                if query_norm in text:
                    findings.append(
                        LeakFinding(split, case_id, "exact_query", doc_name, "blocking", 1.0)
                    )
        if case_id:
            for doc_name, text in normalized_docs.items():
                if case_id in text:
                    findings.append(
                        LeakFinding(split, case_id, "case_id", doc_name, "blocking", 1.0)
                    )

        fact_ids = {
            str(fact.get("fact_id"))
            for fact in case.get("gold_facts") or []
            if fact.get("fact_id")
        }
        for evidence in case.get("required_evidence") or []:
            fact_ids.update(str(fid) for fid in evidence.get("fact_ids") or [])
        for fact_id in sorted(fact_ids):
            for doc_name, text in normalized_docs.items():
                if fact_id in text:
                    findings.append(
                        LeakFinding(split, case_id, "fact_id", f"{doc_name}:{fact_id}", "blocking", 1.0)
                    )

        for claim in case.get("forbidden_claims") or []:
            claim_norm = normalize_span(str(claim))
            if not claim_norm:
                continue
            for doc_name, text in normalized_docs.items():
                if claim_norm in text:
                    findings.append(
                        LeakFinding(
                            split, case_id, "forbidden_claim", f"{doc_name}:{claim}", "blocking", 1.0
                        )
                    )

        query_grams = _shingles(query, ngram)
        if query_grams:
            for doc_name, text in normalized_docs.items():
                doc_grams = _shingles(text, ngram)
                score = len(query_grams & doc_grams) / len(query_grams)
                if score >= block_threshold:
                    findings.append(
                        LeakFinding(split, case_id, "ngram_overlap", doc_name, "blocking", score)
                    )
                elif score >= warn_threshold:
                    findings.append(
                        LeakFinding(split, case_id, "ngram_overlap", doc_name, "warning", score)
                    )

    return findings


def _marker_findings(docs: dict[str, str]) -> list[LeakFinding]:
    findings: list[LeakFinding] = []
    for doc_name, text in docs.items():
        for marker in EVAL_MARKERS:
            if marker in text:
                findings.append(
                    LeakFinding("all", "", "eval_marker", f"{doc_name}:{marker}", "blocking", 1.0)
                )
    return findings


def _eval_file_findings(docs_dir: Path = DOCS_DIR) -> list[LeakFinding]:
    findings: list[LeakFinding] = []
    for path in Path(docs_dir).glob("*"):
        if path.suffix.lower() in {".jsonl", ".json", ".yaml", ".yml"}:
            findings.append(
                LeakFinding("all", "", "eval_file_in_knowledge", path.name, "blocking", 1.0)
            )
    return findings


def scan_split(
    split: str, root: Path = ROOT, *, docs_dir: Path = DOCS_DIR
) -> list[LeakFinding]:
    docs = _doc_texts(docs_dir)
    cases = load_split(split, root)
    return (
        _marker_findings(docs)
        + _eval_file_findings(docs_dir)
        + scan_cases(cases, docs, factbook=load_factbook(), split=split)
    )


def scan_all(root: Path = ROOT, *, docs_dir: Path = DOCS_DIR) -> list[LeakFinding]:
    docs = _doc_texts(docs_dir)
    findings = _marker_findings(docs) + _eval_file_findings(docs_dir)
    try:
        factbook = load_factbook()
    except Exception:
        factbook = None
    entries = split_entries(load_registry(root))
    for split in entries:
        try:
            cases = load_split(split, root)
        except Exception as exc:  # 数据尚未齐备时只报告,不阻断扫描器启动
            findings.append(
                LeakFinding(split, "", "data_error", str(exc), "warning", 0.0)
            )
            continue
        findings.extend(scan_cases(cases, docs, factbook=factbook, split=split))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="扫描评测题与知识文档泄漏")
    parser.add_argument("--split", default="all")
    args = parser.parse_args(argv)
    if args.split == "all":
        findings = scan_all()
    else:
        docs = _doc_texts()
        try:
            cases = load_split(args.split)
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 2
        findings = _marker_findings(docs) + _eval_file_findings() + scan_cases(
            cases, docs, factbook=load_factbook(), split=args.split
        )

    blocking = [item for item in findings if item.severity == "blocking"]
    warnings = [item for item in findings if item.severity == "warning"]
    for item in blocking:
        print(
            f"[blocking] {item.split}/{item.case_id} {item.kind}: {item.detail} "
            f"(score={item.score:.2f})"
        )
    for item in warnings:
        print(
            f"[warning] {item.split}/{item.case_id} {item.kind}: {item.detail} "
            f"(score={item.score:.2f})"
        )
    print(f"blocking={len(blocking)} warning={len(warnings)}")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
