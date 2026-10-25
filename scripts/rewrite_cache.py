"""冻结 query 改写：read / write / refresh 三种模式，四策略共用同一份改写。

缓存文件进 git（默认 tests/eval/ch04_rewrite_cache.json），保证跨机器可复现。
read 模式要求 eval_set_sha256 + model 完全匹配；write 模式只补缺失/变更条目；
refresh 模式全量重写。写文件走临时文件 + fsync + os.replace。
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.rewrite import QueryRewrite

CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_PATH = Path("tests/eval/ch04_rewrite_cache.json")


class RewriteCacheError(RuntimeError):
    """缓存缺失、哈希/模型不匹配、或缺少条目。"""


def eval_set_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_sha256(path: str | Path) -> str:
    return eval_set_sha256(path)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RewriteCacheError(f"缓存不是合法 JSON:{path}: {exc}") from exc


def _validate(cache: dict, eval_sha: str, model: str) -> None:
    if cache.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise RewriteCacheError(
            f"缓存 schema 版本不匹配:{cache.get('schema_version')} != {CACHE_SCHEMA_VERSION}")
    if cache.get("eval_set_sha256") != eval_sha:
        raise RewriteCacheError(
            "评估集已变化,缓存过期;先跑 --rewrite-cache-mode write/refresh")
    if cache.get("model") != model:
        raise RewriteCacheError(
            f"改写模型不匹配:{cache.get('model')} != {model};先跑 refresh")
    if not isinstance(cache.get("entries"), dict):
        raise RewriteCacheError("缓存缺少 entries 字典")


def _to_rewrite(entry: dict) -> QueryRewrite:
    return QueryRewrite(
        normalized=entry.get("normalized") or entry.get("query") or "",
        synonyms=list(entry.get("synonyms") or []),
    )


def _rewrite_entry(rewriter, query: str) -> dict:
    error = None
    try:
        result = rewriter.rewrite(query)
        normalized = (result.normalized or "").strip() or query
        synonyms = [s.strip() for s in result.synonyms if s and s.strip()]
    except Exception as exc:  # 改写失败也落缓存,保证可复现
        error = str(exc)
        normalized, synonyms = query, []
    return {"query": query, "normalized": normalized, "synonyms": synonyms, "error": error}


def load_rewrites(cases: list[dict], cache_path: str | Path,
                  eval_path: str | Path | None,
                  model: str, rewriter, mode: str = "read", *,
                  eval_sha256: str | None = None) -> dict[str, QueryRewrite]:
    """→ {case_id: QueryRewrite}。mode: read | write | refresh。

    `eval_sha256` 显式给出时优先于 `eval_path`;二者至少给一个。
    """
    if mode not in {"read", "write", "refresh"}:
        raise ValueError(f"未知缓存模式:{mode}")
    cache_path = Path(cache_path)
    if eval_sha256 is None:
        if eval_path is None:
            raise ValueError("load_rewrites 需要 eval_path 或 eval_sha256")
        eval_sha = eval_set_sha256(eval_path)
    else:
        eval_sha = eval_sha256
    case_by_id = {case["id"]: case for case in cases}

    if mode == "read":
        cache = _load(cache_path)
        if cache is None:
            raise RewriteCacheError(f"缓存不存在:{cache_path}")
        _validate(cache, eval_sha, model)
        entries = cache["entries"]
        missing = [cid for cid in case_by_id if cid not in entries]
        if missing:
            raise RewriteCacheError(f"缓存缺少条目:{missing[:10]}")
        return {cid: _to_rewrite(entries[cid]) for cid in case_by_id}

    existing = _load(cache_path) if mode == "write" else None
    entries: dict = {}
    if existing is not None:
        same_model = existing.get("model") == model
        same_schema = existing.get("schema_version") == CACHE_SCHEMA_VERSION
        if same_model and same_schema:
            entries = dict(existing.get("entries") or {})

    changed = existing is None or existing.get("eval_set_sha256") != eval_sha
    if existing is not None and entries == {}:
        changed = True

    for cid, case in case_by_id.items():
        old = entries.get(cid)
        if mode == "write" and old and old.get("query") == case["query"]:
            continue
        entries[cid] = _rewrite_entry(rewriter, case["query"])
        changed = True

    entries = {cid: entries[cid] for cid in case_by_id if cid in entries}
    if changed:
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "eval_set_sha256": eval_sha,
            "model": model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "entries": {cid: entries[cid] for cid in case_by_id},
        }
        _atomic_write(cache_path, payload)
    return {cid: _to_rewrite(entries[cid]) for cid in case_by_id}


def identity_rewrites(cases: list[dict]) -> dict[str, QueryRewrite]:
    return {case["id"]: QueryRewrite(normalized=case["query"], synonyms=[])
            for case in cases}


def _load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="冻结 query 改写缓存")
    parser.add_argument("--set", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--mode", choices=("read", "write", "refresh"), default="read")
    parser.add_argument("--model", default=None)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--ledger", type=Path, default=Path("data/ch11_budget.json"))
    args = parser.parse_args(argv)

    cases = _load_cases(args.set)
    model = args.model or os.environ.get("OPENAI_MODEL", "deepseek-flash")
    rewriter = None
    if args.mode in {"write", "refresh"}:
        if not args.allow_paid:
            print("拒绝付费改写:加 --allow-paid", file=sys.stderr)
            return 2
        estimate = 0.0005 * len(cases)
        if estimate > args.max_cost_usd:
            print(
                f"改写估算 ${estimate:.4f} 超过单次上限 ${args.max_cost_usd:.4f}",
                file=sys.stderr,
            )
            return 3
        from app.config import Settings
        from app.rewrite import make_rewriter
        from scripts.eval_cost import BudgetLedger

        settings = Settings()
        model = args.model or settings.openai_model
        rewriter = make_rewriter(settings)
        BudgetLedger(args.ledger).record(estimate, note=f"rewrite:{args.set}")
    load_rewrites(
        cases,
        args.cache,
        args.set,
        model,
        rewriter,
        mode=args.mode,
    )
    print(f"rewrite cache {args.mode}: {args.cache} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
