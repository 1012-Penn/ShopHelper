"""ch06 分流器评估:真模型逐轮跑 resolve+intent(与线上同一节点代码),报告落 reports/。

用法:.venv/bin/python scripts/eval_router.py [--set tests/eval/router_samples.jsonl]
      [--report reports/ch06-router-report.md] [--escalate]
"""
import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.graph.entry import make_intent_node, make_resolve_node
from app.llm import make_extract_model
from app.prompts import ResolvedQuestion


def _normalize(text: str) -> str:
    return "".join(ch for ch in text if ch not in " ,。?!?!、;;:·")


async def _with_retry(coro_fn, attempts: int = 4, delay: float = 20.0):
    """上游 429/超时重试(ch05 同款:该模型访问量过大)。"""
    for i in range(attempts):
        try:
            return await coro_fn()
        except Exception:
            if i == attempts - 1:
                raise
            await asyncio.sleep(delay)


async def _eval_turn(resolve_node, intent_node, history, user_message):
    state = {"session_id": 0, "user_message": user_message, "resolved_message": "",
             "intent": "", "intent_confidence": 0.0, "trace": [], "history": history}

    async def _run():
        out_r = await resolve_node(state, writer=None)
        state.update(out_r)
        await intent_node(state)

    await _with_retry(_run)
    return state["resolved_message"], state["intent"], state["intent_confidence"], state["trace"][-1]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="samples", default="tests/eval/router_samples.jsonl")
    ap.add_argument("--report", default="reports/ch06-router-report.md")
    ap.add_argument("--escalate", action="store_true", help="开启小→大降级路对照")
    args = ap.parse_args()

    settings = Settings()
    resolver = make_extract_model(settings).with_structured_output(
        ResolvedQuestion, method="function_calling")
    if args.escalate:
        small = make_extract_model(settings, model=settings.intent_small_model or None)
        big = make_extract_model(settings)
        intent_node = make_intent_node(small, escalator=big, floor=settings.intent_confidence_floor)
    else:
        intent_node = make_intent_node(make_extract_model(settings))
    resolve_node = make_resolve_node(resolver)

    rows = [json.loads(line) for line
            in Path(args.samples).read_text(encoding="utf-8").splitlines() if line.strip()]
    per_tag: dict[str, Counter] = {}
    mal_count, conf_sum, conf_n, escalated = 0, 0.0, 0, 0
    details = []
    for row in rows:
        history: list[dict] = []
        for i, turn in enumerate(row["turns"]):
            resolved, intent, conf, trace_line = await _eval_turn(
                resolve_node, intent_node, history, turn["user"])
            tag = row["tag"]
            ok_intent = intent == turn["expect_intent"]
            expect_resolved = turn.get("expect_resolved")
            ok_resolved = (expect_resolved is not None
                           and _normalize(resolved) == _normalize(expect_resolved))
            c = per_tag.setdefault(tag, Counter())
            c["intent_ok"] += ok_intent
            c["total"] += 1
            if expect_resolved is not None:
                c["resolved_ok"] += ok_resolved
                c["resolved_total"] += 1
            malformed = "malformed=true" in trace_line
            mal_count += malformed
            conf_sum += conf
            conf_n += 1
            escalated += "escalated=true" in trace_line
            details.append((row["id"], i, turn["user"], turn["expect_intent"], intent,
                            ok_intent, resolved, expect_resolved, ok_resolved, trace_line))
            history.append({"role": "user", "content": turn["user"]})

    total_turns = conf_n
    total_ok = sum(c["intent_ok"] for c in per_tag.values())
    res_total = sum(c.get("resolved_total", 0) for c in per_tag.values())
    res_ok = sum(c.get("resolved_ok", 0) for c in per_tag.values())
    lines = ["# ch06 分流器评估报告", "",
             f"- 用例:{len(rows)} 组对话 / {total_turns} 轮;escalation={'开' if args.escalate else '关'}",
             f"- 意图准确率(总体):{total_ok}/{total_turns}"
             f" = {total_ok / max(total_turns, 1):.2%}",
             f"- 指代消解准确率:{res_ok}/{res_total} = {res_ok / max(res_total, 1):.2%}",
             f"- JSON 畸形率:{mal_count}/{total_turns};平均 confidence:{conf_sum / max(conf_n, 1):.2f}",
             f"- escalation 触发:{escalated} 次", "",
             "| 桶 | 意图准确 | 指代消解准确 |", "|---|---|---|"]
    for tag, c in per_tag.items():
        lines.append(f"| {tag} | {c['intent_ok']}/{c['total']} "
                     f"| {c.get('resolved_ok', 0)}/{c.get('resolved_total', 0)} |")
    lines += ["", "## 个案明细", "",
              "| 组/轮 | 用户消息 | 期望意图 | 判定意图 | 对错 | 补全问法 | 指代对错 | trace |",
              "|---|---|---|---|---|---|---|---|"]
    for rid, i, user, exp, got, ok, resolved, expect_resolved, rok, trace in details:
        mark = "√" if rok else ("×" if expect_resolved is not None else "-")
        shown = resolved if expect_resolved is not None else "-"
        lines.append(f"| {rid}/{i} | {user} | {exp} | {got} | {'√' if ok else '×'} "
                     f"| {shown} | {mark} | {trace} |")
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已写入 {args.report}")
    print(f"意图准确率 {total_ok}/{total_turns};指代 {res_ok}/{res_total};"
          f"畸形 {mal_count};escalated {escalated}")


if __name__ == "__main__":
    asyncio.run(main())
