"""ch09 真机验收:可观测性 + 数据飞轮六条验收标准。

前提:
  docker compose -f docker-compose.langfuse.yml up -d   # Langfuse 自部署栈
  docker compose up -d                                  # MySQL
  .venv/bin/python -m scripts.build_kb                  # 知识库已灌
  .venv/bin/uvicorn app.main:app --port 8000            # 主服务(带 LANGFUSE_* env)
  .venv/bin/python scripts/acceptance_ch09.py --scenario all

场景:
  trace     验收 1:问一轮,Langfuse 里出现完整 trace 树(节点/模型/工具/token/耗时)
  flywheel  验收 2+3:知识库没有的问题 → 兜底 + 入待审队列(详情含原话/召回片段)
            → 审核通过 → 同题再问答对(飞轮整圈)
  feedback  验收 4:聊天后点 👎 → user_feedback 入池 → 标准化查重后出现在待审队列
  cost      验收 5:按意图汇总 token 花销(/api/usage/by-intent)
  eval      验收 6:连跑两轮评估 → /api/eval-runs 趋势对比
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = os.environ.get("ACCEPT_BASE", "http://127.0.0.1:8000")
TIMEOUT = float(os.environ.get("ACCEPT_TIMEOUT", "120"))


def chat(client: httpx.Client, message: str, session_id: int | None = None) -> dict:
    """走一遍 SSE 聊天,回收最终 reply 与帧。"""
    r = client.post(f"{BASE}/api/chat", json={"message": message, "session_id": session_id},
                    timeout=TIMEOUT)
    r.raise_for_status()
    reply, session, frames = "", None, []
    for block in r.text.split("\n\n"):
        line = block.strip()
        if not line.startswith("data: "):
            continue
        frame = json.loads(line[len("data: "):])
        frames.append(frame)
        if frame["type"] == "session":
            session = frame["session_id"]
        elif frame["type"] == "token":
            reply += frame.get("content", "")
    return {"reply": reply, "session_id": session, "frames": frames}


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))
    return ok


def langfuse_client() -> httpx.Client:
    pk = os.environ["LANGFUSE_PUBLIC_KEY"]
    sk = os.environ["LANGFUSE_SECRET_KEY"]
    host = os.environ.get("LANGFUSE_HOST", "http://127.0.0.1:3000")
    return httpx.Client(base_url=host, auth=(pk, sk), timeout=30)


def scenario_trace() -> bool:
    with httpx.Client(trust_env=False) as c, langfuse_client() as lf:
        out = chat(c, "普通订单的邮费是多少")
        ok = check("验收1 聊天请求拿到回复", len(out["reply"]) > 0, out["reply"][:50])
        time.sleep(8)  # 等 OTel batch 导出 + worker 消费
        traces = lf.get("/api/public/traces", params={"limit": 20}).json()["data"]
        mine = [t for t in traces if str(t.get("metadata", {}).get("conversation_id")) ==
                str(out["session_id"])]
        ok &= check("验收1 Langfuse 出现本轮 trace", bool(mine),
                    f"共 {len(traces)} 条,命中 {len(mine)} 条")
        if not mine:
            return bool(ok)
        trace_id = mine[0]["id"]
        tree = lf.get(f"/api/public/traces/{trace_id}").json()
        obs = tree.get("observations", [])
        kinds = {o["type"] for o in obs}
        names = [o.get("name", "") for o in obs]
        ok &= check("验收1 观察树非空(节点+模型调用)", len(obs) >= 3,
                    f"observations={len(obs)} types={sorted(kinds)}")
        usage = [(o.get("usageDetails") or o.get("usage_details") or {})
                 for o in obs if o["type"] == "GENERATION"]
        tokens = sum(int(u.get("total") or u.get("total_tokens") or 0) for u in usage)
        lat = tree.get("latency")
        ok &= check("验收1 token 消耗与耗时已记录", tokens > 0 and bool(lat),
                    f"total_tokens={tokens} latency={lat}s")
        ok &= check("验收1 意图已写进 trace 元数据",
                    bool(tree.get("metadata", {}).get("intent")),
                    f"metadata.intent={tree.get('metadata', {}).get('intent')}")
        print(f"    trace: http://127.0.0.1:3000/project/dev/traces/{trace_id}")
        return bool(ok)


def scenario_flywheel() -> bool:
    ok = True
    marker = f"会飞的手机多少钱一台{uuid.uuid4().hex[:4]}"
    with httpx.Client(trust_env=False) as c:
        out = chat(c, marker)
        ok &= check("验收2 库外问题拿到兜底话术", "无法回答" in out["reply"], out["reply"][:60])
        time.sleep(3)  # 落池内联标准化/查重(LLM)可能晚于 SSE 收尾
        listed = c.get(f"{BASE}/api/review-queue", timeout=30).json()["items"]
        # 标准化会改写掉 marker,按详情里的用户原话匹配
        hit = []
        for r in listed:
            d = c.get(f"{BASE}/api/review-queue/{r['id']}", timeout=30).json()
            if any(marker in q["raw_question"] for q in d.get("original_questions", [])):
                hit = [r]
                break
        ok &= check("验收2 问题出现在待审队列", bool(hit),
                    f"待审 {len(listed)} 条,命中 {bool(hit)}")
        if not hit:
            return bool(ok)
        detail = c.get(f"{BASE}/api/review-queue/{hit[0]['id']}", timeout=30).json()
        originals = detail.get("original_questions", [])
        chunks = originals[0].get("retrieved_chunks") or [] if originals else []
        ok &= check("验收2 详情含用户原话", any(marker in q["raw_question"] for q in originals),
                    f"原话 {len(originals)} 条")
        ok &= check("验收2 详情含当轮召回片段快照", len(chunks) > 0,
                    json.dumps(chunks[:1], ensure_ascii=False)[:120])

        approved = c.post(f"{BASE}/api/review-queue/{hit[0]['id']}/approve",
                          json={"approved_answer": f"「{marker}」是本店概念商品 SH-FLY01(飞行手机),"
                                                 f"售价 99999 元,仅限到店体验,暂不发售线上订单。"},
                          timeout=120)
        ok &= check("验收3 审核通过写回知识库", approved.status_code == 200,
                    str(approved.json()))
        time.sleep(2)
        out2 = chat(c, marker, out["session_id"])
        # 命中新知识:不再走兜底,引用新入库 chunk 作答(答案应带新核准信息)
        ok &= check("验收3 同题再问不再兜底(飞轮整圈)", "无法回答" not in out2["reply"],
                    out2["reply"][:80])
        ok &= check("验收3 回答引用新核准知识", "99999" in out2["reply"],
                    out2["reply"][:60])
    return bool(ok)


def scenario_feedback() -> bool:
    marker = uuid.uuid4().hex[:6]
    question = f"无人机能带上飞机吗{marker}"
    with httpx.Client(trust_env=False) as c:
        out = chat(c, question)  # 先走一轮,留下当轮检索快照
        fb = c.post(f"{BASE}/api/feedback",
                    json={"session_id": out["session_id"], "question": question,
                          "rating": "down"}, timeout=30)
        ok = check("验收4 👎 反馈被后端受理", fb.status_code == 200 and fb.json()["accepted"])
        time.sleep(3)
        listed = c.get(f"{BASE}/api/review-queue", timeout=30).json()["items"]
        hit = []
        for r in listed:
            d = c.get(f"{BASE}/api/review-queue/{r['id']}", timeout=30).json()
            if any(marker in q["raw_question"] for q in d.get("original_questions", [])):
                hit = [r]
                break
        ok &= check("验收4 标准化查重后出现在待审队列", bool(hit),
                    f"待审 {len(listed)} 条,命中 {bool(hit)}")
        if hit:
            detail = c.get(f"{BASE}/api/review-queue/{hit[0]['id']}", timeout=30).json()
            src = [q["source"] for q in detail.get("original_questions", [])]
            ok &= check("验收4 入池入口标 user_feedback", "user_feedback" in src, str(src))
    return bool(ok)


def scenario_cost() -> bool:
    with httpx.Client(trust_env=False) as c:
        rows = c.get(f"{BASE}/api/usage/by-intent", timeout=30).json()["items"]
        ok = check("验收5 按意图汇总的 token 花销可得", bool(rows), f"{len(rows)} 个意图")
        for row in rows:
            print(f"    意图={row['intent'] or '(未识别)'} 次数={row['request_count']} "
                  f"总token={row['total_tokens']} 均token={row['avg_tokens']:.0f}")
        if len(rows) >= 2:
            top = max(rows, key=lambda r: r["total_tokens"])
            ok &= check("验收5 能看出哪类意图最烧钱", top["total_tokens"] > 0,
                        f"最烧钱意图={top['intent']}")
        return bool(ok)


def scenario_eval() -> bool:
    limit = os.environ.get("ACCEPT_EVAL_LIMIT", "4")
    for i in (1, 2):
        proc = subprocess.run(
            [sys.executable, "scripts/run_eval.py", "--triggered-by", "手动", "--limit", limit],
            cwd=ROOT, capture_output=True, text=True, timeout=900)
        ok_i = proc.returncode == 0
        print(f"{'✅' if ok_i else '❌'} 评估第 {i} 轮落表" +
              (f" —— {proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-200:]}" if not ok_i else ""))
        if not ok_i:
            return False
    with httpx.Client(trust_env=False) as c:
        runs = c.get(f"{BASE}/api/eval-runs", timeout=30).json()["items"]
        ok = check("验收6 至少两轮评估可对比", len(runs) >= 2, f"共 {len(runs)} 轮")
        if len(runs) >= 2:
            last, prev = runs[-1]["metrics"], runs[-2]["metrics"]
            deltas = {k: round(last.get(k, 0) - prev.get(k, 0), 4) for k in last}
            ok &= check("验收6 指标趋势可读", all(isinstance(v, (int, float)) for v in deltas.values()),
                        json.dumps(deltas, ensure_ascii=False))
        return bool(ok)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all",
                        choices=["all", "trace", "flywheel", "feedback", "cost", "eval"])
    args = parser.parse_args()
    scenarios = (["trace", "flywheel", "feedback", "cost", "eval"]
                 if args.scenario == "all" else [args.scenario])
    runners = {"trace": scenario_trace, "flywheel": scenario_flywheel,
               "feedback": scenario_feedback, "cost": scenario_cost, "eval": scenario_eval}
    results = {name: runners[name]() for name in scenarios}
    print("\n== 验收结果 ==")
    for name, ok in results.items():
        print(f"{'✅' if ok else '❌'} {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
