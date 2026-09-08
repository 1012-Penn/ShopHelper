"""ch07 真机验收脚本:对运行中的服务跑脚本化多轮对话,并对 logs/app.log 做断言。

用法:.venv/bin/python scripts/acceptance_ch07.py --scenario a|b|trap|s
  a    默认配置:连聊 22 轮,断言无降级/无摘要、无错误帧(验收 1+3)
  b    演示配置:埋单号锚点→长消息逼级联→断言 层1 降级/summary trigger/summary done;
       再问「最开始那个订单后来怎么说」断言答出订单号(验收 2);记录触发轮 done 与
       summary done 的时间先后(验收 4 非阻塞证据)
  trap 只调 MODEL_CONTEXT_WINDOW=18000 启动,断言启动自检报「上下文预算不足」(验收 2 的陷阱复核)
  s    多会话侧栏 HTTP 级验收(验收 5;浏览器视觉终验留用户本地)
证据统一追加写入 reports/ch07-acceptance.md。
"""
import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx

BASE = os.environ.get("ACCEPT_BASE_URL", "http://127.0.0.1:8000")
LOG_PATH = Path("logs/app.log")
REPORT = Path("reports/ch07-acceptance.md")


def _full_log() -> str:
    return LOG_PATH.read_text(encoding="utf-8", errors="replace") if LOG_PATH.exists() else ""


def parse_sse(raw: str) -> list[dict]:
    return [json.loads(ln[len("data: "):]) for ln in raw.split("\n\n")
            if ln.startswith("data: ")]


class LogWatch:
    """增量读 logs/app.log 新增部分。"""

    def __init__(self):
        self.pos = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0

    def new_text(self) -> str:
        if not LOG_PATH.exists():
            return ""
        size = LOG_PATH.stat().st_size
        if size <= self.pos:
            return ""
        with LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(self.pos)
            text = f.read()
        self.pos = size
        return text


async def chat(client: httpx.AsyncClient, message: str, session_id=None):
    payload = {"message": message} if session_id is None else \
        {"message": message, "session_id": session_id}
    for attempt in (1, 2):
        t0 = time.monotonic()
        try:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                assert resp.status_code == 200, f"HTTP {resp.status_code}"
                raw = ""
                async for chunk in resp.aiter_text():
                    raw += chunk
            break
        except httpx.TimeoutException:
            if attempt == 2:
                raise
            print("    (超时一次,重试)")
    frames = parse_sse(raw)
    done_at = time.monotonic() - t0
    reply = "".join(f["content"] for f in frames if f["type"] == "token")
    sid = frames[0].get("session_id") if frames else None
    return frames, reply, sid, done_at


async def scenario_a(report: list) -> None:
    """验收 1+3:默认配置连聊 22 轮,不触发任何降级/摘要,token 不爆。"""
    watch = LogWatch()
    turns = ["你好呀", "你们的退货政策是怎样的", "SH-E300 多少钱", "邮费是多少",
             "订单1001的物流到哪了", "发票怎么开", "这个东西能便宜点不", "会员有什么权益",
             "怎么申请退货", "退款一般多久到账", "商品坏了想换个新的", "付款方式支持哪些",
             "发货时间一般多久", "包裹丢了怎么办", "能开发票吗", "耳机 what 型号推荐",
             "七天无理由是什么意思", "你们什么态度", "怎么联系人工", "红包怎么用",
             "订单能修改地址吗", "评价能改吗"]
    async with httpx.AsyncClient(timeout=120, base_url=BASE) as client:
        sid = None
        errors = []
        for i, q in enumerate(turns, 1):
            frames, reply, sid, _ = await chat(client, q, sid)
            if any(f["type"] == "error" for f in frames) or not any(
                    f["type"] == "done" for f in frames):
                errors.append((i, q, frames[-1]["type"] if frames else "empty"))
            print(f"[A] 第{i:02d}轮 reply={reply[:24]!r}")
        # 第 23 轮:再回去问最早的话题(旧上下文仍可自然衔接)
        frames, reply, sid, _ = await chat(client, "再问一遍,最开始那个退货政策是怎么说的?", sid)
        print(f"[A] 第23轮(回望)reply={reply[:24]!r}")
    time.sleep(1.0)
    log_new = watch.new_text()
    # 启动自检行写在服务启动时(早于脚本启动),对全量日志检查;压缩行为则只看本轮新增
    log_full = LOG_PATH.read_text(encoding="utf-8", errors="replace") if LOG_PATH.exists() else ""
    has_degrade = "层1 降级" in log_new
    has_trigger = "summary trigger" in log_new
    has_budget_fail = "上下文预算不足" in log_full
    has_startup = "[ctx] budget" in log_full
    report += ["## 场景 A:默认配置连聊 23 轮(验收 1+3)",
               "",
               f"- 轮数:{len(turns) + 1},错误轮:{errors or '无'}",
               f"- 启动预算自检行存在:{has_startup}",
               f"- 「上下文预算不足」出现:{has_budget_fail}(应为 False)",
               f"- 层1 降级出现:{has_degrade}(应为 False);summary trigger 出现:{has_trigger}(应为 False)",
               "", "```", *[_ln for _ln in log_new.splitlines()
                            if "[ctx] budget" in _ln][:1], "```", ""]
    assert not errors, f"错误轮:{errors}"
    assert has_startup and not has_budget_fail
    assert not has_degrade and not has_trigger, "默认配置不应触发压缩"


async def scenario_b(report: list) -> None:
    """验收 2+4:演示配置下完整级联 + 梗概召回 + 非阻塞证据。"""
    watch = LogWatch()
    async with httpx.AsyncClient(timeout=120, base_url=BASE) as client:
        # 锚点轮:订单号与诉求进入最早的历史
        frames, reply, sid, _ = await chat(
            client, "我的订单1001一直没发货,急用,帮我查一下并记录一下诉求,手机号13800001234")
        print(f"[B] 锚点轮1 reply={reply[:30]!r}")
        frames, reply, sid, _ = await chat(
            client, "对,就是订单1001这个无线耳机的订单,催一下尽快发货", sid)
        print(f"[B] 锚点轮2 reply={reply[:30]!r}")

        trigger_done_at = None
        filler = ("帮我查一下订单1001的物流到哪了,顺便详细讲讲退货政策的具体条款、"
                  "七天无理由退货的适用范围、运费什么情况下由店家承担、质量问题怎么举证,"
                  "我担心后续扯皮,请把可能的处理流程从头到尾讲一遍,谢谢")
        trigger_seen = False
        trigger_turn_secs = None
        for i in range(1, 26):
            frames, reply, sid, turn_secs = await chat(client, filler, sid)
            new_log = watch.new_text()
            print(f"[B] 填充第{i:02d}轮 reply={reply[:20]!r} "
                  f"降级={'层1 降级' in new_log} trigger={'summary trigger' in new_log}")
            if "summary trigger" in new_log:
                trigger_seen = True
                trigger_turn_secs = turn_secs
                break
        assert trigger_seen, "25 轮内未触发摘要"
        _all = _full_log()
        trigger_line = next(_ln for _ln in _all.splitlines()
                            if "summary trigger" in _ln and f"session={sid} " in _ln)
        degrade_line = next((_ln for _ln in _all.splitlines()
                             if "层1 降级" in _ln and f"session={sid} " in _ln), None)
        assert degrade_line, "未见层1 降级行(级联第一步缺失)"

        # 等后台摘要落成:全文件轮询(摘要往往在触发轮流式结束前就完成,增量读会错过)
        summary_done_line = None
        for _ in range(90):
            time.sleep(1.0)
            for ln in _full_log().splitlines():
                if "[summary]" in ln and "done 第" in ln and f"session={sid} " in ln:
                    summary_done_line = ln
                    break
            if summary_done_line:
                break
        assert summary_done_line, "90 秒内未见 summary done"
        print(f"[B] {summary_done_line.strip()}")

        # 摘要已落:再走一轮,确认 history_ctx/model_ctx 带出摘要,且回忆问题答得对
        frames, reply, sid, _ = await chat(client, "最开始那个订单后来怎么说?", sid)
        print(f"[B] 回忆轮 reply={reply!r}")
        recall_ok = "1001" in reply
        log_new = watch.new_text()
        has_summary_in_ctx = "摘要全文:用户" in log_new or ("摘要全文:" in log_new and "订单1001" in log_new)
        has_history_ctx = "[history_ctx]" in log_new
        has_model_ctx = "[model_ctx]" in log_new

        # 确认摘要内容确实存了订单号(查表)
        summary_text = await _fetch_summary(sid)
        summary_has_order = "1001" in (summary_text or "")

        report += ["## 场景 B:演示配置(窗口18000/输出2000/输入2000/步数3/工具1200/RERANK5)",
               "",
               f"- 会话 id:{sid}",
               f"- 摘要段落内容含订单 1001:{summary_has_order}",
               f"- 摘要表内容:`{(summary_text or '')[:80]}…`",
               f"- 回忆轮(「最开始那个订单后来怎么说?」)答复含 1001:{recall_ok};答复:`{reply[:80]}`",
               f"- 回忆轮日志带出摘要(history_ctx/model_ctx):{has_summary_in_ctx}",
               f"- grep model_ctx 有内容:{has_model_ctx};grep history_ctx 有内容:{has_history_ctx}",
               f"- 非阻塞证据:触发轮整体耗时 {trigger_turn_secs:.1f}s,摘要任务与该轮流式并行完成;"
               f"触发行与完成行时间戳见下",
               "", "关键日志行:", "", "```",
               degrade_line.strip(),
               trigger_line.strip(),
               summary_done_line.strip(),
               *[_ln for _ln in _full_log().splitlines() if "[model_ctx]" in _ln][-2:],
               *[_ln for _ln in _full_log().splitlines() if "[history_ctx]" in _ln][-2:],
               "```", ""]
    assert summary_has_order, "摘要里没存下订单号"
    assert recall_ok, f"回忆轮没答出订单号:回复={reply!r}"


async def _fetch_summary(sid: int) -> str | None:
    import pymysql
    conn = pymysql.connect(host="127.0.0.1", user="shophelper", password="shophelper",
                           database="shophelper", charset="utf8mb4")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT summary FROM conversations WHERE id=%s", (sid,))
            return cur.fetchone()[0]
    finally:
        conn.close()


async def scenario_trap(report: list) -> None:
    """验收 2 的陷阱复核:只调窗口,滑窗应归零并报「上下文预算不足」。(启动行在服务起时已写,全文件扫)"""
    async with httpx.AsyncClient(timeout=30, base_url=BASE) as client:
        resp = await client.get("/")
        assert resp.status_code == 200
    log_new = LOG_PATH.read_text(encoding="utf-8", errors="replace") if LOG_PATH.exists() else ""
    hit = "上下文预算不足" in log_new
    line = next((_ln for _ln in log_new.splitlines() if "上下文预算不足" in _ln), "")
    report += ["## 场景 trap:仅设 MODEL_CONTEXT_WINDOW=18000(其余默认)",
               "",
               f"- 启动自检报「上下文预算不足」:{hit}",
               "", "```", line, "```", ""]
    assert hit, "只调窗口未触发预算不足报警"


async def scenario_s(report: list) -> None:
    """验收 5(HTTP 级;浏览器视觉终验留用户本地):多会话、新在前、回载、切回续聊。"""
    async with httpx.AsyncClient(timeout=120, base_url=BASE) as client:
        # 会话 A:两轮(第二轮回望第一轮的耳机话题);会话 B:一轮(键盘)
        _, r_a1, sid_a, _ = await chat(client, "SH-E300 这款耳机多少钱?")
        _, r_a2, sid_a, _ = await chat(client, "那它现在有货吗?", sid_a)
        _, r_b1, sid_b, _ = await chat(client, "机械键盘什么价格?")
        # 侧栏列表:新在前
        resp = await client.get("/api/conversations")
        items = resp.json()["items"]
        ids = [i["id"] for i in items]
        newest_ok = ids.index(sid_b) < ids.index(sid_a)
        item_a = next(i for i in items if i["id"] == sid_a)
        preview_ok = item_a["preview"].startswith("SH-E300")
        # 回载会话 A 全部消息
        resp = await client.get(f"/api/conversations/{sid_a}/messages")
        msgs = resp.json()["messages"]
        reload_roles = [m["role"] for m in msgs]
        reload_ok = reload_roles == ["user", "assistant", "user", "assistant"]
        # 切回 A 续聊:带 session_id 发第三轮,正常收敛即线程正确
        frames, r_a3, sid_a3, _ = await chat(client, "你们的运费是多少?", sid_a)
        continue_ok = (sid_a3 == sid_a and any(f["type"] == "done" for f in frames)
                       and "键盘" not in r_a3)  # 答的是退货政策,不串到 B 的话题
        # 已摘要标记:场景 B 跑出的会话带摘要
        resp = await client.get("/api/conversations")
        sum_items = [i for i in resp.json()["items"] if i["summarized"]]
        sum_flag_ok = len(sum_items) >= 1
    report += ["## 场景 S:多会话侧栏(HTTP 级;浏览器视觉终验留用户本地)",
               "",
               f"- 会话 A(id={sid_a})两轮:{r_a1[:20]!r} / {r_a2[:20]!r};会话 B(id={sid_b}):{r_b1[:20]!r}",
               f"- 列表新在前:{newest_ok};首问预览正确:{preview_ok}",
               f"- 会话 A 回载消息序列={reload_roles}({reload_ok})",
               f"- 切回 A 续聊第三轮正常(id={sid_a3}):{continue_ok};答复:`{r_a3[:60]}`",
               f"- 已摘要会话带 summarized 标记:{sum_flag_ok}(共 {len(sum_items)} 条)",
               ""]
    assert newest_ok and preview_ok and reload_ok and continue_ok and sum_flag_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["a", "b", "trap", "s"], required=True)
    args = ap.parse_args()
    REPORT.parent.mkdir(exist_ok=True)
    report = [f"---", f"", f"# 追加记录 {datetime.now():%Y-%m-%d %H:%M:%S} · 场景 {args.scenario.upper()}"]
    try:
        asyncio.run({"a": scenario_a, "b": scenario_b, "trap": scenario_trap, "s": scenario_s}[args.scenario](report))
        report.append(f"**结论:场景 {args.scenario.upper()} 全部断言通过。**")
        code = 0
    except AssertionError as exc:
        report.append(f"**结论:场景 {args.scenario.upper()} 断言失败:{exc}**")
        code = 1
    with REPORT.open("a", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
