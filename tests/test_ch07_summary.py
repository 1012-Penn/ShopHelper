"""ch07 后台摘要:段落追加、锚点推进、旧梗概作背景、超长截断、失败不抛、批上界。"""
import logging

import pytest

from app.summarizer import SummaryService

pytestmark = pytest.mark.asyncio


class StubSummaryModel:
    def __init__(self, text="用户询问订单1001的发货时间,诉求是尽快发货。"):
        self.text = text
        self.prompts: list[str] = []

    async def ainvoke(self, prompt):
        self.prompts.append(str(prompt))
        return type("R", (), {"content": self.text})()


class BoomModel:
    async def ainvoke(self, prompt):
        raise RuntimeError("上游挂了")


async def _seed(db_store, msgs=None):
    sid = await db_store.resolve(None)
    await db_store.append(sid, msgs or [{"role": "user", "content": "订单1001多久发货"},
                                        {"role": "assistant", "content": "一般48小时内发出"}])
    rows = await db_store.get_rows(sid)
    return sid, rows


async def test_run_appends_segment_and_advances_anchor(db_store):
    sid, rows = await _seed(db_store)
    await db_store.set_layer1_from(sid, rows[-1]["id"])
    svc = SummaryService(db_store, StubSummaryModel(), None)
    await svc._run(sid, upto_id=rows[-1]["id"])
    a = await db_store.get_anchors(sid)
    assert a["summary_seqs"] == 1 and a["summary_upto"] == rows[-1]["id"]
    assert "订单1001" in a["summary_text"]


async def test_second_run_appends_next_segment(db_store):
    sid, rows = await _seed(db_store)
    svc = SummaryService(db_store, StubSummaryModel(), None)
    await svc._run(sid, upto_id=rows[0]["id"])
    await svc._run(sid, upto_id=rows[-1]["id"])  # 第二段只补增量
    a = await db_store.get_anchors(sid)
    assert a["summary_seqs"] == 2 and a["summary_upto"] == rows[-1]["id"]
    assert len(a["summary_text"].split("\n")) == 2


async def test_prompt_carries_old_summary_and_batch(db_store, caplog):
    sid, rows = await _seed(db_store)
    model = StubSummaryModel()
    svc = SummaryService(db_store, model, None)
    with caplog.at_level(logging.INFO, logger="app.summarizer"):
        await svc._run(sid, upto_id=rows[-1]["id"])
    assert "已有梗概(背景,勿重复):\n(无)" in model.prompts[0]
    assert "本批对话" in model.prompts[0] and "订单1001多久发货" in model.prompts[0]
    lines = [r.getMessage() for r in caplog.records]
    assert any("start 第1段" in ln for ln in lines) and any("done 第1段" in ln for ln in lines)


async def test_upto_id_bounds_batch(db_store):
    sid, rows = await _seed(db_store, [{"role": "user", "content": "第一条"},
                                       {"role": "assistant", "content": "回复"},
                                       {"role": "user", "content": "第二条最新"}])
    model = StubSummaryModel()
    svc = SummaryService(db_store, model, None)
    await svc._run(sid, upto_id=rows[1]["id"])  # 只压到第二条之前
    assert "第二条最新" not in model.prompts[0]
    assert (await db_store.get_anchors(sid))["summary_upto"] == rows[1]["id"]


async def test_overlong_output_hard_truncated(db_store):
    sid, rows = await _seed(db_store, [{"role": "user", "content": "u"}])
    svc = SummaryService(db_store, StubSummaryModel(text="长" * 400), None)
    await svc._run(sid, upto_id=rows[-1]["id"])
    assert len((await db_store.get_anchors(sid))["summary_text"]) <= 300


async def test_upstream_failure_logged_not_raised(db_store, caplog):
    sid, rows = await _seed(db_store, [{"role": "user", "content": "u"}])
    svc = SummaryService(db_store, BoomModel(), None)
    with caplog.at_level(logging.WARNING, logger="app.summarizer"):
        await svc._run(sid, upto_id=rows[-1]["id"])  # 不抛
    assert any("fail" in r.message for r in caplog.records)
    assert (await db_store.get_anchors(sid))["summary_seqs"] == 0


async def test_empty_batch_skips(db_store, caplog):
    sid = await db_store.resolve(None)
    svc = SummaryService(db_store, StubSummaryModel(), None)
    with caplog.at_level(logging.INFO, logger="app.summarizer"):
        await svc._run(sid, upto_id=None)
    assert any("skip" in r.message for r in caplog.records)


async def test_maybe_trigger_skips_when_inflight(db_store, caplog):
    sid, rows = await _seed(db_store, [{"role": "user", "content": "u"}])
    svc = SummaryService(db_store, StubSummaryModel(), None)
    svc._inflight.add(sid)  # 模拟在飞
    with caplog.at_level(logging.INFO, logger="app.summarizer"):
        svc.maybe_trigger(sid, upto_id=rows[-1]["id"])
    assert any("skip 已有任务在跑" in r.message for r in caplog.records)
    assert await db_store.next_seq(sid) == 1  # 没有新段落产生
