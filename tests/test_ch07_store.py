"""ch07 store 扩展:行带 id、锚点读写、分段摘要追加与投影重拼、多会话列表。"""
import pytest

pytestmark = pytest.mark.asyncio


async def test_get_rows_carries_ids_ordered(db_store):
    sid = await db_store.resolve(None)
    await db_store.append(sid, [{"role": "user", "content": "问"},
                                {"role": "assistant", "content": "答"}])
    rows = await db_store.get_rows(sid)
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    assert rows[0]["role"] == "user" and rows[0]["content"] == "问"


async def test_get_rows_upto_id_bounds_batch(db_store):
    sid = await db_store.resolve(None)
    await db_store.append(sid, [{"role": "user", "content": "u1"},
                                {"role": "assistant", "content": "a1"},
                                {"role": "user", "content": "u2"}])
    rows = await db_store.get_rows(sid)
    batch = await db_store.get_rows(sid, upto_id=rows[1]["id"])
    assert len(batch) == 2 and batch[-1]["content"] == "a1"


async def test_anchors_default_none(db_store):
    sid = await db_store.resolve(None)
    a = await db_store.get_anchors(sid)
    assert a["summary_upto"] is None and a["layer1_from"] is None
    assert a["summary_text"] is None and a["summary_seqs"] == 0


async def test_layer1_anchor_roundtrip(db_store):
    sid = await db_store.resolve(None)
    await db_store.set_layer1_from(sid, 42)
    assert (await db_store.get_anchors(sid))["layer1_from"] == 42


async def test_append_summary_advances_anchor_and_projection(db_store):
    sid = await db_store.resolve(None)
    await db_store.append(sid, [{"role": "user", "content": "u1"},
                                {"role": "assistant", "content": "a1"}])
    rows = await db_store.get_rows(sid)
    await db_store.append_summary(sid, seq=1, from_id=rows[0]["id"],
                                  upto_id=rows[1]["id"], content="梗概一")
    await db_store.append_summary(sid, seq=2, from_id=rows[1]["id"] + 1,
                                  upto_id=rows[1]["id"] + 2, content="梗概二")
    a = await db_store.get_anchors(sid)
    assert a["summary_upto"] == rows[1]["id"] + 2
    assert a["summary_seqs"] == 2
    assert a["summary_text"] == "梗概一\n梗概二"
    assert await db_store.next_seq(sid) == 3


async def test_list_conversations_newest_first_with_preview_and_flag(db_store):
    s1 = await db_store.resolve(None)
    await db_store.append(s1, [{"role": "user", "content": "第一通的首问内容"}])
    s2 = await db_store.resolve(None)
    await db_store.append(s2, [{"role": "user", "content": "第二通的首问内容"}])
    await db_store.append_summary(s2, seq=1, from_id=1, upto_id=1, content="梗概")
    items = await db_store.list_conversations()
    assert [i["id"] for i in items][:2] == [s2, s1]  # 新在前
    assert items[0]["preview"] == "第二通的首问内容"
    assert items[0]["summarized"] is True and items[1]["summarized"] is False
