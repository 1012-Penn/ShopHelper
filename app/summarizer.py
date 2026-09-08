"""ch07 后台会话摘要:层 2 攒到超预算,把一批压成一段追加进摘要表;全程不阻塞聊天轮。

一段一行只追加(压完不回炉,一个事实只经历一次有损压缩);旧梗概只作背景给模型看、
不参与合并。触发/开始/完成/跳过/失败全留痕,带覆盖边界与耗时。
"""
import asyncio
import logging
import time

from app.prompts import SUMMARY_PROMPT

logger = logging.getLogger(__name__)

SUMMARY_MAX_CHARS = 300  # 软约束几十到一两百字,超长硬截断兜底(不重问,成本可控)
_ROLE_ZH = {"user": "用户", "assistant": "客服"}


def _render_batch(rows: list[dict]) -> str:
    """摘要批的正文文本:user/assistant 全文(压缩从原文做);tool 残留行一行标识。"""
    lines = []
    for r in rows:
        if r["role"] == "tool":
            lines.append("[工具结果已省略]")
        else:
            lines.append(f"{_ROLE_ZH.get(r['role'], r['role'])}:{r.get('content') or ''}")
    return "\n".join(lines) or "(空)"


class SummaryService:
    """每会话至多一个在飞任务;任务引用挂在 self._tasks 防 GC;进程关停未完成的下轮重触发(锚点单调幂等)。"""

    def __init__(self, store, model) -> None:
        self._store = store
        self._model = model
        self._inflight: set[int] = set()
        self._tasks: set[asyncio.Task] = set()

    def maybe_trigger(self, session_id: int, upto_id: int | None) -> None:
        """同步返回:命中在飞只记 skip;否则投后台任务,批上界 = 触发时的 layer1_from。"""
        if session_id in self._inflight:
            logger.info("[summary] session=%s skip 已有任务在跑", session_id)
            return
        self._inflight.add(session_id)
        task = asyncio.create_task(self._run(session_id, upto_id))
        self._tasks.add(task)
        task.add_done_callback(
            lambda t: (self._tasks.discard(t), self._inflight.discard(session_id)))

    async def _run(self, session_id: int, upto_id: int | None) -> None:
        t0 = time.monotonic()
        try:
            anchors = await self._store.get_anchors(session_id)
            rows = await self._store.get_rows(session_id, upto_id=upto_id)
            rows = [r for r in rows
                    if not anchors["summary_upto"] or r["id"] > anchors["summary_upto"]]
            if not rows:
                logger.info("[summary] session=%s skip 批内无消息", session_id)
                return
            seq = await self._store.next_seq(session_id)
            logger.info("[summary] session=%s start 第%d段 覆盖=[%d,%d]",
                        session_id, seq, rows[0]["id"], rows[-1]["id"])
            prompt = SUMMARY_PROMPT.format(old=anchors["summary_text"] or "(无)",
                                           batch=_render_batch(rows))
            resp = await self._model.ainvoke(prompt)
            text = str(getattr(resp, "content", "") or "").strip()
            if not text:
                raise ValueError("摘要模型返回空")
            if len(text) > SUMMARY_MAX_CHARS:
                text = text[:SUMMARY_MAX_CHARS]
            await self._store.append_summary(session_id, seq=seq, from_id=rows[0]["id"],
                                             upto_id=rows[-1]["id"], content=text)
            logger.info("[summary] session=%s done 第%d段 覆盖=[%d,%d] 耗时=%dms 字数=%d",
                        session_id, seq, rows[0]["id"], rows[-1]["id"],
                        int((time.monotonic() - t0) * 1000), len(text))
        except Exception as exc:
            logger.warning("[summary] session=%s fail err=%s", session_id, exc, exc_info=True)
        finally:
            self._inflight.discard(session_id)
