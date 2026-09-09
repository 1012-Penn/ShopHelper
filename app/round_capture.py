"""ch09 回捞:当轮召回片段快照的会话级暂存,供 👎 反馈事后取用。

进程内缓存:重启即失,回捞不到按「尽力回捞」落 NULL;确认走过检索的当轮
(含闲聊轮的空快照)才会被记录,空快照用于区分「真没走检索」。
"""
from collections import OrderedDict, deque


def _norm(question: str) -> str:
    return " ".join((question or "").split())


class RoundCapture:
    def __init__(self, max_sessions: int = 256, rounds_per_session: int = 5) -> None:
        self._max_sessions = max_sessions
        self._rounds = rounds_per_session
        self._sessions: OrderedDict[int, deque[tuple[str, list[dict]]]] = OrderedDict()

    def record(self, conversation_id: int, question: str, snapshot: list[dict]) -> None:
        rounds = self._sessions.get(conversation_id)
        if rounds is None:
            rounds = deque(maxlen=self._rounds)
            self._sessions[conversation_id] = rounds
            self._touch(conversation_id)
        rounds.append((_norm(question), list(snapshot or [])))

    def lookup(self, conversation_id: int, question: str) -> list[dict] | None:
        rounds = self._sessions.get(conversation_id)
        if rounds is None:
            return None
        self._touch(conversation_id)
        key = _norm(question)
        for stored_question, snapshot in reversed(rounds):
            if stored_question == key:
                return snapshot
        return None

    def _touch(self, conversation_id: int) -> None:
        self._sessions.move_to_end(conversation_id)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
