"""进程内会话存储:单进程假设,asyncio.Lock 保护并发读写。"""
import asyncio
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class SessionData:
    messages: list[dict] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._data: dict[str, SessionData] = {}
        self._lock = asyncio.Lock()

    async def resolve(self, session_id: str | None) -> str:
        """返回可用 session_id:缺省则新建;客户端给的 id(含未知)一并注册。"""
        async with self._lock:
            if session_id is None:
                session_id = uuid4().hex
            self._data.setdefault(session_id, SessionData())
            return session_id

    async def get(self, session_id: str) -> SessionData | None:
        async with self._lock:
            return self._data.get(session_id)

    async def append(self, session_id: str, user_content: str, assistant_content: str) -> None:
        """一轮对话结束后成对落库;id 不存在(异常时序)则静默跳过。"""
        async with self._lock:
            data = self._data.get(session_id)
            if data is None:
                return
            data.messages.append({"role": "user", "content": user_content})
            data.messages.append({"role": "assistant", "content": assistant_content})


session_store = SessionStore()  # 进程级单例(生产入口用;测试自建实例注入 app.state)
