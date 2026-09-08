"""ch08 工具调用审计 Store:统一执行引擎每次调用落一条;不挂外键,写失败由调用方兜。"""
from sqlalchemy.orm import sessionmaker

from app.models import ToolAuditLog

RESULT_SUMMARY_MAX = 500  # result_summary 截断长度


class ToolAuditStore:
    def __init__(self, factory: sessionmaker) -> None:
        self._factory = factory

    def log(self, *, tool_name: str, tool_source: str, status: str,
            conversation_id: int | None = None, tool_call_id: str | None = None,
            mcp_server: str | None = None, arguments: dict | None = None,
            result_summary: str | None = None, error_message: str | None = None,
            retry_count: int = 0, duration_ms: int | None = None) -> None:
        with self._factory() as session:
            session.add(ToolAuditLog(
                conversation_id=conversation_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_source=tool_source,
                mcp_server=mcp_server,
                arguments=arguments,
                result_summary=result_summary[:RESULT_SUMMARY_MAX] if result_summary else None,
                status=status,
                error_message=error_message[:512] if error_message else None,
                retry_count=retry_count,
                duration_ms=duration_ms,
            ))
            session.commit()
