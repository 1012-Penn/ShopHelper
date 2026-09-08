"""ch08 统一执行引擎:所有工具调用的唯一入口。

管线:查表 → JSON Schema 校验(拦下回灌)→ 写权限把门(直调拒绝+推送确认)→
执行(超时包裹;读操作仅暂时性故障重试,写操作不重试)→ 结果格式化 → 审计落行。
审计写入失败只 warning,绝不拦工具执行。
"""
import asyncio
import json
import logging
import time

import jsonschema

from app.tools.audit import ToolAuditStore
from app.tools.base import ToolRegistryV2

logger = logging.getLogger(__name__)

WRITE_CONFIRM_INSTRUCTION = (
    "写入操作需要用户在前端确认:已推送工单预览卡片,请在本轮答复中引导用户在卡片上"
    "确认或取消,不要重复发起创建。"
)

# 暂时性故障:仅这些才值得重试(超时未必没执行,所以写操作一律不自动重试)
_TRANSIENT_ERRORS = (asyncio.TimeoutError, ConnectionError, OSError)

RESULT_SUMMARY_MAX = 500


def format_result(raw) -> str:
    """结果格式化:MCP 内容块列表取 text 拼接;str 原样;其余 JSON 序列化中文不转义。"""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        return "".join(parts)
    return json.dumps(raw, ensure_ascii=False)


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


class ToolEngine:
    def __init__(self, registry: ToolRegistryV2, audit_store: ToolAuditStore,
                 timeout_seconds: float = 10.0, retry_attempts: int = 1) -> None:
        self.registry = registry
        self.audit = audit_store
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts

    async def execute(self, name: str, args_json: str, *, conversation_id: int | None = None,
                      tool_call_id: str | None = None, on_write_intercept=None,
                      confirmed: bool = False) -> str:
        record = self.registry.get(name)
        t0 = time.monotonic()
        if record is None:
            # 无工具实体可归属(DDL 枚举无中立值),不落审计,直接回灌;记 spec 已知边界
            return f"错误:工具 {name} 未注册"

        def audit(status, *, args=None, result_summary=None, error_message=None, retry_count=0):
            try:
                self.audit.log(
                    conversation_id=conversation_id, tool_call_id=tool_call_id,
                    tool_name=record.name, tool_source=record.source,
                    mcp_server=record.mcp_server, arguments=args,
                    result_summary=result_summary[:RESULT_SUMMARY_MAX] if result_summary else None,
                    status=status, error_message=error_message, retry_count=retry_count,
                    duration_ms=_ms(t0))
            except Exception:
                logger.warning("审计写入失败(不拦工具执行):%s", record.name, exc_info=True)

        # ① 参数解析 + JSON Schema 校验
        try:
            args = json.loads(args_json or "{}")
        except json.JSONDecodeError as exc:
            msg = f"参数不是合法 JSON:{exc};请修正后重新调用"
            audit("校验拦下", error_message=msg)
            return f"错误:工具 {name} {msg}"
        if not isinstance(args, dict):
            msg = "参数必须是 JSON 对象;请修正后重新调用"
            audit("校验拦下", args=None, error_message=msg)
            return f"错误:工具 {name} {msg}"
        try:
            jsonschema.validate(args, record.json_schema)
        except jsonschema.ValidationError as exc:
            path = "/".join(str(p) for p in exc.absolute_path) or "(根)"
            msg = f"参数校验失败 [{path}]:{exc.message};请修正参数后重新调用"
            audit("校验拦下", args=args, error_message=msg)
            return f"错误:工具 {name} {msg}"

        # ② 权限把门:写操作必须经前端确认回传,模型直调一律拒绝
        if record.access == "write" and not confirmed:
            audit("权限拒绝", args=args,
                  error_message="写入操作未获用户确认,已推送确认卡片等待用户操作")
            if on_write_intercept is not None:
                on_write_intercept(record, args)
            return WRITE_CONFIRM_INSTRUCTION

        # ③ 执行:读操作暂时性故障重试;写操作任何情况不自动重试
        retries = 0 if record.access == "write" else max(self.retry_attempts, 0)
        last_error = ""
        for attempt in range(retries + 1):
            try:
                raw = await asyncio.wait_for(record.tool.ainvoke(args),
                                             timeout=self.timeout_seconds)
                result = format_result(raw)
                audit("成功", args=args, result_summary=result, retry_count=attempt)
                return result
            except asyncio.TimeoutError:
                last_error = f"工具 {name} 执行超时({self.timeout_seconds}s)"
            except _TRANSIENT_ERRORS as exc:
                last_error = f"工具 {name} 网络类暂时性故障:{exc}"
            except Exception as exc:  # 真故障:不重试,如实回灌
                last_error = f"工具 {name} 执行失败:{exc}"
                audit("失败", args=args, error_message=last_error, retry_count=attempt)
                return f"错误:{last_error}"
            if attempt < retries:
                logger.warning("%s(第 %d 次尝试失败,重试)", last_error, attempt + 1)

        status = "超时" if "超时" in last_error else "失败"
        audit(status, args=args, error_message=last_error, retry_count=retries)
        return f"错误:{last_error}(已重试 {retries} 次),请如实告知用户该查询暂时不可用"

    async def execute_confirmed(self, name: str, args_json: str, *,
                                conversation_id: int | None = None,
                                tool_call_id: str | None = None) -> str:
        """前端确认回传后的写执行入口:与模型直调同管线,但放行权限门。"""
        return await self.execute(name, args_json, conversation_id=conversation_id,
                                  tool_call_id=tool_call_id, confirmed=True)
