"""建工单端点:按钮建单(点按钮即用户确认)+ ch08 建工单确认流(预览卡确认/取消)。

ch12:两端点都支持 confirm_key 幂等;同键但会话/描述/类型不一致 → 409(ADR-3),
绝不把旧工单当成功返回。
"""
import json

from fastapi import APIRouter, HTTPException, Request

from app.schemas import TicketCancelRequest, TicketConfirmRequest, TicketRequest
from app.tickets import CONFIRM_KEY_CONFLICT

router = APIRouter()


def _raise_on_conflict(parsed) -> None:
    if isinstance(parsed, dict) and parsed.get("error") == CONFIRM_KEY_CONFLICT:
        raise HTTPException(status_code=409,
                            detail=parsed.get("detail") or "确认键与已建工单不一致")


@router.post("/api/tickets")
async def create_ticket(body: TicketRequest, request: Request):
    """ch05 按钮建单:点按钮本身就是用户确认,引擎以 confirmed 放行(审计「成功」)。
    ch12:confirm_key 代表一次点击(双击/超时重试同键,再点新键),同键返回同一工单(ADR-3)。"""
    store, engine = request.app.state.store, request.app.state.engine
    if body.session_id is not None and not await store.exists(body.session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    conversation_id = await store.resolve(body.session_id)
    raw = await engine.execute("create_ticket", json.dumps({
        "conversation_id": conversation_id, "description": body.description,
        "ticket_type": body.ticket_type, "confirm_key": body.confirm_key,
    }, ensure_ascii=False), conversation_id=conversation_id, confirmed=True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None  # 引擎失败收敛为错误字符串,不是 JSON → 与下方同归 502
    _raise_on_conflict(parsed)
    if not isinstance(parsed, dict) or "ticket_no" not in parsed:
        raise HTTPException(status_code=502, detail="创建工单失败")
    return parsed


@router.post("/api/tickets/confirm")
async def confirm_ticket(body: TicketConfirmRequest, request: Request):
    """ch08 确认流:前端「确认提交」回传,引擎以 confirmed 放行真实执行。

    收口(I-4):仅「已注册且为写操作」的工具可走确认通道——确认即用户授权;
    create_ticket(缺省)用工单字段组装参数,其他写工具原样透传 arguments,引擎侧仍做 Schema 校验。
    """
    store, engine = request.app.state.store, request.app.state.engine
    registry = request.app.state.registry
    if not await store.exists(body.conversation_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    record = registry.get(body.tool_name)
    if record is None or record.access != "write":
        raise HTTPException(status_code=422, detail="该工具不可经确认通道执行")
    if body.tool_name == "create_ticket":
        args = {"conversation_id": body.conversation_id,
                "description": body.description, "ticket_type": body.ticket_type,
                "confirm_key": body.confirm_key}
        if not body.description:
            raise HTTPException(status_code=422, detail="缺少问题描述")
    else:
        args = body.arguments
    ledger = getattr(request.app.state, "action_ledger", None)
    known_action = (
        ledger.latest_for_confirm_key(body.confirm_key)
        if ledger is not None and body.confirm_key else None
    )
    if known_action is not None:
        if (
            known_action.conversation_id != body.conversation_id
            or known_action.tool_name != body.tool_name
            or known_action.state in {"cancelled", "failed"}
        ):
            raise HTTPException(
                status_code=409,
                detail="确认键不属于当前会话，或对应预览已取消/失败",
            )
        expected = known_action.arguments or {}
        if body.tool_name == "create_ticket" and (
            expected.get("description") != body.description
            or expected.get("ticket_type") != body.ticket_type
        ):
            raise HTTPException(
                status_code=409,
                detail="确认参数与已登记预览不一致",
            )
    raw = await engine.execute(body.tool_name,
                               json.dumps(args, ensure_ascii=False),
                               conversation_id=body.conversation_id, confirmed=True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    _raise_on_conflict(parsed)
    if body.tool_name == "create_ticket" and (
            not isinstance(parsed, dict) or "ticket_no" not in parsed):
        raise HTTPException(status_code=502, detail="创建工单失败")
    if (
        ledger is not None
        and body.tool_name == "create_ticket"
        and body.confirm_key
    ):
        pending = ledger.pending(body.conversation_id)
        if pending is not None and pending.confirm_key == body.confirm_key:
            ledger.mark_confirmed(
                conversation_id=body.conversation_id,
                confirm_key=body.confirm_key,
                ticket=parsed,
            )
    return parsed if isinstance(parsed, dict) else {"result": raw}


@router.post("/api/tickets/cancel")
async def cancel_ticket(body: TicketCancelRequest, request: Request):
    """ch08 建工单确认流:用户点「取消」→ 工单不建,本次调用按「权限拒绝」落审计。"""
    store, audit = request.app.state.store, request.app.state.tool_audit_store
    if not await store.exists(body.conversation_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    audit.log(tool_name="create_ticket", tool_source="builtin",
              conversation_id=body.conversation_id, status="权限拒绝",
              error_message=body.reason or "用户取消确认")
    if not request.app.state.settings.multi_agent_action_contracts_enabled:
        return {"ok": True}
    ledger = getattr(request.app.state, "action_ledger", None)
    if ledger is None:
        return {"ok": True}
    artifact = ledger.cancel(body.conversation_id)[-1]
    return {"ok": True, "action_artifact": artifact.model_dump(mode="json")}
