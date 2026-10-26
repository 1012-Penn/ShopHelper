"""Schema-derived action contracts and deterministic user-intent checks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jsonschema

from app.action_contracts.models import (
    ActionArgumentDecision,
    ActionArtifact,
    ActionIntent,
    NormalizationRecord,
)

_AFTER_SALES_TYPES = {
    "售后", "退款", "退货", "维修", "返修", "换货", "退款处理", "维修处理",
    "refund", "repair", "return",
}
_COMPLAINT_TYPES = {"投诉", "complaint"}
_CONSULT_TYPES = {"咨询", "inquiry", "consultation"}

_FEASIBILITY = re.compile(
    r"(?:能|可以|能否|是否|可不可以|会不会).{0,10}取消"
    r"|取消.{0,10}(?:吗|么|嘛|可以|可行)"
)
_CANCEL_ACTION = re.compile(
    r"(?:算了|不用了|不要了|取消|撤销|停止|不建|别建|不需要).{0,24}"
    r"(?:工单|预览|刚才|这个|算了)"
    r"|(?:工单|预览).{0,16}(?:取消|撤销)"
)
_CONFIRM = re.compile(r"^(?:确认|我确认|确认创建|确认提交|提交工单)[。！!，,\s]*$")
_TICKET = re.compile(r"(?:工单|ticket)", re.IGNORECASE)
_CREATE = re.compile(r"(?:创建|生成|建立|建一个|建个|开一个|开个|起草|处理草稿)")
_PREVIEW = re.compile(r"(?:预览|草稿|先给我看)")


def infer_action_intent(
    message: str,
    *,
    pending: ActionArtifact | None = None,
) -> ActionIntent:
    """Classify only explicit action language; factual questions stay non-actions."""
    text = re.sub(r"\s+", "", message or "")
    if _FEASIBILITY.search(text):
        return ActionIntent(reason="cancel_feasibility_question")
    if pending is not None and re.search(
        r"(?:不要|不再|先不|暂不).{0,6}(?:提交|创建|处理)", text
    ):
        return ActionIntent(
            kind="ticket_cancel", reason="explicit_cancel_pending", explicit=True,
        )
    if _CANCEL_ACTION.search(text):
        return ActionIntent(
            kind="ticket_cancel", reason="explicit_cancel", explicit=True,
        )
    if pending is not None and _CONFIRM.search(text):
        return ActionIntent(
            kind="ticket_confirm", source="pending_state",
            reason="explicit_client_confirmation", explicit=True,
        )
    if _TICKET.search(text) and _CREATE.search(text) and not _FEASIBILITY.search(text):
        return ActionIntent(
            kind="ticket_preview", reason="explicit_ticket_creation", explicit=True,
        )
    if pending is not None and _PREVIEW.search(text) \
            and re.search(r"(?:改|重新|再|更新)", text):
        return ActionIntent(
            kind="ticket_preview", reason="explicit_preview_revision", explicit=True,
        )
    if pending is None and _TICKET.search(text) and _PREVIEW.search(text):
        return ActionIntent(
            kind="ticket_preview", reason="explicit_ticket_preview", explicit=True,
        )
    return ActionIntent(reason="no_explicit_write_action")


@dataclass(frozen=True)
class ActionToolContext:
    """The model-visible parameter contract read from the active registry."""

    name: str
    description: str
    schema: dict[str, Any]
    required_parameters: frozenset[str]
    allowed_parameters: frozenset[str]
    enum_values: dict[str, frozenset[str]]

    @classmethod
    def from_registry(cls, registry, name: str) -> ActionToolContext:
        record = registry.get(name)
        if record is None:
            raise ValueError(f"tool is not registered: {name}")
        schema = dict(record.json_schema or {})
        properties = schema.get("properties") or {}
        enums: dict[str, frozenset[str]] = {}
        for field, definition in properties.items():
            if isinstance(definition, Mapping) and isinstance(definition.get("enum"), list):
                enums[field] = frozenset(str(item) for item in definition["enum"])
        return cls(
            name=name,
            description=getattr(record.tool, "description", "") or "",
            schema=schema,
            required_parameters=frozenset(str(item) for item in schema.get("required", [])),
            allowed_parameters=frozenset(str(item) for item in properties),
            enum_values=enums,
        )

    def model_context(self) -> dict[str, Any]:
        """Small JSON-safe projection for prompts and repair feedback."""
        return {
            "tool_name": self.name,
            "description": self.description,
            "required_parameters": sorted(self.required_parameters),
            "allowed_parameters": sorted(self.allowed_parameters),
            "enum_values": {
                key: sorted(values) for key, values in self.enum_values.items()
            },
        }

    def prepare_arguments(
        self,
        raw_arguments: Mapping[str, Any],
        *,
        conversation_id: int | None,
        task_id: str | None = None,
    ) -> ActionArgumentDecision:
        original = dict(raw_arguments or {})
        args = dict(original)
        normalizations: list[NormalizationRecord] = []
        feedback: list[str] = []

        supplied_conversation = args.get("conversation_id")
        if conversation_id is None:
            feedback.append("当前运行没有可信的 conversation_id，不能生成写预览。")
        else:
            args["conversation_id"] = conversation_id
            if supplied_conversation != conversation_id:
                normalizations.append(NormalizationRecord(
                    field="conversation_id",
                    original=supplied_conversation,
                    normalized=conversation_id,
                    reason="trusted_run_scope",
                ))

        model_key = args.pop("confirm_key", None)
        if model_key not in (None, ""):
            normalizations.append(NormalizationRecord(
                field="confirm_key",
                original=model_key,
                normalized=None,
                reason="confirmation_key_is_server_generated",
            ))

        unknown = sorted(set(args) - self.allowed_parameters)
        if unknown:
            feedback.append(f"不允许的参数: {', '.join(unknown)}")
        for key in unknown:
            args.pop(key, None)

        if "description" in args:
            args["description"] = str(args.get("description") or "").strip()
        if not str(args.get("description") or "").strip():
            feedback.append("description 必须包含明确的业务问题描述，不能留空。")

        if "ticket_type" in args:
            raw_type = str(args.get("ticket_type") or "").strip()
            normalized_type, reason = _normalize_ticket_type(raw_type)
            if normalized_type is None:
                allowed = sorted(self.enum_values.get("ticket_type", ()))
                feedback.append(
                    "ticket_type 不明确或非法；"
                    f"请根据用户明确诉求选择 {allowed} 之一，不能自行猜测。"
                )
            else:
                args["ticket_type"] = normalized_type
                if raw_type != normalized_type:
                    normalizations.append(NormalizationRecord(
                        field="ticket_type",
                        original=raw_type,
                        normalized=normalized_type,
                        reason=reason,
                    ))
        else:
            feedback.append("缺少 ticket_type，不能默认猜测用户诉求。")

        missing = sorted(
            field for field in self.required_parameters
            if field not in args or args[field] in (None, "")
        )
        for field in missing:
            message = f"缺少必填参数 {field}。"
            if message not in feedback:
                feedback.append(message)

        decision = ActionArgumentDecision(
            ok=not feedback,
            arguments=args,
            original_arguments=original,
            normalizations=normalizations,
            feedback=" ".join(feedback),
            needs_clarification=bool(feedback),
        )
        if not decision.ok:
            return decision
        try:
            jsonschema.validate(args, self.schema)
        except jsonschema.ValidationError as exc:
            path = "/".join(str(item) for item in exc.absolute_path) or "(root)"
            return ActionArgumentDecision(
                ok=False,
                original_arguments=original,
                normalizations=normalizations,
                feedback=f"参数 {path} 不满足工具 Schema: {exc.message}",
                needs_clarification=True,
            )
        return decision

    def repair_feedback(self, decision: ActionArgumentDecision) -> str:
        return (
            "ACTION_CONTRACT_FEEDBACK: 本轮没有生成结构化预览。"
            f"{decision.feedback} "
            f"工具契约={self.model_context()}。"
            "如信息完整，请恰好调用一次 create_ticket 生成预览；"
            "不得声称已经创建，不得填写 confirm_key。"
        )

    def missing_preview_feedback(self) -> str:
        return (
            "ACTION_CONTRACT_FEEDBACK: 你声称已有预览，但本轮没有任何结构化 "
            "create_ticket 预览 artifact。若用户已明确要求工单且参数完整，"
            "请恰好调用一次 create_ticket；否则只要求用户补充缺失信息。"
        )


def _normalize_ticket_type(value: str) -> tuple[str | None, str]:
    folded = value.strip().casefold()
    if folded in {item.casefold() for item in _AFTER_SALES_TYPES}:
        reason = (
            "unambiguous_refund_is_after_sales"
            if any(term in folded for term in ("退款", "退货", "退", "refund", "return"))
            else "unambiguous_repair_is_after_sales"
            if any(term in folded for term in ("维修", "返修", "换货", "repair"))
            else "already_canonical"
        )
        return "售后", reason
    if folded in {item.casefold() for item in _COMPLAINT_TYPES}:
        return "投诉", "already_canonical"
    if folded in {item.casefold() for item in _CONSULT_TYPES}:
        return "咨询", "already_canonical"
    return None, "ambiguous_ticket_type"


def guard_final_reply(
    tentative: str,
    *,
    intent: ActionIntent,
    artifacts: list[ActionArtifact],
) -> str:
    """Prevent prose from contradicting the observable action artifact."""
    text = tentative or ""
    awaiting = [
        item for item in artifacts if item.state == "awaiting_confirmation"
    ]
    confirmed = [item for item in artifacts if item.state == "confirmed"]
    failed = [item for item in artifacts if item.state == "failed"]

    if intent.kind == "ticket_preview" and not awaiting and failed:
        detail = failed[-1]
        if detail.error_code == "clarification_required":
            return (
                "暂不能生成工单预览。请补充明确的业务问题描述，并选择工单类型："
                "售后、投诉或咨询。当前没有创建或提交任何工单。"
            )
        return (
            "未能生成结构化工单预览。请补充或确认明确的业务问题描述和工单类型"
            "（售后、投诉、咨询）；当前没有创建或提交任何工单。"
        )
    if (
        intent.kind == "ticket_preview"
        and awaiting
        and not confirmed
        and re.search(r"(?:已经创建|已创建|创建成功|提交成功|已经提交)", text)
    ):
        return "工单预览已生成，尚未创建或提交，等待你在确认卡片上明确确认。"
    if intent.kind == "ticket_cancel" and not awaiting:
        return "已取消待确认的动作；当前没有待确认工单，也没有提交任何写入。"
    if intent.kind == "ticket_confirm" and not confirmed:
        return "尚未提交工单。请使用预览卡片的明确确认操作；模型文本不能代替客户端确认。"
    if intent.kind == "none" and not confirmed and re.search(
            r"(?:已经创建|已创建|创建成功|提交成功|预览已生成|预览已经生成|"
            r"预览草稿已生成)",
            text,
    ):
        return "当前请求没有生成可确认的工单预览，也未创建或提交任何工单。"
    return text
