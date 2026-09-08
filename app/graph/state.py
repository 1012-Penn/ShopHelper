"""ch05 图骨架:State 一路贯穿,checkpointer 只承载图内运行态(历史真源在 MySQL store)。"""
from typing import TypedDict


class ChatState(TypedDict):
    session_id: int
    user_message: str
    resolved_message: str
    intent: str
    evidence: list[dict]          # 检索证据(带引用编号 n),知识类才有
    low_confidence: bool          # 置信度闸输入(检索层判定)
    low_reason: str
    gate_passed: bool
    agent_steps: int              # ReAct 实际步数(验收 5 观察)
    final_reply: str
    suggested_actions: list[str]  # "transfer_human" / "create_ticket" 子集
    trace: list[str]              # 每节点一行,log 节点逐行打日志(验收 1 靠它)
    messages: list                # 本轮新增的落库 pending(role/content/tool_calls/tool_call_id)
    history: list                 # 裁剪后的历史(dict),由路由层注入
    # ---- ch06:分流器正式版 ----
    intent_confidence: float      # 意图置信度(trace/评估用)
    order: dict                   # 子流程拿到的订单数据({}=无)
    expand_queries: list[str]     # 扩写产物
    refund_flow: bool             # 本轮走了退款售后子流程(agent 窄化/actions 决策)
    pending_order_id: str         # prepare_order 抽到的单号(""=缺)
    resume_order_id: str          # 无状态回传:点选带回的订单号
    resume_question: str          # 无状态回传:点选带回的补全问题
