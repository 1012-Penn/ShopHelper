"""ch05 图骨架:State 一路贯穿,checkpointer 只承载图内运行态(持久真源在 MySQL store)。

ch07:messages 挂 add_messages reducer——各节点只吐本轮新消息,框架按 id 并入完整历史,
checkpoint 承载进程内完整轨迹;模型面的分层精简版由路由层预算好放在 layered,两套各走各的。
"""
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


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
    messages: Annotated[list, add_messages]  # 完整历史+本轮轨迹;节点只吐新消息,框架按 id 并入
    turn_user_msg_id: str         # 本轮 user 消息 id(log 节点据此切出本轮新消息落库)
    layered: dict                 # 模型面分层上下文(路由层预算好):层2/层1消息、摘要、history_text
    # ---- ch06:分流器正式版 ----
    intent_confidence: float      # 意图置信度(trace/评估用)
    order: dict                   # 子流程拿到的订单数据({}=无)
    expand_queries: list[str]     # 扩写产物
    refund_flow: bool             # 本轮走了退款售后子流程(agent 窄化/actions 决策)
    pending_order_id: str         # prepare_order 抽到的单号(""=缺)
    resume_order_id: str          # 无状态回传:点选带回的订单号
    resume_question: str          # 无状态回传:点选带回的补全问题
