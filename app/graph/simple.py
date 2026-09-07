"""投诉安抚与闲聊:零模型调用的固定话术节点。"""
COMPLAINT_REPLY = ("非常抱歉给您带来了不好的体验,您的反馈我们很重视。"
                   "您可以选择下面的方式,我们会尽快为您处理。")
CHITCHAT_REPLY = ("我是本店智能客服,专注商品咨询、订单物流和售后问题;"
                  "闲聊虽然不太擅长,但有关购物的任何问题都可以随时问我哦。")
ACTION_LABELS = {"transfer_human": "转人工", "create_ticket": "建工单"}


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def _base_state_update(state, reply: str) -> dict:
    return {"final_reply": reply,
            "messages": [{"role": "user", "content": state["user_message"]},
                         {"role": "assistant", "content": reply}]}


async def complaint_node(state, writer=None) -> dict:
    """不进 Agent、不自动执行任何动作:安抚 + 把两个选项交用户自选。"""
    _emit(writer, {"type": "token", "content": COMPLAINT_REPLY})
    return {**_base_state_update(state, COMPLAINT_REPLY),
            "suggested_actions": ["transfer_human", "create_ticket"],
            "trace": [*state["trace"], "node=complaint fixed_reply actions=2"]}


async def chitchat_node(state, writer=None) -> dict:
    _emit(writer, {"type": "token", "content": CHITCHAT_REPLY})
    return {**_base_state_update(state, CHITCHAT_REPLY),
            "suggested_actions": [],
            "trace": [*state["trace"], "node=chitchat fixed_reply"]}
