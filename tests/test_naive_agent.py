"""裸 Agent 循环:调 LLM → 有工具调用就执行喂回 → 没有就收敛。"""
import pytest
from langchain.tools import tool
from langchain_core.messages import AIMessage

from tests.helpers import FakeChatWithTools


@tool
def echo_city(city: str) -> str:
    """按城市查天气(测试替身)。"""
    return f"{city}:晴"


class ToolThenAnswerModel(FakeChatWithTools):
    """第一次 invoke 出 tool_calls,第二次出文本答案;计 ainvoke 次数。"""

    call_count: int = 0  # 类级字段声明,pydantic 模型实例属性须走字段

    def __init__(self):
        super().__init__(messages=iter([
            AIMessage(content="", tool_calls=[
                {"name": "echo_city", "args": {"city": "杭州"}, "id": "c1", "type": "tool_call"}]),
            AIMessage(content="杭州今天晴。"),
        ]))

    async def ainvoke(self, messages, **kwargs):
        self.call_count += 1
        return await super().ainvoke(messages, **kwargs)


async def test_loop_executes_tool_and_converges():
    from scripts.naive_agent import naive_agent_loop

    model = ToolThenAnswerModel()
    answer = await naive_agent_loop(model, [echo_city], "杭州天气怎么样")
    assert answer == "杭州今天晴。"
    assert model.call_count == 2


async def test_loop_no_tool_single_call():
    from scripts.naive_agent import naive_agent_loop

    model = FakeChatWithTools(messages=iter([AIMessage(content="直接回答。")]))
    answer = await naive_agent_loop(model, [echo_city], "你好")
    assert answer == "直接回答。"
