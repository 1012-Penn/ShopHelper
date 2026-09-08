"""ch07 会话上下文:预算推导、三层切分、渲染拼装、级联降级——纯函数与数据变换,不碰模型。"""
from dataclasses import dataclass

from app.config import Settings


@dataclass
class ContextBudgets:
    """一次推导,处处使用:window/fixed/peak 只为可观测,sliding 是历史的硬上限。"""

    window: int
    fixed: int
    peak: int
    sliding: int
    history: int
    layer1: int
    layer2: int


def derive_budgets(s: Settings) -> ContextBudgets:
    """token 预算从模型窗口倒推:扣输出预留、单轮 ReAct 峰值、固定开销,余量归历史。

    层 1 拿七成、层 2 拿余下;int(历史×0.7) 的浮点截断语义是验收口径(5650→3954/1695)。
    """
    peak = s.max_user_input_tokens + s.max_agent_steps * s.tool_result_max_tokens
    fixed = (s.ctx_prompt_overhead_tokens
             + s.rerank_top_k * s.ctx_evidence_item_tokens
             + s.ctx_summary_allowance_tokens
             + s.ctx_safety_margin_tokens)
    sliding = max(s.model_context_window - s.max_output_tokens - peak - fixed, 0)
    history = min(s.ctx_keep_rounds * s.ctx_per_round_tokens, sliding)
    # 七三开各自 int 截断(验收口径:5650 → 3954 + 1695,和为 5649 留 1 token 余量)
    return ContextBudgets(window=s.model_context_window, fixed=fixed, peak=peak,
                          sliding=sliding, history=history,
                          layer1=int(history * 0.7), layer2=int(history * 0.3))
