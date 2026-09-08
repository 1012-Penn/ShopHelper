"""ch07 会话上下文:预算推导、三层切分、渲染拼装、级联降级——纯函数与数据变换,不碰模型。

层边界靠消息 id 表达,不搬数据:id ≤ summary_upto 已进摘要;summary_upto < id ≤ layer1_from
为层 2(渲染时半压,存储保持原文);id > layer1_from 为层 1(原样)。锚点语义与 db/ch07-layers.sql 一致。
"""
import json
import logging
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import trim_messages

from app.config import Settings
from app.history import estimate_tokens, rows_to_messages

logger = logging.getLogger(__name__)

TOOL_MARK = "[工具结果已省略]"
L2_ELLIPSIS = "…"
_ROLE_ZH = {"user": "用户", "assistant": "客服"}


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

    层 1 七成、层 2 三成,各自 int 截断——验收口径(5650 → 3954 + 1695,和留 1 token 余量)。
    """
    peak = s.max_user_input_tokens + s.max_agent_steps * s.tool_result_max_tokens
    fixed = (s.ctx_prompt_overhead_tokens
             + s.rerank_top_k * s.ctx_evidence_item_tokens
             + s.ctx_summary_allowance_tokens
             + s.ctx_safety_margin_tokens)
    sliding = max(s.model_context_window - s.max_output_tokens - peak - fixed, 0)
    history = min(s.ctx_keep_rounds * s.ctx_per_round_tokens, sliding)
    return ContextBudgets(window=s.model_context_window, fixed=fixed, peak=peak,
                          sliding=sliding, history=history,
                          layer1=int(history * 0.7), layer2=int(history * 0.3))


# ---- 三层切分与渲染 ----

def split_layers(rows: list[dict], summary_upto: int | None,
                 layer1_from: int | None) -> tuple[list[dict], list[dict]]:
    """行按锚点切 (层1, 层2);NULL layer1_from 视为「从未降级」:层 1 从 summary_upto 之后起。"""
    su = summary_upto or 0
    lf = layer1_from if layer1_from is not None else su
    layer1 = [r for r in rows if r["id"] > lf]
    layer2 = [r for r in rows if su < r["id"] <= lf]
    return layer1, layer2


def _layer2_line(row: dict, head_chars: int) -> str | None:
    """层 2 半压口径的单行渲染:user 原文;assistant 截头;tool 一行标识;纯工具调用行跳过。"""
    role, content = row["role"], row.get("content")
    if role == "user":
        return content or ""
    if role == "assistant":
        if not content:
            return None
        return content if len(content) <= head_chars else content[:head_chars] + L2_ELLIPSIS
    return TOOL_MARK


def render_layer2(rows: list[dict], head_chars: int) -> list:
    """层 2 → LC 消息(半压)。tool 残留行渲染成 assistant 一行标识,保证协议合法(不产生孤儿 tool 消息)。"""
    msgs: list = []
    for row in rows:
        if row["role"] == "user":
            msgs.append(HumanMessage(content=row.get("content") or ""))
        elif row["role"] == "assistant":
            line = _layer2_line(row, head_chars)
            if line is not None:
                msgs.append(AIMessage(content=line))
        else:
            msgs.append(AIMessage(content=TOOL_MARK))
    return msgs


def layer2_tokens(rows: list[dict], head_chars: int) -> int:
    """层 2 占用按渲染态计(发给模型的量),与预算同一把尺。"""
    total = 0
    for row in rows:
        line = _layer2_line(row, head_chars)
        if line is not None:
            total += estimate_tokens(line)
    return total


# ---- 层 1 装填与级联降级 ----

def _msg_tokens(m) -> int:
    return estimate_tokens(m.content or "")


def _list_tokens(msgs: list) -> int:
    return sum(_msg_tokens(m) for m in msgs)


def cascade_layer1(rows: list[dict], layer1_from: int | None,
                   budget: int) -> tuple[int | None, int, int]:
    """层 1 超预算:重切层 1,返回 (新锚点, 降级前 token, 降级后 token);未超返回 (原锚点, t, t)。

    trim_messages(strategy="last", start_on="human") 保证后缀从 user 行起整轮切入;
    新锚点 = 首条保留行 id − 1(层 1 语义 id > layer1_from);一圈都装不下时至少保最后一轮。
    """
    layer1_rows, _ = split_layers(rows, None, layer1_from)
    before = sum(estimate_tokens(r.get("content") or "") for r in layer1_rows)
    if before <= budget:
        return layer1_from, before, before
    msgs = rows_to_messages(layer1_rows)
    kept = trim_messages(msgs, max_tokens=budget, token_counter=_list_tokens,
                         strategy="last", start_on="human")
    if not kept:
        idx = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
        kept = msgs[idx:]
    new_anchor = int(kept[0].id) - 1
    return new_anchor, before, _list_tokens(kept)


# ---- 拼装 ----

def build_layered(l1_rows: list[dict], l2_rows: list[dict], summary_text: str | None,
                  head_chars: int, sliding: int) -> dict:
    """组装模型面上下文素材;层 2 渲染总量超滑窗时从最老端丢弃(仅本轮,不动锚点)。"""
    l2_msgs = render_layer2(l2_rows, head_chars)
    l1_msgs = rows_to_messages(l1_rows)
    dropped = 0
    while l2_msgs and _list_tokens(l2_msgs) + _list_tokens(l1_msgs) > sliding:
        l2_msgs.pop(0)
        dropped += 1
    if dropped:
        logger.warning("层2 渲染超滑窗,丢弃最老 %d 条(仅本轮,不动锚点)", dropped)
    return {"layer2_msgs": l2_msgs, "layer1_msgs": l1_msgs,
            "summary_text": summary_text or "",
            "history_text": render_history_text(summary_text, l2_rows, l1_rows, head_chars),
            "n_window": len(l2_msgs) + len(l1_msgs),
            "tokens_l1": _list_tokens(l1_msgs), "tokens_l2": _list_tokens(l2_msgs),
            "dropped": dropped}


def render_history_text(summary_text: str | None, l2_rows: list[dict],
                        l1_rows: list[dict], head_chars: int) -> str:
    """resolve/意图等前置节点的历史文本:摘要行 + 滑窗(层 2 半压、层 1 原文)。"""
    lines = [ln for r in l2_rows if (ln := _layer2_line(r, head_chars)) is not None]
    lines += [f"{_ROLE_ZH.get(r['role'], r['role'])}:{r.get('content')}" for r in l1_rows
              if r.get("content")]
    return (f"【早期对话梗概】{summary_text or '(无)'}\n"
            f"【最近对话】\n" + ("\n".join(lines) if lines else "(无)"))


def build_background(summary_text: str | None, evidence: list[dict] | None,
                     order: dict | None, narrow_instruction: str) -> str | None:
    """梗概+证据(+子流程订单数据与窄化指令)合成一条挂在当前用户消息之后;全空则省略。

    梗概绝不单独占一条 system:上游模板会把所有 system 上提合并渲染,前缀缓存整段作废。
    """
    from app.prompts import build_evidence_block

    blocks = []
    if summary_text:
        blocks.append("【历史梗概】" + summary_text)
    if evidence:
        blocks.append("【参考知识】(回答须带 [n] 角标,编号只能用下面给定的)\n"
                      + build_evidence_block(evidence))
    if order:
        blocks.append("【订单数据】" + json.dumps(order, ensure_ascii=False))
    if narrow_instruction:
        blocks.append("【指令】" + narrow_instruction)
    return "\n\n".join(blocks) or None


def truncate_tool_result(text: str, max_tokens: int) -> str:
    """工具结果按 token 口径截断;CJK 每字记 1 token,故字符数上限即 token 上限(ASCII 只会更省)。"""
    mark = "…(工具结果已截断)"
    if estimate_tokens(text) <= max_tokens:
        return text
    keep = max_tokens - estimate_tokens(mark)
    return (text[:keep] if keep > 0 else "") + mark
