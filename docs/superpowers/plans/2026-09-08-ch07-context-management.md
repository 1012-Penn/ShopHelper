# ch07 会话上下文管理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch03 的整条丢弃式裁剪升级为三层上下文管理:层 1 原文 / 层 2 渲染截短 / 早期异步分段摘要,锚点 id 划界只挪不搬;预算从模型窗口倒推;每轮上下文与摘要任务全留痕;前端多会话侧栏。

**Architecture:** 分层视图以 MySQL 为构建源(messages 行带自增 id + conversations 两个锚点列 + conversation_summaries 分段表);State.messages 挂 `add_messages` 承载图内完整轨迹(checkpoint),与模型输入精简版各走各的;级联(层 1 降级)在每轮开头同步执行,摘要投后台 asyncio 任务不阻塞 SSE;agent 的 system 收敛为纯静态,梗概+证据合成一条 HumanMessage 挂当前用户消息之后。

**Tech Stack:** LangGraph 1.2(`add_messages` + InMemorySaver)+ LangChain 1.4(`trim_messages`)+ MySQL(conversations 加三列 / conversation_summaries 新表);无新增依赖。

**Spec:** `docs/superpowers/specs/2026-09-08-ch07-context-management-design.md`(本计划从 spec 论证,执行者两份都读)

## Global Constraints

- 预算公式(spec「预算推导」节,验收数字逐位复现):滑窗=max(窗口−输出−(输入+步数×工具上限)−固定,0);固定=1800+RERANK_TOP_K×250+200+1500;层1=`int(历史×0.7)`、层2=`int(历史×0.3)`(各算各的,Python 浮点语义)
- 演示配置必须精确产出滑窗 5650 / 层1 3954 / 层2 1695:`MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5`
- messages 表只落 user 行与含文本 assistant 行;tool 行与纯工具调用 assistant 行不落库(ch02 契约收窄)
- 梗概只追加不回炉;旧梗概只作背景不参与合并;摘要 prompt 禁编造、不留寒暄、几十到一两百字、超 300 字硬截断
- agent system 每轮逐字节一致:证据/订单数据/窄化指令/梗概一律不进 system
- 锚点语义(`id > layer1_from` 为层 1):降级 `layer1_from = 首条保留行 id − 1`;摘要批 `(summary_upto, layer1_from]`,完成后 `summary_upto = 批末行 id`
- 日志锚点(grep 口径):`[history_ctx]`、`[model_ctx]`、`层1 降级`、`summary trigger`、`[summary] … done 第N段`、`上下文预算不足`;落 `logs/app.log`
- py3.10;测试替身不出网;每任务收尾跑全量 `pytest`;前端 Vibe Coding 不走 TDD/code review

---

### Task 1: 配置改名退役 + 预算推导纯函数

**Files:**
- Modify: `app/config.py`(ch07 设置组;删 `history_token_budget`;`agent_max_steps`→`max_agent_steps`;`retrieval_final_top_k`→`rerank_top_k`)
- Create: `app/context.py`(本任务只含 `derive_budgets` + `ContextBudgets`)
- Modify: `app/agent.py`(`settings.agent_max_steps`→`settings.max_agent_steps`)、`app/main.py`(build_tools `top_k=settings.rerank_top_k`)、`app/graph/builder.py`(make_retrieval_chain `final_top_k=settings.rerank_top_k`)、`app/graph/refund.py`(构造参数默认值来源不变,调用方传 `settings.rerank_top_k` 的在 builder)
- Modify: `.env.example`(ch07 段替换 HISTORY_TOKEN_BUDGET)
- Test: `tests/test_ch07_budget.py`

**Interfaces:**
- Produces: `ContextBudgets`(dataclass:window/fixed/peak/sliding/history/layer1/layer2 全 int)、`derive_budgets(settings) -> ContextBudgets`;Settings 新增字段 `model_context_window=65536, max_output_tokens=2000, max_user_input_tokens=2000, max_agent_steps=6, tool_result_max_tokens=2000, rerank_top_k=10, ctx_keep_rounds=30, ctx_per_round_tokens=250, ctx_prompt_overhead_tokens=1400, ctx_evidence_item_tokens=250, ctx_summary_allowance_tokens=200, ctx_safety_margin_tokens=1500, ctx_layer2_assistant_head_chars=60`
- 消费方:Task 3(切分/级联用 budgets)、Task 5(chat.py 与 main.py 启动自检)

- [ ] **Step 1: 写失败测试** `tests/test_ch07_budget.py`

```python
"""预算推导:演示配置逐位复现验收数字;只调窗口归零;默认配置不触发压缩。"""
from app.config import Settings
from app.context import derive_budgets


def _settings(**kw) -> Settings:
    return Settings(openai_api_key="sk-test", _env_file=None, **kw)


def test_demo_config_matches_acceptance_exactly():
    b = derive_budgets(_settings(model_context_window=18000, max_output_tokens=2000,
                                 max_user_input_tokens=2000, max_agent_steps=3,
                                 tool_result_max_tokens=1200, rerank_top_k=5))
    assert b.peak == 5600            # 2000 + 3*1200
    assert b.fixed == 4350           # 1400 + 5*250 + 200 + 1500
    assert b.sliding == 5650
    assert b.layer1 == 3954          # int(5650*0.7) 浮点截断语义
    assert b.layer2 == 1695


def test_window_only_collapses_to_zero():
    b = derive_budgets(_settings(model_context_window=18000))
    assert b.sliding == 0
    assert b.sliding < b.history or b.history == 0  # 触发自检报警条件


def test_default_config_roomy_no_compression():
    b = derive_budgets(_settings())
    assert b.sliding > 20000
    assert b.history == 7500 and b.layer1 == 5250 and b.layer2 == 2250


def test_negative_clamps_to_zero():
    b = derive_budgets(_settings(model_context_window=1, ctx_safety_margin_tokens=100000))
    assert b.sliding == 0 and b.layer1 == 0 and b.layer2 == 0
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_ch07_budget.py -v` → FAIL(ImportError: derive_budgets)

- [ ] **Step 3: 实现** config 字段按 Interfaces 清单改;`app/context.py`:

```python
"""ch07 会话上下文:预算推导、三层切分、渲染拼装、级联降级——纯函数与数据变换,不碰模型。"""
from dataclasses import dataclass

from app.config import Settings


@dataclass
class ContextBudgets:
    window: int
    fixed: int
    peak: int
    sliding: int
    history: int
    layer1: int
    layer2: int


def derive_budgets(s: Settings) -> ContextBudgets:
    """token 预算从模型窗口倒推:扣输出预留、单轮 ReAct 峰值、固定开销,余量归历史。"""
    peak = s.max_user_input_tokens + s.max_agent_steps * s.tool_result_max_tokens
    fixed = (s.ctx_prompt_overhead_tokens
             + s.rerank_top_k * s.ctx_evidence_item_tokens
             + s.ctx_summary_allowance_tokens
             + s.ctx_safety_margin_tokens)
    sliding = max(s.model_context_window - s.max_output_tokens - peak - fixed, 0)
    history = min(s.ctx_keep_rounds * s.ctx_per_round_tokens, sliding)
    layer1 = int(history * 0.7)  # 七三开各自 int 截断(验收口径:5650 → 3954 + 1695)
    return ContextBudgets(window=s.model_context_window, fixed=fixed, peak=peak,
                          sliding=sliding, history=history,
                          layer1=int(history * 0.7), layer2=int(history * 0.3))
```

- [ ] **Step 4: 跑测试 + 全量** `pytest tests/test_ch07_budget.py -v && pytest -q` → 全绿(改名波及面 grep 兜底见 Step 5)

- [ ] **Step 5: 旧名清扫** `grep -rn "history_token_budget\|agent_max_steps\|retrieval_final_top_k" app/ tests/ .env.example README.md` → app/tests 内除注释外清零;`chat.py` 的 400 闸改 `settings.max_user_input_tokens`(临时闸,Task 5 重写);`.env.example` 的 `HISTORY_TOKEN_BUDGET=3000` 行换成 ch07 注释段(逐项列新 env)

- [ ] **Step 6: Commit** `git add -A && git commit -m "ch07: 预算从窗口倒推+配置改名退役(演示配置逐位复现 5650/3954/1695)"`

---

### Task 2: DDL / ORM / store 扩展(锚点与分段摘要)

**Files:**
- Create: `db/ch07-summary.sql`、`db/ch07-layers.sql`(用户提供内容原样落盘)
- Modify: `db/init.sql`(conversations 三列 + conversation_summaries 表 + 头注释 ch07 行)
- Modify: `app/models.py`(Conversation 加三列;新增 ConversationSummary)
- Modify: `app/store.py`(锚点/行/摘要方法)
- Test: `tests/test_ch07_store.py`

**Interfaces:**
- Produces(ConversationStore 实例方法,全部 async):
  - `get_rows(conversation_id: int, upto_id: int | None = None) -> list[dict]`,行 `{"id": int, "role": str, "content": str|None, "tool_calls": list|None, "tool_call_id": str|None}`,按 id 升序
  - `get_anchors(conversation_id: int) -> dict`:`{"summary_upto": int|None, "layer1_from": int|None, "summary_text": str|None, "summary_seqs": int}`
  - `set_layer1_from(conversation_id: int, msg_id: int) -> None`
  - `next_seq(conversation_id: int) -> int`(现有最大 seq+1,空则 1)
  - `append_summary(conversation_id: int, seq: int, from_id: int, upto_id: int, content: str) -> None`(单事务:插段 + `summary_upto_msg_id=upto_id` + summary 列重拼为全部段落 "\n".join)
- 消费方:Task 3(行/锚点进纯函数)、Task 4(摘要服务)、Task 5(chat.py)

- [ ] **Step 1: 落盘 DDL** 两个 SQL 文件按用户提供内容;`init.sql` conversations 定义插入三列(summary 在 status 后、summary_upto_msg_id 在 summary 后、layer1_from_msg_id 在 summary_upto_msg_id 后)并在文件尾部加 conversation_summaries 建表(注释注明 ch07)

- [ ] **Step 2: 写失败测试** `tests/test_ch07_store.py`

```python
"""ch07 store 扩展:行带 id、锚点读写、分段摘要追加与投影重拼。"""
import pytest

from tests.test_db import ...  # 不适用;直接用 conftest 的 db_store fixture


@pytest.mark.asyncio
async def test_get_rows_carries_ids_ordered(db_store):
    sid = await db_store.resolve(None)
    await db_store.append(sid, [{"role": "user", "content": "问"},
                                {"role": "assistant", "content": "答"}])
    rows = await db_store.get_rows(sid)
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    assert rows[0]["role"] == "user" and rows[0]["content"] == "问"


@pytest.mark.asyncio
async def test_anchors_default_none(db_store):
    sid = await db_store.resolve(None)
    a = await db_store.get_anchors(sid)
    assert a["summary_upto"] is None and a["layer1_from"] is None
    assert a["summary_text"] is None and a["summary_seqs"] == 0


@pytest.mark.asyncio
async def test_layer1_anchor_roundtrip(db_store):
    sid = await db_store.resolve(None)
    await db_store.set_layer1_from(sid, 42)
    assert (await db_store.get_anchors(sid))["layer1_from"] == 42


@pytest.mark.asyncio
async def test_append_summary_advances_anchor_and_projection(db_store):
    sid = await db_store.resolve(None)
    await db_store.append(sid, [{"role": "user", "content": "u1"},
                                {"role": "assistant", "content": "a1"}])
    rows = await db_store.get_rows(sid)
    await db_store.append_summary(sid, seq=1, from_id=rows[0]["id"],
                                  upto_id=rows[1]["id"], content="梗概一")
    await db_store.append_summary(sid, seq=2, from_id=rows[1]["id"] + 1,
                                  upto_id=rows[1]["id"] + 2, content="梗概二")
    a = await db_store.get_anchors(sid)
    assert a["summary_upto"] == rows[1]["id"] + 2
    assert a["summary_seqs"] == 2
    assert a["summary_text"] == "梗概一\n梗概二"
    assert (await db_store.next_seq(sid)) == 3


@pytest.mark.asyncio
async def test_get_rows_upto_id_bounds_batch(db_store):
    sid = await db_store.resolve(None)
    await db_store.append(sid, [{"role": "user", "content": "u1"},
                                {"role": "assistant", "content": "a1"},
                                {"role": "user", "content": "u2"}])
    rows = await db_store.get_rows(sid)
    batch = await db_store.get_rows(sid, upto_id=rows[1]["id"])
    assert len(batch) == 2
```

- [ ] **Step 3: 跑测试确认失败** → AttributeError

- [ ] **Step 4: 实现** `app/models.py` Conversation 加:

```python
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=True)
```

新增 ConversationSummary(id 自增 / conversation_id / seq int / from_msg_id / upto_msg_id / content Text / created_at,UniqueConstraint(conversation_id, seq))。store 五方法按 Interfaces 实现(append_summary 里插段、UPDATE conversations、重拼投影在同一 `with self._factory() as session` 事务)。

- [ ] **Step 5: 跑测试 + 全量** → 绿

- [ ] **Step 6: Commit** `git commit -m "ch07: 锚点列+分段摘要表 DDL/ORM/store(降级只挪 id、摘要一段一行)"`

---

### Task 3: 分层切分 / 渲染 / 组装 / 级联(context.py 主体)

**Files:**
- Modify: `app/context.py`(主体)
- Modify: `app/history.py`(`trim_history`/`history_to_messages` 退役,保留 `estimate_tokens`;新增 `rows_to_messages`)
- Test: `tests/test_ch07_layers.py`

**Interfaces:**
- Consumes: Task 1 budgets、Task 2 行/锚点形状
- Produces(`app/context.py`):
  - `split_layers(rows, summary_upto, layer1_from) -> tuple[list, list]`(None 锚点:summary_upto 视 0、layer1_from 视全部层 1)
  - `render_layer2(rows, head_chars) -> list[BaseMessage]`(user 原文;assistant 非空文本截 head_chars 加「…」;content 空的 assistant 跳过;tool 行 → `AIMessage("[工具结果已省略]")`)
  - `layer2_tokens(rows, head_chars) -> int`(渲染态估算)
  - `rows_to_messages(rows) -> list`(迁移入 history.py:带 `id=str(row_id)`,完整还原工具轨迹)
  - `cascade_layer1(rows, layer1_from, budget) -> tuple[int | None, int, int]`(`trim_messages(strategy="last", start_on="human", token_counter=…)`;新锚点=首条保留行 id−1;结果空则保最后一轮;返回 (新锚点, 降级前 token, 降级后 token);未超预算返回原锚点)
  - `build_layered(l1_rows, l2_rows, summary_text, head_chars, sliding) -> dict`:`{"layer2_msgs", "layer1_msgs", "summary_text", "history_text", "n_window", "tokens_l1", "tokens_l2"}`;超滑窗时从最老端丢层 2 渲染块并计 `dropped` 键
  - `render_history_text(summary_text, l2_rows, l1_rows, head_chars) -> str`(「【早期对话梗概】…\n【最近对话】\n用户:…/客服:…」;层 2 半压、层 1 原文)
  - `build_background(summary_text, evidence, order, narrow_instruction) -> str | None`(【历史梗概】/【参考知识】带角标规则/【订单数据】/【指令】;全空 None)
  - `truncate_tool_result(text, max_tokens) -> str`(超限按字符截断加「…(工具结果已截断)」,CJK 每字 1 token 故字符数上限即 token 上限)
- 消费方:Task 4(render_batch 复用渲染规则)、Task 5(chat.py/agent.py)

- [ ] **Step 1: 写失败测试** `tests/test_ch07_layers.py`(节选核心断言,实现时补全同构用例)

```python
"""分层切分/渲染/组装/级联:锚点数学、三条截短规则、trim_messages 装填、背景块。"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.context import (build_background, build_layered, cascade_layer1,
                         render_history_text, render_layer2, split_layers,
                         truncate_tool_result)
from app.history import estimate_tokens, rows_to_messages


def _rows(*pairs):
    out, i = [], 100
    for u, a in pairs:
        out.append({"id": i, "role": "user", "content": u}); i += 1
        out.append({"id": i, "role": "assistant", "content": a}); i += 1
    return out


def test_split_layers_null_anchors_all_layer1():
    rows = _rows(("u", "a"))
    l1, l2 = split_layers(rows, None, None)
    assert len(l1) == 2 and l2 == []


def test_split_layers_band_boundaries():
    rows = _rows(("u1", "a1"), ("u2", "a2"), ("u3", "a3"))  # id 100..105
    l1, l2 = split_layers(rows, summary_upto=101, layer1_from=103)
    assert [r["id"] for r in l2] == [102, 103]
    assert [r["id"] for r in l1] == [104, 105]


def test_render_layer2_rules():
    rows = [{"id": 1, "role": "user", "content": "原始用户话" * 30},
            {"id": 2, "role": "assistant", "content": "客服长答复" * 40},
            {"id": 3, "role": "tool", "content": "大块JSON"},
            {"id": 4, "role": "assistant", "content": None,
             "tool_calls": [{"name": "q", "args": {}, "id": "1"}]}]
    msgs = render_layer2(rows, head_chars=60)
    assert isinstance(msgs[0], HumanMessage) and msgs[0].content == "原始用户话" * 30  # user 不动
    assert msgs[1].content.endswith("…") and len(msgs[1].content) <= 61              # assistant 截头
    assert msgs[2].content == "[工具结果已省略]"                                       # tool 一行标识
    assert len(msgs) == 3                                                             # 纯工具调用行跳过


def test_cascade_moves_anchor_to_turn_boundary():
    rows = _rows(*[(f"用户第{i}轮的问题" + "补充" * 80, "客服第%d轮答复" % i + "说明" * 80)
                   for i in range(10)])  # 每轮约 170+ token
    l1_from, t0, t1 = cascade_layer1(rows, None, 600)
    assert l1_from is not None and t1 <= 600 < t0
    kept = [r for r in rows if r["id"] > l1_from]
    assert kept[0]["role"] == "user"          # 整轮边界切入


def test_cascade_noop_within_budget():
    rows = _rows(("u", "a"))
    assert cascade_layer1(rows, None, 10000) == (None, estimate_tokens("u") + estimate_tokens("a"),) * 2[:2] or True
    # 实现时写严格断言:返回 (None, t, t)


def test_build_layered_drops_oldest_on_overflow():
    rows = _rows(*[(f"问题{i}" * 50, "答复%d" % i * 50) for i in range(8)])
    layered = build_layered(rows, [], "", head_chars=60, sliding=400)
    assert layered["dropped"] > 0
    total = sum(estimate_tokens(m.content or "") for m in
                layered["layer2_msgs"] + layered["layer1_msgs"])
    assert total <= 400


def test_background_combines_and_never_empty():
    bg = build_background("梗概", [{"n": 1, "section_path": "p", "question": "q", "answer": "a"}], None, "")
    assert "【历史梗概】梗概" in bg and "【参考知识】" in bg and "[1]" in bg
    assert build_background("", [], None, "") is None


def test_truncate_tool_result_bounds_tokens():
    big = "数" * 5000
    out = truncate_tool_result(big, 200)
    assert estimate_tokens(out) <= 210 and out.endswith(")")
    assert truncate_tool_result("短", 200) == "短"


def test_rows_to_messages_carries_ids():
    msgs = rows_to_messages(_rows(("u", "a")))
    assert msgs[0].id == "100" and isinstance(msgs[0], HumanMessage)
```

- [ ] **Step 2: 跑失败** → ImportError

- [ ] **Step 3: 实现** `app/context.py` 追加(核心代码):

```python
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import trim_messages

from app.history import estimate_tokens

logger = logging.getLogger(__name__)
TOOL_MARK = "[工具结果已省略]"
L2_ELLIPSIS = "…"


def split_layers(rows, summary_upto, layer1_from):
    su = summary_upto or 0
    lf = layer1_from if layer1_from is not None else float("inf")
    return ([r for r in rows if r["id"] > lf],
            [r for r in rows if su < r["id"] <= lf])


def _row_line(row, head_chars):
    """渲染一行(层 2 半压口径):user 原文 / assistant 截头 / tool 一行标识。"""
    role, content = row["role"], row.get("content")
    if role == "user":
        return content or ""
    if role == "assistant":
        if not content:
            return None
        return content if len(content) <= head_chars else content[:head_chars] + L2_ELLIPSIS
    return TOOL_MARK


def render_layer2(rows, head_chars):
    msgs = []
    for row in rows:
        if row["role"] == "user":
            msgs.append(HumanMessage(content=row.get("content") or ""))
        elif row["role"] == "assistant":
            line = _row_line(row, head_chars)
            if line is not None:
                msgs.append(AIMessage(content=line))
        else:
            msgs.append(AIMessage(content=TOOL_MARK))
    return msgs


def layer2_tokens(rows, head_chars):
    return sum(estimate_tokens(line) for r in rows
               if (line := _row_line(r := _row, head_chars) if False else None) is None)  # 实现时改为直白循环


def _msg_tokens(m):
    return estimate_tokens(m.content or "")


def _list_tokens(msgs):
    return sum(_msg_tokens(m) for m in msgs)


def cascade_layer1(rows, layer1_from, budget):
    l1_rows, _ = split_layers(rows, None, layer1_from)
    t0 = sum(estimate_tokens(r.get("content") or "") for r in l1_rows)
    if t0 <= budget:
        return layer1_from, t0, t0
    msgs = rows_to_messages(l1_rows)
    kept = trim_messages(msgs, max_tokens=budget, token_counter=_list_tokens,
                         strategy="last", start_on="human")
    if not kept:  # 一轮都装不下:至少保最后一轮
        idx = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
        kept = msgs[idx:]
    new_anchor = int(kept[0].id) - 1  # 层 1 语义 id > layer1_from
    return new_anchor, t0, _list_tokens(kept)


def build_layered(l1_rows, l2_rows, summary_text, head_chars, sliding):
    l2_msgs, l1_msgs = render_layer2(l2_rows, head_chars), rows_to_messages(l1_rows)
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


def render_history_text(summary_text, l2_rows, l1_rows, head_chars):
    zh = {"user": "用户", "assistant": "客服"}
    lines = [ln for r in l2_rows if (ln := _row_line(r, head_chars)) is not None]
    lines += [f"{zh.get(r['role'], r['role'])}:{r.get('content') or ''}" for r in l1_rows
              if r.get("content")]
    return (f"【早期对话梗概】{summary_text or '(无)'}\n"
            f"【最近对话】\n" + ("\n".join(lines) if lines else "(无)"))


def build_background(summary_text, evidence, order, narrow_instruction):
    from app.prompts import build_evidence_block
    blocks = []
    if summary_text:
        blocks.append("【历史梗概】" + summary_text)
    if evidence:
        blocks.append("【参考知识】(回答须带 [n] 角标,编号只能用下面给定的)\n"
                      + build_evidence_block(evidence))
    if order:
        import json
        blocks.append("【订单数据】" + json.dumps(order, ensure_ascii=False))
    if narrow_instruction:
        blocks.append("【指令】" + narrow_instruction)
    return "\n\n".join(blocks) or None


def truncate_tool_result(text, max_tokens):
    mark = "…(工具结果已截断)"
    if estimate_tokens(text) <= max_tokens:
        return text
    keep = max_tokens - estimate_tokens(mark)
    return (text[:keep] if keep > 0 else "") + mark
```

`app/history.py`:`rows_to_messages(rows)` 按 `history_to_messages` 旧体迁移 + `id=str(m["id"])`;删 `trim_history`、`history_to_messages`。

- [ ] **Step 4: 跑测试 + 全量** `pytest tests/test_ch07_layers.py -v && pytest -q`(trim_history 退役会红一批旧测试,记录清单,Tasks 5 统一迁移;若 test_history 失败先改 `tests/test_history.py` 为 rows_to_messages/estimate_tokens 口径)

- [ ] **Step 5: Commit** `git commit -m "ch07: 三层切分/渲染/组装/级联纯函数层(trim_messages 装填层1)"`

---

### Task 4: 后台摘要服务 + SUMMARY_PROMPT

**Files:**
- Modify: `app/prompts.py`(SUMMARY_PROMPT)
- Create: `app/summarizer.py`
- Test: `tests/test_ch07_summary.py`

**Interfaces:**
- Consumes: Task 2 store(`get_rows(upto_id)`/`get_anchors`/`next_seq`/`append_summary`)、Task 3 渲染规则
- Produces: `SummaryService(store, model, settings)`;`asyncio` 组内方法 `maybe_trigger(session_id: int, upto_id: int | None) -> None`(同步返回,内部 create_task;在飞 set 防重入);日志行:`[summary] session=%s start 第%d段 覆盖=[%d,%d]` / `done 第%d段 覆盖=[%d,%d] 耗时=%dms 字数=%d` / `skip …` / `fail err=…`;任务引用挂 `self._tasks` 防 GC,done callback 清理在飞集合
- `SUMMARY_PROMPT = PromptTemplate.from_template(...)`,占位符 `{old}`(旧梗概背景)`{batch}`(本批对话原文,user/assistant 全文、tool 行渲染标识)
- 消费方:Task 5(chat.py 接线、main.py 挂 app.state)

- [ ] **Step 1: 写失败测试** `tests/test_ch07_summary.py`

```python
"""后台摘要:触发条件、段落追加、锚点推进、旧梗概作背景、超长截断、失败不抛、在飞跳过。"""
import logging

import pytest

from app.summarizer import SummaryService


class StubSummaryModel:
    def __init__(self, text="用户询问订单1001发货,诉求催单。"):
        self.text = text
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(str(prompt))
        return type("R", (), {"content": self.text})()


class BoomModel:
    async def ainvoke(self, prompt):
        raise RuntimeError("上游挂了")


@pytest.mark.asyncio
async def test_run_appends_segment_and_advances_anchor(db_store):
    sid = await db_store.resolve(None)
    await db_store.append(sid, [{"role": "user", "content": "订单1001多久发货"},
                                {"role": "assistant", "content": "一般48小时内"}])
    rows = await db_store.get_rows(sid)
    await db_store.set_layer1_from(sid, rows[-1]["id"])
    svc = SummaryService(db_store, StubSummaryModel(), None)
    await svc._run(sid, upto_id=rows[-1]["id"])
    a = await db_store.get_anchors(sid)
    assert a["summary_seqs"] == 1 and a["summary_upto"] == rows[-1]["id"]
    assert "订单1001" in a["summary_text"]


@pytest.mark.asyncio
async def test_prompt_carries_old_summary_and_batch(db_store, caplog):
    sid = await db_store.resolve(None)
    await db_store.append(sid, [{"role": "user", "content": "新款问题"},
                                {"role": "assistant", "content": "好的"}])
    rows = await db_store.get_rows(sid)
    model = StubSummaryModel()
    svc = SummaryService(db_store, model, None)
    with caplog.at_level(logging.INFO, logger="app.summarizer"):
        await svc._run(sid, upto_id=rows[-1]["id"])
    assert "【早期对话梗概】" in model.prompts[0] and "本批对话" in model.prompts[0]
    assert any("done 第1段" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_overlong_output_hard_truncated(db_store):
    sid = await db_store.resolve(None)
    await db_store.append(sid, [{"role": "user", "content": "u"}])
    svc = SummaryService(db_store, StubSummaryModel(text="长" * 400), None)
    await svc._run(sid, upto_id=(await db_store.get_rows(sid))[-1]["id"])
    assert len((await db_store.get_anchors(sid))["summary_text"]) <= 300


@pytest.mark.asyncio
async def test_upstream_failure_logged_not_raised(db_store, caplog):
    sid = await db_store.resolve(None)
    await db_store.append(sid, [{"role": "user", "content": "u"}])
    svc = SummaryService(db_store, BoomModel(), None)
    with caplog.at_level(logging.WARNING, logger="app.summarizer"):
        await svc._run(sid, upto_id=(await db_store.get_rows(sid))[-1]["id"])  # 不抛
    assert any("fail" in r.message for r in caplog.records)
    assert (await db_store.get_anchors(sid))["summary_seqs"] == 0


@pytest.mark.asyncio
async def test_empty_batch_skips(db_store, caplog):
    sid = await db_store.resolve(None)
    svc = SummaryService(db_store, StubSummaryModel(), None)
    with caplog.at_level(logging.INFO, logger="app.summarizer"):
        await svc._run(sid, upto_id=None)
    assert any("skip" in r.message for r in caplog.records)
```

- [ ] **Step 2: 跑失败** → ModuleNotFoundError

- [ ] **Step 3: 实现** `app/prompts.py` 追加:

```python
# ch07 会话摘要:只提炼事实与诉求;旧梗概仅作背景,不参与合并
SUMMARY_PROMPT = PromptTemplate.from_template(
    "你是电商客服的会话摘要器。把本批客服对话压成一段梗概,只提炼事实与诉求:"
    "用户问过哪款商品、报过哪些关键信息(订单号、手机号等)、明确诉求是什么、还有哪些问题没解决。\n"
    "规则:对话里没出现的内容一个字不许编;寒暄闲聊不写;不要逐句复述客服答复;\n"
    "已有梗概只是帮你理解前情的背景,新梗概只概括本批对话,不要重复已有梗概的内容;\n"
    "输出一段连贯的话,长度在几十到一两百字之间,不要分点,不要任何前后缀说明。\n"
    "已有梗概(背景,勿重复):\n{old}\n\n本批对话:\n{batch}"
)
```

`app/summarizer.py`:

```python
"""ch07 后台会话摘要:层 2 攒到超预算,一批压成一段追加落库;不阻塞聊天轮。"""
import asyncio
import logging
import time

from app.prompts import SUMMARY_PROMPT

logger = logging.getLogger(__name__)

SUMMARY_MAX_CHARS = 300  # 软约束几十到一两百字,硬截断兜底
_ROLE_ZH = {"user": "用户", "assistant": "客服", "tool": "[工具结果已省略]"}


def _render_batch(rows) -> str:
    lines = []
    for r in rows:
        if r["role"] == "tool":
            lines.append(_ROLE_ZH["tool"])
        else:
            lines.append(f"{_ROLE_ZH.get(r['role'], r['role'])}:{r.get('content') or ''}")
    return "\n".join(lines) or "(空)"


class SummaryService:
    def __init__(self, store, model, settings):
        self._store = store
        self._model = model
        self._inflight: set[int] = set()
        self._tasks: set[asyncio.Task] = set()

    def maybe_trigger(self, session_id: int, upto_id: int | None) -> None:
        if session_id in self._inflight:
            logger.info("[summary] session=%s skip 已有任务在跑", session_id)
            return
        self._inflight.add(session_id)
        task = asyncio.create_task(self._run(session_id, upto_id))
        self._tasks.add(task)
        task.add_done_callback(lambda t: (self._tasks.discard(t),
                                          self._inflight.discard(session_id)))

    async def _run(self, session_id: int, upto_id: int | None) -> None:
        t0 = time.monotonic()
        try:
            anchors = await self._store.get_anchors(session_id)
            rows = await self._store.get_rows(session_id, upto_id=upto_id)
            rows = [r for r in rows if not anchors["summary_upto"]
                    or r["id"] > anchors["summary_upto"]]
            if not rows:
                logger.info("[summary] session=%s skip 批内无消息", session_id)
                return
            seq = await self._store.next_seq(session_id)
            logger.info("[summary] session=%s start 第%d段 覆盖=[%d,%d]",
                        session_id, seq, rows[0]["id"], rows[-1]["id"])
            prompt = SUMMARY_PROMPT.format(old=anchors["summary_text"] or "(无)",
                                           batch=_render_batch(rows))
            resp = await self._model.ainvoke(prompt)
            text = str(getattr(resp, "content", "") or "").strip()
            if not text:
                raise ValueError("摘要模型返回空")
            if len(text) > SUMMARY_MAX_CHARS:
                text = text[:SUMMARY_MAX_CHARS]
            await self._store.append_summary(session_id, seq=seq, from_id=rows[0]["id"],
                                             upto_id=rows[-1]["id"], content=text)
            logger.info("[summary] session=%s done 第%d段 覆盖=[%d,%d] 耗时=%dms 字数=%d",
                        session_id, seq, rows[0]["id"], rows[-1]["id"],
                        int((time.monotonic() - t0) * 1000), len(text))
        except Exception as exc:
            logger.warning("[summary] session=%s fail err=%s", session_id, exc, exc_info=True)
        finally:
            self._inflight.discard(session_id)
```

- [ ] **Step 4: 跑测试 + 全量** → 绿

- [ ] **Step 5: Commit** `git commit -m "ch07: 后台摘要服务+SUMMARY_PROMPT(一段一行只追加,旧梗概仅作背景)"`

---

### Task 5: 管线重接线(State/messages + 节点 LC 化 + log 契约 + agent 组装 + chat.py 编排 + main 自检)

**Files:**
- Modify: `app/graph/state.py`、`app/graph/builder.py`(initial_state)、`app/graph/simple.py`、`app/graph/logging_node.py`、`app/graph/agent.py`、`app/graph/entry.py`、`app/routers/chat.py`、`app/main.py`
- Test: `tests/test_ch07_pipeline.py` + 迁移 `tests/test_history.py`、`tests/helpers.py`(graph_state)、`tests/test_ch05_agent.py`、`tests/test_ch06_agent.py`、`tests/test_ch06_refund.py`、`tests/test_ch06_chat_api.py`、`tests/test_ch05_chat_api.py`、`tests/test_chat_toolflow.py`、`tests/test_ch05_simple.py` 等按新契约
- Modify: `tests/conftest.py`(make_client 增 settings 覆盖参数)

**Interfaces:**
- `ChatState.messages: Annotated[list, add_messages]`;新增 `turn_user_msg_id: str`、`layered: dict`;删除 `history`
- `initial_state(session_id, user_message, resume=None, history_rows=None, layered=None) -> dict`:history_rows 非空时 `rows_to_messages` 回灌 + `HumanMessage(content=user_message, id=f"u{uuid4().hex}")`;layered 缺省 `{}`
- 节点消息契约:agent 吐 `[AIMessage(tool_calls)…, ToolMessage…, AIMessage(终答)]`;complaint/chitchat 吐 `[AIMessage(话术)]`;ask_order 吐 `[]`(user 已在 State);其余节点零吐
- log 节点:按 `turn_user_msg_id` 定位本轮起点,切片落库(HumanMessage→user;AIMessage 且 content 非空→assistant;其余不落)
- `chat.py` 轮前编排(顺序固定):长度闸(max_user_input_tokens)→ resolve → get_anchors/get_rows → cascade_layer1(变了则 set_layer1_from + `[ctx] … 层1 降级 A→B token X→Y`)→ split_layers → 层2 渲染态 token 超预算则 `[ctx] … summary trigger 层2 约 N token > 预算 M(后台执行,不阻塞本轮回复)` + `summary_service.maybe_trigger` → `build_layered` → `[history_ctx] session=%s 摘要=%d段 滑窗=%d条 层2=%d 层1=%d tokens≈%d\n<history_text>` → hydration 判定(app.state.hydrated_threads)→ initial_state → 跑图(SSE 管道不变)
- `agent.py`:system=SERVICE_PROMPT_TEMPLATE.format() 纯静态;messages=[System, *layer2_msgs, *layer1_msgs, Human(resolved), (背景块 HumanMessage?)];背景块=`build_background(summary_text, evidence, order if refund_flow, NARROW_INSTRUCTIONS.get(intent))`;工具结果过 `truncate_tool_result`;每步 `[model_ctx] session=%s step=%d 条数=%d tokens≈%d\n摘要全文:%s\n滑窗逐条:\n…`(system 只计 token 不落正文)
- `main.py`:根 logger 挂 `logs/app.log` FileHandler(utf-8,幂等);`derive_budgets` 存 `app.state.budgets`;`滑窗 < ctx_per_round_tokens` 时 `logger.error("[ctx] 上下文预算不足:…")` 否则打预算明细行;`app.state.hydrated_threads=set()`;`app.state.summary_service=SummaryService(store, make_extract_model(settings), settings)`;挂 conversations 路由(Task 6 前先留空实现或同任务带上——本任务直接带 Task 6 的路由文件创建,见 Task 6)
- 消费方:Task 6(端点)、Task 8(真机验收)

- [ ] **Step 1: 写失败测试** `tests/test_ch07_pipeline.py`(节点级 + e2e,节选核心)

```python
"""管线重接线:State 合并、log 切片落库契约、agent 组装顺序、chat 编排级联、可观测日志。"""
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.builder import initial_state
from app.graph.logging_node import make_log_node
from app.history import rows_to_messages
from tests.helpers import graph_state, FakeChatModel  # helpers 既有底座按新契约微调


def test_initial_state_hydrates_with_ids_and_user_marker():
    rows = [{"id": 7, "role": "user", "content": "旧问"},
            {"id": 8, "role": "assistant", "content": "旧答"}]
    st = initial_state(1, "新问", history_rows=rows, layered={"k": "v"})
    assert [m.id for m in st["messages"][:2]] == ["7", "8"]
    assert st["messages"][-1].content == "新问" and st["messages"][-1].id.startswith("u")
    assert st["turn_user_msg_id"] == st["messages"][-1].id
    assert st["layered"] == {"k": "v"}


@pytest.mark.asyncio
async def test_log_node_persists_only_user_and_text_assistant(db_store):
    sid = await db_store.resolve(None)
    msgs = rows_to_messages([{"id": 1, "role": "user", "content": "旧"}])
    user_id = "u99"
    state = graph_state(session_id=sid,
                        messages=[*msgs, HumanMessage(content="新问", id=user_id),
                                  AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
                                  ToolMessage(content="结果", tool_call_id="c1"),
                                  AIMessage(content="终答")],
                        turn_user_msg_id=user_id, final_reply="终答")
    await make_log_node(db_store, None)(state)
    hist = await db_store.get_history(sid)
    assert [(m["role"], m["content"]) for m in hist] == [("user", "旧"), ("user", "新问"),
                                                         ("assistant", "终答")]


@pytest.mark.asyncio
async def test_agent_assembly_static_system_and_background(test_ctx):
    """system 不含证据/订单数据;背景块含梗概与证据;顺序 system→l2→l1→human→背景。"""
    ...  # 用 helpers 的捕获型假模型断言 messages 形状(实现时按 helpers 既有 FakeChatModel 扩展 captured)


@pytest.mark.asyncio
async def test_chat_e2e_cascade_trigger_and_logs(test_ctx, caplog):
    """小预算 settings 下:连聊触发 层1 降级 → summary trigger;messages 表无 tool 行;history_ctx 每轮必打。"""
    ...  # make_client(settings=演示小预算, summary_service=Stub)连发 8 轮长消息


@pytest.mark.asyncio
async def test_tool_rows_never_persisted_even_when_tools_run(test_ctx, db_store):
    ...  # 假模型带 tool_calls 走完一轮,断言 get_history 无 role=tool
```

- [ ] **Step 2: 跑失败**

- [ ] **Step 3: 实现**(按 Interfaces 逐文件;`state.py` 头部 `from typing import Annotated` + `from langgraph.graph.message import add_messages`;`agent.py` 循环内 pending dict 全部换 LC 消息,citations/n_offset/拒答/窄化逻辑不动;`chat.py` 按 Interfaces 顺序编排,SSE 管道原样)

- [ ] **Step 4: 旧测试迁移** `grep -rn "state\[.history.\]\|history_to_messages\|trim_history\|\"messages\": \[{" tests/ | grep -v ch07` 逐个按新契约改(graph_state 默认 `layered={"layer2_msgs": [], "layer1_msgs": [], "summary_text": "", "history_text": "(无)", "n_window": 0, "tokens_l1": 0, "tokens_l2": 0, "dropped": 0}` + `turn_user_msg_id`;节点断言 dict→LC 消息)

- [ ] **Step 5: 全量** `pytest -q` 全绿

- [ ] **Step 6: Commit** `git commit -m "ch07: 管线重接线——add_messages State/节点LC化/log新契约/agent静态system+背景块/chat轮前编排+启动自检"`

---

### Task 6: 多会话只读端点

**Files:**
- Create: `app/routers/conversations.py`
- Modify: `app/schemas.py`(ConversationItem/ConversationListResponse/ConversationMessagesResponse)、`app/main.py`(挂路由;若 Task 5 未并,则此处并)、`app/store.py`(`list_conversations(user_id) -> list[dict]`)
- Test: `tests/test_ch07_conversations_api.py`

**Interfaces:**
- `GET /api/conversations` → `{"items": [{"id": int, "preview": str, "summarized": bool, "updated_at": str}]}`(updated_at DESC;preview=首条 user 消息截 100 字)
- `GET /api/conversations/{id}/messages` → `{"conversation_id": id, "messages": [MessageItem…]}`;404 同 sessions 语义
- 消费方:Task 7 前端

- [ ] **Step 1: 失败测试** `tests/test_ch07_conversations_api.py`

```python
"""多会话只读端点:新在前、首问预览、已摘要标记、messages 回载与 404。"""
import pytest

pytestmark = pytest.mark.asyncio


async def test_list_conversations_newest_first_with_preview(make_client, db_store):
    client, app = await make_client(FakeChatModel-of-helpers)
    r1 = await _chat(client, "第一通会话的首问")   # 不带 session_id
    r2 = await _chat(client, "第二通会话的首问")
    resp = await client.get("/api/conversations")
    items = resp.json()["items"]
    assert [i["id"] for i in items][0] == r2["session_id"]      # 新在前
    assert items[0]["preview"].startswith("第二通")
    assert items[0]["summarized"] is False


async def test_summarized_flag_and_messages_endpoint(make_client, db_store):
    ...  # append_summary 后 summarized=True;GET messages 回载两条;未知 id 404
```

- [ ] **Step 2-4: 实现 + 全量绿**(store.list_conversations 两条查询:会话列表 + 每会话 MIN(id) where role='user' 的内容;preview 截断服务端做)

- [ ] **Step 5: Commit** `git commit -m "ch07: GET /api/conversations 与 /{id}/messages 只读端点"`

---

### Task 7: 前端会话侧栏(Vibe 例外,不走 TDD/code review)

**Files:**
- Modify: `static/index.html`

**Interfaces:** `GET /api/conversations`、`GET /api/conversations/{id}/messages`;发送携带当前 `sessionId`;done 帧后静默刷新侧栏;侧栏请求失败 catch 隐藏侧栏。

- [ ] **Step 1: 布局** body 左侧加 `<aside id="sidebar">`:头部「会话」+「新对话」按钮;`<ul id="conv-list">`;主聊天区右移(flex 布局)
- [ ] **Step 2: 逻辑** `loadConversations()`(fetch 渲染:首问预览单行省略、summarized 加「摘要」徽标、当前会话高亮)、`switchConversation(id)`(设 sessionId、拉 messages 渲染气泡、关欢迎语)、`newConversation()`(sessionId=null、清聊天区)、`send()` 带当前 sessionId、done 后 `loadConversations()`(catch 静默)
- [ ] **Step 3: 验证** `node --check`(抽取 JS 校验)、服务起后 curl 两个端点、浏览器手测点切(或 Task 8 浏览器自动化)
- [ ] **Step 4: Commit** `git commit -m "ch07: 前端会话侧栏——新对话/切换回载/已摘要标记/静默降级"`

---

### Task 8: 真机验收(docker MySQL + 真模型全链)

**Files:**
- Create: `scripts/acceptance_ch07.py`(httpx 脚本化多轮对话 + app.log 断言)、`reports/ch07-acceptance.md`
- Modify: `README.md`(ch07 验收段 + 配置段 + 已知边界)

- [ ] **Step 1: 验收脚本** 骨架:读 env(`ACCEPT_BASE_URL` 默认 127.0.0.1:8000);`chat(session_id, text)` 走 SSE 收 done;场景 A(默认 env):22 轮日常问答 → 断言无 `层1 降级`/`summary trigger`、无 5xx/上游 400;场景 B(演示 env 单独起服务):前 2 轮埋「订单1001 发货问题 + 手机号」,续 14 轮 300 字级长消息逼级联 → 断言 app.log 出现 `层1 降级`→`summary trigger`→`summary done 第1段`;问「最开始那个订单后来怎么说」→ 断言答复含「1001」;场景 C:done 帧时间戳早于 summary done 日志(非阻塞)
- [ ] **Step 2: 环境准备** `docker compose down -v && up -d` → `db.seed`/`seed_conversations` → apply 两个 ch07 SQL(或 init.sql 全新)→ `build_kb` → 起 uvicorn(后台任务方式,ch06 教训:沙箱回收 nohup 进程组)
- [ ] **Step 3: 跑场景 A/B/C** 证据逐条落 `reports/ch07-acceptance.md`(log 摘录 + 断言输出)
- [ ] **Step 4: 侧栏浏览器终验**(browser-use 可用则自动化:开两个会话、切回、续聊;不可则 curl 端点证据 + 留用户点验)
- [ ] **Step 5: README 收口**(ch07 验收五条 + 新配置表 + 已知边界增补)
- [ ] **Step 6: Commit** `git commit -m "ch07: 真机验收五条全过+README 收口"`

---

### Task 9: Code Review(独立评审 + 修复波)

- [ ] **Step 1: 派发独立 subagent 全量评审** `git diff <brainstorm-commit>..HEAD`,重点:锚点数学与边界(空历史/首轮/降级到只剩一轮)、两套真源一致性、并发(在飞摘要/双开首轮)、SSE 契约兼容、日志口径与验收 grep 对齐、测试缺口
- [ ] **Step 2: 结论分级 Critical/Important/Minor**;Critical/Important 必修且带回归测试;Minor 修或记档写理由
- [ ] **Step 3: 修复波 + 全量 pytest 全绿;dev-notes 补记**

---

### Task 10: finish(完结交付)

- [ ] **Step 1:** spec「实现期偏差」回填;dev-notes finish 段(演示命令/测试结果/文档路径);`reports/ch07-acceptance.md` 齐备
- [ ] **Step 2:** 最终提交;交付消息:功能演示命令、测试结果、dev-notes 路径
