# ch06 分流器正式版 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch05 占位分流器做成正式版:LLM 指代消解+Query 改写、八类意图识别(七类+其他,带置信度与降级路)、退款/售后确定性子流程(订单数据→Query 扩写→强制政策检索→窄化交主力 Agent)、订单选择器与退款单前端。

**Architecture:** LangGraph 图在 ch05 骨架上扩展:resolve 节点升级为 LLM 结构化改写(带 resume 旁路),intent 升级为四件套 prompt+置信度,新增 `app/graph/refund.py` 五节点确定性子流程;订单选择器停走用**无状态回传**(选择器数据随 SSE 帧下发、点选随请求带回,`interrupt()` 已被 spike 排除);退款单复用 `/api/tickets`。

**Tech Stack:** FastAPI + LangChain 1.4 + LangGraph 1.2(py3.10 async 约束见 spec)+ MySQL/Milvus 既有链路;无新增依赖。

**Spec:** `docs/superpowers/specs/2026-09-08-ch06-router-production-design.md`(本计划从 spec 论证,执行者两份都读)

## Global Constraints

- 无新增组件/依赖:延用现有聊天上游、Milvus Lite、MySQL;订单不建表(固定 mock 目录)
- py3.10 async:禁用 `get_stream_writer`/`interrupt()`(同款官方守卫),发帧一律走 `configurable.sink` 队列桥接(ch05 既有 `_with_sink`)
- 意图输出契约:裸 JSON 文本 + 宽松解析,畸形/超纲 → `{"intent": "其他", "confidence": 0.0}`,trace 记 `malformed=true`
- Query 扩写只在退款退货/售后子流程内,检索侧现查现用;知识库入库侧不动
- 缺订单号不让模型猜:regex 抽取(「订单N」3-6 位优先,裸 4-6 位独立数字兜底),抽不到发选择器
- 需求澄清留在主力 Agent;意图识别只判「要干什么」
- 纯 Prompt/数据类任务用评估集真跑替代 TDD;前端 Vibe Coding 不走流程
- 每任务收尾跑全量 `pytest`;测试替身不出网

---

### Task 1: 订单目录单一来源 + 稳定化 query_order + GET /api/orders

**Files:**
- Create: `app/orders.py`
- Modify: `app/tools/definitions.py`(query_order 改调 get_order)
- Create: `app/routers/orders.py`
- Modify: `app/main.py`(挂路由)
- Modify: `app/config.py`(ch06 三个意图设置)、`app/llm.py`(model 覆盖参数)
- Test: `tests/test_ch06_orders.py`

**Interfaces:**
- Produces: `get_order(order_id: str) -> dict`(命中目录副本,未命中 `{"order_id": ..., "error": "未找到该订单"}`)、`list_orders() -> list[dict]`(固定顺序)、`extract_order_id(text: str) -> str | None`;`GET /api/orders` → `{"items": [...]}`;`make_extract_model(settings, model: str | None = None)`;Settings 新增 `intent_escalation_enabled: bool = False`、`intent_confidence_floor: float = 0.6`、`intent_small_model: str = ""`
- 消费方:Task 3(extract_order_id 由 prepare_order 用)、Task 4(fetch_order/list_orders)、Task 6(builder 构造意图模型)

- [ ] **Step 1: 写失败测试** `tests/test_ch06_orders.py`

```python
"""订单目录:固定三单、未命中话术、单号抽取正则;query_order 稳定化契约。"""
from app.orders import ORDER_CATALOG, extract_order_id, get_order, list_orders


def test_catalog_three_fixed_orders():
    assert len(ORDER_CATALOG) == 3
    assert list_orders()[0]["order_id"] == "1001"
    assert list_orders()[0]["product"] == "无线耳机"
    assert list_orders()[1]["status"] == "已发货"


def test_get_order_hit_returns_copy_without_mutation():
    o = get_order("1002")
    o["product"] = "改坏"
    assert get_order("1002")["product"] == "机械键盘"


def test_get_order_miss_returns_error_dict():
    o = get_order("9999")
    assert o["error"] == "未找到该订单"
    assert o["order_id"] == "9999"


def test_extract_contextual_first():
    assert extract_order_id("订单1001的物流到哪了") == "1001"
    assert extract_order_id("order 1002 什么状态") == "1002"


def test_extract_bare_fallback_4_to_6_digits():
    assert extract_order_id("1001能退吗") == "1001"
    assert extract_order_id("SH-E300 多少钱") is None      # 3 位不误伤
    assert extract_order_id("普通一句话没有数字") is None
    assert extract_order_id("123456789太长不抓") is None    # 9 连位不带边界


async def test_query_order_tool_stable():
    from tests.conftest import make_session_factory
    from app.tools.definitions import build_tools
    from app.embedding import FakeEmbedding if False else None  # 占位防误导入,实现时删除此行
```

(末一个用例实现时写成:)

```python
import json


async def test_query_order_tool_stable():
    from tests.conftest import make_session_factory
    from app.tools.definitions import build_tools

    tools = {t.name: t for t in build_tools(make_session_factory(), embedder=object(), vectors=object())}
    raw = tools["query_order"].invoke({"order_id": "1001"})
    data = json.loads(raw)
    assert data["product"] == "无线耳机" and data["amount"] == 299.0
    assert "未找到" in json.loads(tools["query_order"].invoke({"order_id": "424242"}))["error"]
```

注意:build_tools 非生产分支(embedder/vectors 已注入)不会构造真检索链,`embedder=object()` 即可,query_order 不触网。

- [ ] **Step 2: 跑测试确认失败**:`.venv/bin/pytest tests/test_ch06_orders.py -v` → FAIL(`No module named app.orders`)
- [ ] **Step 3: 实现** `app/orders.py`

```python
"""订单固定目录(ch06):不建表、不接真实 API,选择器 /api/orders、query_order、子流程三处共用单一来源。"""
import re

ORDER_CATALOG: dict[str, dict] = {
    "1001": {"order_id": "1001", "product": "无线耳机", "amount": 299.0, "status": "已签收", "created_at": "2026-08-30"},
    "1002": {"order_id": "1002", "product": "机械键盘", "amount": 399.0, "status": "已发货", "created_at": "2026-09-03"},
    "1003": {"order_id": "1003", "product": "硅胶手机壳", "amount": 19.9, "status": "待发货", "created_at": "2026-09-07"},
}

# 「订单1001」「order 1002」带上下文优先;裸 4-6 位独立数字兜底(SH-E300 的 3 位数字不误伤)
_ORDER_ID_CONTEXTUAL = re.compile(r"(?:订单|order)\s*号?\s*(\d{3,6})", re.IGNORECASE)
_ORDER_ID_BARE = re.compile(r"(?<!\d)(\d{4,6})(?!\d)")


def get_order(order_id: str) -> dict:
    order = ORDER_CATALOG.get(str(order_id))
    if order is None:
        return {"order_id": str(order_id), "error": "未找到该订单"}
    return dict(order)


def list_orders() -> list[dict]:
    return [dict(o) for o in ORDER_CATALOG.values()]


def extract_order_id(text: str) -> str | None:
    m = _ORDER_ID_CONTEXTUAL.search(text or "")
    if m:
        return m.group(1)
    m = _ORDER_ID_BARE.search(text or "")
    return m.group(1) if m else None
```

`app/tools/definitions.py` query_order 函数体替换(random/datetime 若仅他处使用则保留导入):

```python
    @tool
    def query_order(order_id: str) -> str:
        """按订单号查询订单信息:商品、金额、状态。用户问订单相关问题时使用。"""
        from app.orders import get_order

        return json.dumps(get_order(order_id), ensure_ascii=False)
```

`app/routers/orders.py`:

```python
"""ch06:订单列表(固定 mock 目录),订单选择器的数据源。"""
from fastapi import APIRouter

from app.orders import list_orders

router = APIRouter()


@router.get("/api/orders")
async def orders() -> dict:
    return {"items": list_orders()}
```

`app/main.py`:`from app.routers import chat, extract, orders, sessions, tickets` 与 `app.include_router(orders.router)`。`app/config.py` 追加:

```python
    # ch06:意图识别(默认直接主模型;escalation 开启后小模型先判、置信度低于阈值大模型重判一次)
    intent_escalation_enabled: bool = False
    intent_confidence_floor: float = 0.6
    intent_small_model: str = ""  # 缺省同 openai_model
```

`app/llm.py` `make_extract_model` 加覆盖参数:

```python
def make_extract_model(settings: Settings, model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.extract_temperature,
    )
```

- [ ] **Step 4: 全量测试**:`.venv/bin/pytest` → 若 `tests/test_tools.py` 有 query_order 随机断言,按新契约迁移(固定值/未命中 error)
- [ ] **Step 5: Commit** `git commit -m "ch06: 订单目录单一来源+稳定化 query_order+GET /api/orders"`

---

### Task 2: Prompt 三件套定稿 + 意图宽松解析(纯 Prompt/数据任务,解析函数走 TDD,prompt 文案由 Task 8 评估集真跑验证)

**Files:**
- Modify: `app/prompts.py`(INTENT_PROMPT 四件套重写;新增 RESOLVE_PROMPT/EXPAND_PROMPT 与 ResolvedQuestion/ExpandedQueries 模型)
- Test: `tests/test_ch06_prompts.py`

**Interfaces:**
- Produces: `parse_intent(raw: str) -> tuple[str, float, bool]`(intent, confidence, malformed;畸形/超纲 → ("其他", 0.0, True));`ResolvedQuestion(resolved: str, changed: bool)`;`ExpandedQueries(queries: list[str])`;`INTENTS` 含「其他」八项
- 消费方:Task 3(entry 节点)、Task 4(expander 结构化模型)

- [ ] **Step 1: 写失败测试** `tests/test_ch06_prompts.py`

```python
"""意图宽松解析(裸 JSON 契约)与三件套 prompt 常量。"""
from app.graph.entry import INTENTS, parse_intent
from app.prompts import EXPAND_PROMPT, INTENT_PROMPT, RESOLVE_PROMPT, ExpandedQueries, ResolvedQuestion


def test_intents_eight_with_other():
    assert INTENTS == ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")


def test_parse_intent_normal():
    intent, conf, bad = parse_intent('{"intent": "退款退货", "confidence": 0.92}')
    assert (intent, conf, bad) == ("退款退货", 0.92, False)


def test_parse_intent_in_code_fence():
    intent, conf, bad = parse_intent('```json\n{"intent": "物流", "confidence": 0.8}\n```')
    assert (intent, conf, bad) == ("物流", 0.8, False)


def test_parse_intent_confidence_missing_or_bad_defaults_zero():
    assert parse_intent('{"intent": "投诉"}') == ("投诉", 0.0, False)
    assert parse_intent('{"intent": "闲聊", "confidence": "很高"}') == ("闲聊", 0.0, False)
    assert parse_intent('{"intent": "闲聊", "confidence": 1.7}')[1] == 1.0  # 截断到 [0,1]


def test_parse_intent_garbage_and_out_of_enum_go_other():
    assert parse_intent("我说不好") == ("其他", 0.0, True)
    assert parse_intent('{"intent": "聊天气", "confidence": 0.5}') == ("其他", 0.0, True)
    assert parse_intent("") == ("其他", 0.0, True)


def test_resolve_prompt_contract():
    # 占位符只有 history/query;文案不含花括号(PromptTemplate 敏感)
    assert set(RESOLVE_PROMPT.input_variables) == {"history", "query"}
    rendered = RESOLVE_PROMPT.format(history="用户:耳机能退吗", query="它能退吗")
    assert "耳机能退吗" in rendered and "它能退吗" in rendered


def test_expand_prompt_contract():
    assert set(EXPAND_PROMPT.input_variables) == {"query"}
    assert ExpandedQueries(queries=["a", "b"]).queries == ["a", "b"]
    assert "3" in INTENT_PROMPT or len(INTENT_PROMPT) > 200  # 四件套文案实质存在(细则见评估)
```

- [ ] **Step 2: 跑测试确认失败**:`.venv/bin/pytest tests/test_ch06_prompts.py -v` → FAIL(`cannot import name 'parse_intent'`)
- [ ] **Step 3: 实现**——`app/prompts.py` 追加(INTENT_PROMPT 整体替换;注意 RESOLVE/EXPAND 文案内禁花括号):

```python
class ResolvedQuestion(BaseModel):
    """ch06 指代消解+改写:resolved 为补全后完整问法;changed=false 表示原样透传。"""
    resolved: str
    changed: bool = False


class ExpandedQueries(BaseModel):
    """ch06 检索扩写:3-4 条侧重点不同的标准查询。"""
    queries: list[str] = []


RESOLVE_PROMPT = PromptTemplate.from_template(
    "你是电商客服的问法澄清器。根据对话历史处理用户最新消息:\n"
    "1. 消息里有代词或省略指代(它、这个、上面那单等)时,结合历史把指代替换成具体对象,"
    "补全成不看历史也能看懂的完整问题;\n"
    "2. 口语、模糊的问法改写成规范问法;\n"
    "3. 消息本身已完整、指代明确时,原样返回且 changed 取 false;\n"
    "只依据历史里出现过的事实补全,不新增假设;changed 取 true 表示你改写过。只输出两个字段:resolved(字符串)、changed(布尔)。\n"
    "对话历史:\n{history}\n最新消息:{query}"
)

EXPAND_PROMPT = PromptTemplate.from_template(
    "你是电商客服的检索查询扩写器。把下面这条退款/售后相关问题泛化成 3 到 4 条侧重点不同的标准检索查询,"
    "分别覆盖:退货退款条件、申请时效限制、特殊品类限制、运费或费用承担等角度;"
    "查询里不要出现订单号、金额等个人信息;不要重复相同角度;只输出 queries 字段(字符串数组)。\n"
    "问题:{query}"
)
```

INTENT_PROMPT(纯字符串,非 PromptTemplate,花括号无碍):

```python
INTENT_PROMPT = """你是电商客服的意图分类器。从下面八个选项里选一个,只输出 JSON 对象,只有两个字段:
{"intent": "物流|订单|商品咨询|退款退货|售后|投诉|闲聊|其他", "confidence": 0到1之间的小数}

判类口径:问包裹/快递到哪了、多久收到→物流;查订单状态/订单信息→订单;商品参数、价格、库存→商品咨询;
退钱、退货流程与政策→退款退货;安装、维修、换货等售后问题→售后;不满、要讨说法→投诉;寒暄或与购物无关的话题→闲聊;
拿不准、超出店铺业务、无法归入以上七类→其他,不要硬塞。

边界样例:
「东西坏了,我想退钱」→退款退货;「东西坏了,想换个新的」→售后
「你们什么态度,我要投诉」→投诉;「哈哈你们客服真逗」→闲聊
「订单1001到哪了」→物流;「帮我看看订单1001是什么状态」→订单
「这个能退吗」→退款退货;「量子力学是怎么回事」→其他

拿不准时 confidence 给低值并选「其他」。只输出 JSON,不要任何其他文字。"""
```

`app/graph/entry.py` 加(常量先加,Task 3 再改节点):

```python
import re

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")
INTENT_FALLBACK = "其他"

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_intent(raw: str) -> tuple[str, float, bool]:
    """裸 JSON 契约的宽松解析:剥代码围栏 → json.loads → 枚举校验;畸形/超纲归其他。"""
    text = (raw or "").strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return INTENT_FALLBACK, 0.0, True
    intent = obj.get("intent") if isinstance(obj, dict) else None
    if intent not in INTENTS:
        return INTENT_FALLBACK, 0.0, True
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return intent, min(max(conf, 0.0), 1.0), False
```

- [ ] **Step 4: 全量测试**(旧 `test_ch05_entry` 的 INTENTS 断言会红,本任务先只跑新文件,Task 3 统一迁移):`.venv/bin/pytest tests/test_ch06_prompts.py -v` → PASS
- [ ] **Step 5: Commit** `git commit -m "ch06: 意图四件套 prompt+宽松解析+resolve/expand 结构化模型"`

---

### Task 3: entry 节点升级(resolve 正式版 / intent 置信度+降级路 / route 八出口)

**Files:**
- Modify: `app/graph/entry.py`
- Test: `tests/test_ch06_router.py`(新建;`tests/test_ch05_entry.py` 按新契约迁移)

**Interfaces:**
- Consumes: `parse_intent`(Task 2)、`ResolvedQuestion`/`RESOLVE_PROMPT`(Task 2)、`make_extract_model`(Task 1)
- Produces: `make_resolve_node(resolver)` → 节点 `(state, writer=None)`;`make_intent_node(model, escalator=None, floor=0.6)`;`route(state) -> "retrieve"|"agent"|"complaint"|"chitchat"|"prepare_order"`;resolve 旁路:`state["resume_question"]` 非空时不过 LLM
- 消费方:Task 6(builder 组图)、Task 8(eval 脚本复用同一节点代码)

- [ ] **Step 1: 写失败测试** `tests/test_ch06_router.py`

```python
"""resolve 正式版 / intent 置信度与降级路 / route 八出口。"""
from app.graph.entry import INTENT_FALLBACK, make_intent_node, make_resolve_node, route
from app.prompts import ResolvedQuestion
from tests.helpers import EchoResolver, fake_chat


def _state(**kw):
    base = dict(session_id=1, user_message="它能退吗", resolved_message="", intent="",
                intent_confidence=0.0, evidence=[], low_confidence=False, low_reason="",
                gate_passed=False, agent_steps=0, final_reply="", suggested_actions=[],
                trace=[], messages=[], history=[], order={}, expand_queries=[],
                refund_flow=False, pending_order_id="", resume_order_id="", resume_question="")
    base.update(kw)
    return base


async def test_resolve_rewrites_with_history():
    resolver = EchoResolver({"它能退吗": ResolvedQuestion(resolved="订单1001的无线耳机能退吗", changed=True)})
    frames = []
    node = make_resolve_node(resolver)
    out = await node(_state(history=[{"role": "user", "content": "SH-E300耳机多少钱"}]),
                     writer=frames.append)
    assert out["resolved_message"] == "订单1001的无线耳机能退吗"
    assert frames and frames[0]["type"] == "resolved" and frames[0]["changed"] is True
    assert "node=resolve changed=True" in out["trace"][-1]


async def test_resolve_passthrough_when_complete():
    node = make_resolve_node(EchoResolver({}))  # 未命中映射 → 原样透传 changed=False
    frames = []
    out = await node(_state(user_message="退货政策是什么"), writer=frames.append)
    assert out["resolved_message"] == "退货政策是什么"
    assert frames == []  # changed=False 不发帧
    assert "changed=False" in out["trace"][-1]


async def test_resolve_resume_bypass_skips_llm():
    node = make_resolve_node(EchoResolver({}))
    frames = []
    out = await node(_state(resume_question="订单1002的机械键盘能退吗",
                            resume_order_id="1002"), writer=frames.append)
    assert out["resolved_message"] == "订单1002的机械键盘能退吗"
    assert frames[0]["type"] == "resolved"
    assert "resume_bypass" in out["trace"][-1]


async def test_resolve_llm_error_falls_back_passthrough():
    class Boom:
        async def ainvoke(self, text):
            raise RuntimeError("上游挂了")

    node = make_resolve_node(Boom())
    out = await node(_state(), writer=None)
    assert out["resolved_message"] == "它能退吗"


async def test_intent_outputs_confidence_and_trace():
    node = make_intent_node(fake_chat('{"intent": "退款退货", "confidence": 0.9}'))
    out = await node(_state(resolved_message="订单1001的无线耳机能退吗"))
    assert out["intent"] == "退款退货" and out["intent_confidence"] == 0.9
    assert "confidence=0.90" in out["trace"][-1]


async def test_intent_malformed_goes_other_with_flag():
    node = make_intent_node(fake_chat("我说不好"))
    out = await node(_state())
    assert out["intent"] == "其他" and out["intent_confidence"] == 0.0
    assert "malformed=true" in out["trace"][-1]


async def test_intent_escalation_low_confidence_rejudges():
    class SeqModel:
        def __init__(self, replies):
            self.replies = list(replies)

        async def ainvoke(self, messages, **kw):
            return fake_chat(self.replies.pop(0)) if False else _aimsg(self.replies.pop(0))

    from langchain_core.messages import AIMessage as _AM

    def _aimsg(text):
        return _AM(content=text)

    small = SeqModel(['{"intent": "商品咨询", "confidence": 0.3}'])
    big = SeqModel(['{"intent": "退款退货", "confidence": 0.95}'])
    node = make_intent_node(small, escalator=big, floor=0.6)
    out = await node(_state())
    assert out["intent"] == "退款退货" and out["intent_confidence"] == 0.95
    assert "escalated=true" in out["trace"][-1]


async def test_intent_escalation_skipped_when_confident():
    small = fake_chat('{"intent": "闲聊", "confidence": 0.9}')
    big = fake_chat('{"intent": "退款退货", "confidence": 0.9}')  # 不应被消费
    node = make_intent_node(small, escalator=big, floor=0.6)
    out = await node(_state())
    assert out["intent"] == "闲聊"


def test_route_refund_flows_to_prepare_order():
    assert route(_state(intent="退款退货")) == "prepare_order"
    assert route(_state(intent="售后")) == "prepare_order"


def test_route_other_and_unchanged_outlets():
    assert route(_state(intent="其他")) == "agent"
    for intent, expect in [("商品咨询", "retrieve"), ("物流", "agent"), ("订单", "agent"),
                           ("投诉", "complaint"), ("闲聊", "chitchat")]:
        assert route(_state(intent=intent)) == expect
    assert INTENT_FALLBACK == "其他"
```

- [ ] **Step 2: 跑测试确认失败**:`.venv/bin/pytest tests/test_ch06_router.py -v` → FAIL(`cannot import name 'make_resolve_node'`)
- [ ] **Step 3: 实现** `app/graph/entry.py`(整文件重写):

```python
"""入口节点:指代消解+改写(ch06 正式版,带 resume 旁路)+ 意图识别(四件套,置信度+降级路)+ 写死分流。"""
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts import INTENT_PROMPT, RESOLVE_PROMPT, ResolvedQuestion

logger = logging.getLogger(__name__)

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")
INTENT_FALLBACK = "其他"

KNOWLEDGE_INTENTS = {"商品咨询"}
REFUND_INTENTS = {"退款退货", "售后"}
BUSINESS_INTENTS = {"物流", "订单"}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

_ROLE_ZH = {"user": "用户", "assistant": "客服"}


def parse_intent(raw: str) -> tuple[str, float, bool]:
    """裸 JSON 契约的宽松解析:剥代码围栏 → json.loads → 枚举校验;畸形/超纲归其他(malformed=true)。"""
    text = (raw or "").strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return INTENT_FALLBACK, 0.0, True
    intent = obj.get("intent") if isinstance(obj, dict) else None
    if intent not in INTENTS:
        return INTENT_FALLBACK, 0.0, True
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return intent, min(max(conf, 0.0), 1.0), False


def _history_text(history: list) -> str:
    return "\n".join(f"{_ROLE_ZH.get(h['role'], h['role'])}:{h['content']}"
                     for h in history if h.get("content")) or "(无)"


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def make_resolve_node(resolver):
    """resolver = with_structured_output(ResolvedQuestion) 产物;resume 旁路不走 LLM;异常兜底透传。"""

    async def resolve_node(state, writer=None) -> dict:
        if state.get("resume_question"):
            q = state["resume_question"]
            _emit(writer, {"type": "resolved", "changed": True, "question": q})
            return {"resolved_message": q,
                    "trace": [*state["trace"], f"node=resolve resume_bypass q={q}"]}
        try:
            result = await resolver.ainvoke(
                RESOLVE_PROMPT.format(history=_history_text(state["history"]),
                                      query=state["user_message"]))
            resolved = (result.resolved or "").strip() if isinstance(result, ResolvedQuestion) \
                else str(getattr(result, "resolved", "") or "").strip()
        except Exception:
            logger.warning("resolve 上游失败,原样透传", exc_info=True)
            resolved = ""
        if not resolved:
            resolved = state["user_message"]
        changed = bool(getattr(result, "changed", False)) and resolved != state["user_message"]
        if changed:
            _emit(writer, {"type": "resolved", "changed": True, "question": resolved})
        return {"resolved_message": resolved,
                "trace": [*state["trace"], f"node=resolve changed={changed} q={resolved}"]}

    return resolve_node


def make_intent_node(model, escalator=None, floor: float = 0.6):
    """默认单次大模型;escalator 给定时小模型先判、confidence<floor 大模型重判一次取其结果。"""

    async def intent_node(state) -> dict:
        messages = [SystemMessage(content=INTENT_PROMPT),
                    HumanMessage(content=state["resolved_message"])]
        resp = await model.ainvoke(messages)
        intent, conf, malformed = parse_intent(resp.content)
        escalated = False
        if escalator is not None and conf < floor:
            resp2 = await escalator.ainvoke(messages)
            intent, conf, malformed = parse_intent(resp2.content)
            escalated = True
        line = f"node=intent intent={intent} confidence={conf:.2f}"
        if escalated:
            line += " escalated=true"
        if malformed:
            line += " malformed=true"
        return {"intent": intent, "intent_confidence": conf,
                "trace": [*state["trace"], line]}

    return intent_node


def route(state) -> str:
    if state["intent"] in KNOWLEDGE_INTENTS:
        return "retrieve"
    if state["intent"] in REFUND_INTENTS:
        return "prepare_order"
    if state["intent"] in BUSINESS_INTENTS:
        return "agent"
    return "complaint" if state["intent"] == "投诉" else "chitchat" \
        if state["intent"] == "闲聊" else "agent"
```

route 末行实现时写清晰版:

```python
def route(state) -> str:
    if state["intent"] in KNOWLEDGE_INTENTS:
        return "retrieve"
    if state["intent"] in REFUND_INTENTS:
        return "prepare_order"
    if state["intent"] in BUSINESS_INTENTS:
        return "agent"
    if state["intent"] == "投诉":
        return "complaint"
    if state["intent"] == "闲聊":
        return "chitchat"
    return "agent"  # 其他/兜底:交主力 Agent 需求澄清
```

`tests/helpers.py` 追加 EchoResolver:

```python
class EchoResolver:
    """resolve 替身:映射命中返回预置 ResolvedQuestion,未命中原样透传 changed=False。"""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    async def ainvoke(self, text):
        from app.prompts import ResolvedQuestion

        query = text.split("最新消息:")[-1].strip()
        hit = self.mapping.get(query)
        if hit is not None:
            return hit
        return ResolvedQuestion(resolved=query, changed=False)
```

helpers `__all__` 追加 `"EchoResolver"`。

- [ ] **Step 4: 迁移旧测试** `tests/test_ch05_entry.py`:fallback 断言 商品咨询→其他;`route_four_outlets` 中 退款退货/售后 → "prepare_order";resolve_node 直接函数已变工厂——改为 `node = make_resolve_node(EchoResolver({}))` 后调用;`test_intent_parses_llm_json` 断言补 confidence trace。文件头注释同步改「ch06 契约」。旧文件与 test_ch06_router 有重叠的用例(route 七出口等)直接删除,以 ch06 文件为准。
- [ ] **Step 5: 全量测试**:`.venv/bin/pytest` — 注意 builder 仍引用旧 resolve_node/intent_node 会 ImportError;本任务同步把 `app/graph/builder.py` 的 `from app.graph.entry import intent_node, resolve_node, route` 改为 `from app.graph.entry import make_intent_node, make_resolve_node, route`,组图部分先最小接线(resolve 用 `make_resolve_node(_default_resolver(settings))`、intent 用 `make_intent_node(model)`,其中 `_default_resolver` 在 builder 内用 `make_extract_model(settings).with_structured_output(ResolvedQuestion, method="function_calling")`;详细组图 Task 6 完成)。`_state()` 不含新字段的旧图测试若红,按 ChatState 默认值补。
- [ ] **Step 6: Commit** `git commit -m "ch06: resolve 正式版/intent 置信度降级路/route 八出口"`

---

### Task 4: 退款售后确定性子流程五节点(prepare_order/ask_order/fetch_order/expand/policy_retrieve)

**Files:**
- Create: `app/graph/refund.py`
- Modify: `app/graph/state.py`(ChatState 六个新字段)
- Test: `tests/test_ch06_refund.py`

**Interfaces:**
- Consumes: `extract_order_id`/`get_order`/`list_orders`(Task 1)、`ExpandedQueries`/`EXPAND_PROMPT`(Task 2)、`RetrievalService.retrieve(query, strategy, category, use_rewrite)`(ch04 既有)
- Produces: `make_refund_nodes(expander, service, kb, pool, top_k) -> (prepare_order_node, ask_order_node, fetch_order_node, expand_node, policy_retrieve_node)`;State 新字段 `intent_confidence: float`、`order: dict`、`expand_queries: list[str]`、`refund_flow: bool`、`pending_order_id: str`、`resume_order_id: str`、`resume_question: str`
- 消费方:Task 6(builder 组图)、Task 5(agent 窄化读 `order`/`refund_flow`)

- [ ] **Step 1: 写失败测试** `tests/test_ch06_refund.py`

```python
"""退款售后确定性子流程:槽位→取数→扩写→强制政策检索。"""
import json

from app.graph.refund import make_refund_nodes
from app.prompts import ExpandedQueries
from tests.helpers import FakeReranker, StubExtractModel


class StubService:
    """retrieve 替身:按查询文本返回预置 chunk 序列(带分),记录调用。"""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def retrieve(self, query, strategy="hybrid_rerank", category=None, use_rewrite=True):
        self.calls.append((query, use_rewrite))
        for key, items in self.mapping.items():
            if key in query:
                from app.retrieval import Retrieved
                return type("R", (), {"items": [Retrieved(chunk_id=c, score=s) for c, s in items],
                                      "low_confidence": False, "reason": "",
                                      "filter_fallback": False, "rewritten": "",
                                      "search_text": "", "degraded": False})()
        from app.retrieval import Retrieved
        return type("R", (), {"items": [], "low_confidence": True, "reason": "空",
                              "filter_fallback": False, "rewritten": "",
                              "search_text": "", "degraded": False})()


class StubKB:
    def get_chunks(self, ids):
        return [type("Row", (), {"questions": f"问题{i}\n变体", "answer": f"答案{i}",
                                 "category": "退货政策", "section_path": f"退货政策.md > 节{i}"})()
                for i, ids_i in enumerate(ids)]  # 实现时按 ids 顺序生成对应行


def _state(**kw):
    base = dict(session_id=1, user_message="这个能退吗", resolved_message="订单1001的无线耳机能退吗",
                intent="退款退货", intent_confidence=0.9, evidence=[], low_confidence=False,
                low_reason="", gate_passed=False, agent_steps=0, final_reply="",
                suggested_actions=[], trace=[], messages=[], history=[], order={},
                expand_queries=[], refund_flow=False, pending_order_id="",
                resume_order_id="", resume_question="")
    base.update(kw)
    return base


def _nodes(mapping=None, top_k=3):
    service = StubService(mapping or {"退货": [(11, 0.9), (12, 0.5)]})
    kb = StubKB()
    pool = _StubPool()
    expander = StubExtractModel(result=ExpandedQueries(
        queries=["无线耳机退货条件", "七天无理由时效", "耳机类特殊品类限制"]))
    return make_refund_nodes(expander, service, kb, pool, top_k=top_k), service, pool


class _StubPool:
    def __init__(self):
        self.inserted = []

    def insert(self, source, session_id, question, reason):
        self.inserted.append((source, question, reason))


async def test_prepare_order_extracts_from_resolved():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    out = await prep(_state(), writer=None)
    assert out["pending_order_id"] == "1001" and out["refund_flow"] is True
    assert "node=prepare_order order=1001" in out["trace"][-1]


async def test_prepare_order_resume_id_priority():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    out = await prep(_state(resume_order_id="1003", resolved_message="订单1003的硅胶手机壳能退吗"), writer=None)
    assert out["pending_order_id"] == "1003"


async def test_prepare_order_missing_goes_ask():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    out = await prep(_state(resolved_message="这个能退吗"), writer=None)
    assert out["pending_order_id"] == ""


async def test_ask_order_emits_selector_frame_and_user_only_messages():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    frames = []
    out = await ask(_state(resolved_message="这个能退吗"), writer=frames.append)
    assert frames[0]["type"] == "order_selector"
    assert len(frames[0]["items"]) == 3
    assert frames[0]["items"][0]["order_id"] == "1001"
    assert frames[0]["question"] == "这个能退吗"
    assert out["messages"] == [{"role": "user", "content": "这个能退吗"}]
    assert "node=ask_order" in out["trace"][-1]


async def test_fetch_order_hit_and_miss():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    frames = []
    out = await fetch(_state(pending_order_id="1001"), writer=frames.append)
    assert out["order"]["product"] == "无线耳机"
    assert frames and frames[0]["label"] == "订单查询"
    out2 = await fetch(_state(pending_order_id="9999"), writer=None)
    assert out2["order"]["error"] == "未找到该订单"


async def test_expand_emits_queries_and_badge():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    frames = []
    out = await expand(_state(), writer=frames.append)
    assert len(out["expand_queries"]) == 3
    assert frames[0]["label"] == "查询扩写"
    assert "node=expand n=3" in out["trace"][-1]


async def test_expand_failure_falls_back_to_resolved():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    bad = make_refund_nodes(StubExtractModel(error=RuntimeError("挂了")),
                            StubService({}), StubKB(), _StubPool(), top_k=3)[3]
    out = await bad(_state(), writer=None)
    assert out["expand_queries"] == ["订单1001的无线耳机能退吗"]


async def test_policy_retrieve_merges_dedup_renumbers():
    mapping = {"无线耳机退货条件": [(11, 0.9), (12, 0.5)],
               "七天无理由时效": [(11, 0.7), (13, 0.85)],
               "耳机类特殊品类限制": [(14, 0.6)]}
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes(mapping, top_k=3)
    frames = []
    st = _state(expand_queries=list(mapping.keys()))
    out = await retrieve(st, writer=frames.append)
    # 11 出现在两条查询里,去重留最高分 0.9;合并序 11(0.9) > 13(0.85) > 12(0.5),截 top_k=3
    assert [e["n"] for e in out["evidence"]] == [1, 2, 3]
    assert [e["chunk_id"] for e in out["evidence"]] == [11, 13, 12]
    assert all(e["question"] and e["answer"] for e in out["evidence"])
    assert frames and frames[0]["type"] == "citations" and len(frames[0]["items"]) == 3
    assert "node=policy_retrieve" in out["trace"][-1]
    assert all(use_rewrite is False for _, use_rewrite in service.calls)  # 扩写产物不二次改写
    assert pool.inserted == []


async def test_policy_retrieve_empty_evidence_lands_pool():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes({})
    out = await retrieve(_state(expand_queries=["无人问津的问题"]), writer=None)
    assert out["evidence"] == [] and out["low_confidence"] is True
    assert pool.inserted and pool.inserted[0][0] == "retrieval_low_conf"
```

(StubKB.get_chunks 实现时写成按 ids 逐个生成:`return [type("Row", (), {...f"问题{cid}"...})() for cid in ids]`,保证 evidence 的 question 与 chunk_id 对应。)

- [ ] **Step 2: 跑测试确认失败**:`.venv/bin/pytest tests/test_ch06_refund.py -v` → FAIL(`No module named app.graph.refund`)
- [ ] **Step 3: 实现** `app/graph/refund.py`:

```python
"""退款/售后确定性子流程(ch06):槽位检查→订单数据→Query 扩写→强制政策检索;窄化问题交主力 Agent。

扩写只在检索侧现查现用;缺订单号不让模型猜——发 order_selector 帧结束本轮,点选后 resume 旁路再进。
"""
import logging

from app.orders import extract_order_id, get_order, list_orders
from app.prompts import EXPAND_PROMPT

logger = logging.getLogger(__name__)


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def make_refund_nodes(expander, service, kb, pool, top_k: int = 10):
    async def prepare_order_node(state, writer=None) -> dict:
        oid = state.get("resume_order_id") or extract_order_id(state["resolved_message"]) or ""
        line = f"node=prepare_order order={oid or 'missing'}"
        return {"refund_flow": True, "pending_order_id": oid,
                "trace": [*state["trace"], line]}

    async def ask_order_node(state, writer=None) -> dict:
        _emit(writer, {"type": "order_selector",
                       "items": [{"order_id": o["order_id"], "product": o["product"],
                                  "amount": o["amount"], "status": o["status"]}
                                 for o in list_orders()],
                       "question": state["resolved_message"],
                       "original": state["user_message"]})
        return {"messages": [{"role": "user", "content": state["user_message"]}],
                "trace": [*state["trace"], "node=ask_order n=3"]}

    async def fetch_order_node(state, writer=None) -> dict:
        order = get_order(state["pending_order_id"])
        _emit(writer, {"type": "tool_status", "name": "query_order", "label": "订单查询"})
        return {"order": order,
                "trace": [*state["trace"], f"node=fetch_order found={'error' not in order}"]}

    async def expand_node(state, writer=None) -> dict:
        queries: list[str] = []
        try:
            result = await expander.ainvoke(EXPAND_PROMPT.format(query=state["resolved_message"]))
            raw = result.queries if hasattr(result, "queries") else (result or {}).get("queries", [])
            queries = [q.strip() for q in raw if isinstance(q, str) and q.strip()][:4]
        except Exception:
            logger.warning("expand 上游失败,退化为原句", exc_info=True)
        if not queries:
            queries = [state["resolved_message"]]
        _emit(writer, {"type": "tool_status", "name": "expand", "label": "查询扩写"})
        return {"expand_queries": queries,
                "trace": [*state["trace"], f"node=expand n={len(queries)}"]}

    async def policy_retrieve_node(state, writer=None) -> dict:
        merged: dict[int, float] = {}
        for q in state["expand_queries"]:
            try:
                result = service.retrieve(q, strategy="hybrid_rerank", use_rewrite=False)
            except Exception:
                logger.warning("政策检索失败,该查询按空结果处理", exc_info=True)
                continue
            for r in result.items:
                if r.chunk_id not in merged or r.score > merged[r.chunk_id]:
                    merged[r.chunk_id] = r.score
        top = sorted(merged.items(), key=lambda kv: -kv[1])[:top_k]
        rows = kb.get_chunks([cid for cid, _ in top])
        evidence = [
            {"n": i + 1, "chunk_id": cid,
             "question": (row.questions.splitlines() or [""])[0], "answer": row.answer,
             "category": row.category, "section_path": row.section_path or ""}
            for i, (cid, score), row in zip(range(len(top)), top, rows)
        ]
        _emit(writer, {"type": "tool_status", "name": "query_faq", "label": "政策检索"})
        if evidence:
            _emit(writer, {"type": "citations", "items": list(evidence)})
            low = False
        else:
            low = True
            pool.insert("retrieval_low_conf", state["session_id"],
                        state["user_message"], "子流程政策检索无证据")
        top1 = f"{top[0][1]:.2f}" if top else "0.00"
        return {"evidence": evidence, "low_confidence": low,
                "low_reason": "" if not low else "子流程政策检索无证据",
                "trace": [*state["trace"],
                          f"node=policy_retrieve queries={len(state['expand_queries'])} "
                          f"merged={len(merged)} top1={top1} n={len(evidence)}"]}

    return prepare_order_node, ask_order_node, fetch_order_node, expand_node, policy_retrieve_node
```

`app/graph/state.py` ChatState 追加字段(带注释):

```python
    intent_confidence: float      # ch06 意图置信度(trace/评估用)
    order: dict                   # ch06 子流程拿到的订单数据({}=无)
    expand_queries: list[str]     # ch06 扩写产物
    refund_flow: bool             # ch06 本轮走了退款售后子流程(agent 窄化/actions 决策)
    pending_order_id: str         # ch06 prepare_order 抽到的单号(""=缺)
    resume_order_id: str          # ch06 无状态回传:点选带回的订单号
    resume_question: str          # ch06 无状态回传:点选带回的补全问题
```

- [ ] **Step 4: 全量测试**:`.venv/bin/pytest tests/test_ch06_refund.py -v` → PASS(全量此时可能仍红在 builder,Task 6 收口)
- [ ] **Step 5: Commit** `git commit -m "ch06: 退款售后子流程五节点(槽位/取数/扩写/政策检索)"`

---

### Task 5: agent 窄化注入 + refund_form action + 引用编号防撞

**Files:**
- Modify: `app/graph/agent.py`
- Modify: `app/graph/simple.py`(ACTION_LABELS 加 refund_form)
- Modify: `app/graph/logging_node.py`(actions item 带 order_id)
- Test: `tests/test_ch06_agent.py`

**Interfaces:**
- Consumes: `state["order"]`/`state["refund_flow"]`/`state["intent"]`/`state["evidence"]`(Task 4/State)
- Produces: agent 收敛时 `suggested_actions=["refund_form"]`(意图=退款退货、非拒答、子流程轮);System 注入订单 JSON + 窄化指令;`n_offset = len(state.get("evidence") or [])`;ACTION_LABELS["refund_form"]="申请退款";actions 帧 item 形如 `{"action": "refund_form", "label": "申请退款", "order_id": "1001"}`
- 消费方:Task 6(组图后端到端)、Task 7(前端按钮与表单)

- [ ] **Step 1: 写失败测试** `tests/test_ch06_agent.py`

```python
"""agent 子流程窄化:订单数据+政策证据注入、refund_form action、n_offset 续排。"""
import json

from app.graph.agent import make_agent_node
from tests.helpers import ChunkStub, GraphChatModel


def _settings():
    from app.config import Settings

    return Settings(openai_api_key="sk-test", _env_file=None,
                    agent_max_steps=4, agent_token_budget=2000)


class _Registry:
    tools = []
    def labels(self):
        return {}


def _state(**kw):
    base = dict(session_id=1, user_message="这个能退吗", resolved_message="订单1001的无线耳机能退吗",
                intent="退款退货", intent_confidence=0.9, evidence=[], low_confidence=False,
                low_reason="", gate_passed=False, agent_steps=0, final_reply="",
                suggested_actions=[], trace=[], messages=[],
                history=[{"role": "user", "content": "SH-E300耳机多少钱"},
                         {"role": "assistant", "content": "299 元"}],
                order={"order_id": "1001", "product": "无线耳机", "amount": 299.0,
                       "status": "已签收", "created_at": "2026-08-30"},
                expand_queries=[], refund_flow=True, pending_order_id="1001",
                resume_order_id="", resume_question="")
    base.update(kw)
    return base


async def test_narrowing_injects_order_and_instruction():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "这一单可以退。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(evidence=[{"n": 1, "chunk_id": 11, "question": "退货条件",
                                       "answer": "签收后7天内", "category": "退货政策",
                                       "section_path": "退货政策.md"}]), writer=None)
    system = model.prompts[0][0].content
    assert "订单数据" in system and "无线耳机" in system
    assert "能不能退" in system and "1001" in system
    assert "退货条件" in system  # 证据块照旧注入


async def test_aftersale_narrowing_wording():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "可以换新。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    await node(_state(intent="售后"), writer=None)
    assert "修/换/退" in model.prompts[0][0].content


async def test_non_refund_converge_emits_refund_form_action():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "这一单可以退。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(), writer=None)
    assert out["suggested_actions"] == ["refund_form"]


async def test_refusal_keeps_transfer_actions():
    from app.guard import REFUSAL_MARKER

    model = GraphChatModel(intent_reply="{}", turns=[("text", REFUSAL_MARKER + "证据不足,建议转人工。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(), writer=None)
    assert out["suggested_actions"] == ["transfer_human", "create_ticket"]


async def test_non_refund_flow_turn_gets_no_refund_form():
    model = GraphChatModel(intent_reply="{}", turns=[("text", "无线耳机 299 元。")])
    node = make_agent_node(model, _Registry(), _settings(), pool=None)
    out = await node(_state(refund_flow=False, order={}, intent="商品咨询"), writer=None)
    assert out["suggested_actions"] == []


async def test_query_faq_numbering_continues_after_evidence():
    """子流程证据已有 n=1..2 时,agent 内 query_faq 的引用编号从 3 续排(防撞号)。"""
    tool_chunks = [{"name": "query_faq", "args": '{"keyword": "退货"}', "id": "t1", "index": 0}]
    model = GraphChatModel(intent_reply="{}", turns=[
        ("tools", tool_chunks),
        ("text", "结合政策这一单可以退[3]。"),
    ])
    calls = []

    class Reg(_Registry):
        tools = ["query_faq"]

        def labels(self):
            return {"query_faq": "FAQ 检索"}

        async def execute(self, name, args):
            calls.append((name, args))
            return json.dumps({"items": [{"n": 1, "chunk_id": 99, "question": "运费",
                                          "answer": "8 元", "category": "退货政策",
                                          "section_path": "p"}],
                               "low_confidence": False, "reason": "",
                               "filter_fallback": False, "degraded": False},
                              ensure_ascii=False)

    frames = []
    node = make_agent_node(model, Reg(), _settings(), pool=None)
    out = await node(_state(evidence=[{"n": 1, "chunk_id": 11, "question": "a", "answer": "a",
                                       "category": "退货政策", "section_path": "a"},
                                      {"n": 2, "chunk_id": 12, "question": "b", "answer": "b",
                                       "category": "退货政策", "section_path": "b"}]),
                     writer=frames.append)
    cite = [f for f in frames if f["type"] == "citations"][-1]
    assert [it["n"] for it in cite["items"]] == [1, 2, 3]  # 既有证据 + 续排的工具证据
```

- [ ] **Step 2: 跑测试确认失败**:`.venv/bin/pytest tests/test_ch06_agent.py -v` → FAIL(无订单注入/无 refund_form)
- [ ] **Step 3: 实现** `app/graph/agent.py` 改动点:

```python
NARROW_INSTRUCTIONS = {
    "退款退货": "用户处于退款退货流程。请仅依据下方订单数据与参考知识,判断这一单(订单{order_id})能不能退,并说明依据条款;不要回答与这一单无关的问题。",
    "售后": "用户处于售后流程。请仅依据下方订单数据与参考知识,说明这一单(订单{order_id})的售后问题该怎么处理(修/换/退);不要回答与这一单无关的问题。",
}
```

agent_node 内 system 组装后追加:

```python
        if state.get("refund_flow") and state.get("order"):
            instruction = NARROW_INSTRUCTIONS.get(state.get("intent"), "")
            if instruction:
                system += ("\n\n【订单数据】" + json.dumps(state["order"], ensure_ascii=False)
                           + "\n" + instruction.format(order_id=state["order"].get("order_id", "")))
```

`n_offset` 初始化改:`spent, steps, n_offset = 0, 0, len(state.get("evidence") or [])`。

收敛分支 actions 改:

```python
                actions = ACTIONS_ON_REFUSAL if is_refusal(text) else (
                    ["refund_form"] if state.get("refund_flow")
                    and state.get("intent") == "退款退货" else [])
```

`app/graph/simple.py`:`ACTION_LABELS = {"transfer_human": "转人工", "create_ticket": "建工单", "refund_form": "申请退款"}`。

`app/graph/logging_node.py` actions 帧构造改:

```python
        if state["suggested_actions"]:
            items = []
            for a in state["suggested_actions"]:
                item = {"action": a, "label": ACTION_LABELS.get(a, a)}
                if a == "refund_form" and state.get("order"):
                    item["order_id"] = state["order"].get("order_id", "")
                items.append(item)
            _emit(writer, {"type": "actions", "items": items})
```

- [ ] **Step 4: 全量测试**:`.venv/bin/pytest tests/test_ch06_agent.py tests/test_ch05_agent.py -v`;ch05 的 suggested_actions 断言若受影响(知识路径非子流程轮不受影响)按新契约核对
- [ ] **Step 5: Commit** `git commit -m "ch06: agent 窄化注入+refund_form action+引用编号续排防撞"`

---

### Task 6: builder 组图 + resume 契约(/api/chat)+ GET /api/orders 接线 + 端到端测试

**Files:**
- Modify: `app/graph/builder.py`(组图/initial_state/resolver 构造)
- Modify: `app/schemas.py`(OrderResume + ChatRequest.resume)
- Modify: `app/routers/chat.py`(resume 注入)
- Modify: `tests/conftest.py`(make_client 注 resolver/expander 替身)、`tests/helpers.py`(StubExpand)
- Test: `tests/test_ch06_chat_api.py`

**Interfaces:**
- Consumes: Task 3 节点工厂、Task 4 子流程、Task 5 agent
- Produces: `build_graph(model, registry, settings, store, pool, retrieval_service, kb, *, resolver=None, intent_model=None, escalator=None, expander=None)`(缺省 None 时生产路径自建);`initial_state(session_id, user_message, history, resume=None)`;`ChatRequest(message, session_id=None, resume: OrderResume|None)`;`OrderResume(order_id: str(pattern=^\d{3,6}$), question: str)`
- 消费方:Task 7 前端、Task 9 真机验收

- [ ] **Step 1: 写失败测试** `tests/test_ch06_chat_api.py`

```python
"""端到端:选择器轮→resume 旁路→子流程→agent;GET /api/orders;畸形 resume 422。"""
import json

from app.prompts import ExpandedQueries, ResolvedQuestion
from tests.helpers import GraphChatModel, EchoResolver, post_chat_sse, StubExpand


class ToollessGraphModel(GraphChatModel):
    """agent 轮恒返纯文本(无工具),意图 JSON 可配。"""


def _make_client(make_client):
    resolver = EchoResolver({"它能退吗": ResolvedQuestion(
        resolved="订单1001的无线耳机能退吗", changed=True)})
    expander = StubExpand({"订单1001的无线耳机能退吗": ["无线耳机退货条件", "七天无理由时效"]})
    model = ToollessGraphModel(intent_reply='{"intent": "退款退货", "confidence": 0.9}',
                               turns=[("text", "这一单签收未超7天,可以退。")])
    return make_client(model, resolver=resolver, expander=expander)


async def test_ask_order_round_then_resume_completes(make_client):
    client, app = await _make_client(make_client)
    frames = await post_chat_sse(client, {"message": "它能退吗"})
    types = [f["type"] for f in frames]
    assert "resolved" in types and "order_selector" in types
    assert "token" not in types  # 选择器轮没有答案流
    sel = next(f for f in frames if f["type"] == "order_selector")
    assert sel["question"] == "订单1001的无线耳机能退吗"

    frames2 = await post_chat_sse(client, {
        "message": "它能退吗", "session_id": None,
        "resume": {"order_id": "1001", "question": "订单1001的无线耳机能退吗"}})
    types2 = [f["type"] for f in frames2]
    assert "order_selector" not in types2
    assert "citations" in types2
    assert any(f["type"] == "token" and "可以退" in f.get("content", "") for f in frames2)
    actions = next(f for f in frames2 if f["type"] == "actions")
    assert actions["items"][0]["action"] == "refund_form"
    assert actions["items"][0]["order_id"] == "1001"
    assert frames2[-1]["type"] == "done"


async def test_order_number_in_text_skips_selector(make_client):
    resolver = EchoResolver({})
    expander = StubExpand({})
    model = ToollessGraphModel(intent_reply='{"intent": "退款退货", "confidence": 0.9}',
                               turns=[("text", "订单1001可以退。")])
    client, app = await make_client(model, resolver=resolver, expander=expander)
    frames = await post_chat_sse(client, {"message": "订单1001能退吗"})
    types = [f["type"] for f in frames]
    assert "order_selector" not in types and "token" in types


async def test_other_intent_goes_agent(make_client):
    model = ToollessGraphModel(intent_reply='{"intent": "其他", "confidence": 0.2}',
                               turns=[("text", "这个问题我需要进一步了解,您能说详细些吗")])
    client, app = await make_client(model, resolver=EchoResolver({}), expander=StubExpand({}))
    frames = await post_chat_sse(client, {"message": "量子力学怎么解释"})
    assert any(f["type"] == "token" and "详细" in f.get("content", "") for f in frames)


async def test_orders_endpoint(make_client):
    client, app = await _make_client(make_client)
    resp = await client.get("/api/orders")
    body = resp.json()
    assert resp.status_code == 200
    assert [o["order_id"] for o in body["items"]] == ["1001", "1002", "1003"]


async def test_resume_invalid_order_id_422(make_client):
    client, app = await _make_client(make_client)
    resp = await client.post("/api/chat", json={
        "message": "x", "resume": {"order_id": "abc", "question": "q"}})
    assert resp.status_code == 422
```

- [ ] **Step 2: 跑测试确认失败**:`.venv/bin/pytest tests/test_ch06_chat_api.py -v` → FAIL(make_client 不收 resolver/expander 参数)
- [ ] **Step 3: 实现**——

`tests/helpers.py` 追加:

```python
class StubExpand:
    """expand 结构化替身:映射命中返回预置 ExpandedQueries,未命中返回单条原句外壳。
    ainvoke 收到的 text 里「问题:」之后是 query。"""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    async def ainvoke(self, text):
        query = text.split("问题:")[-1].strip()
        return ExpandedQueries(queries=self.mapping.get(query, [query]))
```

`__all__` 追加 `"StubExpand"`。

`tests/conftest.py` `_make` 签名改 `async def _make(chat_model, extract_model=None, resolver=None, expander=None):`,缺省 `resolver = resolver or EchoResolver({})`、`expander = expander or StubExpand({})`,并传给 build_graph:

```python
        app.state.graph = build_graph(
            chat_model, app.state.registry, app.state.settings,
            app.state.store, app.state.pool, service, kb,
            resolver=resolver, expander=expander,
        )
```

`app/schemas.py`:

```python
class OrderResume(BaseModel):
    """ch06 订单选择器无状态回传:点选后前端原样带回的槽位数据。"""
    order_id: str = Field(pattern=r"^\d{3,6}$", description="点选的订单号")
    question: str = Field(min_length=1, description="选择器帧带回的补全问题")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="用户本轮输入")
    session_id: int | None = Field(default=None, description="会话 id,缺省则服务端新建")
    resume: OrderResume | None = Field(default=None, description="订单选择器点选回传,非空走子流程旁路")
```

`app/graph/builder.py` 组图(`build_graph` 全量替换;生产 resolver/expander 自建,测试可注入):

```python
def build_graph(model, registry, settings, store, pool, retrieval_service, kb, *,
                resolver=None, intent_model=None, escalator=None, expander=None):
    retrieve_node, gate_node, fallback_node = make_knowledge_nodes(retrieval_service, kb, pool)
    from app.graph.refund import make_refund_nodes

    if resolver is None:
        resolver = make_extract_model(settings).with_structured_output(
            ResolvedQuestion, method="function_calling")
    if expander is None:
        expander = make_extract_model(settings).with_structured_output(
            ExpandedQueries, method="function_calling")
    if settings.intent_escalation_enabled:
        small = make_extract_model(settings, model=settings.intent_small_model or None)
        big = escalator or make_extract_model(settings)
        _intent = make_intent_node(small, escalator=big, floor=settings.intent_confidence_floor)
    else:
        _intent = make_intent_node(intent_model or model)
    _resolve = make_resolve_node(resolver)
    prep, ask, fetch, expand, policy = make_refund_nodes(
        expander, retrieval_service, kb, pool, top_k=settings.retrieval_final_top_k)

    g = StateGraph(ChatState)
    g.add_node("resolve", _with_sink(_resolve))
    g.add_node("intent", _intent)
    g.add_node("retrieve", _with_sink(retrieve_node))
    g.add_node("gate", _with_sink(gate_node))
    g.add_node("fallback", _with_sink(fallback_node))
    g.add_node("agent", _with_sink(make_agent_node(model, registry, settings, pool)))
    g.add_node("complaint", _with_sink(complaint_node))
    g.add_node("chitchat", _with_sink(chitchat_node))
    g.add_node("prepare_order", _with_sink(prep))
    g.add_node("ask_order", _with_sink(ask))
    g.add_node("fetch_order", _with_sink(fetch))
    g.add_node("expand", _with_sink(expand))
    g.add_node("policy_retrieve", _with_sink(policy))
    g.add_node("log", _with_sink(make_log_node(store, pool)))

    g.add_edge(START, "resolve")
    g.add_conditional_edges("resolve",
                            lambda s: "prepare_order" if s.get("resume_question") else "intent",
                            {"intent": "intent", "prepare_order": "prepare_order"})
    g.add_conditional_edges("intent", route, {
        "retrieve": "retrieve", "agent": "agent", "complaint": "complaint",
        "chitchat": "chitchat", "prepare_order": "prepare_order"})
    g.add_conditional_edges("prepare_order",
                            lambda s: "fetch_order" if s.get("pending_order_id") else "ask_order",
                            {"fetch_order": "fetch_order", "ask_order": "ask_order"})
    g.add_edge("fetch_order", "expand")
    g.add_edge("expand", "policy_retrieve")
    g.add_edge("policy_retrieve", "agent")
    g.add_edge("retrieve", "gate")
    g.add_conditional_edges("gate", lambda s: "fallback" if not s["gate_passed"] else "agent",
                            {"fallback": "fallback", "agent": "agent"})
    for n in ("fallback", "agent", "complaint", "chitchat", "ask_order"):
        g.add_edge(n, "log")
    g.add_edge("log", END)
    return g.compile(checkpointer=InMemorySaver())
```

builder 顶部 import 调整:`from app.graph.entry import make_intent_node, make_resolve_node, route`;`from app.llm import make_extract_model`;`from app.prompts import ExpandedQueries, ResolvedQuestion`。`initial_state` 替换:

```python
def initial_state(session_id: int, user_message: str, history: list, resume=None) -> dict:
    """路由层每回合注入的全量默认值:防 checkpointer 上一回合残留字段泄漏;resume 非空时带槽位。"""
    return {"session_id": session_id, "user_message": user_message, "resolved_message": "",
            "intent": "", "intent_confidence": 0.0, "evidence": [], "low_confidence": False,
            "low_reason": "", "gate_passed": False, "agent_steps": 0, "final_reply": "",
            "suggested_actions": [], "trace": [], "messages": [], "history": history,
            "order": {}, "expand_queries": [], "refund_flow": False,
            "pending_order_id": "", "resume_order_id": resume.order_id if resume else "",
            "resume_question": resume.question if resume else ""}
```

`app/routers/chat.py` 一行改动:

```python
                await graph.ainvoke(
                    initial_state(session_id, body.message, trimmed[1:], resume=body.resume),
                    config)
```

- [ ] **Step 4: 全量测试 + 旧契约迁移**:`.venv/bin/pytest` → 全绿。已知需迁移面:`tests/test_ch05_chat_api.py`(GraphChatModel 的意图回复无 confidence 字段——parse_intent 容忍,confidence 缺省 0.0,断言 intent 的用例应不受影响;凡断言「解析失败归商品咨询」的改为「其他」)、`tests/test_ch05_agent.py`(make_agent_node 直接调用处 _state 需补新 State 字段,或 agent 内全部用 `state.get(...)` 已天然兼容——以实际红名单为准逐个修)
- [ ] **Step 5: Commit** `git commit -m "ch06: 组图接线退款子流程+resume 无状态回传契约+orders 端点"`

---

### Task 7: 前端订单选择器 + resolved 灰字 + 退款表单(Vibe Coding,不走 TDD/评审)

**Files:**
- Modify: `static/index.html`

**Interfaces:**
- Consumes: SSE 帧 `resolved`/`order_selector`/`actions(refund_form)`;`GET /api/orders` 不直接调(选择器数据在帧里);`POST /api/tickets`
- 效果迭代:用户描述→直接改,无评审门;本任务验收 = 浏览器自动化手测(Task 9 验收 4)

- [ ] **Step 1: resolved 帧**——事件分派加分支:

```js
} else if (evt.type === "resolved") {
  if (evt.changed) addResolvedNote(bubble, evt.question);
}
```

```js
// ch06:指代消解可视提示——灰字「已理解:…」,验收 3 的前端证据
function addResolvedNote(bubble, question) {
  const note = document.createElement("div");
  note.className = "resolved-note";
  note.textContent = "已理解:" + question;
  bubble.insertBefore(note, bubble.querySelector(".content"));
  scrollDown();
}
```

样式:`.resolved-note { font-size: 12px; color: #8a919e; margin-bottom: 6px; }`

- [ ] **Step 2: order_selector 帧**——分派分支 + 渲染:

```js
} else if (evt.type === "order_selector") {
  renderOrderSelector(bubble, evt);
}
```

```js
// ch06:订单选择器——可点订单卡,点选后带原问题+补全问题回传,resume 旁路续走子流程
function renderOrderSelector(bubble, evt) {
  const bar = document.createElement("div");
  bar.className = "order-bar";
  for (const o of evt.items || []) {
    const card = document.createElement("button");
    card.className = "order-card";
    card.innerHTML = '<div class="o-title"></div><div class="o-meta"></div>';
    card.querySelector(".o-title").textContent = "订单" + o.order_id + " · " + o.product;
    card.querySelector(".o-meta").textContent = "¥" + o.amount + " · " + o.status;
    card.onclick = () => pickOrder(bar, o, evt);
    bar.appendChild(card);
  }
  bubble.appendChild(bar);
  scrollDown();
}
async function pickOrder(bar, order, evt) {
  if (state.busy) return;
  [...bar.children].forEach((c) => { c.disabled = true; if (c !== bar) c.classList.add("picked"); });
  addUserMsg(evt.original || "");
  const bubble = addBotMsg();
  setBusy(true);
  try {
    await streamChat({ message: evt.original || "", session_id: state.sessionId,
                       resume: { order_id: order.order_id, question: evt.question } }, bubble);
  } finally { setBusy(false); text.focus(); }
}
```

实现时把 `send()` 里「fetch + SSE 循环 + finally 收尾」抽成 `streamChat(payload, bubble)` 复用(send 与 pickOrder 共用);本步允许顺手重构,帧处理逻辑不变。

样式:

```css
.order-bar { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.order-card {
  border: 1px solid #d6dae2; background: #fff; border-radius: 10px;
  padding: 8px 12px; cursor: pointer; text-align: left; font-size: 13px;
}
.order-card:hover { border-color: #4f7cff; background: #eef3ff; }
.order-card.picked { border-color: #4f7cff; background: #eef3ff; }
.order-card:disabled { cursor: default; opacity: .7; }
.order-card .o-title { font-weight: 600; color: #1a1d24; margin-bottom: 2px; }
.order-card .o-meta { color: #8a919e; font-size: 12px; }
```

- [ ] **Step 3: refund_form action**——`renderActions` 的 `btn.onclick` 分派加分支:

```js
btn.onclick = () => {
  if (it.action === "transfer_human") transferHuman(bar);
  else if (it.action === "create_ticket") createTicket(btn);
  else if (it.action === "refund_form") renderRefundForm(bar, it.order_id);
};
```

```js
// ch06:退款单表单——原因固定类目下拉 + 备注,提交复用 /api/tickets(类型:售后)
function renderRefundForm(bar, orderId) {
  bar.remove();
  const form = document.createElement("div");
  form.className = "refund-form";
  form.innerHTML =
    '<div class="rf-title">申请退款 · 订单' + orderId + '</div>' +
    '<select class="rf-reason"></select>' +
    '<input class="rf-note" placeholder="补充说明(可空)" />' +
    '<div class="rf-row"><button class="rf-submit">提交退款申请</button><span class="rf-status"></span></div>';
  const REASONS = ["质量问题", "七天无理由", "尺寸不合适", "发错货", "与描述不符", "不想要了"];
  for (const r of REASONS) form.querySelector(".rf-reason").add(new Option(r, r));
  form.querySelector(".rf-submit").onclick = async () => {
    const reason = form.querySelector(".rf-reason").value;
    const note = form.querySelector(".rf-note").value.trim();
    const status = form.querySelector(".rf-status");
    try {
      const resp = await fetch("/api/tickets", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.session_id_placeholder, ticket_type: "售后",
          description: "订单" + orderId + "申请退款;原因:" + reason + (note ? ";备注:" + note : "") }),
      });
      const body = await resp.json();
      if (resp.ok) {
        form.querySelector(".rf-submit").disabled = true;
        form.querySelector(".rf-reason").disabled = true;
        form.querySelector(".rf-note").disabled = true;
        status.textContent = "工单 " + body.ticket_no + " 已创建";
      } else { status.textContent = "提交失败"; }
    } catch (e) { status.textContent = "提交失败"; }
  };
  bubble_append(form);   // 实现时:把 form 挂到当前 actions 所在气泡(冒泡找 .bubble)
  scrollDown();
}
```

(实现注意:renderRefundForm 需要 bubble 引用——renderActions 已有 bubble 参数,把它透传进 renderRefundForm;`session_id` 用 `state.sessionId`。)

样式:

```css
.refund-form {
  margin-top: 8px; padding: 10px; border: 1px solid #e0e5ee;
  border-radius: 10px; background: #f6f8fc; display: flex; flex-direction: column; gap: 8px;
}
.refund-form .rf-title { font-size: 13px; font-weight: 600; color: #1a1d24; }
.refund-form select, .refund-form input {
  border: 1px solid #d6dae2; border-radius: 8px; padding: 7px 10px; font-size: 13px; background: #fff;
}
.refund-form .rf-row { display: flex; align-items: center; gap: 8px; }
.refund-form .rf-submit {
  border: none; background: linear-gradient(135deg, #4f7cff, #6a5cff);
  color: #fff; border-radius: 8px; padding: 7px 16px; cursor: pointer; font-size: 13px;
}
.refund-form .rf-submit:disabled { opacity: .5; cursor: default; }
.refund-form .rf-status { font-size: 12px; color: #34c759; }
```

- [ ] **Step 4: 手测**:起服务(替身不可用,直接真机 Task 9 一并验;本步至少 `python -m http.server` 干渲染不报 JS 错——以 Task 9 浏览器自动化为准)
- [ ] **Step 5: Commit** `git commit -m "ch06: 前端订单选择器/resolved 灰字/退款表单(Vibe)"`

---

### Task 8: 评估集 + eval_router.py 真跑报告(替代 TDD 的质量验证)

**Files:**
- Create: `tests/eval/router_samples.jsonl`(评估集,数据任务)
- Create: `scripts/eval_router.py`
- Create: `tests/test_ch06_eval_set.py`(评估集 schema 校验,普通单测)
- Create: `reports/ch06-router-report.md`(真跑产出,提交)

**Interfaces:**
- Consumes: `make_resolve_node`/`make_intent_node`(Task 3,与线上一同代码路径)、`Settings`/`make_extract_model`
- Produces: `python scripts/eval_router.py [--set tests/eval/router_samples.jsonl] [--report reports/ch06-router-report.md]`;指标:意图准确率(总体/分桶)、指代消解质量(expected_resolved 归一化全等)、JSON 可解析率(malformed 率)、confidence 分布、escalation 开关对照

- [ ] **Step 1: 写评估集** `tests/eval/router_samples.jsonl`——每行一个对话,`turns` 数组按序演化(history 累积),字段:`id`、`tag`(七类/多轮切换/指代/口语/边界/怪问题)、`turns: [{user, expect_intent, expect_resolved?, note?}]`。必须覆盖:

```jsonl
{"id": "R01", "tag": "多轮切换", "turns": [
  {"user": "订单1001的物流到哪了", "expect_intent": "物流"},
  {"user": "它能退吗", "expect_intent": "退款退货", "expect_resolved": "订单1001的无线耳机能退吗"},
  {"user": "那现在到哪个城市了", "expect_intent": "物流", "expect_resolved": "订单1001的物流现在到哪个城市了"}]}
{"id": "R02", "tag": "多轮切换", "turns": [
  {"user": "SH-E300多少钱", "expect_intent": "商品咨询"},
  {"user": "这个能退吗", "expect_intent": "退款退货", "expect_resolved": "SH-E300能退吗"},
  {"user": "多久能发货", "expect_intent": "商品咨询", "expect_resolved": "SH-E300多久能发货"}]}
{"id": "R03", "tag": "指代", "turns": [
  {"user": "你们的退货政策是怎样的", "expect_intent": "退款退货"},
  {"user": "上面说的时效是多久", "expect_intent": "退款退货", "expect_resolved": "退货政策里说的时效是多久"}]}
{"id": "R04", "tag": "口语", "turns": [
  {"user": "咋把钱整回来啊", "expect_intent": "退款退货"},
  {"user": "东西咋还没到啊", "expect_intent": "物流"}]}
{"id": "R05", "tag": "边界", "turns": [
  {"user": "东西坏了想退钱", "expect_intent": "退款退货"},
  {"user": "东西坏了想换个新的", "expect_intent": "售后"},
  {"user": "我要投诉你们", "expect_intent": "投诉"},
  {"user": "帮我看看订单1002什么状态", "expect_intent": "订单"}]}
{"id": "R06", "tag": "边界", "turns": [
  {"user": "你们什么态度啊我要个说法", "expect_intent": "投诉"},
  {"user": "哈哈你们客服真逗", "expect_intent": "闲聊"}]}
{"id": "R07", "tag": "怪问题", "turns": [
  {"user": "量子力学怎么解释", "expect_intent": "其他"},
  {"user": "你会写诗吗", "expect_intent": "其他"},
  {"user": "明天股票会涨吗", "expect_intent": "其他"}]}
{"id": "R08", "tag": "七类", "turns": [
  {"user": "耳机怎么安装", "expect_intent": "售后"},
  {"user": "发票怎么开", "expect_intent": "商品咨询"},
  {"user": "你好呀", "expect_intent": "闲聊"}]}
```

- [ ] **Step 2: 评估集 schema 校验测试** `tests/test_ch06_eval_set.py`

```python
"""评估集 schema 校验:字段齐全、意图标签在枚举内、多轮切换用例存在。"""
import json
from pathlib import Path

from app.graph.entry import INTENTS

PATH = Path("tests/eval/router_samples.jsonl")


def _rows():
    return [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_rows_schema():
    for row in _rows():
        assert row["id"] and row["tag"]
        for t in row["turns"]:
            assert t["user"] and t["expect_intent"] in INTENTS
            if "expect_resolved" in t:
                assert t["expect_resolved"]


def test_covers_multiling_switch_and_other():
    rows = _rows()
    assert any(r["tag"] == "多轮切换" for r in rows)
    assert any(t["expect_intent"] == "其他" for r in rows for t in r["turns"])
    # 验收 1 的来回切:同对话内物流→退款退货→物流
    assert any([t["expect_intent"] for t in r["turns"]][:3] == ["物流", "退款退货", "物流"]
               for r in rows)
```

- [ ] **Step 3: 跑校验**:`.venv/bin/pytest tests/test_ch06_eval_set.py -v` → PASS
- [ ] **Step 4: 实现 `scripts/eval_router.py`**(读 `scripts/eval_retrieval.py` 的 argparse/报告落盘风格;核心循环):

```python
"""ch06 分流器评估:真模型逐轮跑 resolve+intent(与线上同一节点代码),报告落 reports/。

用法:.venv/bin/python scripts/eval_router.py [--set tests/eval/router_samples.jsonl]
      [--report reports/ch06-router-report.md] [--escalate]
"""
import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from app.config import Settings
from app.llm import make_extract_model
from app.graph.entry import make_intent_node, make_resolve_node, parse_intent

from app.prompts import ResolvedQuestion


def _normalize(text: str) -> str:
    return "".join(ch for ch in text if ch not in " ,。?!?!、;;:")


async def _eval_turn(resolve_node, intent_node, history, turn):
    state = {"session_id": 0, "user_message": turn["user"], "resolved_message": "",
             "intent": "", "intent_confidence": 0.0, "trace": [], "history": history}
    out_r = await resolve_node(state, writer=None)
    state.update(out_r)
    out_i = await intent_node(state)
    return state["resolved_message"], state["intent"], state["intent_confidence"], out_i["trace"][-1]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="samples", default="tests/eval/router_samples.jsonl")
    ap.add_argument("--report", default="reports/ch06-router-report.md")
    ap.add_argument("--escalate", action="store_true", help="开启小→大降级路对照(需 intent_small_model)")
    args = ap.parse_args()

    settings = Settings()
    resolver = make_extract_model(settings).with_structured_output(
        ResolvedQuestion, method="function_calling")
    if args.escalate:
        small = make_extract_model(settings, model=settings.intent_small_model or None)
        big = make_extract_model(settings)
        intent_node = make_intent_node(small, escalator=big, floor=settings.intent_confidence_floor)
    else:
        intent_node = make_intent_node(make_extract_model(settings))
    resolve_node = make_resolve_node(resolver)

    rows = [json.loads(l) for l in Path(args.samples).read_text(encoding="utf-8").splitlines() if l.strip()]
    per_tag: dict[str, Counter] = {}
    mal_count, conf_sum, conf_n, escalated = 0, 0.0, 0, 0
    details = []
    for row in rows:
        history: list[dict] = []
        for i, turn in enumerate(row["turns"]):
            resolved, intent, conf, trace_line = await _eval_turn(resolve_node, intent_node, history, turn)
            tag = row["tag"]
            ok_intent = intent == turn["expect_intent"]
            ok_resolved = True
            if "expect_resolved" in turn:
                ok_resolved = _normalize(resolved) == _normalize(turn["expect_resolved"])
            per_tag.setdefault(tag, Counter())["total"] += 1
            per_tag[tag]["intent_ok"] += ok_intent
            if "expect_resolved" in turn:
                per_tag[tag]["resolved_total"] = per_tag[tag].get("resolved_total", 0) + 1
                per_tag[tag]["resolved_ok"] += ok_resolved
            malformed = "malformed=true" in trace_line
            mal_count += malformed
            conf_sum += conf
            conf_n += 1
            escalated += "escalated=true" in trace_line
            details.append((row["id"], i, turn["user"], turn["expect_intent"], intent,
                            ok_intent, resolved, ok_resolved, conf, trace_line))
            history.append({"role": "user", "content": turn["user"]})

    lines = ["# ch06 分流器评估报告", "",
             f"- 用例:{len(rows)} 组对话 / {conf_n} 轮;escalation={'开' if args.escalate else '关'}",
             f"- 意图准确率(总体):{sum(c['intent_ok'] for c in per_tag.values())}/{conf_n}",
             f"- 指代消解准确率:{sum(c.get('resolved_ok', 0) for c in per_tag.values())}"
             f"/{sum(c.get('resolved_total', 0) for c in per_tag.values())}",
             f"- JSON 畸形率:{mal_count}/{conf_n};平均 confidence:{conf_sum / max(conf_n, 1):.2f}",
             f"- escalation 触发:{escalated} 次", "",
             "| 桶 | 意图 | 指代消解 |", "|---|---|---|"]
    for tag, c in per_tag.items():
        lines.append(f"| {tag} | {c['intent_ok']}/{c['total']} "
                     f"| {c.get('resolved_ok', 0)}/{c.get('resolved_total', 0)} |")
    lines += ["", "## 个案明细", "",
              "| 组/轮 | 用户消息 | 期望意图 | 判定 | 对错 | 补全问法 | 指代对错 | trace |",
              "|---|---|---|---|---|---|---|---|"]
    for rid, i, user, exp, got, ok, resolved, rok, conf, trace in details:
        lines.append(f"| {rid}/{i} | {user} | {exp} | {got} | {'√' if ok else '×'} "
                     f"| {resolved} | {'√' if rok else '×' if 'expect_resolved' else '-'} "
                     f"| {trace} |")
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已写入 {args.report}")
    total_ok = sum(c["intent_ok"] for c in per_tag.values())
    print(f"意图准确率 {total_ok}/{conf_n};畸形 {mal_count};escalated {escalated}")


if __name__ == "__main__":
    asyncio.run(main())
```

(实现时对照 `scripts/eval_retrieval.py` 统一 CLI/落盘风格;个案明细里的 expect_resolved 有无要与 turns 数据一致。)

- [ ] **Step 5: 真跑留数**:`.venv/bin/python scripts/eval_router.py` → 确认 `reports/ch06-router-report.md` 生成、指标非空;意图准确率与指代质量若 < 90%,回改 prompt 文案(prompt 属「拿标注样例跑一遍验证」的迭代对象,迭代过程记 dev-notes)
- [ ] **Step 6: 全量测试 + Commit** `git commit -m "ch06: 路由评估集+eval_router 真跑报告"`

---

### Task 9: code review + 修复波

- [ ] **Step 1**:按 requesting-code-review 技能发起独立 subagent 全量评审(spec 对照 + 全量 pytest 复跑)
- [ ] **Step 2**:按 receiving-code-review 技能逐条核实修复(带回归测试);接受不修的记档说明理由
- [ ] **Step 3**:dev-notes/ch06.md 补「Code Review 结论与修复波」段;全量测试绿后 Commit

### Task 10: 真机验收四条 + README 收口 + finish 留痕

**Files:**
- Modify: `README.md`(ch06 功能/验收/已知边界)
- Modify: `dev-notes/ch06.md`(finish 段)

- [ ] **Step 1: 起真环境**:

```bash
docker compose down -v && docker compose up -d   # 等约 30 秒
.venv/bin/python -m db.seed && .venv/bin/python -m db.seed_conversations
.venv/bin/python -m scripts.build_kb && .venv/bin/python -m scripts.mine_qa
.venv/bin/uvicorn app.main:app --port 8000       # 后台任务方式(ch05 教训:nohup 会被沙箱回收)
```

- [ ] **Step 2: 验收 3(指代→子流程链路)**:浏览器/HTTP 先问「SH-E300 多少钱」,再问「这个能退吗」——检查点:①回复气泡灰字「已理解:SH-E300能退吗」(resolved 帧);②服务日志 trace 链 `node=resolve changed=True → node=intent intent=退款退货 → node=prepare_order order=1001 → node=fetch_order found=True → node=expand n=3~4 → node=policy_retrieve ... → node=agent`;③徽章序列 订单查询→查询扩写→政策检索;④答案带 [n] 角标可点来源卡
- [ ] **Step 3: 验收 4(选择器点选全流程)**:新会话直接问「这个能退吗」(无前置上下文,resolve 透传)→ 气泡出现三张订单卡(browser-use 自动化点选「订单1001」)→ 新气泡收到最终退款判定 + 「申请退款」按钮 → 点开表单选「七天无理由」提交 → 按钮转「工单 T… 已创建」,tickets 表落行(type=售后):

```bash
docker exec shophelper-mysql mysql -ushophelper -pshophelper --default-character-set=utf8mb4 shophelper \
  -e "SELECT ticket_no, ticket_type, description FROM tickets ORDER BY created_at DESC LIMIT 3"
```

- [ ] **Step 4: 验收 1/2(评估报告)**:`.venv/bin/python scripts/eval_router.py` 真跑——多轮来回切组(R01/R02)意图全对、指代补全全对;怪问题组(R07)全落「其他」;JSON 畸形率 0(报告数字即证据)
- [ ] **Step 5: README 收口**:ch06 功能段落 + 「ch06 四条验收」操作清单 + 已知边界(选择器旧卡片可重放/订单 mock 三单/escalation 默认关)
- [ ] **Step 6: dev-notes/ch06.md 补齐各任务段(随做随记,不收尾一次性补)+ finish 段(演示命令/测试结果/路径)**;全量 `.venv/bin/pytest` 绿;Commit `git commit -m "ch06: 真机验收四条全过+finish 留痕"`

---

## Self-Review 记录

1. **Spec coverage**:指代消解+改写(Task 3)、意图四件套+其他+降级路(Task 2/3)、扩写+去重合并(Task 4)、确定性子流程(Task 4/6)、槽位处理不猜单号(Task 4)、订单选择器/退款单前端(Task 7)、稳定化 query_order+orders 端点(Task 1)、评估集替代 TDD(Task 8)、验收映射(Task 8/10)——spec 各节均有对应任务;spec「实现期偏差」预留节由执行期回填。
2. **Placeholder scan**:Task 1 Step 1 测试草稿里的占位行已在同一步内给出正式版;Task 7 的 `bubble_append`/`state.session_id_placeholder` 已标注实现时的正确写法(透传 bubble 引用/`state.sessionId`);无 TBD。
3. **Type consistency**:`make_refund_nodes` 五返回值顺序在 Task 4/6 一致;`parse_intent` 三元组在 Task 2/3/8 一致;`OrderResume` 字段与前端 `resume: {order_id, question}` 一致;`initial_state(..., resume=None)` 与 chat.py 调用一致;ChatState 七个新字段在 state.py/initial_state/各测试 _state 助手中字段名一致。
