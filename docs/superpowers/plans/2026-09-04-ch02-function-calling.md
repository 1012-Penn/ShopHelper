# ch02 Function Calling 工具链 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 ch01 的 SSE 流式客服聊天装上 Function Calling:模型自主选工具 → 执行 → 结果回灌 → 单轮流式收敛,聊天记录(含工具轨迹)落 MySQL 四张表。

**Architecture:** Docker 起 MySQL 8 承载 faq/conversations/messages/tickets 四表;SQLAlchemy 2.x 同步 ORM(演示场景,同步调用直接放在 async handler 中,阻塞可忽略);五个 LangChain `@tool` + 自建注册表(校验/超时/重试/错误回灌);chat 路由两段式:第一段 `bind_tools` 非流式拿 tool_calls,有则推 `tool_status` 帧、执行、回灌,第二段不绑工具 `astream` 逐 token 出最终回答。内存 SessionStore 删除,多轮上下文全走 MySQL。

**Tech Stack:** FastAPI + SQLAlchemy 2.x + PyMySQL + MySQL 8(Docker Compose)+ LangChain 1.x `@tool` / `bind_tools`(API 用法已经 Context7 核对)。

**Spec:** `docs/superpowers/specs/2026-09-04-ch02-function-calling-design.md`

## Global Constraints

- 全库 InnoDB + utf8mb4;四表 DDL 以用户提供的为准,tickets 缺失字段按用户确认补齐(status ENUM('待处理','已处理') DEFAULT '待处理'、created_at、PRIMARY KEY (ticket_no)、KEY idx_conversation_id)。
- 只做单轮工具调用:第二次模型调用强制不绑工具;不做 Agent Loop、不做向量检索。
- query_order / query_product / query_logistics 工具内部 random mock,不接真实 API、不建表。
- conversations.user_id 固定写 `'guest'`。
- 落库契约:整轮成功才落,中途出错整轮不落(ch01 契约延续)。
- 聊天页 `static/index.html` 改造走 Vibe Coding,不套 TDD/brainstorm/code review。
- 测试不依赖真实 MySQL(SQLite 内存库)和真实上游 key(替身模型)。
- 过程留痕:每完成一个任务追记 `dev-notes/ch02.md`。

---

### Task 1: Docker MySQL + 四表 DDL + db.py

**Files:**
- Create: `docker-compose.yml`
- Create: `db/init.sql`
- Create: `app/db.py`
- Modify: `app/config.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces: `Settings.database_url: str`、`app.db.make_engine(settings) -> Engine`、`app.db.make_session_factory(engine) -> sessionmaker`;Task 2 的 ORM、Task 4/6/7 都消费。

- [ ] **Step 1: 安装依赖**

```bash
.venv/bin/pip install "sqlalchemy>=2.0" pymysql cryptography
```

- [ ] **Step 2: 写 docker-compose.yml 与 init.sql(用户 DDL 原文,补齐 tickets)**

`docker-compose.yml`:

```yaml
services:
  mysql:
    image: mysql:8
    container_name: shophelper-mysql
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: shophelper
      MYSQL_USER: shophelper
      MYSQL_PASSWORD: shophelper
    ports:
      - "3306:3306"
    volumes:
      - mysql-data:/var/lib/mysql
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    command: --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci

volumes:
  mysql-data:
```

`db/init.sql`:用户提供的四表 DDL 全文(conversations / messages / faq / tickets),其中 tickets 表按确认补为:

```sql
CREATE TABLE tickets (
  ticket_no       VARCHAR(32)     NOT NULL                COMMENT '工单号,如 T20260701008',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '关联会话,可倒查当时聊了什么',
  description     TEXT            NOT NULL                COMMENT '问题描述',
  ticket_type     ENUM('售后','投诉','咨询') NOT NULL     COMMENT '工单类型',
  status          ENUM('待处理','已处理') NOT NULL DEFAULT '待处理' COMMENT '处理状态',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (ticket_no),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_tickets_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人工工单';
```

- [ ] **Step 3: config.py 加 DATABASE_URL**

```python
database_url: str = "mysql+pymysql://shophelper:shophelper@127.0.0.1:3306/shophelper?charset=utf8mb4"
```

- [ ] **Step 4: 写失败测试 tests/test_db.py**

```python
from sqlalchemy import text
from app.config import Settings
from app.db import make_engine, make_session_factory


def test_make_engine_and_session_roundtrip():
    settings = Settings(openai_api_key="test", database_url="sqlite:///:memory:")
    engine = make_engine(settings)
    factory = make_session_factory(engine)
    with factory() as session:
        assert session.execute(text("SELECT 1")).scalar() == 1
```

- [ ] **Step 5: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_db.py -v`
Expected: FAIL,`ModuleNotFoundError: app.db`(config 字段已存在则该项无失败,以 app.db 缺失为准)

- [ ] **Step 6: 实现 app/db.py**

```python
"""数据库引擎与会话工厂——同步 SQLAlchemy,演示场景直接在 async handler 里调用。"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings


def make_engine(settings: Settings):
    return create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=3600)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
```

- [ ] **Step 7: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 8: Docker 起 MySQL 并验证建表**

```bash
docker compose up -d && sleep 30
docker exec shophelper-mysql mysql -ushophelper -pshophelper shophelper -e "SHOW TABLES;"
```

Expected: 四张表 conversations / faq / messages / tickets。(本环境无 Docker 则此步改为记录到 dev-notes 并标注待用户环境执行,不阻塞后续任务——测试全走 SQLite。)

- [ ] **Step 9: Commit**

```bash
git add docker-compose.yml db/init.sql app/db.py app/config.py tests/test_db.py
git commit -m "ch02: Docker MySQL + 四表 DDL + db.py 引擎工厂"
```

---

### Task 2: ORM 四模型

**Files:**
- Create: `app/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: Task 1 的 session factory。
- Produces: `Base`(DeclarativeBase)、`Faq(id, question, answer, category)`、`Conversation(id, user_id, status, created_at, updated_at)`、`Message(id, conversation_id, role, content, tool_calls, tool_call_id, created_at)`、`Ticket(ticket_no, conversation_id, description, ticket_type, status, created_at)`。

- [ ] **Step 1: 写失败测试**

```python
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Base, Conversation, Faq, Message, Ticket


def _seed(session: Session) -> int:
    Base.metadata.create_all(session.bind)
    conv = Conversation(user_id="guest")
    session.add(conv)
    session.flush()
    session.add(Faq(question="退货政策是什么", answer="七天无理由退货", category="售后"))
    session.add(Message(conversation_id=conv.id, role="user", content="退货政策是什么"))
    session.add(
        Message(
            conversation_id=conv.id,
            role="assistant",
            content=None,
            tool_calls='[{"name":"query_faq","args":{"keyword":"退货"},"id":"call_1"}]',
        )
    )
    session.add(Message(conversation_id=conv.id, role="tool", content='{"items":[]}', tool_call_id="call_1"))
    session.add(Ticket(ticket_no="T20260904001", conversation_id=conv.id, description="想退货", ticket_type="售后"))
    session.commit()
    return conv.id


def test_models_roundtrip():
    engine = create_memory_engine()
    with Session(engine) as session:
        conv_id = _seed(session)
        conv = session.get(Conversation, conv_id)
        assert conv.status == "进行中"
        assert session.scalar(select(Faq).limit(1)).answer == "七天无理由退货"
        msgs = session.scalars(select(Message).order_by(Message.id)).all()
        assert [m.role for m in msgs] == ["user", "assistant", "tool"]
        assert msgs[2].tool_call_id == "call_1"
        assert session.get(Ticket, "T20260904001").status == "待处理"


def create_memory_engine():
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: FAIL,`ModuleNotFoundError: app.models`

- [ ] **Step 3: 实现 app/models.py**

```python
"""ORM 四模型,字段与 db/init.sql 一一对应。"""
from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, Enum, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(None, "sqlite"), primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), default="guest")
    status: Mapped[str] = mapped_column(Enum("进行中", "已转人工", "已结束", name="conv_status"), default="进行中")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(None, "sqlite"), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(BigInteger().with_variant(None, "sqlite"), nullable=False)
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool", name="msg_role"), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[str | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(None, "sqlite"), primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(BigInteger().with_variant(None, "sqlite"), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询", name="ticket_type"), nullable=False)
    status: Mapped[str] = mapped_column(Enum("待处理", "已处理", name="ticket_status"), default="待处理")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "ch02: ORM 四模型 faq/conversations/messages/tickets"
```

---

### Task 3: FAQ 种子数据(数据类任务,标注样例验证)

**Files:**
- Create: `db/seed.py`

**Interfaces:**
- Consumes: Task 1/2 的 engine 与 Faq 模型。
- Produces: faq 表至少 8 条数据,必须包含问题含「退货」的条目(验收 2);**不得**包含任何问题含「邮费」直接措辞的条目以外的说法——「邮费」验收 3 要求 LIKE 查不到,故种子数据中不能出现「邮费」二字(运费说明条目用「多久发货」「七天无理由」等措辞,不含「邮费/运费」字样)。

- [ ] **Step 1: 写 db/seed.py(幂等:已有数据则跳过)**

```python
"""灌 FAQ 测试数据:python -m db.seed"""
from app.config import Settings
from app.db import make_engine, make_session_factory
from app.models import Faq

FAQ_ITEMS = [
    ("退货政策是什么?", "支持七天无理由退货,商品需保持完好,凭订单号联系客服即可发起退货。", "售后"),
    ("怎么申请退货?", "在订单详情页点击「申请售后」,选择退货原因后提交,客服会在 24 小时内审核。", "售后"),
    ("多久能发货?", "现货商品 48 小时内发出,预售商品按页面标注时间发货。", "物流"),
    ("物流一般几天到?", "普通快递 3-5 天送达,偏远地区 5-7 天。", "物流"),
    ("支持哪些付款方式?", "支持微信支付、支付宝、银联卡。", "交易"),
    ("可以开发票吗?", "支持电子发票,下单时备注抬头,发货后 24 小时内开出。", "交易"),
    ("会员有什么权益?", "会员享 95 折、生日礼包、优先客服通道。", "会员"),
    ("商品有质量问题的退货流程是?", "质量问题可凭照片凭证免费退货,运费由商家承担,退款原路返回。", "售后"),
]


def main() -> None:
    engine = make_engine(Settings())
    with make_session_factory(engine)() as session:
        if session.query(Faq).count() > 0:
            print("faq 表已有数据,跳过")
            return
        session.add_all(Faq(question=q, answer=a, category=c) for q, a, c in FAQ_ITEMS)
        session.commit()
    print(f"已灌入 {len(FAQ_ITEMS)} 条 FAQ")


if __name__ == "__main__":
    main()
```

注意:第 8 条含「运费」二字——验收 3 用「邮费」查询,LIKE '%邮费%' 不命中,符合预期;若实现后测试发现「运费」也该漏召回,再把该条改写去掉。

- [ ] **Step 2: 用标注样例验证(代替单测)**

先跑一条内存库冒烟,再对真库验证:

```bash
.venv/bin/python -c "
from sqlalchemy import create_engine, select
from app.models import Base, Faq
Base.metadata.create_all(engine := create_engine('sqlite:///:memory:'))
from app.db import make_session_factory
import db.seed as s
with make_session_factory(engine)() as sess:
    sess.add_all(Faq(question=q, answer=a, category=c) for q, a, c in s.FAQ_ITEMS)
    sess.commit()
    hit = sess.scalar(select(Faq).where(Faq.question.like('%退货%')))
    miss = sess.scalars(select(Faq).where(Faq.question.like('%邮费%'))).all()
    assert hit and not miss, '标注样例失败'
    print('样例通过: 退货命中 / 邮费漏召回')
"
```

Expected: 输出「样例通过: 退货命中 / 邮费漏召回」。

Docker 可用时再对真库执行:`.venv/bin/python -m db.seed` 然后同样断言;不可用则记录 dev-notes。

- [ ] **Step 3: Commit**

```bash
git add db/seed.py
git commit -m "ch02: FAQ 种子数据(退货可命中,邮费刻意漏召回)"
```

---

### Task 4: 五个 @tool 业务工具

**Files:**
- Create: `app/tools/__init__.py`(空)
- Create: `app/tools/definitions.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: Task 2 的 `Faq` / `Ticket` 模型;`db.seed.FAQ_ITEMS` 可直接复用做测试数据。
- Produces: `build_tools(session_factory) -> list`(五个 tool 对象,供 registry 与 bind_tools 消费)、`TOOL_LABELS: dict[str, str]`(name → 中文显示名,如 `"query_logistics": "物流查询"`)。工具签名:`query_order(order_id: str) -> str`、`query_product(keyword: str) -> str`、`query_logistics(order_id: str) -> str`、`query_faq(keyword: str) -> str`、`create_ticket(conversation_id: int, description: str, ticket_type: str) -> str`,全部返回 JSON 字符串。

- [ ] **Step 1: 写失败测试**

```python
import json

from sqlalchemy import select

from app.models import Faq, Ticket
from app.tools.definitions import TOOL_LABELS, build_tools
from tests.test_models import create_memory_engine


def _make():
    engine = create_memory_engine()
    factory = make_session_factory(engine)
    with factory() as s:
        s.add_all(Faq(question=q, answer=a, category=c) for q, a, c in [
            ("退货政策是什么?", "七天无理由退货。", "售后"),
            ("多久能发货?", "48 小时内发出。", "物流"),
        ])
        s.add(Faq(question="能不能便宜点?", "会员享 95 折。", "会员"))
        s.commit()
    return build_tools(factory)


def test_build_tools_names_and_labels():
    tools = _make()
    names = {t.name for t in tools}
    assert names == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}
    assert TOOL_LABELS["query_logistics"] == "物流查询"


def test_mock_tools_return_structured_json():
    tools = {t.name: t for t in _make()}
    order = json.loads(tools["query_order"].invoke({"order_id": "1001"}))
    assert order["order_id"] == "1001" and "status" in order and "amount" in order
    product = json.loads(tools["query_product"].invoke({"keyword": "手机"}))
    assert "name" in product and "price" in product
    logistics = json.loads(tools["query_logistics"].invoke({"order_id": "1001"}))
    assert logistics["order_id"] == "1001" and len(logistics["traces"]) >= 1


def test_query_faq_hit_and_miss():
    tools = {t.name: t for t in _make()}
    hit = json.loads(tools["query_faq"].invoke({"keyword": "退货"}))
    assert len(hit["items"]) == 1 and "七天无理由" in hit["items"][0]["answer"]
    miss = json.loads(tools["query_faq"].invoke({"keyword": "邮费"}))
    assert miss["items"] == []


def test_create_ticket_persists():
    tools = {t.name: t for t in _make()}
    engine = create_memory_engine()  # 同一个内存库?—— 不,需要复用 factory,见下
```

注意:为让 create_ticket 断言落库,`_make()` 返回 `(tools, factory)`,测试改为:

```python
def _make():
    engine = create_memory_engine()
    factory = make_session_factory(engine)
    with factory() as s:
        s.add_all(Faq(question=q, answer=a, category=c) for q, a, c in [
            ("退货政策是什么?", "七天无理由退货。", "售后"),
        ])
        s.commit()
    return build_tools(factory), factory


def test_create_ticket_persists():
    tools, factory = _make()
    result = json.loads(tools["create_ticket"].invoke({
        "conversation_id": 1, "description": "想退货", "ticket_type": "售后",
    }))
    assert result["ticket_no"].startswith("T")
    with factory() as s:
        ticket = s.get(Ticket, result["ticket_no"])
        assert ticket is not None
        assert ticket.status == "待处理"
        assert ticket.conversation_id == 1


def test_create_ticket_rejects_bad_type():
    tools, _ = _make()
    try:
        tools["create_ticket"].invoke({
            "conversation_id": 1, "description": "x", "ticket_type": "砍价",
        })
        raised = False
    except Exception:
        raised = True
    assert raised  # pydantic/LangChain 参数校验拒绝非法枚举
```

(测试文件顶部补 `from app.db import make_session_factory`;`tests/test_models.py` 的 `create_memory_engine` 保持原样导出。)

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: FAIL,`ModuleNotFoundError: app.tools`

- [ ] **Step 3: 实现 app/tools/definitions.py**

```python
"""五个业务工具:三个 mock(不接真实 API、不建表)+ query_faq(LIKE 查表)+ create_ticket(写表)。"""
import json
import random
from datetime import datetime

from langchain.tools import tool
from sqlalchemy import select

from app.models import Faq, Ticket

TOOL_LABELS = {
    "query_order": "订单查询",
    "query_product": "商品查询",
    "query_logistics": "物流查询",
    "query_faq": "FAQ 检索",
    "create_ticket": "创建工单",
}

CITIES = ["北京", "上海", "广州", "杭州", "成都"]


def build_tools(session_factory) -> list:
    """session_factory 由调用方注入;返回 tool 对象列表(供 bind_tools 与 registry)。"""

    @tool
    def query_order(order_id: str) -> str:
        """按订单号查询订单信息:商品、金额、状态。用户问订单相关问题时使用。"""
        return json.dumps({
            "order_id": order_id,
            "product": random.choice(["无线耳机", "机械键盘", "硅胶手机壳", "智能手环"]),
            "amount": round(random.uniform(19.9, 999.0), 2),
            "status": random.choice(["待发货", "已发货", "已签收"]),
            "created_at": datetime.now().strftime("%Y-%m-%d"),
        }, ensure_ascii=False)

    @tool
    def query_product(keyword: str) -> str:
        """按关键词查询商品:名称、价格、库存。用户咨询商品信息时使用。"""
        return json.dumps({
            "keyword": keyword,
            "name": f"{keyword}精选款",
            "price": round(random.uniform(9.9, 499.0), 2),
            "stock": random.randint(0, 500),
        }, ensure_ascii=False)

    @tool
    def query_logistics(order_id: str) -> str:
        """按订单号查询物流轨迹:承运商与节点列表。用户问包裹到哪了时使用。"""
        node_count = random.randint(2, 4)
        traces = [
            f"{datetime.now().strftime('%m-%d %H:%M')} 包裹已到达{random.choice(CITIES)}转运中心"
            for _ in range(node_count)
        ]
        traces.append(f"{datetime.now().strftime('%m-%d %H:%M')} 派送中,快递员 {random.randint(100, 999)} 号")
        return json.dumps({
            "order_id": order_id,
            "carrier": random.choice(["顺丰速运", "中通快递", "京东物流"]),
            "traces": traces,
        }, ensure_ascii=False)

    @tool
    def query_faq(keyword: str) -> str:
        """按关键词检索常见问题库(FAQ)。用户问退货政策、发货时间等常见问题时使用。"""
        with session_factory() as session:
            rows = session.scalars(
                select(Faq).where(Faq.question.like(f"%{keyword}%")).limit(3)
            ).all()
            return json.dumps(
                {"items": [{"question": r.question, "answer": r.answer, "category": r.category} for r in rows]},
                ensure_ascii=False,
            )

    @tool
    def create_ticket(conversation_id: int, description: str, ticket_type: str) -> str:
        """创建人工工单转人工处理。问题超出自动客服能力、用户强烈要求人工时使用。

        Args:
            conversation_id: 当前会话 id
            description: 问题描述
            ticket_type: 工单类型,只能是 '售后'、'投诉'、'咨询' 之一
        """
        ticket_no = f"T{datetime.now().strftime('%Y%m%d')}{random.randint(0, 999):03d}"
        with session_factory() as session:
            session.add(Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            ))
            session.commit()
        return json.dumps({"ticket_no": ticket_no, "status": "待处理"}, ensure_ascii=False)

    return [query_order, query_product, query_logistics, query_faq, create_ticket]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: PASS(5 个测试全绿)

- [ ] **Step 5: Commit**

```bash
git add app/tools/ tests/test_tools.py
git commit -m "ch02: 五个 @tool 业务工具(mock 三件套 + query_faq + create_ticket)"
```

---

### Task 5: 工具注册表(校验/超时/重试/错误回灌)

**Files:**
- Create: `app/tools/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: Task 4 的 tool 对象与 `TOOL_LABELS`。
- Produces: `class ToolRegistry`,构造 `ToolRegistry(tools: list)`;方法 `has(name) -> bool`、`labels() -> dict[str, str]`、`async execute(name: str, args_json: str) -> str`(永不抛异常,失败信息作为字符串返回供回灌)。常量 `TOOL_TIMEOUT_SECONDS = 10`、`TOOL_RETRIES = 1`。

- [ ] **Step 1: 写失败测试**

```python
import asyncio
import json

import pytest
from langchain.tools import tool

from app.tools.registry import ToolRegistry


@tool
def ok_tool(x: int) -> str:
    """doubles x"""
    return json.dumps({"result": x * 2})


@tool
def bad_tool(x: int) -> str:
    """always raises"""
    raise RuntimeError("下游炸了")


@tool
def slow_tool(x: int) -> str:
    """sleeps past timeout"""
    import time

    time.sleep(5)
    return json.dumps({"ok": True})


def test_execute_ok():
    reg = ToolRegistry([ok_tool])
    assert json.loads(asyncio.run(reg.execute("ok_tool", '{"x": 21}')))["result"] == 42


def test_unknown_tool_returns_error_string():
    reg = ToolRegistry([ok_tool])
    out = asyncio.run(reg.execute("nope", "{}"))
    assert "nope" in out and "未注册" in out


def test_bad_json_returns_error_string():
    reg = ToolRegistry([ok_tool])
    out = asyncio.run(reg.execute("ok_tool", "{not json"))
    assert "参数" in out


def test_invalid_args_returns_error_string():
    reg = ToolRegistry([ok_tool])
    out = asyncio.run(reg.execute("ok_tool", '{"x": "not-a-number"}'))
    assert "参数" in out


def test_runtime_error_returns_error_string_after_retry():
    reg = ToolRegistry([bad_tool])
    out = asyncio.run(reg.execute("bad_tool", '{"x": 1}'))
    assert "下游炸了" in out


def test_timeout_returns_error_string():
    reg = ToolRegistry([slow_tool])
    reg.TOOL_TIMEOUT_SECONDS = 0.1  # 测试里收紧超时
    out = asyncio.run(reg.execute("slow_tool", '{"x": 1}'))
    assert "超时" in out


def test_labels():
    reg = ToolRegistry([ok_tool])
    assert reg.labels() == {"ok_tool": ok_tool.description}
```

(实现后若 pydantic 对 `"not-a-number"` 宽容转型,则该断言改为检查输出含「参数」或按实际校验行为调整——以跑出来的行为为准,错误字符串风格保持统一。)

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: FAIL,`ModuleNotFoundError: app.tools.registry`

- [ ] **Step 3: 实现 app/tools/registry.py**

```python
"""工具注册表:统一执行入口,校验/超时/重试/错误都收敛为字符串回灌,不让请求 500。"""
import asyncio
import json
import logging

from app.tools.definitions import TOOL_LABELS

logger = logging.getLogger(__name__)


class ToolRegistry:
    TOOL_TIMEOUT_SECONDS = 10.0
    TOOL_RETRIES = 1

    def __init__(self, tools: list) -> None:
        self._by_name = {t.name: t for t in tools}

    def has(self, name: str) -> bool:
        return name in self._by_name

    def labels(self) -> dict[str, str]:
        return {name: TOOL_LABELS.get(name, name) for name in self._by_name}

    async def execute(self, name: str, args_json: str) -> str:
        if name not in self._by_name:
            return f"错误:工具 {name} 未注册"
        try:
            args = json.loads(args_json or "{}")
        except json.JSONDecodeError:
            return f"错误:工具 {name} 的参数不是合法 JSON"
        if not isinstance(args, dict):
            return f"错误:工具 {name} 的参数必须是 JSON 对象"
        tool = self._by_name[name]
        last_error = ""
        for attempt in range(self.TOOL_RETRIES + 1):
            try:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(tool.invoke, args), timeout=self.TOOL_TIMEOUT_SECONDS
                )
                return raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            except asyncio.TimeoutError:
                last_error = f"错误:工具 {name} 执行超时({self.TOOL_TIMEOUT_SECONDS}s)"
            except Exception as exc:
                last_error = f"错误:工具 {name} 执行失败:{exc}"
            logger.warning("%s(第 %d 次)", last_error, attempt + 1)
        return last_error
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/tools/registry.py tests/test_registry.py
git commit -m "ch02: 工具注册表——校验/超时/重试/错误字符串回灌"
```

---

### Task 6: MySQL 会话存储替换内存 store

**Files:**
- Create: `app/store.py`
- Delete: `app/sessions.py`
- Modify: `app/history.py`(content 可能为 None 的防御)
- Modify: `app/routers/chat.py`(仅历史读写部分,完整工具流在 Task 7)、`app/routers/sessions.py`、`app/main.py`
- Modify: `tests/conftest.py`,删除 `tests/test_sessions.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: Task 1/2 的 factory 与 `Conversation` / `Message` 模型。
- Produces: `class ConversationStore(session_factory)`:
  - `async resolve(session_id: int | None) -> int` —— None 则新建 conversation(user_id='guest'),返回 id;
  - `async get_history(conversation_id: int) -> list[dict]` —— 按 id 升序返回 `{"role","content","tool_calls","tool_call_id"}`(tool_calls 为已解析的 list,无则 None);
  - `async append(conversation_id: int, msgs: list[dict]) -> None` —— 批量落一条消息(dict 结构同上)。
- `app/history.py` 的 `trim_history` / `estimate_tokens` 签名不变,仅对 `content=None` 消息按空串计费。

- [ ] **Step 1: 删 app/sessions.py 与 tests/test_sessions.py,改 conftest**

`tests/conftest.py` 现有内容先读一遍;把其中 session_store 注入相关 fixture 换成:

```python
@pytest.fixture
def db_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def store(db_session_factory):
    return ConversationStore(db_session_factory)


@pytest.fixture
def client(db_session_factory):
    settings = Settings(openai_api_key="test")
    app = create_app(settings)
    app.state.session_factory = db_session_factory
    app.state.store = ConversationStore(db_session_factory)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)
```

(以 conftest 实际结构为准做最小替换;create_app 的依赖注入方式在 Task 7 定稿——`app.state.store` + `app.state.registry` 是 chat 路由取依赖的唯一入口。)

- [ ] **Step 2: 写失败测试 tests/test_store.py**

```python
import pytest

from app.store import ConversationStore


@pytest.mark.asyncio
async def test_resolve_creates_new_conversation(store):
    conv_id = await store.resolve(None)
    history = await store.get_history(conv_id)
    assert history == []


@pytest.mark.asyncio
async def test_resolve_existing_id(store):
    conv_id = await store.resolve(None)
    assert await store.resolve(conv_id) == conv_id


@pytest.mark.asyncio
async def test_append_and_history_order(store):
    conv_id = await store.resolve(None)
    await store.append(conv_id, [
        {"role": "user", "content": "退货政策是什么"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1"}]},
        {"role": "tool", "content": '{"items":[]}', "tool_call_id": "call_1"},
        {"role": "assistant", "content": "支持七天无理由退货。"},
    ])
    history = await store.get_history(conv_id)
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[1]["tool_calls"][0]["name"] == "query_faq"
    assert history[2]["tool_call_id"] == "call_1"
```

- [ ] **Step 3: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_store.py -v`
Expected: FAIL,`ModuleNotFoundError: app.store`

- [ ] **Step 4: 实现 app/store.py**

```python
"""MySQL 会话存储:conversations/messages 表承载多轮上下文与工具轨迹。"""
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.models import Conversation, Message


class ConversationStore:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._factory = session_factory

    async def resolve(self, session_id: int | None) -> int:
        if session_id is not None:
            return session_id
        with self._factory() as session:
            conv = Conversation(user_id="guest")
            session.add(conv)
            session.commit()
            return conv.id

    async def get_history(self, conversation_id: int) -> list[dict]:
        with self._factory() as session:
            rows = session.scalars(
                select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
            ).all()
            return [
                {
                    "role": r.role,
                    "content": r.content,
                    "tool_calls": r.tool_calls,
                    "tool_call_id": r.tool_call_id,
                }
                for r in rows
            ]

    async def append(self, conversation_id: int, msgs: list[dict]) -> None:
        with self._factory() as session:
            for m in msgs:
                session.add(Message(
                    conversation_id=conversation_id,
                    role=m["role"],
                    content=m.get("content"),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                ))
            session.commit()
```

- [ ] **Step 5: history.py 防御 None content,路由与 main 换依赖**

`app/history.py` 中 `trim_history` 的计费行改为 `cost = estimate_tokens(msg.get("content") or "")`。

`app/routers/sessions.py`:历史接口改用 `request.app.state.store.get_history(int(session_id))`,session_id 类型从 str 改 int(前端同步改,Task 8 一并处理);不存在时返回空列表。

`app/main.py`:`create_app` 中去掉 `app.state.sessions = session_store` 与 `make_chat_model` 之外的工具相关部分,新增:

```python
engine = make_engine(settings)
session_factory = make_session_factory(engine)
app.state.session_factory = session_factory
app.state.store = ConversationStore(session_factory)
app.state.registry = ToolRegistry(build_tools(session_factory))
```

`app/routers/chat.py` 本任务先做最小替换(store 取历史、成功后 append),工具流留给 Task 7——本任务结束时 chat 行为与 ch01 等价(仅存储换了),全部现有测试适配后通过。

- [ ] **Step 6: 全量测试通过(含 ch01 测试适配)**

Run: `.venv/bin/pytest -v`
Expected: PASS(test_sessions.py 已删,其余测试改用 store/client fixture)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "ch02: MySQL 会话存储替换内存 store,多轮上下文走 conversations/messages"
```

---

### Task 7: chat 路由工具流(单轮收敛)

**Files:**
- Modify: `app/routers/chat.py`
- Test: `tests/test_chat_toolflow.py`

**Interfaces:**
- Consumes: Task 5 `ToolRegistry.execute(name, args_json) -> str`、Task 6 `ConversationStore`、`app.llm.make_chat_model`。
- Produces: SSE 帧序 `session → [tool_status] → token* → done|error`;`tool_status` 帧结构 `{"type":"tool_status","name":"query_faq","label":"FAQ 检索"}`。

替身模型:测试里用一个 `FakeChatModel`,第一次 `bind_tools(...).ainvoke` 返回带 tool_calls 的 `AIMessage`,第二次 `astream` 吐两个文本 chunk;`bind_tools` 记录是否被调用以断言「第二次不绑工具」。

- [ ] **Step 1: 写失败测试**

```python
import json

import pytest
from langchain_core.messages import AIMessage

from app.routers.chat import sse_frame


class FakeFirstCallModel:
    """第一次调用返回 tool_calls,第二次流式吐文本。"""

    def __init__(self):
        self.bind_calls = 0

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="", tool_calls=[
            {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1"}
        ])

    async def astream(self, messages):
        for piece in ["支持", "七天无理由退货"]:
            yield type("Chunk", (), {"text": piece})()


@pytest.fixture
def tool_client(client, db_session_factory):
    from app.store import ConversationStore
    from app.tools.definitions import build_tools
    from app.tools.registry import ToolRegistry

    client.app.state.store = ConversationStore(db_session_factory)
    client.app.state.registry = ToolRegistry(build_tools(db_session_factory))
    client.app.state.chat_model = FakeFirstCallModel()
    return client


def _frames(response_text: str) -> list[dict]:
    return [json.loads(line.removeprefix("data: "))
            for line in response_text.split("\n\n") if line.startswith("data: ")]


def test_tool_flow_frame_order_and_persistence(tool_client, db_session_factory):
    resp = tool_client.post("/api/chat", json={"message": "退货政策是什么"})
    frames = _frames(resp.text)
    assert frames[0]["type"] == "session"
    status = [f for f in frames if f["type"] == "tool_status"]
    assert status and status[0]["name"] == "query_faq" and status[0]["label"] == "FAQ 检索"
    tokens = "".join(f["content"] for f in frames if f["type"] == "token")
    assert "七天无理由退货" in tokens
    assert frames[-1]["type"] == "done"

    history = _run(db_session_factory)
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[1]["tool_calls"][0]["id"] == "call_1"
    assert history[2]["tool_call_id"] == "call_1"
    assert "七天无理由退货" in history[3]["content"]


def _run(factory):
    import asyncio
    from app.store import ConversationStore

    return asyncio.run(ConversationStore(factory).get_history(1))


def test_plain_path_no_tool_frames(client):
    client.app.state.chat_model = _TextOnlyModel()
    resp = client.post("/api/chat", json={"message": "你们几点营业?"})
    frames = _frames(resp.text)
    assert not [f for f in frames if f["type"] == "tool_status"]
    assert frames[-1]["type"] == "done"


class _TextOnlyModel(FakeFirstCallModel):
    async def ainvoke(self, messages):
        raise AssertionError("无工具路径不应走 ainvoke 两段式")

    async def astream(self, messages):
        if self.bind_calls == 0:
            yield AIMessage(content="每天 9 点到 21 点。")  # 实现按 chunk.text 取值,见下
        else:
            raise AssertionError("无工具路径只有一次调用")
```

(替身 chunk 统一用带 `.text` 属性的简单对象,`_TextOnlyModel.astream` 里 yield 两个 `type("Chunk", (), {"text": "…"})()`,与 ch01 消费方式对齐;写测试时以 chat.py 实际取值属性为准。)

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_chat_toolflow.py -v`
Expected: FAIL(无 tool_status 帧,断言失败)

- [ ] **Step 3: 实现 chat 路由工具流**

`app/routers/chat.py` 重写 `chat` endpoint:

```python
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.history import estimate_tokens, trim_history
from app.prompts import SERVICE_PROMPT_TEMPLATE
from app.schemas import ChatRequest

router = APIRouter()


def sse_frame(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _to_prompt_messages(trimmed: list[dict], new_user_content: str) -> list:
    """支持 assistant(tool_calls) 与 tool 消息的完整回灌还原。"""
    messages: list = []
    for m in trimmed:
        if m["role"] == "system":
            messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(
                content=m.get("content") or "",
                tool_calls=m.get("tool_calls") or None,
            ))
        elif m["role"] == "tool":
            messages.append(ToolMessage(content=m["content"] or "", tool_call_id=m["tool_call_id"]))
    messages.append(HumanMessage(content=new_user_content))
    return messages


@router.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    settings = request.app.state.settings
    store = request.app.state.store
    registry = request.app.state.registry
    model = request.app.state.chat_model

    if estimate_tokens(body.message) > settings.history_token_budget:
        raise HTTPException(status_code=400, detail="消息过长,超出会话历史预算")

    conversation_id = await store.resolve(body.session_id)
    history = await store.get_history(conversation_id)
    full_history = [
        {"role": "system", "content": SERVICE_PROMPT_TEMPLATE.format()},
        *history,
    ]
    prompt_messages = _to_prompt_messages(
        trim_history(full_history, settings.history_token_budget),
        body.message,
    )

    async def event_stream():
        yield sse_frame({"type": "session", "session_id": conversation_id})
        pending_msgs: list[dict] = [{"role": "user", "content": body.message}]
        try:
            # 第一段:绑工具,非流式,拿 tool_calls
            bound = model.bind_tools(registry_tool_list(registry))
            ai_msg = await bound.ainvoke(prompt_messages)
            if getattr(ai_msg, "tool_calls", None):
                for tc in ai_msg.tool_calls:
                    yield sse_frame({"type": "tool_status", "name": tc["name"],
                                     "label": registry.labels().get(tc["name"], tc["name"])})
                    result = await registry.execute(tc["name"], json.dumps(tc["args"], ensure_ascii=False))
                    pending_msgs.append({
                        "role": "assistant", "content": ai_msg.content or None, "tool_calls": ai_msg.tool_calls,
                    })
                    pending_msgs.append({"role": "tool", "content": result, "tool_call_id": tc["id"]})
                    prompt_messages.append(AIMessage(content=ai_msg.content or "", tool_calls=ai_msg.tool_calls))
                    prompt_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                # 第二段:不绑工具,强制逐 token 收敛(单轮)
                async for chunk in model.astream(prompt_messages):
                    if chunk.text:
                        yield sse_frame({"type": "token", "content": chunk.text})
                        pending_msgs.append-accumulate(chunk.text)
            else:
                # 无工具:第一段结果即最终回答(ai_msg.content 整体作为回答下发)
                if ai_msg.content:
                    yield sse_frame({"type": "token", "content": ai_msg.content})
                    pending_msgs.append({"role": "assistant", "content": ai_msg.content})
            await store.append(conversation_id, pending_msgs)
            yield sse_frame({"type": "done"})
        except Exception as exc:
            yield sse_frame({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

实现注意(伪码处落实):
- `registry_tool_list(registry)`:registry 增加只读属性 `tools -> list` 返回原始 tool 对象(Task 5 接口微扩展,补一个测试)。
- 第二段 token 用局部 `parts: list[str]` 累积,结束后 `pending_msgs.append({"role": "assistant", "content": "".join(parts)})`。
- 无工具路径:ch01 是流式吐 token;为保持 ch01 体验,无工具路径改为 `model.astream(prompt_messages)` 直接流式(第一段不用 ainvoke)——**定稿实现**:先 `astream` 累积;若流中检测到文本则当普通回答流式吐出(此路径 LangChain 绑了工具但模型不调工具时,chunk 里只有 text);若整个流没有任何 text 且产生不了 tool_calls,则退化为一次性 `ainvoke` 判断。简化定稿:第一段用 `astream`,收集完整 `AIMessage` 片段(`chunk.tool_call_chunks` 累积)与文本;文本出现即边收边吐 token;流结束后看累积的 tool_call_chunks:有则走工具分支(此时文本帧已吐过的场景极少,DeepSeek 工具调用时 content 为空,可接受)。替身模型的 `astream` 第一段吐一个 `Chunk(text="")` 且带 `tool_call` 属性模拟。**计划以这条 astream 单趟方案为定稿**,实现时若 tool_call_chunks 累积过于繁琐,允许回退到「ainvoke 判断 + astream 二段」的替身驱动方案,但必须保持测试断言不变。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_chat_toolflow.py -v && .venv/bin/pytest -v`
Expected: PASS(含 ch01 回归)

- [ ] **Step 5: request schema 与会话 id 类型**

`app/schemas.py` 的 `ChatRequest.session_id: int | None = None`(原 str,conversations.id 是 BIGINT);`/api/sessions/{session_id}/history` 同步改 int。

- [ ] **Step 6: 全量测试 + Commit**

```bash
.venv/bin/pytest -v
git add -A
git commit -m "ch02: chat 路由单轮工具流——tool_status 帧/回灌/落库/不绑工具收敛"
```

---

### Task 8: 聊天页工具徽章(Vibe Coding,不套流程)

**Files:**
- Modify: `static/index.html`

**Interfaces:**
- Consumes: Task 7 的 `tool_status` SSE 帧;session_id 变 int。

- [ ] **Step 1: 前端改造**

1. SSE 分发处新增 `tool_status` 分支:当轮助手气泡(未渲染 token 前的位置)插入徽章元素 `<span class="tool-badge">⚙ FAQ 检索</span>`,带一个 CSS 动画(执行中闪烁小圆点);同轮多个工具调用则追加多个徽章。
2. `done` 后徽章变完成态(圆点变绿,不再闪烁)。
3. 「新对话」按钮确认发送时不带 `session_id`;渲染历史/会话逻辑里的 session_id 直接透传后端返回值(int),无类型假设。
4. 样式:徽章用浅色圆角小块,置于气泡内容上方,不占对话流宽度。

按效果描述直接改,改完起服务浏览器肉眼验收(三种问法各跑一遍),不写测试。

- [ ] **Step 2: Commit**

```bash
git add static/index.html
git commit -m "ch02: 聊天页工具轨迹徽章 + tool_status 帧消费"
```

---

### Task 9: dev-notes 留痕 + README 更新 + 终验收

**Files:**
- Create: `dev-notes/ch02.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: 全部前序任务。

- [ ] **Step 1: 全量测试**

Run: `.venv/bin/pytest -v`
Expected: 全绿,记录输出到 dev-notes。

- [ ] **Step 2: 真实环境验收(Docker MySQL + 真实 key)**

```bash
docker compose up -d && sleep 30
.venv/bin/python -m db.seed
.venv/bin/uvicorn app.main:app --port 8000
```

浏览器三条验收(spec §7.2):
1. 「订单 1001 的物流到哪了」→ 工具徽章(物流查询)+ 按 mock 轨迹作答;
2. 「退货政策是什么」→ FAQ 检索徽章 + 命中作答;
3. 「邮费是多少」→ 观察模型行为(可能空手作答或调 query_faq 查不到),LIKE 确实查不到 → 记为预期漏召回,留给 ch03。

再补 curl:

```bash
curl http://127.0.0.1:8000/api/sessions/<conv_id>/history
```

断言含 user / assistant(tool_calls) / tool / assistant 四类消息。

- [ ] **Step 3: dev-notes/ch02.md 追记全程 + README 更新**

dev-notes 按阶段记录:每任务完成点、踩坑(如 SQLite ENUM 差异、tool_call_chunks 处理方案取舍)、验收 3 的漏召回结论、评估/验证输出。README 增补:MySQL Docker 启动、`python -m db.seed`、新环境变量 `DATABASE_URL`、验收步骤。

- [ ] **Step 4: Commit 收口**

```bash
git add -A
git commit -m "ch02: 留痕收口——dev-notes/README/验收记录"
```

---

## Self-Review 结论

- Spec 覆盖:数据层(T1-T3)、工具层(T4-T5)、会话替换(T6)、工具流(T7)、前端(T8)、验收与留痕(T9)——spec §3-§7 全部有任务对应;§8 不做项未引入。
- 占位符:T7 Step 3 中两处「定稿」二选一是显式决策点而非 TBD,替身写法以实际 chunk 属性为准已注明;其余步骤均有完整代码。
- 类型一致性:`ConversationStore.resolve/get_history/append`、`ToolRegistry.execute/labels/tools`、`TOOL_LABELS`、`build_tools(session_factory)` 在 T4-T7 间签名一致;`ChatRequest.session_id` int 化在 T6/T7/T8 三处同步。
