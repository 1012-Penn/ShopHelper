# ch02 设计文档:Function Calling 工具链

日期:2026-09-04
状态:已与用户对齐(brainstorming 三问已拍板)

## 1. 目标

在 ch01 纯对话客服系统上叠加 Function Calling 能力:模型自主决定是否调工具,工具结果回灌后由模型组织最终回答,全程长在现有 SSE 流式聊天页里。单轮收敛(调一次工具即答),不做 Agent Loop、不做向量检索/RAG。

## 2. 技术栈(定死)

- FastAPI + SQLAlchemy 2.x + MySQL 8(Docker Compose 起)
- LangChain 1.x `@tool` 装饰器定义工具;`bind_tools` 绑定
- 动手前用 Context7 核对 langchain 1.x / sqlalchemy 最新 API 用法

## 3. 数据层

### 3.1 Docker MySQL

- `docker-compose.yml`:mysql:8,库名 `shophelper`,端口 3306,volume 持久化,`db/init.sql` 挂载到 `/docker-entrypoint-initdb.d/` 首次建表。
- `db/init.sql`:四张表 DDL(用户已提供,tickets 表按用户确认补齐 `status ENUM('待处理','已处理') NOT NULL DEFAULT '待处理'`、`created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP`、`PRIMARY KEY (ticket_no)`、`KEY idx_conversation_id (conversation_id)`)。全库 InnoDB + utf8mb4。
- 商品/订单/物流不建表,由工具内部 mock。

### 3.2 ORM 与配置

- `app/config.py` 增加 `database_url`(默认 `mysql+pymysql://shophelper:shophelper@127.0.0.1:3306/shophelper?charset=utf8mb4`)。
- `app/db.py`:`create_engine` + `sessionmaker`;提供 `get_session` 依赖。
- `app/models.py`:Faq / Conversation / Message / Ticket 四个 ORM 模型,与 DDL 字段一一对应(ENUM 用 SQLAlchemy Enum,JSON 用 `sa.JSON`)。

### 3.3 会话存储替换

- **MySQL 全面替换内存 SessionStore**(用户拍板):
  - 新对话 = 新建一条 conversations 记录,`user_id` 固定 `'guest'`(用户拍板),`status='进行中'`。
  - `/api/chat` 每轮:历史从 messages 表按 conversation_id 读出拼 prompt;整轮成功后落 user / assistant(含 tool_calls 时一并落)/ tool 消息;失败不落(延续 ch01 契约)。
  - `/api/sessions/{session_id}/history` 改读 messages 表;「新对话」= 不带 session_id 即新建。
  - `app/sessions.py` 内存 store 删除,相关测试改为 MySQL-schema(SQLite 内存库承载)。

### 3.4 测试数据库

- 测试统一用 SQLite(`sqlite+aiosqlite` 或同步 sqlite)内存库建 ORM schema,不依赖真 MySQL;ENUM 在 SQLite 退化为 VARCHAR,行为等价于本演示场景。

## 4. 工具层(`app/tools/`)

### 4.1 五个业务工具(`definitions.py`)

| 工具 | 数据来源 | 说明 |
|---|---|---|
| `query_order(order_id)` | random mock | 返回订单号、商品、金额、状态 |
| `query_product(keyword)` | random mock | 返回商品名、价格、库存 |
| `query_logistics(order_id)` | random mock | 返回承运商、轨迹节点列表 |
| `query_faq(keyword)` | faq 表 SQL LIKE | `question LIKE '%kw%'` 关键词查表;查不到返回空列表(不抛错) |
| `create_ticket(conversation_id, description, ticket_type)` | 写 tickets 表 | 工单号 `T+yyyyMMdd+3位序号`;返回工单号 |

- 工具返回值为字符串(JSON 序列化),由模型组织成自然语言回答。

### 4.2 工具基础设施(`registry.py`)

- 注册表:名字 → 工具对象;`bind_tools` 用 LangChain `.tools` 列表,SSE 状态帧与落库用注册表反查中文显示名。
- 每次执行统一走 `execute_tool(name, args_json)`:
  1. 未注册工具 → 返回错误字符串回灌(不抛 500);
  2. pydantic 参数校验(由 `@tool` 的 args_schema 承担),校验失败同样转错误字符串;
  3. 执行超时(默认 10s,`asyncio.wait_for`)+ 失败重试 1 次;
  4. 最终失败把「错误信息」作为 tool 消息内容回灌,模型据此向用户致歉/说明。

## 5. 聊天链路(`app/routers/chat.py` 改造)

流程(单轮):

1. 解析/新建 conversation(落库会话壳),读历史拼 prompt(System 在首位,沿用 trim_history 预算裁剪)。
2. `model.bind_tools(tools)` 流式调用。累积 chunk:若出现 `tool_calls`,先推 SSE 帧 `{"type":"tool_status","name":..., "label":...}`(前端显示「正在查询×××」)。
3. 第一轮流结束判 `tool_calls`:
   - **有**:执行工具 → user / assistant(tool_calls JSON) / tool(tool_call_id + 结果) 落库 → 把 assistant(tool_calls) + tool 消息追加进消息列表,**第二次调用不绑工具**,强制逐 token 吐最终回答。
   - **无**:直接逐 token 吐出(= ch01 原路径)。
4. 最终 assistant 回答落库(若第 3 步已落 tool 消息,把最终回答 UPDATE/追加到对应 assistant 记录或单独落一条 assistant 消息——以实现简洁为准,计划阶段定死:最终回答单独落一条 assistant 消息,纯工具调用的 assistant 消息 content 为空)。
5. 中途任一环节异常:推 `error` 帧收流,本轮不落最终回答(已落的 user 消息回滚——实现上把「读历史→全部落库」放到流成功路径,失败整体不落,与 ch01 契约一致)。

SSE 帧协议在 ch01 基础上新增 `tool_status`;`session` / `token` / `done` / `error` 不变。

## 6. 前端(`static/index.html`,Vibe Coding,不套流程)

- 处理 `tool_status` 帧:在当轮助手气泡内渲染工具轨迹小徽章(工具中文名 + 状态点)。
- token 流式、多轮、新对话逻辑沿用 ch01。

## 7. 测试与验收

### 7.1 TDD 覆盖(聊天页 UI 除外)

- 工具:三个 mock 工具返回结构可断言;query_faq LIKE 命中「退货」、查不到「邮费」返回空;create_ticket 落库且工单号格式正确。
- registry:未注册工具、坏参数、超时、重试、最终错误回灌。
- chat 路由:替身模型先返回 tool_calls 再返回文本,断言 SSE 帧序(session → tool_status → token* → done)、三条消息落库、单轮收敛(第二次调用未绑工具);不调工具路径回归 ch01。
- 历史接口:从 MySQL schema 读多轮历史。

### 7.2 人工验收(浏览器 + 真实上游)

1. 问「订单 1001 的物流到哪了」→ 气泡带工具徽章,按 mock 结果作答。
2. 问「退货政策是什么」→ query_faq 命中并作答。
3. 问「邮费是多少」→ LIKE 查不到,确认漏召回为预期,记入 dev-notes 留给 ch03。

### 7.3 过程要求

- `dev-notes/ch02.md` 每阶段追记;LLM/SQLAlchemy API 动手前 Context7 核对;完结交付演示命令 + 测试结果 + dev-notes 路径。

## 8. 明确不做

- 多轮 Agent Loop、向量检索/RAG、真实电商/物流 API、用户体系。
