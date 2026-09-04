# ShopHelper

电商智能客服系统。ch01 纯对话:FastAPI + LangChain 1.x 的 SSE 流式聊天 + 售后信息结构化抽取;ch02 叠加 Function Calling 工具链:模型自主选工具 → 执行 → 结果回灌 → 单轮流式收敛,会话与工具轨迹全量落 MySQL。

## 环境要求

- Python 3.10+
- Docker(起 MySQL 8)

## 安装

```bash
pip install virtualenv && python3 -m virtualenv .venv
.venv/bin/pip install fastapi==0.141.1 uvicorn==0.52.4 langchain==1.4.0 langchain-openai==1.6.0 pydantic-settings==2.15.0 httpx==0.28.1 "sqlalchemy>=2.0" pymysql cryptography pytest==9.1.1 pytest-asyncio==1.4.0
```

## 配置

`cp .env.example .env` 后填入:`OPENAI_API_KEY`(必填,上游模型 key)、`OPENAI_BASE_URL`(OpenAI 兼容地址,默认 DeepSeek)、`OPENAI_MODEL`(默认 deepseek-chat)、`DATABASE_URL`(默认指向本机 3306 的 Docker MySQL)、`HISTORY_TOKEN_BUDGET`(默认 3000)、`CHAT_TEMPERATURE`(默认 0.7)、`EXTRACT_TEMPERATURE`(默认 0)。换厂商只改 `OPENAI_*` 三个值,其余不动。

## 数据库

首次启动先建库灌数据(init.sql 建四张表 faq/conversations/messages/tickets,seed 灌 8 条 FAQ):

```bash
docker compose up -d        # 等约 30 秒初始化
.venv/bin/python -m db.seed # 幂等,已有数据则跳过
```

## 跑测试

无需真实 key 和真实 MySQL(替身模型 + SQLite 内存库):

```bash
.venv/bin/pytest
```

## 跑评估

抽取质量评估走真实上游,需要 `.env` 里有有效 key:

```bash
.venv/bin/python scripts/eval_extract.py
```

## 启动

```bash
.venv/bin/uvicorn app.main:app --port 8000
```

## 验收

聊天页入口:浏览器打开 <http://127.0.0.1:8000/>。模型调用工具时,回答气泡顶部会出现工具轨迹小徽章(执行中闪烁,完成后转绿)。

ch02 三条验收(服务启动后):

1. 物流问题 → 触发 `query_logistics`(mock 数据),气泡带「物流查询」徽章,按返回轨迹作答:

```bash
curl -sN -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' -d '{"message": "订单 1001 的物流到哪了"}'
```

2. FAQ 命中 → 触发 `query_faq` LIKE 查表并作答:

```bash
curl -sN -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' -d '{"message": "退货政策是什么"}'
```

3. **预期漏召回**:「邮费」二字不在 faq 表任何问题里,LIKE 查不到(`{"items": []}`)——这是 ch02 的刻意设计,留作 ch03 向量检索升级的动机:

```bash
curl -sN -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' -d '{"message": "邮费是多少"}'
```

多轮上下文与历史(含工具调用轨迹)落 MySQL,可回查:

```bash
curl http://127.0.0.1:8000/api/sessions/<sid>/history
```

ch01 原有验收(纯对话流式、两轮上下文、售后抽取)继续有效,会话 id 已从字符串改为数字。

## 已知边界

- 工具链只做单轮:模型调一次工具即收敛,不做多轮 Agent Loop、不做向量检索(ch03 方向)。
- query_order / query_product / query_logistics 为工具内 mock 随机数据,不接真实电商/物流 API。
- 「邮费」类同义改写会漏召回(LIKE 关键词检索的固有局限),ch03 升级向量检索。

API 契约细节见
`docs/superpowers/specs/2026-09-04-ch01-pure-chat-design.md` 与
`docs/superpowers/specs/2026-09-04-ch02-function-calling-design.md`。
