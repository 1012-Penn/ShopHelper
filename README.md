# ShopHelper

电商智能客服系统,ch01 阶段为纯对话:FastAPI + LangChain 1.x 的 SSE 流式聊天,带进程内多轮会话与售后信息结构化抽取。

## 环境要求

- Python 3.10+

## 安装

```bash
pip install virtualenv && python3 -m virtualenv .venv
.venv/bin/pip install fastapi==0.141.1 uvicorn==0.52.4 langchain==1.4.0 langchain-openai==1.6.0 pydantic-settings==2.15.0 httpx==0.28.1 pytest==9.1.1 pytest-asyncio==1.4.0
```

## 配置

`cp .env.example .env` 后填入 6 项:`OPENAI_API_KEY`(必填,上游模型 key)、`OPENAI_BASE_URL`(OpenAI 兼容地址,默认 DeepSeek)、`OPENAI_MODEL`(默认 deepseek-chat)、`HISTORY_TOKEN_BUDGET`(默认 3000)、`CHAT_TEMPERATURE`(默认 0.7)、`EXTRACT_TEMPERATURE`(默认 0)。换厂商只改 `OPENAI_*` 三个值,其余不动。

## 跑测试

无需真实 key(测试全用替身):

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

聊天页入口:浏览器打开 <http://127.0.0.1:8000/>(GET / 返回单文件聊天页)。

三条验收 curl(服务启动后执行):

1. 流式对话(先回 `session` 帧给出 SID,再逐 token,最后 `done`):

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' -d '{"message": "你们几点营业?"}'
```

2. 两轮上下文(带上第 1 步返回的 `<sid>` 再问一轮,再查历史验证落库):

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' -d '{"message": "我刚才问了什么?", "session_id": "<sid>"}'
curl http://127.0.0.1:8000/api/sessions/<sid>/history
```

3. 售后结构化抽取:

```bash
curl -X POST http://127.0.0.1:8000/api/extract -H 'Content-Type: application/json' -d '{"text": "订单 202609040001 的快递三天没动,我要退款"}'
```

## 已知边界

会话存储在进程内存且按单进程假设实现,重启即丢,多副本部署需外置存储。

API 契约细节(事件流格式、错误语义、接口字段)见
`docs/superpowers/specs/2026-09-04-ch01-pure-chat-design.md`。
