# ShopHelper

电商智能客服系统。ch01 纯对话:FastAPI + LangChain 1.x 的 SSE 流式聊天 + 售后信息结构化抽取;ch02 叠加 Function Calling 工具链:模型自主选工具 → 执行 → 结果回灌 → 单轮流式收敛,会话与工具轨迹全量落 MySQL;ch03 叠加 RAG 基础:`query_faq` 从关键词查表升级为 BGE-M3 + Milvus 向量语义检索(契约不变),配套离线建库(结构感知切分 + MySQL/Milvus 双写幂等)与历史对话挖知识两条管道。

## 环境要求

- Python 3.10+
- Docker(起 MySQL 8;向量库用 Milvus Lite 内嵌文件,无需容器)

## 安装

```bash
pip install virtualenv && python3 -m virtualenv .venv
.venv/bin/pip install fastapi==0.141.1 uvicorn==0.52.4 langchain==1.4.0 langchain-openai==1.6.0 pydantic-settings==2.15.0 httpx==0.28.1 "pymilvus[milvus-lite]==3.0.1" "sqlalchemy>=2.0" pymysql cryptography pytest==9.1.1 pytest-asyncio==1.4.0
```

## 配置

`cp .env.example .env` 后填入:

- `OPENAI_API_KEY`(必填,对话/抽取/挖知识的上游模型 key)、`OPENAI_BASE_URL`(默认 DeepSeek)、`OPENAI_MODEL`(默认 deepseek-chat)
- `EMBEDDING_API_KEY`(必填,BGE-M3 嵌入,硅基流动申请)、`EMBEDDING_API_BASE` / `EMBEDDING_MODEL`(默认 BAAI/bge-m3)
- `DATABASE_URL`(默认指向本机 3306 的 Docker MySQL)、`MILVUS_DB_PATH`(默认 `./data/milvus_knowledge.db`)
- `HISTORY_TOKEN_BUDGET` / `CHAT_TEMPERATURE` / `EXTRACT_TEMPERATURE` / `RETRIEVAL_TOP_K` / `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP_CHARS` / `DEDUP_THRESHOLD` 均有默认值

## 数据库

首次启动先建库灌数据(init.sql 建六张表 faq/conversations/messages/tickets/knowledge_chunks/qa_extraction_staging,seed 灌 8 条 FAQ 与 4 通历史对话):

```bash
docker compose up -d        # 等约 30 秒初始化
.venv/bin/python -m db.seed
.venv/bin/python -m db.seed_conversations
```

> 已有旧数据卷时需 `docker compose down -v` 重建,ch03 新表才会由 init.sql 创建。

## 知识库建库与挖知识

```bash
.venv/bin/python -m scripts.build_kb               # knowledge/*.md 切分 → MySQL(pending)→ Milvus → done
.venv/bin/python -m scripts.build_kb --max-chunks 5   # 演示中断:只向量化 5 块
.venv/bin/python -m scripts.build_kb --resume-only    # 中断续跑:只补 pending,漏多少补多少
.venv/bin/python -m scripts.mine_qa                # 历史对话 → LLM 抽 QA → 暂存 → 去重 → 入库向量化
```

挖知识定时化(示例,每小时一批):

```cron
0 * * * * cd /path/to/ShopHelper && .venv/bin/python -m scripts.mine_qa >> logs/mine_qa.log 2>&1
```

## 跑测试

无需真实 key、MySQL 与嵌入 API(替身模型 + FakeEmbedding + SQLite 内存库 + 内存向量库):

```bash
.venv/bin/pytest
```

## 跑评估

- 抽取质量(ch01):`.venv/bin/python scripts/eval_extract.py`
- **检索质量(ch03,需已建库)**:`.venv/bin/python scripts/eval_retrieval.py`,输出 hit@3(ch03 基线:12/12,「邮费是多少」Top-1 召回运费说明)

## 启动

```bash
.venv/bin/uvicorn app.main:app --port 8000
```

## 验收

聊天页入口:浏览器打开 <http://127.0.0.1:8000/>。模型调用工具时,回答气泡顶部会出现工具轨迹小徽章(执行中闪烁,完成后转绿)。

ch03 三条验收(先 `docker compose up -d` → seed → `build_kb` → 启动服务):

1. **换说法召回**:「邮费是多少」→ `query_faq` 召回运费说明,答对 8 元/满 99 包邮(ch02 的 LIKE 漏召回场景翻案):

```bash
curl -sN -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' -d '{"message": "邮费是多少"}'
```

2. **中断续跑**:`build_kb --max-chunks 5` 模拟中断(剩 pending)→ `build_kb --resume-only` 补齐,输出 `剩 pending 0 条`;
3. **挖知识增量**:`mine_qa` 后暂存表 kept/discarded 落地,挖出的「国外的快递费」换说法可召回:

```bash
.venv/bin/python -m scripts.mine_qa
curl -sN -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' -d '{"message": "海外的运费怎么算"}'
```

ch02 验收(物流 mock、FAQ 命中、多轮上下文)与 ch01 验收(纯对话流式、售后抽取)继续有效;ch02 的「邮费漏召回」已知边界自 ch03 起移除。

## 已知边界

- 检索只跑 dense 向量单路:无关键词召回、无混合检索、无重排(ch04 方向)。
- 挖知识去重基于 embedding 余弦阈值(0.88),极近义但不同义的问答对可能误判。
- query_order / query_product / query_logistics 为工具内 mock 随机数据,不接真实电商/物流 API。
- 多轮 Agent Loop、用户体系不做(ch02 边界延续)。

API 契约细节见
`docs/superpowers/specs/2026-09-04-ch01-pure-chat-design.md`、
`docs/superpowers/specs/2026-09-04-ch02-function-calling-design.md` 与
`docs/superpowers/specs/2026-09-05-ch03-rag-knowledge-base-design.md`。
