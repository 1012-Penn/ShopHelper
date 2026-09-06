# ShopHelper

电商智能客服系统。ch01 纯对话:FastAPI + LangChain 1.x 的 SSE 流式聊天 + 售后信息结构化抽取;ch02 叠加 Function Calling 工具链:模型自主选工具 → 执行 → 结果回灌 → 单轮流式收敛,会话与工具轨迹全量落 MySQL;ch03 叠加 RAG 基础:`query_faq` 从关键词查表升级为 BGE-M3 + Milvus 向量语义检索(契约不变),配套离线建库(结构感知切分 + MySQL/Milvus 双写幂等)与历史对话挖知识两条管道;ch04 叠加 RAG 进阶:Milvus 原生 BM25(dense+BM25 各召回 Top-50,hybrid_search RRF 融合)+ bge-reranker-v2-m3 精排 Top-10 + query 改写归一,回答带可点引用编号(角标 → 来源 chunk 章节路径与原文),检索低置信/生成自评不足显式拒答并落低置信度问题池,四策略评估体系(Recall@K / MRR / Faithfulness 分桶报告 + 编造个案台账),聊天页满意度反馈(纯前端采集)。

## 环境要求

- Python 3.10+
- Docker(起 MySQL 8;向量库用 Milvus Lite 内嵌文件,无需容器)

## 安装

```bash
pip install virtualenv && python3 -m virtualenv .venv
.venv/bin/pip install fastapi==0.141.1 uvicorn==0.52.4 langchain==1.4.0 langchain-openai==1.6.0 pydantic-settings==2.15.0 httpx==0.28.1 "pymilvus[milvus-lite]==3.0.1" jieba==0.42.1 "sqlalchemy>=2.0" pymysql cryptography pytest==9.1.1 pytest-asyncio==1.4.0
```

## 配置

`cp .env.example .env` 后填入:

- `OPENAI_API_KEY`(必填,对话/抽取/挖知识/query 改写/忠实度裁判的上游模型 key)、`OPENAI_BASE_URL`(默认 DeepSeek)、`OPENAI_MODEL`(默认 deepseek-chat)
- `EMBEDDING_API_KEY`(必填,BGE-M3 嵌入,硅基流动申请)、`EMBEDDING_API_BASE` / `EMBEDDING_MODEL`(默认 BAAI/bge-m3)
- `RERANK_API_KEY`(ch04 重排,留空复用 EMBEDDING_API_KEY)、`RERANK_API_BASE` / `RERANK_MODEL`(默认 BAAI/bge-reranker-v2-m3)
- `DATABASE_URL`(默认指向本机 3306 的 Docker MySQL)、`MILVUS_DB_PATH`(默认 `./data/milvus_knowledge.db`)
- `HISTORY_TOKEN_BUDGET` / `CHAT_TEMPERATURE` / `EXTRACT_TEMPERATURE` / `RETRIEVAL_TOP_K` / `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP_CHARS` / `DEDUP_THRESHOLD` / `RERANK_SCORE_FLOOR` / `HYBRID_CANDIDATES` / `RETRIEVAL_FINAL_TOP_K` / `QUERY_REWRITE_ENABLED` 均有默认值

## 数据库

首次启动先建库灌数据(init.sql 建八张表 faq/conversations/messages/tickets/knowledge_chunks/qa_extraction_staging/low_confidence_questions/faith_cases,seed 灌 8 条 FAQ 与 4 通历史对话):

```bash
docker compose up -d        # 等约 30 秒初始化
.venv/bin/python -m db.seed
.venv/bin/python -m db.seed_conversations
```

> 已有旧数据卷时需 `docker compose down -v` 重建,ch04 新表(low_confidence_questions / faith_cases)才会由 init.sql 创建;首次 `build_kb` 会检测到 ch03 旧 Milvus 集合并自动重建为 v2(dense+BM25+category)。

## 知识库建库与挖知识

```bash
.venv/bin/python -m scripts.build_kb               # knowledge/*.md 切分 → MySQL(pending)→ Milvus → done
.venv/bin/python -m scripts.build_kb --max-chunks 5   # 演示中断:只向量化 5 块
.venv/bin/python -m scripts.build_kb --resume-only    # 中断续跑:只补 pending,漏多少补多少
.venv/bin/python -m scripts.mine_qa                # 历史对话 → LLM 抽 QA → 暂存 → 去重 → 入库向量化
```

挖知识定时化(示例,每小时一批):

```cron
0 * * * * cd /path/to/ShopHelper && mkdir -p logs && .venv/bin/python -m scripts.mine_qa >> logs/mine_qa.log 2>&1
```

## 跑测试

无需真实 key、MySQL 与嵌入 API(替身模型 + FakeEmbedding + SQLite 内存库 + 内存向量库):

```bash
.venv/bin/pytest
```

## 跑评估

- 抽取质量(ch01):`.venv/bin/python scripts/eval_extract.py`
- 检索质量(ch03 基线集):`.venv/bin/python scripts/eval_retrieval.py --set tests/eval/retrieval_samples.jsonl --strategy dense`(基线 hit@3 = 12/12)
- **检索四策略对比(ch04,需已建库)**:`.venv/bin/python scripts/eval_retrieval.py`——dense / bm25 / hybrid / hybrid_rerank 四策略 × Recall@3/5/10 + MRR@10,按 A_policy / B_model / C_colloquial / E_multi 四桶分列,报告落 `reports/ch04-retrieval-report.md`(`--strategy bm25` 单策略、`--no-rewrite` 关改写对照)。真跑基线(ch04 校准后):hybrid_rerank R@10 A=1.00 / B=1.00 / C=0.88 / E=1.00,C 桶 MRR@10=0.875 四策略最高;唯一漏网 C04「这个东西能便宜点不」(四策略共用硬题)
- **生成忠实度(ch04)**:`.venv/bin/python scripts/eval_faithfulness.py`——hybrid_rerank 生成带引用答案 → LLM 裁判判编造 → Faithfulness 分桶报告 + 编造个案进 faith_cases 台账(真跑 31/32,A 桶 0.88)

## 启动

```bash
.venv/bin/uvicorn app.main:app --port 8000
```

## 验收

聊天页入口:浏览器打开 <http://127.0.0.1:8000/>。模型调用工具时,回答气泡顶部会出现工具轨迹小徽章(执行中闪烁,完成后转绿);ch04 起答案中的 [n] 角标可点击,展开来源卡(章节路径 + chunk 原文),气泡左下角 👍/👎 满意度反馈点击即点亮并锁定(纯前端采集)。

ch04 四条验收(先 `docker compose down -v && up -d` → seed → `build_kb` → 启动服务):

1. **四策略对比报告有数字**:`.venv/bin/python scripts/eval_retrieval.py`,四策略 × 四桶 Recall@K / MRR@10 表落 `reports/`;
2. **带型号的问题 BM25 命中**:页面问「SH-E300 多少钱」→ 答 299 且带 [1] 角标;评估 B 桶 bm25 R@10 = 1.00;
3. **引用可点回原文**:页面问「邮费是多少」→ 点答案里的 [1] 角标,展开来源卡显示 `mined:1` / `退货政策.md > 邮费与运费` 等章节路径与原文,再点收起;
4. **库外问题明确拒答 + 落池**:问「会飞的手机怎么买」/「量子力学怎么退货」→ 回答以 `【无法回答】` 开头,`low_confidence_questions` 表落行(检索低置信 source=retrieval_low_conf,自评拒答 source=self_check):

```bash
docker exec shophelper-mysql mysql -ushophelper -pshophelper --default-character-set=utf8mb4 shophelper \
  -e "SELECT id, source, raw_question, reason FROM low_confidence_questions"
```

ch03 验收(换说法召回/中断续跑/挖知识增量)与 ch02、ch01 验收继续有效。

## 已知边界

- 评估 C04「这个东西能便宜点不」四策略均未命中(「便宜」与「95 折」语义/词面都跳距),需知识侧补「优惠」问法别名或改写增强,留后续。
- 词面重叠的库外题(如「量子力学怎么退货」带「退货」二字)检索闸挡不住,靠生成自评拒答兜底——prompt 服从非 100%,偶有软拒答不带标记进不了池。
- 忠实度裁判是 LLM,存在误判可能;个案走 faith_cases 台账人工处置(误判标「无需解决」)。
- milvus-lite 库文件单进程锁:服务运行时勿并行跑建库/评估。
- 满意度反馈纯前端内存采集,刷新即失;落库与数据飞轮留后续章节。
- query_order / query_product / query_logistics 为工具内 mock 随机数据,不接真实电商/物流 API。
- 多轮 Agent Loop、用户体系不做(ch02 边界延续)。

API 契约细节见
`docs/superpowers/specs/2026-09-04-ch01-pure-chat-design.md`、
`docs/superpowers/specs/2026-09-04-ch02-function-calling-design.md`、
`docs/superpowers/specs/2026-09-05-ch03-rag-knowledge-base-design.md` 与
`docs/superpowers/specs/2026-09-06-ch04-rag-advanced-design.md`。
