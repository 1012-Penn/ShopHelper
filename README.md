# ShopHelper

电商智能客服系统。ch01 纯对话:FastAPI + LangChain 1.x 的 SSE 流式聊天 + 售后信息结构化抽取;ch02 叠加 Function Calling 工具链:模型自主选工具 → 执行 → 结果回灌 → 单轮流式收敛,会话与工具轨迹全量落 MySQL;ch03 叠加 RAG 基础:`query_faq` 从关键词查表升级为 BGE-M3 + Milvus 向量语义检索(契约不变),配套离线建库(结构感知切分 + MySQL/Milvus 双写幂等)与历史对话挖知识两条管道;ch04 叠加 RAG 进阶:Milvus 原生 BM25(dense+BM25 各召回 Top-50,hybrid_search RRF 融合)+ bge-reranker-v2-m3 精排 Top-10 + query 改写归一,回答带可点引用编号(角标 → 来源 chunk 章节路径与原文),检索低置信/生成自评不足显式拒答并落低置信度问题池,四策略评估体系(Recall@K / MRR / Faithfulness 分桶报告 + 编造个案台账),聊天页满意度反馈(纯前端采集);ch05 叠加 Workflow 确定性编排:LangGraph 图骨架(指代消解→意图识别→写死分流→知识检索→置信度闸→ReAct 主力 Agent→日志落库),七类意图分流四出口(知识类强制 RAG+置信度闸/业务数据类 Agent 自调工具/投诉安抚+自选按钮/闲聊固定话术),前端「转人工」「建工单」独立按钮自选互不绑定,checkpointer 会话态 + MySQL 双写;ch06 分流器正式版:LLM 指代消解+Query 改写(已完整原样透传)、意图识别四件套(七类+其他枚举/JSON intent+confidence/边界 few-shot/置信度与可选小→大降级路)、退款售后确定性子流程(槽位检查→订单数据→Query 扩写→多路政策检索去重合并→窄化「这一单能不能退」交主力 Agent)、订单选择器(SSE 帧下发可点订单卡,点选无状态回传续走)与退款单表单(固定原因类目,复用工单链路);ch07 会话上下文管理:三层历史(层 1 原文/层 2 渲染截短/早期异步分段摘要,锚点 id 划界降级只挪 id)、token 预算从模型窗口倒推(滑窗=窗口−输出−单轮峰值−固定开销,层 1 七成层 2 三成,启动自检装不下一轮报警)、摘要走后台任务不阻塞回复(旧梗概只作背景不回炉)、State.messages 挂 add_messages 承载图内完整轨迹(与模型面精简版各走各的)、每轮上下文原样落 logs/app.log(model_ctx/history_ctx)、messages 表只落 user/assistant 文本(工具轨迹活在当轮)、前端会话侧栏(新在前/首问预览/已摘要标记/切换回载续聊)。

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
- `CHAT_TEMPERATURE` / `EXTRACT_TEMPERATURE` / `RETRIEVAL_TOP_K` / `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP_CHARS` / `DEDUP_THRESHOLD` / `RERANK_SCORE_FLOOR` / `HYBRID_CANDIDATES` 均有默认值
- ch07 上下文预算(均有默认值,见 spec「预算推导」):`MODEL_CONTEXT_WINDOW` / `MAX_OUTPUT_TOKENS` / `MAX_USER_INPUT_TOKENS` / `MAX_AGENT_STEPS` / `TOOL_RESULT_MAX_TOKENS` / `RERANK_TOP_K`(顶替旧 RETRIEVAL_FINAL_TOP_K)及 `CTX_*` 固定开销组件;只调 `MODEL_CONTEXT_WINDOW` 不够——其余项按默认值算,窗口过小会滑窗归零并在启动时报「上下文预算不足」

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

ch05 五条验收(服务启动后逐条打):

1. **知识类强制检索**:问「你们的退货政策是怎样的」→ 气泡出现「知识检索」徽章,服务日志见 `node=retrieve strategy=hybrid_rerank top1=...`;
2. **Agent 自调工具**:问「订单1001的物流到哪了」→ 回答气泡带「物流查询」徽章;
3. **投诉自选按钮**:说「我要投诉」→ 安抚话术下出现「转人工」「建工单」两个独立按钮;点「转人工」本地展示「已转接人工客服」并蹦客服小猫问候(纯前端);点「建工单」确认后才 `POST /api/tickets` 写 tickets 表;都不点就当普通对话继续;
4. **闲聊固定话术**:说「你好呀」→ 固定回复,零工具徽章;
5. **ReAct 多步**:问「先帮我查一下订单1001,然后告诉我它的物流到哪了」→ 依次出现「订单查询」「物流查询」两个徽章,日志 `node=agent steps=2`。

```bash
docker exec shophelper-mysql mysql -ushophelper -pshophelper --default-character-set=utf8mb4 shophelper \
  -e "SELECT ticket_no, ticket_type, description FROM tickets"
```

祛魅热身(不依赖 LangGraph 的裸 Agent 循环):`.venv/bin/python -m scripts.naive_agent "订单1001的物流到哪了"`。

ch06 四条验收(服务启动后逐条打;路由评估:`.venv/bin/python scripts/eval_router.py` 真模型跑,报告落 `reports/ch06-router-report.md`):

1. **多轮来回切意图全对、指代全补对**:评估报告——意图准确率 21/22(95.5%),多轮切换桶 6/6(物流→退款退货→物流逐轮全对),指代消解关键词口径 5/5;
2. **意图 JSON 稳定可解析、怪问题落「其他」**:畸形率 0/22(宽松解析+兜底类单测 7 个);「量子力学怎么解释」「明天股票会涨吗」→其他(「你会写诗吗」→闲聊,非业务出口口径边缘个案,记档);
3. **「这个能退吗」先补全指代、再走子流程**:先问「SH-E300 多少钱」再问「这个能退吗」→ 气泡灰字「已理解:SH-E300…能退吗」(resolved 帧)→ 因无订单号弹**订单选择器** → 点选订单卡后子流程续走:徽章链 订单查询→查询扩写→政策检索,答案带 [n] 角标;服务日志 `node=resolve changed=True → node=intent intent=退款退货 → node=prepare_order → node=fetch_order → node=expand → node=policy_retrieve → node=agent`;
4. **不带订单号问退款弹选择器、点选走完 + 退款单**:新会话直接问「这个能退吗」→ 三张订单卡(单号/商品/金额/状态),点选后自动回填、流程接着走;判定可退后点「申请退款」→ 表单(原因固定六类下拉+备注)→ 提交写 tickets 表:

```bash
docker exec shophelper-mysql mysql -ushophelper -pshophelper --default-character-set=utf8mb4 shophelper \
  -e "SELECT ticket_no, ticket_type, description FROM tickets ORDER BY created_at DESC LIMIT 1"
```

真机验收证据:`reports/ch06-acceptance.md`。

ch07 五条验收(先 `docker compose down -v && up -d` → seed → `build_kb` → 启动服务;老库需另 apply `db/ch07-summary.sql` 与 `db/ch07-layers.sql`):

1. **默认配置连聊 20+ 轮不爆不崩、且不触发任何压缩(装得下就不压)**:`.venv/bin/python scripts/acceptance_ch07.py --scenario a`(23 轮,断言无 `层1 降级`/`summary trigger`、无错误轮,启动行见 `[ctx] budget 窗口=…`);
2. **演示配置完整级联 + 梗概召回**:演示配置 `MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5` 起服务(算出滑窗 5650/层1 3954/层2 1695),`.venv/bin/python scripts/acceptance_ch07.py --scenario b`——日志完整链:`层1 降级 …→…` → `summary trigger 层2 约 N token > 预算 1695` → `[summary] … done 第1段`;此时问「最开始那个订单后来怎么说?」能靠梗概里的订单号与诉求答对;
3. **陷阱复核**:只设 `MODEL_CONTEXT_WINDOW=18000` 起服务 → 启动即报 `[ctx] 上下文预算不足:滑窗 0 …`(`--scenario trap`);
4. **后台摘要不阻塞 + 每轮上下文可观测**:`grep model_ctx logs/app.log`(摘要全文+滑窗逐条+条数+tokens)、`grep history_ctx logs/app.log`(每轮必打,闲聊轮也有);场景 B 报告附触发轮耗时与摘要完成时间戳(并行完成);
5. **多会话侧栏**:`--scenario s`(HTTP 级:新在前/首问预览/回载/切回续聊/已摘要标记);浏览器操作:聊天页左侧栏点「会话」项切换回载、可接着聊;「新对话」开新会话,旧会话仍在侧栏可切回;侧栏加载失败自动隐藏不影响聊天。

真机验收证据:`reports/ch07-acceptance.md`。

ch03 验收(换说法召回/中断续跑/挖知识增量)与 ch02、ch01 验收继续有效。ch04 评估脚本继续可用(注意 ch05 图内检索与 query_faq 工具同链)。

## 已知边界

- 评估 C04「这个东西能便宜点不」四策略均未命中(「便宜」与「95 折」语义/词面都跳距),需知识侧补「优惠」问法别名或改写增强,留后续。
- 词面重叠的库外题(如「量子力学怎么退货」带「退货」二字)检索闸挡不住,靠生成自评拒答兜底——prompt 服从非 100%,偶有软拒答不带标记进不了池。
- 忠实度裁判是 LLM,存在误判可能;个案走 faith_cases 台账人工处置(误判标「无需解决」)。
- milvus-lite 库文件单进程锁:服务运行时勿并行跑建库/评估。
- 满意度反馈纯前端内存采集,刷新即失;落库与数据飞轮留后续章节。
- query_order / query_product / query_logistics 为工具内 mock 随机数据,不接真实电商/物流 API。
- 多轮 Agent Loop、用户体系不做(ch02 边界延续)。
- ch05:意图识别/指代消解是最简版(简单 prompt / 原样透传),判错意图即走错出口(如「发货时间」被判「订单」),正式版后置;InMemorySaver 进程内无界增长,重启即清;同一会话并发请求共用 thread,无会话锁;ReAct 中间思考文本对用户可见(祛魅主题下如实呈现);沙箱无浏览器后端,前端按钮视觉终验由用户本地点开页面确认(SSE 帧/工单接口已 curl 验证)。
- ch06:订单为固定 mock 三单(`app/orders.py`,未知单号走「未找到」话术);被搁置的选择器旧卡片仍可点,点选视作对该订单重新发起退款咨询(语义合理不设过期);意图降级路(小→大)已实现默认关(`INTENT_ESCALATION_ENABLED=true` 开启,需配 `INTENT_SMALL_MODEL`);评估怪问题桶 2/3(「你会写诗吗」判闲聊,未硬塞业务意图,口径边缘);商品名(如 SH-E300)不绑定具体订单,退款一律经订单选择器确认。
- ch07:checkpointer 仍为 InMemorySaver(进程内无界增长,重启即清)——持久真源在 MySQL,重启后按会话从 messages 表回灌 State,轮内工具轨迹不跨重启存活(事实靠摘要延续);messages 表只落 user 与最终 assistant 文本(ch02「全量落库」契约收窄),ReAct 中间文本碎片不落库;层 2 的「截短」是渲染规则,存储保持原文;摘要质量依赖模型(prompt 禁编造+超 300 字硬截断,无事后校验);层 2 超预算到摘要完成之间当轮以半压渲染顶住,必要时组装端丢弃最老层 2 块(仅当轮);崩溃在 log 节点前的回合 checkpoint 有而 MySQL 无,模型输入以 MySQL 为准(该轮视作未发生);跨会话记忆/用户画像不做;多进程多 worker 部署不在本章范围(InMemorySaver/hydrated_threads/摘要 seq 均为单进程假设,多 worker 下摘要任务可能撞 uk_conv_seq 记 fail 后下轮重试,无数据损坏)。

API 契约细节见
`docs/superpowers/specs/2026-09-04-ch01-pure-chat-design.md`、
`docs/superpowers/specs/2026-09-04-ch02-function-calling-design.md`、
`docs/superpowers/specs/2026-09-05-ch03-rag-knowledge-base-design.md` 与
`docs/superpowers/specs/2026-09-06-ch04-rag-advanced-design.md`。
