# ch03 设计文档:RAG 基础——知识库 + 向量语义检索

日期:2026-09-05
状态:已对齐(四问已拍板:BGE-M3 走硅基流动 API、挖知识独立脚本、防幻觉纳入;Milvus 形态用户未答,按推荐项 Milvus Lite 执行,见 §9)

## 1. 目标

把 ch02 的 `query_faq` 内核从 `LIKE '%keyword%'` 关键词查表升级为向量语义检索(dense 单路),同时补齐知识库生产链条:Markdown 知识文档结构感知切分、历史客服对话 LLM 挖知识、MySQL(原文权威源)+ Milvus(向量)双写。「邮费是多少」类换说法问题从漏召回变为能召回运费说明并答对。工具入参出参契约不变。

## 2. 技术栈(定死)

- 嵌入模型 **BGE-M3**(`BAAI/bge-m3`,dense 1024 维),走**硅基流动 OpenAI 兼容 embeddings API**(用户拍板提供 key;已实测:批量入参、1024 维、「邮费是多少」vs「运费怎么算」余弦 0.78)。key 只进 `.env`(gitignore),`.env.example` 放占位符。
- 向量库 **Milvus**,运行形态 **Milvus Lite**(pip 内嵌,官方 `MilvusClient` API;Docker Hub 实锤拉不到 Standalone 三镜像,上生产只改连接串)
- MySQL 8 知识原文权威源(复用 ch02 docker compose);FastAPI + SQLAlchemy 2.x + LangChain 1.x 沿用
- 动手前用 Context7 核对 pymilvus 最新 API;嵌入客户端用 httpx 直连 `/v1/embeddings`(避开 langchain-openai 对非 OpenAI 模型的 tiktoken 分词陷阱,httpx 已是现有依赖)

## 3. 数据层

### 3.1 MySQL(用户已定稿 DDL,原样进 `db/init.sql`)

- `knowledge_chunks`:原文权威源。`category` / `questions` / `answer` 三格拼向量化文本;`section_path` / `content_type` / `is_key_clause` / `prev_chunk_id` / `next_chunk_id` 五项元数据只存不进向量;`vector_id` 回填 Milvus 主键;`vectorize_status ENUM('pending','done')` 承载双写幂等。
- `qa_extraction_staging`:挖知识中转暂存。`batch_no` 分批追溯;`status ENUM('extracted','kept','discarded')` 承载去重流转;`source_ref` 记来源会话 id(不入最终知识库)。
- 建表顺序:knowledge_chunks 在前(自引用 FK),staging 独立。沿用 `SET NAMES utf8mb4` 防双层编码。
- ORM 按惯例加进 `app/models.py`(`BigInteger().with_variant(Integer, "sqlite")` 承载 SQLite 测试)。

### 3.2 Milvus(Milvus Lite 文件库)

- 集合 `knowledge`:字段 `id`(Int64 主键,**= MySQL chunk id**,auto_id=False)+ `vector`(FloatVector, dim=1024);COSINE 度量,默认索引。**元数据一律不进 Milvus**,检索只取 id + 分数,回 MySQL 取原文。
- 库文件路径走配置(`MILVUS_DB_PATH`,默认 `./data/milvus_knowledge.db`,`data/` 进 .gitignore)。
- 封装 `app/vector_store.py`:`ensure_collection()` / `upsert(ids, vectors)` / `search(vector, top_k) -> [(id, score)]` / `delete(ids)`;接口先定,测试可替换身。

## 4. 离线建库:切分器(`app/chunking.py`,纯逻辑 TDD)

输入 Markdown 文本 + 文档标识,输出 Chunk 列表(category / questions / answer / section_path / content_type / is_key_clause)。

规则(每条对应至少一个测试用例):

1. **结构感知**:`#{1,6}` 标题开新节;每节一块;`section_path` = `{文件名} > H1 > H2 > …`(带文档前缀,供按文档删旧;下述规则里的「标题路径」均指去掉文件名前缀后的部分)。
2. **category / questions 填法**:
   - FAQ 型文档(content_type=faq,H3 为条目问法):`questions` = H3 问法(+正文里「其他问法:」列表行并入,换行分隔);`category` = 所在 H2 分组(无 H2 则 H1),与 ch02 faq 表短分类口径一致。
   - 政策/手册型(content_type=policy / manual):`questions` = 所在章节标题(最深一级),`category` = 上级标题路径(section_path 去掉最后一级;仅一级时等于自身)。
3. **超长递归切**:节内容超 `CHUNK_MAX_CHARS`(默认 500)按 段落 → 句读(。!?;;)→ 硬切 递归下刀。
4. **块间重叠**:相邻内容块尾部带 `CHUNK_OVERLAP_CHARS`(默认 80)重叠进下一块开头,重叠区**回退裁到最近句号之后**,不留半截话;标题行、表格块不参与重叠。
5. **大表格按行切**:连续 `|` 行为表格;超限按数据行分组,每块复制表头行 + 分隔行。
6. **is_key_clause 启发**:标题或正文含硬性条款词(必须/不得/禁止/仅限/不予/免费/七天无理由)记 1;只存不进向量。

## 5. 离线建库:双写管道(`scripts/build_kb.py`)

对 `knowledge/*.md` 每个文档:

1. **落原文(阶段一)**:切分 → 按文档标识(`section_path` 以 `{文件名} >` 前缀匹配)先删旧块及 Milvus 对应向量,再整批 INSERT `pending`,同事务回填 prev/next 指针。
2. **向量化(阶段二)**:`SELECT ... WHERE vectorize_status='pending'` → 三格拼文本(`{category}\n{questions}\n{answer}`)embed → Milvus `upsert(id=chunk id)` → 回填 `vector_id`、状态转 `done`。
3. **幂等语义(验收标准 2)**:Milvus 主键即 MySQL 主键,重复 upsert 覆盖不重复;中断重跑 = 阶段一按文档重建(确定结果),阶段二只补 pending。`--max-chunks N` 参数模拟中断,重跑不带参数即补齐。
4. 知识文档种子三件(仓库 `knowledge/` 目录):`退货政策.md`(含运费说明:普通订单邮费 8 元、满 99 元包邮、质量问题退货运费商家承担——验收 1 靶子)、`商品FAQ.md`(H2 分组,H3 问法,涵盖 ch02 八条 FAQ 等价内容)、`售后手册.md`(含超长章节与两张表格,触发递归切与表格切)。

## 6. 离线挖知识(`scripts/mine_qa.py`,定时批处理)

形态:独立批处理脚本 + 系统 crontab 外挂调度(README 附示例行);LLM 用现有 `make_extract_model` + `with_structured_output`。

1. **选料**:messages 表里 user 提问 + 其后首个含正文 assistant 回复(跳过 tool_calls 空正文),按会话组装;`source_ref` 记 conversation_id,已有 staging 记录的会话不重挖。
2. **分批抽取**:每批 `--batch-size`(默认 5)通会话,`batch_no` = 时间戳,LLM 抽 `[{question, answer}]` 写 staging(`extracted`)。
3. **整体去重**:全部 `extracted` 载入,question embed 后 a) 批内两两、b) 与 knowledge_chunks 已有块三格文本,余弦 ≥ `DEDUP_THRESHOLD`(默认 0.88)判重复置 `discarded`;存活置 `kept` 并 INSERT knowledge_chunks(`pending`,content_type='qa_mined',category='对话挖掘',section_path='mined:<source_ref>')。
4. 收尾复用阶段二向量化循环,补齐 pending。
5. 历史会话种子 `db/seed_conversations.py`:灌 4 通含真实问法(「国外的快递费怎么算」「大件寄过来邮费谁出」等)的历史对话,与知识库互补不重复,挖出即增量。

## 7. 在线检索(`app/tools/definitions.py` 改造)

`query_faq(keyword)` 入参出参**不变**;内部替换为:embed(keyword) → Milvus search top_k(默认 3,对齐 ch02 limit 3)→ 按 id 回 MySQL 取块 → `{"items": [{"question", "answer", "category"}]}`。questions 首行充当 question 字段。空库 / 集合不存在 / 检索异常一律收敛为 `{"items": []}`(契约,不抛错)。

`build_tools` 注入扩展:embedding 后端与 vector_store 由调用方注入(默认真实现),测试替身可换。

**防幻觉约束(纳入本章)**:`app/prompts.py` system prompt 增加一条——工具查无结果或无相关信息时如实说明并建议转人工,严禁编造政策/价格/承诺;配 prompt 断言测试与空结果契约测试,真上游行为在人工验收确认(ch02 已实测过编造「满 99 包邮」,此处即补该账)。

## 8. 测试与验收

### 8.1 TDD 覆盖

- 切分器:标题层级 / category-questions 填法 / 超长递归 / 重叠裁句号 / 表格表头复制 / is_key_clause。
- 双写:落库 pending → 向量化 done → vector_id 回填;`--max-chunks` 中断 → 重跑补齐(SQLite + 真 Milvus Lite tmp 文件 + FakeEmbedding)。
- 挖知识:替身抽取器;staging 流转 extracted → kept/discarded;去重阈值两侧各一例;已挖会话不重挖。
- query_faq:契约测试(命中形状、空结果 `{"items": []}`);换说法命中用 FakeEmbedding 近义词典(邮费↔运费映射)。
- prompts:防幻觉约束文本断言。
- 测试基建:FakeEmbedding(确定性 hash + 近义词典,dim 可配),不下载真模型、不依赖真上游。

### 8.2 评估集验证(替代纯数据类任务的 TDD 步)

`tests/eval/retrieval_samples.jsonl` 标注样例(query / expected_doc / expected_keywords)+ `scripts/eval_retrieval.py`:真建库数据 + 真 BGE-M3 跑 hit@top3 报告,「邮费是多少」「快递费怎么算」等换说法样例必须命中运费说明。

### 8.3 人工验收(真库真模型)

1. 「邮费是多少」→ query_faq 召回运费说明,答对金额与包邮门槛(ch02 漏召回场景翻案)。
2. `build_kb --max-chunks` 中断 → 重跑 → pending 清零、Milvus 计数齐全。
3. `mine_qa` 跑一轮 → staging 有 kept/discarded,挖出的 QA 可被换说法问题召回。

### 8.4 过程要求

`dev-notes/ch03.md` 每阶段追记(已开篇);涉及库 API 动手前 Context7 核对;完结交付演示命令 + 测试结果 + dev-notes 路径。

## 9. 拍板记录

- **BGE-M3 走硅基流动 API**:用户原话「这是我硅基流动的API key……你看看是不是从中可以获得 bge-M3 模型」——已实测:批量入参、1024 维、「邮费是多少」vs「运费怎么算」余弦 0.78(低于去重阈值 0.88,换说法不会被误判重复)。
- **挖知识定时任务 = 独立批处理脚本**(推荐项,用户确认)。
- **防幻觉纳入本章**(推荐项,用户确认)。
- **Milvus 形态 = Milvus Lite**:用户把 key 贴进了这题(自述「好像没太理解你的意思,可能答得驴头不对马嘴」),澄清后补充定位——「这一章只跑通『向量检索』这一条路径,让 query_faq 先能查着,更精细的检索优化留到下一章」,随后明确选 Milvus Lite。
- 技术选型本身(BGE-M3、Milvus、MySQL 权威源、dense 单路)全程未动。

## 10. 明确不做

- 关键词召回、混合检索、重排(本章 dense 单路)。
- Milvus Standalone / 集群、embedding 远程 API、知识库增量更新的管理界面。
- 多轮 Agent Loop、用户体系(ch02 边界延续)。
