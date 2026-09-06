# ch04 设计文档:RAG 进阶——混合检索 + 重排 + 评估体系

日期:2026-09-06
状态:已对齐(设计整体批准;四问拍板见 §11)

## 1. 目标

检索质量上台阶:dense 单路升级为**混合检索**(Milvus 原生 BM25 + dense,`hybrid_search` 配 RRF 融合)+ **bge-reranker-v2-m3 精排**;回答带可溯源的引用编号;检索不到或证据不足显式拒答并落低置信度问题池;system prompt 列明负面知识;配套**四策略评估体系**(Recall@K / MRR / Faithfulness,按 query 类型分桶报告);聊天页引用可点 + 满意度反馈(纯前端)。

## 2. 技术栈(定死 + 实测校正)

- **Milvus 原生 BM25** + `hybrid_search` + `RRFRanker`,运行形态延续 Milvus Lite。**真库探针已全部通过**(milvus-lite 3.2.1 + pymilvus 3.0.1):BM25 Function 建集合、insert 自动生成 sparse、纯 BM25 检索、hybrid RRF 融合、dense/sparse 两侧 `expr` 过滤、upsert 幂等。不需要 Milvus Standalone(Docker Hub 本环境拉不到镜像,ch03 已实证,本章也不再依赖)。
- 实测校正三条(与文档的差异,实现以真库为准):
  1. 文档里的 `analyzer_params={"type": "chinese"}` 在 Lite 上不存在,报错明示 supported: `standard`/`jieba`——**中文分词用 `{"tokenizer": "jieba"}`**(即内置 chinese analyzer 的 Lite 等价配置),需安装 `jieba==0.42.1`(milvus-lite 的 JiebaAnalyzer 运行时 import 它);
  2. pymilvus 3.0.1 的 `MilvusClient.hybrid_search(collection, reqs, ranker=…, limit, output_fields)` 关键字是 `ranker`(文档另有 `reranker` 写法,不适用);`MilvusClient.search` 的参数名是 `search_params`(不是 ORM 的 `param`);
  3. BM25 打分为**负值**(实测同库 -2.6 ~ -2.3),只按返回序取名次,任何阈值判断不得依赖 BM25 分数方向。
- **重排 bge-reranker-v2-m3 走硅基流动 `POST /v1/rerank`**(用户拍板):现有 EMBEDDING_API_KEY 实测可用,返回 `results:[{index, relevance_score}]` 按相关度降序,区分度实测 0.993/0.204/0.0003。
- MySQL 8 / FastAPI / SQLAlchemy 2.x / LangChain 1.x 沿用;LLM(改写、生成、忠实度裁判)沿用 DeepSeek(现有 OPENAI_* 配置)。
- 不引入 Langfuse(用户库名清单提及但本章需求无观测项,留后续章节)。

## 3. 数据层

### 3.1 MySQL(用户已定稿 DDL,原样进 `db/init.sql`)

- `low_confidence_questions`:低置信度问题池。本章写两个入口——`retrieval_low_conf`(检索无结果/证据低于阈值)与 `self_check`(生成自评不足);`user_feedback` 枚举值保留但本章不写(反馈纯前端采集,数据飞轮入口留给后续)。`conversation_id` 关联来源会话,`raw_question` 存用户原话,`reason` 存判不能原因。
- `faith_cases`:忠实度编造个案台账(评估裁判驱动)。一题一行(`uk_eval_id`);`seen_count` 跨轮累加;已解决/无需解决的题再被判出 → 状态自动退回「未解决」(复发),`resolution` 清空,`resolved_at` 保留作复发标记;`citations` JSON 存该轮喂给模型的 Top-K 证据**全集**快照(`[{n, chunk_id, section_path, question, answer}]`),答案角标 `[n]` 即这份列表的序号。
- ORM 两模型进 `app/models.py`(`BigInteger().with_variant(Integer, "sqlite")` 承载 SQLite 测试);建表顺序:low_confidence_questions 在 conversations 之后(FK),faith_cases 独立;沿用 `SET NAMES utf8mb4`。

### 3.2 Milvus 集合 v2(schema 不兼容,需重建)

集合 `knowledge` 重建,字段:

| 字段 | 类型 | 用途 |
|---|---|---|
| `id` | INT64 主键(auto_id=False) | = MySQL chunk 主键 |
| `vector` | FLOAT_VECTOR dim=1024 | BGE-M3 dense,COSINE |
| `text` | VARCHAR(max_length=8192, enable_analyzer, `{"tokenizer":"jieba"}`) | BM25 源文本(三格拼文本) |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 Function 输出,禁止手工写入 |
| `category` | VARCHAR(255) | 元数据过滤字段 |

- BM25 Function:`Function(name="text_bm25", input_field_names=["text"], output_field_names=["sparse"], function_type=FunctionType.BM25)`,建集合前挂上。
- 索引:dense `FLAT` + COSINE(库小);sparse `SPARSE_INVERTED_INDEX` + BM25。
- upsert 数据带 `{id, vector, text, category}`(sparse 由函数自动生成)。
- **旧集合处置**:`ensure_collection` 检测已有集合缺 BM25 函数/字段不齐 → drop 重建;`build_kb` 全量重跑补数据(数据权威源在 MySQL,~23 块重嵌入成本可忽略)。演示流程里明确这一步。
- VARCHAR max_length 按字节计,8192 字节对 chunk_max_chars=500 的三格拼文本留足余量,实现时以超长块真插入验证。

### 3.3 元数据过滤

- `query_faq` 入参 `category` 可选 → dense/sparse 两侧 `AnnSearchRequest` 都带 `expr: category == "…"`。
- 模型推断的品类词与库内 `category` 值不一定字面一致:过滤检索为空时**回退无过滤重检**(记入返回 JSON 的 `filter_fallback: true`),宁可多召回不空转。

## 4. 检索模块 `app/retrieval.py`(策略模式)

四个策略同一接口 `retrieve(query, category=None) -> list[Retrieved]`(`Retrieved`: chunk_id, score, 策略内名次),依赖(embedder / reranker / milvus store / rewriter)全部构造注入,测试可替换身:

1. **dense**:embed(query)→ `search(anns_field="vector")`,沿用 `RETRIEVAL_SCORE_FLOOR`(0.5,对 dense 相似度);
2. **bm25**:`search(anns_field="sparse", data=[query_text])`(查询给文本不给向量);
3. **hybrid**:dense_req(50)+ sparse_req(50)→ `hybrid_search(ranker=RRFRanker(), limit=50)`,两 req 的 `expr` 同步带过滤;
4. **hybrid_rerank**:hybrid 候选 → reranker 对候选原文重排 → Top-10。rerank API 失败**降级用 RRF 序**并记 warning,不炸主链路。

**Query 理解(只在检索侧)**:`rewrite(query) -> {normalized, synonyms[0..3]}`(LLM structured output,复用 extract model;失败/超时/`query_rewrite_enabled=false` → 原 query 直用)。dense 吃 `normalized`;BM25 吃 `normalized + synonyms` 拼接扩词面。**不做指代消解、不做多轮改写**(用户明令),改写只见当前这一句。

**反漏斗排列**(纯函数,单测):精排名次 1..N → 摆放序为奇数名次依次从头、偶数名次依次从尾(1→首,2→尾,3→次首,4→次尾……)。编号 n = 精排名次(即 [1] 恒为最强证据),摆放与编号解耦,对抗 lost-in-the-middle。

**Reranker 客户端 `app/rerank.py`**:httpx 直连 `/v1/rerank`(仿 embedder:transport 可注入、非 200/格式异常收敛 RuntimeError、`return_documents=false`、返回 `[(候选下标, score)]` 按分数降序)。测试替身 FakeReranker 按词面重合度确定性打分。

## 5. `query_faq` 工具 v3(契约演进,用户放行)

- 入参:`keyword: str`、`category: str | None`(模型从用户话推断,不强制)。
- 流程:rewrite → hybrid_rerank(带过滤,空则回退无过滤)→ Top-10 → 反漏斗排列。
- 出参 JSON:

```json
{
  "items": [
    {"n": 1, "chunk_id": 12, "question": "…", "answer": "…", "category": "…", "section_path": "退货政策.md > 邮费与运费"}
  ],
  "low_confidence": false,
  "reason": "",
  "filter_fallback": false
}
```

- **低置信判定**:`items` 为空,或精排 Top-1 分数 < `RERANK_SCORE_FLOOR` → `low_confidence=true`、`reason` 写明(如「知识库无相关内容」「证据置信度低」)。阈值默认 0.30,实现期拿评估集 rerank 分数分布校准定稿。**【校准定稿 2026-09-06】0.30 会误杀合法口语题(真模型分布:合法题 Top-1 ∈ [0.039, 0.99],纯噪声 junk = 0.0,词面重叠 junk ≈ 0.17)→ 定稿 `0.03`;词面重叠 junk 交给生成自评拒答兜底(分层防御)。**
- 任何内部异常仍收敛为 `{"items": [], "low_confidence": true, "reason": …}`(契约:不抛错,但本章起空结果明确标低置信,不再静默)。
- `build_tools` 注入面扩展:rewriter / reranker 可注入,缺省按 Settings 构造真实现。

## 6. 生成质量控制

### 6.1 引用编号与溯源

- system prompt 引用规范:引用知识库作答时必须带 `[n]` 角标,n 只能取工具返回 items 里给定的编号,禁止编造编号或引用未提供的编号。
- SSE 新帧 `citations`(在工具执行后、最终答案流之前发):`{"type":"citations","items":[{n, chunk_id, section_path, question, answer}]}`,前端据此渲染可点角标与来源面板。不新增落库(工具结果全文已在 messages 表 tool 消息里)。

### 6.2 拒答、自评与低置信度池

- system prompt 自评指令:回答前自评召回证据是否足以支撑答案;不足时以固定标记 **`【无法回答】`**(常量 `REFUSAL_MARKER`,存 `app/guard.py`)开头,正文说明不知道并建议转人工。
- 落池收拢在 `app/guard.py`(`LowConfidencePool.insert(source, conversation_id, question, reason)`),由 chat 路由调用:
  - 工具结果 JSON `low_confidence=true` → `source='retrieval_low_conf'`,raw_question=本轮用户原话,reason 取工具返回;
  - 最终答案文本以 `【无法回答】` 开头 → `source='self_check'`,reason=「模型自评证据不足」+ 答案前 200 字。
- 不去重(台账语义,同一问题反复出现本身就是信号);`user_feedback` 本章不写。

### 6.3 负面知识清单(system prompt,断言测试锚定)

不承诺退款到账的具体时间点与银行处理时长;不承诺具体送达日期(库内时效区间仅可作「一般」口径引用);不承诺任何赔付、补偿、额外优惠金额;不编造价格、库存、型号参数。库内没有的事实一律按不知道处理。

## 7. 评估体系

### 7.1 评估集 `tests/eval/ch04_eval_set.jsonl`(32 题,数据任务)

- 四桶 × 8:`A_policy` 政策条款理解 / `B_model` 带具体型号关键词(BM25 靶点)/ `C_colloquial` 口语模糊(改写靶点)/ `E_multi` 多约束组合。D 缺号:无 D 桶(用户确认)。
- 字段:`{id, bucket, query, expect_doc: str|[str], expect_section?: str, expect_keywords: [str], notes}`。ground truth 标注到「文档 + 章节 + 关键词」粒度,足以唯一判定单个 chunk 正确与否(不硬标 chunk_id,重建后 id 会漂移);MRR 用首个正确 chunk 的倒数排名。
- 知识文档配套扩充(数据任务):`knowledge/商品FAQ.md` 增店铺自有型号 SKU(如耳机 SH-E300 / SH-E120、键盘 SH-K870 / SH-K104 等,含价格与参数,风格与现有文档一致),B 桶问题围绕这些型号;`退货政策.md` / `售后手册.md` 保持,是 A 桶靶子。

### 7.2 检索评估(`scripts/eval_retrieval.py` 升级)

- `--strategy dense|bm25|hybrid|hybrid_rerank|all`,指标 **Recall@3/5/10 + MRR@10**,按桶分列 + 总表;
- 报告写 `reports/ch04-retrieval-report.md`(`reports/` 进 .gitignore),关键数字进 dev-notes;
- ch03 的 `tests/eval/retrieval_samples.jsonl` 保留作回归。

### 7.3 生成评估(`scripts/eval_faithfulness.py`,新)

- hybrid_rerank 策略为 32 题生成带引用答案(单轮简化链路:system prompt(引用规范+自评)+ 检索 items 拼装 + user query),**citations 证据全集全程记录**;
- LLM 裁判(DeepSeek,`judge_model` 落库)输入 = 问题 + 证据全集 + 生成答案,structured output 输出 `{fabricated: bool, reason, fabricated_claims[]}`;判据:事实性断言是否都有证据支撑、`[n]` 映射是否属实;
- **Faithfulness** = 非 fabricated 占比,按桶报告;判出的个案按 §3.1 语义 upsert 进 `faith_cases`(INSERT 或 seen_count+1/复发退回),`strategy='hybrid_rerank'`,`citations` 存证据全集快照;
- 前置:MySQL + Milvus v2 库已建、硅基流动嵌入/重排可用、DeepSeek key 有效。

## 8. 前端(`static/index.html`,Vibe Coding,例外流程)

- **引用可点**:citations 帧暂存于气泡 state;答案流结束后把正文 `[n]` 渲染成可点角标;点击在气泡下方展开来源卡(section_path + chunk 原文 question/answer),再点收起;
- **满意度反馈**:每个机器人答案内容区下方左侧挂 👍 / 👎;点击点亮所选、显示「已反馈」、两个按钮一次性锁死;纯前端内存采集,不调任何接口;
- 流式期间角标按原样文本渲染,结束后统一重渲染(避免流中替换打断)。

## 9. 配置新增(`app/config.py`)

`rerank_api_base`(默认同嵌入 base)/ `rerank_api_key`(默认空 → 回落 `embedding_api_key`)/ `rerank_model=BAAI/bge-reranker-v2-m3` / `rerank_top_n=10` / `rerank_score_floor=0.30` / `hybrid_candidates=50` / `retrieval_final_top_k=10` / `query_rewrite_enabled=true`。`.env.example` 同步。

## 10. 测试与验收

### 10.1 TDD 覆盖(可单测逻辑)

- 反漏斗排列纯函数;检索四策略(FakeEmbedding/FakeReranker + 真 milvus-lite tmp 库,含过滤与回退);Milvus v2 封装(BM25 upsert 幂等、expr 过滤、旧 schema 检测重建);reranker 客户端(MockTransport,含失败降级路径);rewrite(替身 + 失败兜底);query_faq v3 契约(形状、低置信标志、异常收敛);落池 guard(两入口);prompt 断言(引用规范/自评标记/负面知识);citations SSE 帧;faith_cases upsert 语义(新增/累加/复发退回,SQLite)。
- 数据类任务(评估集 32 题、知识文档型号扩充)按工作约定以「断言 + 评估集真跑」验证替代 TDD。

### 10.2 评估集真跑(替代纯数据类任务的 TDD 步)

四策略全量真跑出数字;B 桶在 bm25 单路下必须命中(验收 2);hybrid_rerank 为四策略最优或并列最优,否则复盘检索链路。

### 10.3 人工验收(真库真模型,对应验收标准)

1. 四策略对比报告有数字(Recall@K / MRR 分桶表);
2. 问带具体型号(如「SH-E300 降噪怎么样」),BM25 路命中型号 chunk(单路演示 + hybrid 提名);
3. 聊天页答案角标可点,点开见来源原文与章节路径;
4. 问知识库没有的内容(如「量子力学怎么退货」):得到 `【无法回答】` 开头的明确拒答,且 `SELECT * FROM low_confidence_questions` 出现该题(自评入口);配合「检索低置信」场景验证 retrieval_low_conf 入口。

### 10.4 过程要求

`dev-notes/ch04.md` 每阶段追记(brainstorm 定稿即开篇);动手前 Context7 核对(Milvus 已核,BM25/jieba/rerank 差异以真库探针实证回填);完结交付演示命令 + 测试结果 + dev-notes 路径。

## 11. 拍板记录

- **Reranker 托管 = 硅基流动 API**:推荐项直接确认(现有 key 实测 `/v1/rerank` 可用)。
- **评估集分桶**:确认解读——A=政策条款理解、B=带型号关键词、C=口语模糊、E=多约束组合;「D 缺号视为无 D 桶」;每桶 8 题共 32 题(推荐项确认)。
- **落池机制 = 拒答标记检测**(推荐项确认):`【无法回答】` 固定标记 + 路由层检测落池,不做二次结构化裁决调用。
- **query_faq 契约演进放行**(推荐项确认):出参加 n/chunk_id/section_path、入参加可选 category、SSE 加 citations 帧。
- **设计整体批准**(含默认边界:不引入 Langfuse;query 改写走 LLM 且可开关)。
- 技术选型全程未动:Milvus 原生 BM25 + hybrid_search RRF + bge-reranker-v2-m3。

## 12. 明确不做

- 指代消解、多轮改写(用户明令)。
- Milvus Standalone / 集群(Lite 探针实证够用)。
- `user_feedback` 入口落库与数据飞轮(枚举与前端入口先行)。
- Langfuse / tracing(本章无观测需求)。
- 多轮 Agent Loop、用户体系(ch02 边界延续)。

## 13. 风险与已知边界

- milvus-lite 库文件单进程锁:服务运行时勿并行跑建库/评估(ch03 已知边界延续)。
- BM25 负分:排序只按返回序;阈值一律架在 dense 相似度或 rerank 分数上。
- rerank / rewrite 外调失败均有降级路径(降级 RRF 序 / 原 query 直检),但降级轮的检索质量会掉,靠评估报告发现。
- 集合 v2 重建后 dense 向量需全量重嵌入(旧向量不迁移);规模小,一次性成本。
- 忠实度裁判本身是 LLM,存在误判;靠 faith_cases 台账的人工处置状态兜底(误判个案走「无需解决」)。
