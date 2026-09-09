# ch09 可观测性与数据飞轮设计

## 目标

给客服系统接入自部署 Langfuse，让任意一轮请求能展开查看 Graph 节点、模型调用、工具调用、检索证据、token 和耗时；同时把低置信问题变成可审核、可去重、可回写知识库的改进闭环，并让 ch04 评估集能按 cron/手动命令持续跑出趋势。

## 已确认的约束

- Langfuse 自部署，链路数据留在本机/自家服务器；仓库已有 `docker-compose.langfuse.yml`，补齐应用侧 SDK 接线。
- LangGraph 的置信度闸仍位于知识检索之后、Agent 之前；拦截时继续返回兜底话术。
- 三个问题池入口固定为：检索证据低置信、生成阶段自评知识不足、聊天页用户点 👎。
- `low_confidence_questions.source` 记录入池入口；落池时保存本轮检索 Top-N 原文和分数，未走检索的入口允许为空。
- 评估任务采用脚本，支持手动运行和外部 cron，不在客服主进程内运行长任务。
- 后台页是本地演示系统，不新增用户认证；前端按 Vibe Coding 直接实现。
- 具体库/API 在当前环境没有 Context7 MCP 可调用，因此以 Langfuse、LangGraph、LangChain 官方文档核对为 fallback。已核对：`langfuse.langchain.CallbackHandler`、LangGraph invocation `config.callbacks/metadata`、Langfuse `start_as_current_observation` 与 `propagate_attributes`。

## 总体架构

```text
                ┌───────────────┐
HTTP /api/chat ─┤ LangGraph 图   ├─ resolve → intent → route
                └──────┬────────┘                 │
                       │                           ├─知识检索→证据闸→Agent
                       │                           ├─业务 Agent→工具引擎
                       │                           ├─投诉/闲聊
                       │                           └─退款售后子流程
                       │
          已编译 Graph 默认 Langfuse callback
                       │
        ┌──────────────┴──────────────┐
        │                             │
  Langfuse 自部署 trace          本地 request_usage
  节点/LLM/工具/检索/耗时          按 intent 汇总 token

低置信三入口 ─→ LowConfidencePool ─→ FlywheelService
                  │                    ├─标准化问题+示例答案
                  │                    ├─LLM 语义查重/累加次数
                  │                    └─review_queue
                  │
                  └─retrieved_chunks 快照 ─→ 后台详情页

review_queue 通过 ─→ ch03 KnowledgeBaseStore(pending)
                  └─向量化后再次可检索

ch04 eval set ─→ scripts/eval_pipeline.py ─→ eval_runs ─→ 趋势 API/后台图表
```

## 可观测性设计

### Langfuse 接线

新增 `app/observability.py`，提供：

- `ObservabilityService(settings, usage_store)`：应用启动时最多创建一次 Langfuse client、`CallbackHandler` 和 token 统计 callback；缺少开关或 key 时使用 no-op，不影响客服功能。
- `bind_graph(compiled_graph)`：在 `StateGraph.compile(...)` 得到的 runnable 上绑定一次默认 callbacks。LangGraph 运行时仍通过 invocation `config` 传入 `thread_id`、SSE sink 和动态 metadata；不在每个节点里重复创建 handler。
- `start_request(conversation_id, question)`：每轮建立一个根 observation，传播 `session_id/conversation_id/source`，让 Graph callback 产生的节点、模型和工具观察挂到同一棵树上。
- `finish_request(...)`：用最终意图补写根 observation metadata，并把本轮累计 token/duration 记录到 `request_usage`；Langfuse flush 失败只记录 warning。
- `observe_retrieval(...)`：为检索调用建立 `retriever` observation，input 记录查询，output 记录 Top-N 的 chunk id、原文、section path、dense/BM25/精排分数；内容按统一截断策略控制体积。

Langfuse 配置：

```text
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://127.0.0.1:3000
LANGFUSE_RELEASE=shophelper-ch09
LANGFUSE_TRACING_ENVIRONMENT=development
LANGFUSE_RETRIEVAL_TOP_K=3
```

意图识别完成后，最终 trace metadata 至少包含：`intent`、`intent_confidence`、`conversation_id`、`entry_route`、`low_confidence_source`（若有）。metadata 值使用短字符串，长 prompt/召回原文放在 observation input/output，不塞进 metadata。

### Token 成本统计

新增 `request_usage` 表，每个请求一行：会话、Langfuse trace id、意图、输入/输出/总 token、耗时、创建时间。`UsageCallback` 从 `LLMResult.llm_output.token_usage` 和消息 usage metadata 兼容提取 token；无法从替身模型拿到用量时记 0，不阻塞回答。

新增接口 `GET /api/admin/usage/by-intent`，返回每个意图的请求数、总 token、平均 token 和占比；后台页用表格/横条显示，Langfuse UI 仍可按 trace metadata 过滤查看明细。

### 降级原则

- Langfuse 服务不可用、SDK 未安装、flush 超时都不能影响聊天回答、问题入池或知识入库。
- trace 记录失败要有日志，但不能把异常抛出 Graph 节点。
- prompt、工具参数、召回原文按现有本地演示系统口径记录，不额外脱敏；生产部署由操作者通过 Langfuse 网络边界和 key 管理负责。

## 数据飞轮设计

### 数据表

`review_queue` 和 `eval_runs` 按用户提供 DDL 建表，保持 utf8mb4、中文 ENUM、MySQL 外键顺序。`low_confidence_questions` 增加：

- `matched_review_id BIGINT UNSIGNED NULL`，索引 + 外键，命中后指向 `review_queue.id`，删除审阅项时 `SET NULL`。
- `retrieved_chunks JSON NULL`，存落池时 Top-N 快照；形如 `[{"rank":1,"chunk_id":12,"text":"...","section_path":"...","score":0.91,"rerank_score":0.87}]`。

为了支持反馈入口事后回捞，新增 `turn_retrievals`：会话、当轮用户问题、检索快照、创建时间；每次聊天轮在 log 节点保存，反馈按会话 + 原话取最近一轮，找不到则返回空快照。

为了支持本地按意图统计，新增 `request_usage`：会话、trace id、意图、三种 token、耗时、创建时间，并为意图和创建时间建索引。

同时更新 SQLAlchemy ORM、SQLite 测试建表、`db/init.sql` 和可独立执行的 `db/ch09.sql`。迁移脚本顺序为先 `review_queue`，再 ALTER `low_confidence_questions`，避免外键引用不存在。

### 统一入池接口

```python
await flywheel.ingest(
    source="retrieval_low_conf" | "self_check" | "user_feedback",
    conversation_id=session_id,
    raw_question=question,
    reason=reason,
    retrieved_chunks=chunks_or_none,
)
```

入口行为：

1. 先写 `low_confidence_questions`，保证模型标准化/查重失败时原始问题不丢。
2. 将本轮 Top-N 召回快照写入 `retrieved_chunks`；反馈入口从 `turn_retrievals` 尽力回捞，业务数据/闲聊等无检索轮为空。
3. 标准化模型输出 FAQ 式问题和一条示例答案；解析失败使用截短原话作为标准化问题，示例答案为空并保留待审记录。
4. 将候选与现有“待审”及“通过”队列交给查重模型判断语义是否相同。命中则锁定并累加 `occurrence_count`，把本次原话的 `matched_review_id` 写回；未命中才新建一行。
5. 查重/模型失败不删除原话；队列中保证至少有一行可人工处理，后续可重跑标准化。

### 正式 evidence confidence

`RetrievalService` 在已有 hybrid + rerank 结果上提供 `evidence_confidence` 和信号明细：

- `top1_relevance`：精排 Top1 相关性分。
- `effective_evidence_count`：超过有效证据分数线的证据条数。
- `top1_top2_margin`：Top1 与 Top2 精排分差；只有一个结果时按校准约定处理。

阈值来自 ch04 评估集校准脚本：遍历候选阈值，按“拦下库外/低证据题、尽量不误拦金标准题”的目标选出阈值并输出校准报告；配置项只加载校准产物/默认已校准值，不在代码中拍脑袋写多个独立阈值。闸位置不变：检索完成后进入 gate，失败返回原有兜底并调用 `flywheel.ingest("retrieval_low_conf", ...)`。

### 生成自评与用户反馈

- Agent 从 `query_faq` 工具结果收到 `low_confidence` 或自评拒答时，沿用 ch04 的 self-check 规则，但把入池改为统一 `FlywheelService`，确保标准化/查重/召回快照一致。
- 后端新增 `POST /api/feedback`，请求包含 `session_id/question/answer/value`；只接受 `value="down"`，幂等防重复。用户点 👎 后前端调用该接口，接口从 `turn_retrievals` 回捞快照并以 `user_feedback` 入池。

## 审核与知识回写

新增管理接口：

- `GET /api/admin/review-queue?status=待审`：按出现次数降序列出标准问题、次数、示例答案、状态。
- `GET /api/admin/review-queue/{id}`：返回队列行、归并的全部原话和每条原话的召回快照。
- `POST /api/admin/review-queue/{id}/approve`：输入核准答案，调用 `KnowledgeBaseStore` 写入 `knowledge_chunks(pending)`，再复用现有向量化服务完成可检索落库；只有成功后状态才改为“通过”。
- `POST /api/admin/review-queue/{id}/reject`：状态改为“驳回”，不写知识库。
- `GET /api/admin/eval-runs`：返回按时间升序的评估轮次和指标 JSON。

后台页新增导航区：待审队列表格、详情抽屉/展开行、召回证据卡片、通过/驳回操作、评估指标趋势区、按意图 token 统计区。通过后的知识写入沿用 `KnowledgeBaseStore` + `vectorize_pending`，为同步验收提供可重复流程；若向量服务不可用，保留 pending 并将审核操作返回失败，不谎报通过。

## 评估流水线

新增 `app/evaluation.py` 封装 ch04 已有 retrieval/faithfulness 计算，新增 `scripts/eval_pipeline.py`：

```bash
python scripts/eval_pipeline.py --trigger manual
python scripts/eval_pipeline.py --trigger scheduled
```

每次脚本运行固定读取 `tests/eval/ch04_eval_set.jsonl`，计算 Recall@K、MRR、Faithfulness，向 `eval_runs` 写入触发方式、样本数、指标 JSON、时间。命令重复运行两次即可产生两行趋势；脚本失败不写“成功轮次”。后台趋势接口按 `created_at` 返回，前端对每个指标画简单折线/变化箭头。

## 测试与验收映射

后端用 TDD 增加：

- evidence confidence 信号计算、校准阈值和 gate 分支；
- 三入口落池、快照回捞、标准化 fallback、查重合并/次数累加；
- review queue 详情、通过写入知识块、驳回、向量化失败不改状态；
- feedback API 幂等；
- Langfuse disabled/no-op、metadata、token 统计 callback；
- eval_runs 两轮落表与趋势接口；
- SQLite ORM/DDL 字段和外键关系。

前端不套 brainstorm/TDD/code review，直接按验收效果实现；用 `node --check static/index.html`（抽取脚本时）和 FastAPI HTTP 测试验证接口。

验收顺序：

1. 启动 Langfuse Compose，问任意问题，在 Langfuse UI 展开完整 trace。
2. 问知识库外问题，得到兜底，审核页看到原话和召回片段。
3. 审核通过后跑向量化，再问同义问题能命中。
4. 在聊天页点 👎，问题进入池并经标准化查重进入待审队列。
5. 运行评估脚本两轮，后台看到 `eval_runs` 趋势。
6. 查看按意图 token 汇总和 Langfuse metadata。

## 非目标与已知边界

- 不实现按主题分类微调模型。
- 不把评估调度器嵌进客服主进程；cron 负责周期触发。
- 不新增登录/权限体系。
- Langfuse 本地 UI 账号、key 和 Compose 密钥仍是开发示例，生产必须替换。
- Langfuse traces 是可观测副本；客服主流程不依赖其可用性。


