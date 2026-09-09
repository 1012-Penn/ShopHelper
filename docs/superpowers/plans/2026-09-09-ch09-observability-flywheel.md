# ch09 可观测性与数据飞轮实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development（本会话已选择）；每个任务按测试先行、实现、任务评审、修复闭环执行。

**目标：** 为现有客服系统接入自部署 Langfuse、统一低置信问题飞轮、后台审核/知识回写和可重复的 ch04 评估趋势。

**架构：** 应用启动时创建一次 Langfuse CallbackHandler 并绑定到已编译 Graph；每轮请求只传递动态 metadata 和根 observation。检索证据、问题池、审核队列、评估轮次和 token 汇总分别由聚焦服务负责，FastAPI 后台接口与静态页只做管理展示和操作编排。

**技术栈：** Python 3.10、FastAPI 0.141.1、SQLAlchemy 2.x、LangGraph 1.2.x、LangChain 1.x、Langfuse Python SDK（>=3.8，<5）、MySQL/SQLite、Milvus Lite、pytest/pytest-asyncio。

**Spec：** docs/superpowers/specs/2026-09-09-ch09-observability-flywheel-design.md

## 全局约束

- Langfuse 自部署；缺少 key、SDK异常、flush失败不得阻断聊天。
- LangGraph 置信度闸仍在检索之后、Agent 之前；低置信继续兜底。
- 三入口 source 固定为 retrieval_low_conf、self_check、user_feedback。
- 落池必须尽力保存本轮 Top-N 原文和分数；未检索入口允许 NULL。
- 评估只由脚本运行，支持 --trigger manual 和 --trigger scheduled，不把长任务嵌进客服主进程。
- 后端新增逻辑先按 TDD 完成 RED→GREEN→REFACTOR；前端是 Vibe Coding 例外。
- 所有 SQLAlchemy ORM 必须可由现有 SQLite 测试 fixture 建表；MySQL DDL 按 utf8mb4 和外键顺序执行。
- 不碰主工作树已有未提交改动；所有实现只在 .worktrees/ch09-observability-flywheel。
- 每个任务完成后立即在 dev-notes/ch09.md 追加关键原话、产出、纠偏、翻车/返工。

---

### 任务 1：可观测性核心、配置与 token 用量底座

**文件：**

- 新建：app/observability.py
- 新建：tests/test_observability.py
- 修改：app/config.py、app/main.py、app/graph/builder.py、app/routers/chat.py、app/models.py、app/store.py、db/init.sql、db/ch09.sql、pyproject.toml
- 修改：dev-notes/ch09.md

**接口：**

- ObservabilityService(settings, usage_store)
- ObservabilityService.bind_graph(compiled_graph) -> Runnable
- ObservabilityService.start_request(conversation_id: int, question: str) -> RequestTrace
- RequestTrace.metadata(intent: str | None = None, entry_route: str | None = None) -> dict
- RequestTrace.finish(*, intent: str, answer: str, duration_ms: int) -> None
- RequestTrace.observe_retrieval(query: str, snapshot: list[dict])
- UsageStore.record(...) 与 UsageStore.aggregate_by_intent()

**步骤：**

- [ ] 先写测试：Langfuse 未启用时 Graph 可绑定 no-op；启用配置能只创建一次 handler；request metadata 含会话字段和意图；usage callback 能从两种 token metadata 形态累计。
- [ ] 运行 wsl.exe -d Ubuntu-22.04 -- bash -lc 'cd /root/workplace/ShopHelper/.worktrees/ch09-observability-flywheel && /root/workplace/ShopHelper/.venv/bin/pytest tests/test_observability.py -q'，确认因接口未实现而失败。
- [ ] 添加 Langfuse 配置、依赖和 request_usage ORM/DDL；实现 no-op 安全降级、CallbackHandler 一次性绑定、根 observation 和 token callback。
- [ ] 把已编译 Graph 绑定到 create_app，聊天 invocation 传递 metadata，请求结束写入 usage；不改变现有 SSE sink。
- [ ] 重跑专项测试和既有测试，确认 RED→GREEN；追加 dev-notes 任务记录并提交。

---

### 任务 2：检索快照与正式 evidence confidence 闸

**文件：**

- 新建：tests/test_evidence_confidence.py
- 修改：app/retrieval.py、app/rerank.py、app/config.py、app/graph/state.py、app/graph/knowledge.py、app/graph/builder.py、tests/test_ch05_knowledge.py、tests/test_retrieval.py
- 新建：scripts/calibrate_evidence_confidence.py
- 修改：dev-notes/ch09.md

**接口：**

- evidence_confidence(items: list[RetrievalItem], calibration: EvidenceCalibration) -> EvidenceConfidence
- EvidenceConfidence(top1_relevance: float, effective_count: int, top1_top2_margin: float, score: float, passed: bool, signals: dict)
- RetrievalResult.snapshot(top_k: int) -> list[dict]

**步骤：**

- [ ] 写测试覆盖 Top1/Top2 分差、有效证据数、只有一条证据、空结果和校准阈值；确认初始失败。
- [ ] 从 ch04 评估集跑候选阈值校准脚本，输出 JSON/Markdown 校准产物；配置加载校准结果且有已校准默认值。
- [ ] 在现有检索结果中生成 evidence confidence 和快照，保留当前 low_confidence/reason 兼容字段。
- [ ] 保持 graph 边 retrieve → gate → fallback/agent 不变；gate 失败时把快照放入状态，供后续飞轮调用。
- [ ] 跑专项、ch04/ch05 检索测试，追加 dev-notes 并提交。

---

### 任务 3：统一 FlywheelService、标准化查重和 review_queue 数据层

**文件：**

- 新建：app/flywheel.py
- 新建：tests/test_flywheel.py
- 修改：app/models.py、app/guard.py、app/store.py、db/init.sql、db/ch09.sql、app/config.py
- 修改：dev-notes/ch09.md

**接口：**

- FlywheelService.ingest(*, source, conversation_id, raw_question, reason, retrieved_chunks=None) -> int
- FlywheelService.normalize(raw_question, reason) -> NormalizedQuestion
- FlywheelService.find_duplicate(candidate, rows) -> DuplicateDecision
- FlywheelStore.get_review_queue(status: str | None) -> list[dict]
- FlywheelStore.get_review_detail(review_id: int) -> dict
- FlywheelStore.save_turn_retrieval(conversation_id, question, snapshot)
- FlywheelStore.recover_turn_retrieval(conversation_id, question) -> list[dict] | None

**步骤：**

- [ ] 写测试：三种 source 都先落原话；快照 JSON 原样保存；标准化失败有原话 fallback；重复问题累加次数并回写 matched id；新问题只创建一行；feedback 能回捞最近检索快照；查重失败仍保留原话。
- [ ] 运行专项测试确认失败。
- [ ] 按用户 DDL 建 review_queue，给 low_confidence_questions 加 matched_review_id/retrieved_chunks；新增 turn_retrievals，补 ORM 和 store。
- [ ] 实现标准化和查重模型适配层，使用结构化输出/宽松解析；所有模型失败都降级为可审核队列行。
- [ ] 运行专项和 tests/test_guard.py/tests/test_models.py，追加 dev-notes 并提交。

---

### 任务 4：把三个入口接入飞轮、反馈 API 和审核知识回写

**文件：**

- 新建：app/routers/admin.py
- 新建：app/routers/feedback.py
- 新建：tests/test_ch09_admin_api.py
- 新建：tests/test_ch09_feedback_api.py
- 修改：app/graph/knowledge.py、app/graph/agent.py、app/graph/logging_node.py、app/main.py、app/schemas.py、app/kb.py、app/store.py
- 修改：dev-notes/ch09.md

**接口：**

- POST /api/feedback
- GET /api/admin/review-queue
- GET /api/admin/review-queue/{id}
- POST /api/admin/review-queue/{id}/approve
- POST /api/admin/review-queue/{id}/reject
- POST /api/admin/review-queue/{id}/retry-normalize

**步骤：**

- [ ] 写 API 测试：检索闸、自评拒答、down feedback 都进入同一服务；反馈幂等；详情返回原话+快照；通过写 pending 知识块且失败不改状态；驳回只改状态。
- [ ] 运行专项测试确认失败。
- [ ] 替换 graph/agent 直接 pool.insert 调用为 FlywheelService.ingest；log 节点每轮保存 retrieval snapshot。
- [ ] 新增反馈 schema/路由，校验只接受 down，按会话+原话回捞快照。
- [ ] 扩展 KnowledgeBaseStore：将核准问题/答案写入 knowledge_chunks(pending)，复用 vectorize_pending；审核状态只在写入成功后变为“通过”。
- [ ] 挂载 admin/feedback routers，跑专项与 ch05/ch06/chat API 回归；追加 dev-notes 并提交。

---

### 任务 5：评估脚本、eval_runs 趋势接口和意图 token 汇总

**文件：**

- 新建：app/evaluation.py
- 新建：scripts/eval_pipeline.py
- 新建：tests/test_ch09_evaluation.py
- 修改：app/routers/admin.py、app/models.py、app/store.py、db/init.sql、db/ch09.sql、dev-notes/ch09.md

**接口：**

- run_evaluation(*, triggered_by: Literal["定时", "手动"], dataset_path: Path, deps) -> EvalRunResult
- GET /api/admin/eval-runs
- GET /api/admin/usage/by-intent

**步骤：**

- [ ] 写测试：替身评估集能算 Recall@K/MRR/Faithfulness；成功轮写一行，失败轮不写成功结果；连续执行两轮按时间返回；按意图聚合 token。
- [ ] 运行专项测试确认失败。
- [ ] 抽取/复用 ch04 评估指标函数，实现 CLI 的 --trigger manual|scheduled 和 eval_runs 持久化。
- [ ] 增加趋势/usage API，指标 JSON 保持可扩展。
- [ ] 运行专项和评估相关旧测试，追加 dev-notes 并提交。

---

### 任务 6：后台管理页与聊天 👎 后端接线（Vibe Coding 前端例外）

**文件：**

- 修改：static/index.html
- 新建：tests/test_static_ch09.py（只做脚本/关键字符串检查，不套完整 TDD）
- 修改：README.md、dev-notes/ch09.md

**效果：**

- 聊天气泡 👎 发送 POST /api/feedback，提交成功后锁定按钮并显示状态。
- 增加“数据飞轮后台”入口：待审队列显示标准化问题、出现次数、示例答案、状态。
- 点行展开详情：显示归并的用户原话和当时 Top-N 召回片段/分数。
- 通过弹出核准答案输入；驳回直接操作；处理结果实时刷新。
- 评估趋势和意图 token 汇总使用简单表格/横条/折线 SVG，不引入前端依赖。

**步骤：**

- [ ] 先实现页面结构和交互，以现有配色/事件流为基础，不改已有聊天/工单/引用行为。
- [ ] 用 node --check 对提取出的脚本检查语法，并以 FastAPI HTTP 测试确认 URL/JSON 契约。
- [ ] 追加 dev-notes 并提交；不走 brainstorm、TDD、code review。

---

### 任务 7：系统验收脚本、README 和全链路回归

**文件：**

- 新建：scripts/acceptance_ch09.py
- 新建：tests/test_acceptance_ch09.py
- 修改：README.md、docker-compose.langfuse.yml（仅补缺失配置时）、dev-notes/ch09.md

**验收场景：**

- 启动 Langfuse Compose 后请求有可展开 trace；未启动 Langfuse 时聊天仍可用。
- 库外问题返回兜底、问题池/待审详情有原话和召回快照。
- 审核通过写入 pending/向量后，同义问题能再次答对。
- 聊天点 👎 后进入 user_feedback 并查重入队。
- 评估脚本连续两轮，趋势 API 返回两行。
- usage API 能看到至少两个意图及 token 汇总。

**步骤：**

- [ ] 先写 HTTP/替身验收测试，确认缺少验收脚本/接口而失败。
- [ ] 实现可重复的 acceptance_ch09.py，支持 --scenario flywheel|feedback|eval|usage，不强依赖真实 Langfuse key。
- [ ] 更新 README 的依赖、启动、迁移、演示和 cron 命令；补齐 docker-compose.langfuse.yml 的应用连接说明。
- [ ] 运行专项、完整 pytest、git diff --check，追加 dev-notes 任务完成记录并提交。

---

### 任务 8：代码评审、验收证据和 finish

**文件：**

- 修改：dev-notes/ch09.md
- 产出：reports/ch09-acceptance.md（若 reports 被忽略则只保留本地证据并在 notes 记录）

**步骤：**

- [ ] 生成全分支 review package，执行独立 code review；Critical/Important 必须带回归测试修复，Minor 需修复或在 dev-notes 记档。
- [ ] 运行完整 pytest、关键 acceptance 场景、必要的脚本命令，保存真实输出。
- [ ] 在 dev-notes 追加 code review 结论和 finish 记录，包含用户关键原话、产出路径、拒绝/纠偏和翻车返工。
- [ ] 使用 finishing-a-development-branch skill，先再次验证，再向用户提供合并/PR/保留分支三选一；不自行合并或推送。

