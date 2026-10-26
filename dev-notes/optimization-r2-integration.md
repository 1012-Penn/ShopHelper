# 第二轮统一集成记录

## 基点与独立交付

起点是已验收的 `446fbb7`。四条源提交分别保留：

- routing `40482fa`；
- claim grounding `6f68073`；
- typed handoff `6334e43`；
- E2E `3d6fd9f`。

四个方向没有互相依赖未提交代码，均先和 `446fbb7` 做单变量对照。统一集成后新增的效果数字不和四组百分比相加。

## 观察到的问题与修复

### 1. 结构化输出在真实 thinking 模型上不可用

E2E 默认路径逐题收到 `Thinking mode does not support this tool_choice`。这是实现级失败，
不是 RAG 或模型回答能力失败。

处理：

- 在 `app/llm.py` 集中增加 `make_structured_extractor`；
- `auto` 对 DeepSeek 模型使用 JSON Schema + Pydantic，其他模型保留原生 function calling；
- 替换生产 graph、证据判断、改写、gap analyzer、flywheel、after-sale、multi-agent Planner；
- 用真实 provider 单调用 smoke 验证，成本 `$0.000309`；
- 再用真实应用 dev 8 题确认技术失败为 0/8。

真实 dev smoke：

- 严格成功 2/8；
- 事实 6/6、条件 2/2、实体 3/3；
- 引用 3 uncertain、2 unsupported、2 tool-only、1 not-required；
- 成本 `$0.017735`，P95 22.448s；
- 这是混合真实 API 的进程内 ASGI 路径，不是公网 SLA。

### 2. adaptive 只有独立 app，未接入真实工厂

新增 `AppAdaptiveRoutingService`，通过 `create_app` 注入：

- `ADAPTIVE_ROUTING_ENABLED=false` 为默认；
- 开后挂载 `POST /api/adaptive-routing/chat`；
- 单 Agent 分支复用 execution store/run/usage 和现有 tool engine，返回最终 JSON 快照；
- multi 分支复用现有 multi-agent runtime，仍是进程内状态；
- dispatcher 单次执行，不双跑。

测试覆盖真实 app factory 的单路持久 run、跨订单多路选择、会话历史落库。

### 3. typed handoff 的审计和缓存边界

源评测的 25/30 = 25/30 结论保留，不重写。集成审查另修：

- 重复等价工具观测的 sequence 会在 ledger 内单调递增，后值 supersede 前值；
- `错误:` 工具结果不进入可复用读缓存；
- Supervisor typed prompt 也剥离 `raw_result`，原文只留在审计记录；
- `MULTI_AGENT_HANDOFF_MODE=raw` 仍是默认；typed 仅显式开关。

这些修复没有重新调用模型，所以不声称 typed 修复后的新质量数字。

### 4. claim grounding 的默认安全

`ClaimGroundingAnswerSemantic` 接入现有 `AnswerVerifier`，但必须同时满足：

- `ANSWER_VERIFICATION_ENABLED=true`；
- `CLAIM_GROUNDING_ENABLED=true`；
- `CLAIM_GROUNDING_LEDGER_PATH` 非空；
- 预算是独立 SQLite ledger。

否则默认行为不变。真实实验表明 supported recall 只有 3/8，且会增加拒答，不能默认启用。

### 5. 其它真实 E2E 输入/工具问题

- 纯退款/售后政策问题不再一律进入固定索要订单流程；有明确订单号时仍走确定性子流程。
- `query_product` 改用确定性商品目录；未知商品返回未找到，不再随机生成价格/库存。
- `app.orders` 与知识库订单事实仍冲突，未在集成中强行统一；必须下一轮以新数据版本重跑。

## TDD 与回归

先看到失败再修：

- DeepSeek structured helper 不存在；
- typed 重复观测未 supersede；
- tool error 被缓存；
- Supervisor typed prompt 泄露 `raw_result`；
- adaptive 真实 app route 404、单路初版 NameError；
- `query_product` 两次结果不同；
- 通用退款政策仍被固定流程截获。

完成后：

- 定向集成：`219 passed`；
- reliability acceptance：通过，32/32 隔离 SQLite run，0 模型调用；
- 全量：`1011 passed, 2 skipped, 1 xfailed`；
- Ruff 修改文件：通过；
- `git diff --check`：通过。

## 预算

各源实验使用各自账本：

- routing：$0.17100027；
- typed：$0.09256836；
- claim：$0.00813324；
- E2E：$0.073185168。

它们不能相加成一次运行的效率。集成会话独立增加 `$0.018044`，未复用旧 pending，未 push。

## 下一轮只值得追的真实失败

优先级最高：

1. **选择性 Agent 路由**：当前规则只在多订单显式比较上收益，写预览/取消全臂失败，且 held-out 已查看。
   下一轮应使用全新 held-out 评“证据依赖复杂度 + 写风险 + 失败升级”，不要继续调本轮阈值。
2. **引用支持的实际语义判断**：当前 28 题兼容路径严格知识引用 supported 为 0，旧 uncertain 很多。
   下一轮应做人工/小模型联合标签，先解决 supported recall，再讨论自动拒答。
3. **统一业务事实源**：订单工具、知识库、评测 fixture 必须一致；这是当前最直接的回答错误来源。

不推荐新一轮重复：

- 再证明 DeepSeek thinking 不支持 tool_choice；已修复；
- 再加入全量反思/多次修订；
- 在没有新证据前把 typed/claim 默认开启；
- 继续使用已查看的 34/30/32/28 题调参并称泛化。
