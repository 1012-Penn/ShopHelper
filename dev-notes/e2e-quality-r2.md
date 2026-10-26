# E2E Quality R2 Dev Notes

## 观察

历史 RAG 报告中的事实正确与严格正确差异很大，且集成报告已经指出旧 prompt 带行为提示。本轮先不
重跑模型，直接对 v3.1 的 450 条只读轨迹做证据映射和三层归类。

## 定位

1. `scripts/eval_prompts.py` 的回答 prompt 直接写 `{behavior}`，历史 contract 的 prompt SHA 与之匹配。
2. `ResponseCache.response_key` 把 `expected\0query\0evidence_block` 放入 key；三模式在证据相同的情况下
   复用回答。dev 有 43/60 三模式完全同答，另有 13 题只在后两模式同答。
3. dev 基线 60 题中，12 正确、17 确定模型/行为错误、26 citation uncertain、5 unsupported。
   因此把全部低严格分归为模型能力失败不成立。
4. 证据 ID 都能映射到 snapshot/metadata，未发现历史 evidence-ID 断链。
5. 旧 trace 不保存完整原 prompt 和 finish reason，无法证明输出截断；本报告没有伪造缺失字段。

## 新基线

新题先冻结 8 dev + 20 held-out，按来源模板隔离。真实 `create_app()` + `POST /api/chat` 首次完全运行
在 Milvus 目录后缀/MCP 隔离处失败；失败产物和已发生费用保留。只修复这两处评测接线后进行了一次
完整 held-out 运行。

默认生产路径随后暴露真实协议问题：`deepseek-flash` thinking 模式拒绝
`with_structured_output(method="function_calling")` 所需的 `tool_choice`。100 条结构化请求 400，
28/28 题按严格口径技术失败。保留为生产基线，不把回答器的部分输出冒充成功。

第二臂只替换结构化输出传输为纯文本 JSON + Pydantic 校验。真实 app/graph/检索/工具保持不变。
结果是技术失败 0/28，严格成功 4/28，事实 16/21、条件 1/6、实体 11/11、工具 3/3、路由 3/3。
20 个知识回答的引用分别为 9 uncertain、5 unsupported、6 missing，0 supported。

## 取舍

- 没有为追求显眼提升加入 reflection、改写 prompt 或调阈值。
- 没有把 uncertain 算成正确，也没有把它叫幻觉。
- 没有在评测里修正订单工具、随机商品工具或固定退款路由；这些是生产行为，保留并给集成说明。
- 评分器事后修正了两个明确错误：完整回答末尾追问误判 clarify、tool-only 仍加 citation_missing。
  所有组统一重评，原响应与原标签均保留；因此 v2 不能当作独立 held-out 验证。

## 成本

整个会话共花费 `$0.073185168`，2 USD 上限内仍有 `$1.926814832`。共 373 条账本记录、0 pending。
兼容臂中 generation `$0.033052272`、证据判断 `$0.020763024`、低置信飞轮 `$0.0025092`。
embedding/rerank 使用历史精确缓存，费用为 0；P95 是混合缓存时延，不是冷启动。

## 风险

- 实体/事实的金标来自模拟语料，订单工具 fixture 与语料存在真实不一致。
- 当前确定性引用器低估同义支持，strict success 只能视为下限。
- 工具调用仍可能引入随机上游事实，例如 `query_product` 的随机价格。
- ASGI 进程内调用不能替代公网 HTTP、并发数据库和人工语义复核。
