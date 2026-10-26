# End-to-End RAG Quality Baseline R2

日期：2026-10-27
分支：`codex/opt-e2e-quality-r2-20261027`
基线：`446fbb7f9972875b7649555cb9b4b19e402cea29`

## 结论

1. 旧 RAG 数字不能直接解释为线上质量。历史回答 prompt 包含 `expected_behavior`，三个模式的 60/90 题中
   分别有 43/67 题三模式回答逐字相同，dev 有 56 题至少两模式同答。旧 10 题中“事实 1/9”和
   “严格 0/9”是回答事实层与引用判尺的共同结果，不是三个独立模型复现。
2. 默认生产结构化输出与当前 `deepseek-flash` thinking 模式不兼容：真实调用收到
   `Thinking mode does not support this tool_choice`，28/28 题被记为技术失败。最小 JSON 文本兼容臂将
   技术失败降到 0/28，证明这是接线/协议兼容问题，不是 RAG 能力收益。
3. 兼容臂的严格任务成功为 dev 2/8、held-out 2/20。事实层 16/21，实体层 11/11，条件层 1/6；
   知识回答的 20 题没有任何一例达到确定性 supported 引用。9 题只是语义 `uncertain`，进入人工复核，
   不能算确定正确，也不能算幻觉。
4. 明确发现三个产品问题：退款/售后意图把 7 个纯政策题提前导向索要订单；`query_product` 能返回随机价格；
   `app.orders` 与《订单与支付》中的订单金额/状态/商品不一致。

## 实际测量路径

### 默认生产基线

真实 `create_app()` 与 `POST /api/chat`，使用生产 compiled graph、检索、证据判断和工具。
隔离输入包括：

- 校验后复制的 Milvus snapshot，树 SHA
  `b6b1fa237d03d2c0f8ddcfe8e003f13c4cb584f0764ac04c2d57ae834bbdca73`；
- 复制到私有 SQLite 的 CH16 metadata，不写共享 MySQL；
- 复制 embedding/rerank 精确缓存；
- `app.orders.py:ORDER_CATALOG` 的确定性订单 fixture；
- `deepseek-flash`，生产温度，`hybrid_rerank`，答案复核和 gap trigger 均为默认关闭。

这是应用路由的进程内 ASGI 调用，不是部署后的公网 HTTP 全链路；真实模型、embedding、rerank
调用是外部 API。所谓 P95 因而不包含浏览器、公网 TLS 或负载均衡。

### 诊断修复臂

除结构化输出传输外，app factory、graph、prompt、检索、工具、题集和评分与默认路径相同。结构化调用
从不受支持的 `tool_choice` 改为“带 JSON Schema 的纯文本 JSON + Pydantic 校验”。该臂是供验收会话
应用的最小接线候选，不是默认生产开关，也不代表 RAG 策略升级。

首次运行在进入答案测量前失败：私有 Milvus 目录名没有 `.db` 后缀，且空 MCP URL 仍被客户端发现。
失败尝试保留在
[`attempt-01`](../../tests/eval/e2e_quality_r2/results/baseline-v1/attempt-01/infrastructure-failure.json)。
第 2 次只修复上述评测接线，题集、策略、评分和 held-out 均未改动。

## 历史低分诊断

零调用重评使用只读
`tests/eval/rag_quality/v3_1/runs/**/rag/**/rescore.jsonl`，共诊断 6 个 split/mode 组、450 条轨迹。
逐题证据 ID、原文、原回答和归类在
[`history-diagnosis.jsonl`](../../tests/eval/e2e_quality_r2/history-v1/history-diagnosis.jsonl)。

### 失败层

| split / mode | 正确 | 模型/行为错误 | 引用 uncertain | 引用 unsupported |
|---|---:|---:|---:|---:|
| dev / baseline | 12/60 | 17 | 26 | 5 |
| dev / decompose_once | 12/60 | 16 | 26 | 6 |
| dev / controlled_loop | 12/60 | 16 | 26 | 6 |
| frozen-v1 / baseline | 17/90 | 28 | 33 | 12 |
| frozen-v1 / decompose_once | 20/90 | 24 | 33 | 13 |
| frozen-v1 / controlled_loop | 20/90 | 24 | 33 | 13 |

这里的“基线”是旧回答器，不是本轮生产路径。引用 `uncertain` 只表示当前确定性尺子不能确认支持，
不自动判为编造；“模型/行为错误”仅在行为、金标事实或必要条件出现确定失败时计入。

同一只读重评里，dev 基线的事实正确为 29/41 determinate，严格正确仅 1/18；frozen-v1 基线的事实
正确为 43/64，严格正确 0/33。两者之间的差异主要来自引用不支撑或不确定，因此旧报告中的
“43/64”是事实层，而不是线上无行为提示的任务成功率。

### 历史 prompt 与缓存

- 回答 prompt SHA：`a98ec0e6894bc148a08bdf933bf3ea8efc7583850dd382bb5578ac7fa5aaf861`。
- prompt 直接包含 `{behavior}`，并逐一列出 `answer / clarify / refuse` 等预期行为。
- 缓存 key 同时包含模型、prompt SHA、温度、`expected\0query\0evidence_block` 和证据 ID。
- 历史 trace 没保存完整原始 prompt 或 LLM finish reason；报告保存的是模板、SHA、证据映射和原回答。
  不能补造当时未保存的推理文本。
- 历史 `llm_calls=0` 是精确缓存重放，不是模型未跑。

### 证据、截断和评分

- 所有被检索到的历史 chunk ID 都能从 snapshot/metadata 找到原文，未发现 ID 缺失；问题不在
  “编号找不到原文”。
- 历史记录没有明确 finish-reason；无法证明输出截断。当前新运行未出现 `工具结果已截断` 标记，
  但流式 chunk 同样没有暴露可审计的 provider finish-reason，所以截断状态仍应按“未观测”处理。
- 旧 10 题验证子集在复核前是事实正确 1/9、严格正确 0/9；复核后事实正确 0/9。
- 旧分类器把完整回答末尾的“还要了解型号吗”误判为 clarify；本轮已在新评分器中修正并统一重评，
  旧结果和旧评分文件未覆盖。

## 新题集与输入边界

新题集有 8 dev + 20 held-out，按业务事实/来源模板分组，dev/held-out 组不重叠：

- case manifest SHA：`41e17bc316b88f2cbeb74ae421118ee2edaa5bd57d42014e4f9e449cc95c0e62`；
- dev 文件 SHA：`06ad620f92803b4adb9231288d93ef98f33185162f4a1cbec65fcb9c44418029`；
- held-out 文件 SHA：`0e133a917088fb5be585eb37f031cf507b6875535926c0b8b16097380f542919`；
- 声明的订单 fixture SHA：`0911fe37b40a9b0893afd1ac4684f45de328d67f60fe2c9aa4da0748259b7ef0`。

兼容臂所有 168 条记录的模型调用都经过输入投影检查：没有 `expected_behavior`、gold fact ID、
评分字段或参考答案字段进入模型输入。完整 prompt/响应、tool schema、工具参数/返回和证据映射都在逐题
`cases.jsonl` 中。

冻结后 held-out 只执行一次。之后发现两个评分器缺陷：完整回答的末尾追问被误判为 clarify，以及
tool-only 题仍加 `citation_missing`。因此 `e2e-quality-r2-score-v2` 是对同批已保存响应的统一事后
重评，不是第二次 held-out 模型运行，也不能再称为独立验证。v1 逐题响应和标签完整保留。

## 新运行结果

### 默认生产基线

| split | 严格成功 | 技术失败 | generation | judge | 输入/输出 token | P50/P95 |
|---|---:|---:|---:|---:|---:|---:|
| dev | 0/8 | 8/8 | $0.006977544 | $0 | 57,161 / 1,208 | 2.690s / 13.693s |
| held-out | 0/20 | 20/20 | $0.009768528 | $0 | 89,726 / 1,602 | 2.926s / 6.383s |

原因不是答案全部不可用，而是每次 resolve/rewrite/证据检查都触发
`tool_choice` 400。默认路径在本轮严格口径下不可用于质量排名；该 0/28 是技术接线结论。

### JSON 兼容修复臂

| split | 严格成功 | 待人工复核 | 技术失败 | 事实 | 条件 | 实体 | 工具 | 路由 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | 2/8 | 2 | 0 | 5/6 | 1/2 | 3/3 | 1/1 | 1/1 |
| held-out | 2/20 | 7 | 0 | 11/15 | 0/4 | 8/8 | 2/2 | 2/2 |
| 合并 | 4/28 | 9 | 0 | 16/21 | 1/6 | 11/11 | 3/3 | 3/3 |

引用三态：

| split | supported | uncertain | unsupported | missing | not_required | tool_only |
|---|---:|---:|---:|---:|---:|---:|
| dev | 0 | 2 | 2 | 1 | 2 | 1 |
| held-out | 0 | 7 | 3 | 5 | 3 | 2 |

因此 4 个成功全部来自无知识引用的工具/拒答题；20 个需要确定性知识引用的回答没有一例达到
strict supported。答案事实层明显好于严格成功，但评测没有把 uncertain 包装成正确。

行为分布：

- `answer -> answer` 18；
- `answer_with_conditions -> answer` 5；
- `refuse -> refuse` 2；
- `clarify -> refuse` 2；
- `clarify -> answer` 1。

实际 clarify 为 0。两题应澄清却拒绝，一题应澄清却空回答；这不是 JSON 兼容修复能解决的问题。

### 主要错例

| case | 结果 | 根因 |
|---|---|---|
| `DEV-DIRECT-E300` | 事实正确，引用 unsupported | 回答附带的质保/在售声明绑定到不完整支持；不是价格错 |
| `DEV-POLICY-V2` | 空回答 | “退款退货”固定子流程先要订单，没有回答通用 v2 规则 |
| `DEV-CLARIFY-SKU` | refuse | 低证据闸走通用拒答，未问商品型号 |
| `HELD-CUSTOM-RETURN` | 空回答 | 同固定售后流程 |
| `HELD-FRESH-SECOND-DAY` | 空回答 | 同固定售后流程 |
| `HELD-VIRTUAL-ACTIVATED` | 空回答 | 同固定售后流程 |
| `HELD-REPAIR-QUOTE` | 空回答 | 同固定售后流程 |
| `HELD-QUALITY-SHIPPING` | 空回答 | 同固定售后流程 |
| `HELD-CLARIFY-RETURN` | 空回答 | 同固定售后流程 |
| `HELD-ORDER-1003-TOOL` | 事实 mismatch | 工具说“硅胶手机壳/19.9/待发货”，知识库说“SH-K68/149/已发货” |
| `HELD-S10-MULTI` | 事实正确，引用 uncertain；另调用随机商品工具 | RAG 答案没有受影响，但 `query_product` 返回随机 `411.55` 元，若模型采用会产生风险 |
| `HELD-ORDER-CORRECTION` | 成功 | 同会话先查 1001，再明确改口 1002；回答与工具参数均绑定最新实体 |

## 订单 fixture 审计

当前 `app.orders.py` 与运行时知识文件不是同一业务事实：

- 1002：工具是“机械键盘 / 399 / 已发货”，知识是 `ORD-1002`
  “SH-E300 + SH-C60 / 328 / 已支付未发货”；
- 1003：工具是“硅胶手机壳 / 19.9 / 待发货”，知识是 `ORD-1003`
  “SH-K68 / 149 / 已发货”。

本轮没有在评测中静默替换生产工具，只在 `tests/eval/e2e_quality_r2/fixtures/orders.json` 声明当前
工具目录并加测试防止漂移。是否统一为知识库事实应由验收会话决定；修改 catalog 后必须用新版本重跑。

## 成本与延迟

整个会话只有一个 2 USD ledger，跨修复臂原子预留；共 373 条 settled 记录，4 次缓存命中，
106 次结构化 400 失败，0 pending，总计 `$0.073185168`，剩余 `$1.926814832`。

兼容臂精确成本：

| 项目 | dev | held-out | 合计 |
|---|---:|---:|---:|
| generation | $0.008777244 | $0.024275028 | $0.033052272 |
| evidence judge | $0.011256108 | $0.009506916 | $0.020763024 |
| 低置信飞轮规范化/去重 | $0.0012744 | $0.0012348 | $0.0025092 |
| embedding/rerank | $0 | $0 | $0 |
| 总计 | $0.021307752 | $0.035016744 | $0.056324496 |

兼容臂 token：176,573 input / 26,287 output。P95 为 dev 39.591s、held-out 16.745s。
embedding/rerank 使用预先复制的精确缓存；这仍是混合缓存的端到端时延，不是冷启动 P95。

## 可解释的限制

- 严格成功少不代表全部答案都“幻觉”；9 个 uncertain 已独立列入人工复核队列。
- 引用支持仍是保守的确定性尺子。markdown、附加说明和多个 `[n]` 会使本来正确的事实句进入
  uncertain/unsupported，需人工或独立模型标签进一步判断。
- 没有根据 held-out 再改 prompt、路由或工具调用；评分 v2 是事后修正，必须保留原结果并降低其独立性。
- 没有运行答案复核策略臂：本轮已有一个由真实兼容 bug 触发的第二臂，再叠加验证器会同时改变两个变量。
- 本轮不证明 MySQL/公网并发、生产 RAG 召回全链路或人工客服任务成功率。

## 复跑

```bash
/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_e2e_quality_r2 \
  --diagnose-history \
  --output tests/eval/e2e_quality_r2/history-v1

/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_e2e_quality_r2 \
  --run --allow-paid --splits dev,heldout --arm structured-json --max-cost-usd 2.0

/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_e2e_quality_r2 \
  --rescore --results tests/eval/e2e_quality_r2/results/structured-json-v1/attempt-01/dev
```

完整逐题记录：

- [默认基线 attempt-02](../../tests/eval/e2e_quality_r2/results/baseline-v1/attempt-02/aggregate.json)
- [JSON 兼容臂原始结果](../../tests/eval/e2e_quality_r2/results/structured-json-v1/attempt-01/aggregate.json)
- [兼容臂 v2 人工复核队列](../../tests/eval/e2e_quality_r2/results/structured-json-v1/attempt-01/heldout/manual-review-v2.jsonl)
- [历史零调用诊断聚合](../../tests/eval/e2e_quality_r2/history-v1/history-aggregate.json)

## 简历候选句

1. “为 ShopHelper 建立真实 `POST /api/chat` 端到端质量基线，冻结 8+20 题与 SHA，记录完整 prompt、
   工具 I/O、证据映射、成本和 P95；统一审出历史行为提示泄漏。”
2. “定位并隔离 DeepSeek thinking 模式不支持 `tool_choice` 的生产兼容故障：默认路径 28/28 技术降级，
   JSON 文本 Schema 修复臂降至 0/28；同批回答由确定性三层评分复核。”

限制：样本是模拟电商数据，兼容臂只有 4/28 严格成功；两句话只能证明评测与接线诊断，不能证明线上
RAG 准确率或业务 ROI。
