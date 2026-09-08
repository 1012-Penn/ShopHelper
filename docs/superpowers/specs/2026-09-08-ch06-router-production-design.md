# ch06 设计:分流器正式版——指代消解 / Query 改写扩写 / 意图识别 / 退款售后确定性子流程(2026-09-08)

## 背景与目标

ch05 的图骨架中,指代消解是原样透传、意图识别是单字段 JSON 简单 prompt、退款退货直接走知识检索路径。本章把 Workflow 里最关键的分流节点做成正式版:

- **指代消解 + Query 改写**:LLM 结合对话历史,把「它能退吗」补全成不依赖上下文的完整问法,口语归一同步完成;问题已完整、指代已明确的原样透传。
- **Query 扩写**:一句问题泛化成多条侧重点不同的检索查询,强制 JSON `{queries: []}`,几条一起检索再去重合并;只对退款退货、售后这类高频、容错低的核心场景做,简单 FAQ 不扩;扩写放检索侧现查现用,知识库只留一份,不在入库侧拆存多份。
- **意图识别**:LLM prompt 路线(不训分类模型),prompt 按四件套写——七类意图枚举成选择题、强制 JSON `{intent, confidence}`、边界 few-shot、留「其他」兜底类。
- **分流精化**:退款退货、售后走一条确定性子流程——先拿订单数据、再 Query 扩写并强制检索政策条款,最后只把「这一单能不能退」交给主力 Agent 判断。
- **槽位处理**:缺订单号不让模型猜,弹订单选择器让用户点选回填;退款原因不追问,提交退款单时从固定类目自选;需求澄清留在主力 Agent,意图识别只判「要干什么」。

不做:微调小模型 / BERT 意图分类、跨会话记忆。

## 定稿决策

| 决策点 | 结论 | 依据 |
|---|---|---|
| 订单数据源 | 固定 mock 订单目录,不建表。`app/orders.py` 单一来源(1001 无线耳机 ¥299 已签收 / 1002 机械键盘 ¥399 已发货 / 1003 硅胶手机壳 ¥19.9 待发货),`GET /api/orders`、稳定化 `query_order`、子流程 `fetch_order` 三处共用;未知单号返回 `{"error": "未找到该订单"}` | 延续 ch02「订单不接真实 API、不建表」mock 哲学;选择器卡片与子流程数据一致,演示不打架 |
| 订单选择器停走机制 | **无状态回传**:选择器数据 + 补全后的问题随 `order_selector` SSE 帧下发,点选时前端随新 `/api/chat` 请求原样带回(`resume` 字段),服务端零跨回合状态 | `interrupt()` 被 spike 实测排除(见选型理由 1);无状态回传重启无感、测试面最小、与 `initial_state` 防残留机制无冲突 |
| 退款单落点 | 复用 ch05 工单链路 `POST /api/tickets`,`ticket_type=售后`,描述 = 订单号 + 退款原因类目 + 备注;不建新表 | 「无新增组件」约束;ch05 确认交互模式复用 |
| 意图模型 | 默认直接用主模型(`openai_model`,温度 0);降级路配置化:`intent_escalation_enabled`(默认 false)、`intent_confidence_floor`(默认 0.6)、`intent_small_model`(缺省同主模型) | 用户要求:默认配够格大模型先把准确率做上去,成本吃紧时再走降级路 |
| 意图输出契约 | 原始 JSON 文本 + 宽松解析(剥 markdown 代码围栏后 `json.loads`);畸形/超纲一律 `{"intent": "其他", "confidence": 0.0}` 并 trace 记 `malformed=true` | 验收 2 要求「JSON 稳定可解析」可被诚实检验;`with_structured_output` 会把解析藏进框架 |
| resolve/expand 输出契约 | 沿用 ch04 `with_structured_output(method="function_calling")` 模式(同 rewriter) | 辅助抽取类,ch04 已验证的模式;意图识别保留裸 JSON 是验收需要,非风格分裂 |

## 选型理由

1. **停走机制弃 `interrupt()`/`Command(resume)`**:spike 实测(2026-09-08,`/tmp/spike_interrupt.py`,venv langgraph 1.2.11 + py3.10)——`interrupt()` 内部调 `langgraph/config.py::get_config`,async 上下文被官方守卫直接 `RuntimeError("Called get_config outside of a runnable context")`,与 ch05 `get_stream_writer` 被禁是**同一函数同一守卫**。服务端 pending slot 方案(备选 B)引入跨回合状态机且与 `initial_state` 全量默认防残留机制相冲突,一并放弃。
2. **意图识别保留裸 JSON 文本契约**:验收 2「意图输出的 JSON 稳定可解析」必须能被检验——解析器就是被测对象。宽松解析 + 兜底类让畸形输出有确定归宿。
3. **子流程不设置信度闸**:ch05 知识路径的 gate 是「证据弱就不进 Agent」;子流程的产出是订单事实 + 政策证据,Agent 自评(system prompt 已强制拒答标记)是足够兜底,双闸会让「订单能查到但政策证据弱」的正常单也掉兜底。政策证据完全为空时落低置信池(数据飞轮口径),但仍进 Agent。
4. **Context7 核对**:条件边读 state、节点返回部分 dict、TypedDict State 语义与官方文档(choosing-apis 章节)一致;新增 API 面为零——本章只用 repo 已验证的 langgraph/langchain 用法。

## 图骨架

### State 增量(ChatState 新增字段)

```python
class ChatState(TypedDict):
    # ...ch05 既有 15 字段不动...
    intent_confidence: float       # 意图置信度(trace/评估用)
    order: dict                    # 子流程拿到的订单数据({} = 无)
    expand_queries: list[str]      # 扩写产物(trace/评估用)
    refund_flow: bool              # 本轮走了退款售后子流程(actions 决策用)
    resume_order_id: str           # 无状态回传:点选带回的订单号
    resume_question: str           # 无状态回传:点选带回的补全问题
```

`initial_state` 同步增默认值(防 checkpointer 残留);`resume_*` 由路由层从 `ChatRequest.resume` 注入。

### 节点与边

```
START → resolve ─(resume_question 非空,旁路)──────────────────────→ prepare_order
          └─(正常)→ intent → route(条件边)
                ├ 商品咨询 → retrieve → gate ─(弱)→ fallback → log → END   (ch05 原路径不动)
                │                        └(够)→ agent
                ├ 物流/订单 → agent
                ├ 退款退货/售后 → prepare_order
                ├ 投诉 → complaint → log → END                              (不动)
                ├ 闲聊 → chitchat → log → END                               (不动)
                └ 其他 → agent(需求澄清兜底,trace 记 fallback=其他)
prepare_order ─(抽到单号)→ fetch_order → expand → policy_retrieve → agent
              └(缺单号)→ ask_order(发 order_selector 帧)→ log → END
agent → log → END;其余同 ch05
```

### 各节点规格

- **resolve(正式版,`app/graph/entry.py`)**:LLM(温度 0)+ 裁剪后历史,指代消解 + 口语归一一次完成;输出 `{resolved: str, changed: bool}`;`changed=false`(问题已完整)时 `resolved_message = user_message` 原样透传。发 `resolved` SSE 帧(前端灰字「已理解:…」,仅 changed=true 时)。旁路:`resume_question` 非空时跳过 LLM,`resolved_message = resume_question`。trace:`node=resolve changed=<bool> q=<resolved>`。历史为空 + 消息完整时也走 LLM 判定(不改写),保持单一代价路径;LLM 异常兜底原样透传(澄清是增强不是依赖,同 ch04 rewriter 哲学)。
- **intent(四件套,`app/prompts.py` INTENT_PROMPT 重写 + `entry.py`)**:①八选一选择题(七类 + 其他,枚举在 prompt 里);②强制 JSON `{"intent": "...", "confidence": 0.0~1.0}`;③边界 few-shot 4-6 条(退款退货 vs 售后、投诉 vs 闲聊、订单 vs 物流等易混边界);④「其他」口径:拿不准、超纲、与店铺业务无关一律归它,不硬塞业务意图。温度 0。confidence 随 trace 落行:`node=intent intent=退款退货 confidence=0.92`。降级路(`intent_escalation_enabled=true` 时):小模型先判,confidence < floor 则大模型重判一次,取大模型结果;trace 记 `escalated=true`。
- **route(改造)**:`退款退货/售后 → prepare_order`;新增 `其他 → agent`;其余不变。
- **prepare_order(新)**:纯确定性。单号抽取正则:①`(?:订单|order)\s*号?\s*(\d{3,6})` 优先;②裸 4-6 位独立数字兜底(`SH-E300` 的 3 位数字不误伤)。来源优先级:`resume_order_id` > resolved_message 正则。抽到 → 下一步;抽不到 → `ask_order`。trace:`node=prepare_order order=<id|missing>`。
- **ask_order(新)**:发 `order_selector` 帧(items = 固定订单目录三单的 `{order_id, product, amount, status}`,question = 补全问题,original = 原始消息),本轮结束:无 final_reply、user 消息照常落库(log 节点兼容)、assistant 侧不落。trace:`node=ask_order n=3`。
- **fetch_order(新)**:`app/orders.py get_order(order_id)` 取数(确定性,无随机);未知单号也把 `{"order_id": ..., "error": "未找到该订单"}` 放进 `order` 继续走(Agent 如实相告)。trace:`node=fetch_order found=<bool>`。
- **expand(新)**:LLM(温度 0)把补全问题泛化成 3-4 条侧重点不同的检索查询,`with_structured_output(ExpandedQueries{queries: list[str]})`;失败兜底 `[resolved_message]`(扩写是增强)。发 `tool_status` 帧(label「查询扩写」)。trace:`node=expand n=<len>`。
- **policy_retrieve(新)**:逐条 `service.retrieve(q, strategy="hybrid_rerank")`(扩写产物已是标准问法,`use_rewrite=False` 不二次改写)→ 按 chunk_id 去重留最高分 → 按 rerank 分降序截 `retrieval_final_top_k` → 证据重编号(n = 合并后名次,沿用 ch04 citations 条目格式)→ 发 citations 帧 + `tool_status` 帧(label「政策检索」)。证据全空 → `pool.insert("retrieval_low_conf", ...)` 落池,仍进 Agent。trace:`node=policy_retrieve queries=<n> merged=<m> top1=<score>`。
- **agent(窄化注入,`app/graph/agent.py` 小改)**:子流程轮(`refund_flow=true`)在 System 追加订单数据 JSON + 窄化指令——退款退货:「请仅依据下方订单数据与政策证据,判断这一单(订单{id})能不能退,并说明依据条款」;售后:「……说明这一单的售后问题该怎么处理(修/换/退)」。收敛且非拒答且意图=退款退货时 `suggested_actions=["refund_form"]`(带 order_id);拒答仍走 `ACTIONS_ON_REFUSAL`。**顺手修既有缺陷**:`n_offset` 初值改 `len(state.get("evidence") or [])`,消除子流程证据与 agent 内 query_faq 的引用编号撞号。
- **logging_node(小改)**:actions 帧的 item 支持携带 `order_id`;`waiting_selector` 轮(有 user 消息、无 assistant 回复)照常落 user 行。

### 订单目录(`app/orders.py`,新文件)

```python
ORDER_CATALOG = {
    "1001": {"order_id": "1001", "product": "无线耳机",   "amount": 299.0,  "status": "已签收", "created_at": "2026-08-30"},
    "1002": {"order_id": "1002", "product": "机械键盘",   "amount": 399.0,  "status": "已发货", "created_at": "2026-09-03"},
    "1003": {"order_id": "1003", "product": "硅胶手机壳", "amount": 19.9,   "status": "待发货", "created_at": "2026-09-07"},
}
def get_order(order_id) -> dict   # 命中返回目录项副本,未命中 {"order_id": ..., "error": "未找到该订单"}
def list_orders() -> list[dict]   # 固定顺序三单
```

`query_order` 工具改为调 `get_order`(随机 mock 退役);`GET /api/orders` 返回 `list_orders()`。ch02 时代 query_order 的随机行为测试按新契约迁移。

## 契约变化

### SSE 新增帧

- `{"type": "resolved", "changed": true, "question": "我买的SH-E300耳机能退吗"}`——前端灰字提示,验收 3 可视证据
- `{"type": "order_selector", "items": [{order_id, product, amount, status}], "question": "...", "original": "这个能退吗"}`——前端渲染可点订单卡

### actions 帧扩展

items 可含 `{"action": "refund_form", "label": "申请退款", "order_id": "1001"}`;既有 `transfer_human`/`create_ticket` 不动。

### ChatRequest 扩展

```python
class OrderResume(BaseModel):
    order_id: str      # 3-6 位数字校验
    question: str      # 补全后的问题(order_selector 帧原样带回)
class ChatRequest(BaseModel):
    message: str
    session_id: int | None = None
    resume: OrderResume | None = None   # 非空时走子流程旁路
```

resume 轮的消息契约:`message` = 原始问题(落库与展示),`resume.question` = 子流程工作问题(不再过 resolve/intent 的 LLM)。

### 新端点

`GET /api/orders` → `{"items": [...]}`(固定三单,选择器数据源)。

## Prompt 定稿要点

- **RESOLVE_PROMPT**:给对话历史 + 当前消息;任务:①把代词/省略指代补全成独立完整问题;②口语模糊问法归一为标准问法;③问题已完整且指代明确时原样返回、changed=false;不新增假设,补全仅用历史中出现过的事实;只输出 JSON。
- **INTENT_PROMPT(四件套)**:八选一枚举 + 判类口径 + 4-6 条边界 few-shot(至少覆盖:退钱 vs 售后换修、抱怨语气 vs 正式投诉、查订单状态 vs 查物流、明确闲聊)+「其他」口径 + 只输出 JSON 两个字段。
- **EXPAND_PROMPT**:面向退款退货/售后政策检索;3-4 条侧重点不同(如退货条件/时效、特殊品类限制、运费承担、质量问题举证);不出现具体订单号(政策是通用的);只输出 JSON。

## 评估体系(替代 TDD 的部分)

- **评估集** `tests/eval/router_samples.jsonl`:多轮对话用例,每轮标注 `{turns: [{user, expect_intent, expect_resolved?, expect_route?}]}`;覆盖:物流→退款→物流来回切(验收 1)、「它能退吗/这个能退吗」类指代、口语归一、拿不准怪问题→其他(验收 2)、七类各覆盖、边界混淆对。
- **评估脚本** `scripts/eval_router.py`:真模型逐轮跑 resolve + intent;指标——意图准确率(分桶:总体/七类各自/其他兜底/边界对)、指代消解质量(expected_resolved 归一化比对:去空白/标点后全等)、JSON 可解析率、confidence 分布、escalation 触发率(开启时)。报告落 `reports/ch06-router-report.md`。真跑一次留数。
- **单元测试(假模型)**:意图畸形 JSON→其他+malformed trace;resolve 旁路不过 LLM;prepare_order 正则(含 SH-E300 不误伤);子流程编排(缺单号→选择器帧→落库无 assistant;有单号→fetch→expand→retrieve→agent 链路);expand 失败兜底;policy_retrieve 去重合并与编号;agent 窄化注入与 refund_form action;resume 请求契约。LLM 用替身,不出网。

## 前端(static/index.html,Vibe Coding 例外,不走 brainstorm/TDD/code review)

- `resolved` 帧:气泡内灰字「已理解:…」(changed=true 才显示)
- `order_selector` 帧:订单卡片行(单号/商品/金额/状态徽章),点选后卡片组置灰锁定,自动 `POST /api/chat {message: original, session_id, resume: {order_id, question}}`,新气泡正常收 SSE
- `refund_form` action:气泡尾部「申请退款」按钮 → 展开内嵌表单(退款原因下拉六类:质量问题/七天无理由/尺寸不合适/发错货/与描述不符/不想要了 + 备注输入框)→ 提交 `POST /api/tickets {session_id, description: "订单{id}申请退款;原因:{reason};备注:{note}", ticket_type: "售后"}` → 按钮转「工单 xxx 已创建」;不重提交
- 效果迭代模式:用户描述 → 直接改,不设评审门

## 测试策略

单测全替身不出网(假模型/FakeEmbedding/内存向量库/SQLite,沿用 ch04/ch05 模式);意图与指代消解的质量验证走评估脚本真跑;真机验收四条在 docker MySQL + 真 DeepSeek/BGE 全链下逐条打。

## 验收标准 → 验证方式映射

1. 多轮来回切意图全对、指代全补对 → `scripts/eval_router.py` 报告(物流→退款→物流用例组)
2. JSON 稳定可解析、怪问题落其他 → 单测(畸形输入)+ 评估报告(JSON 可解析率 / 其他兜底命中率)
3. 「这个能退吗」先补全指代再走子流程拿订单和政策 → 页面灰字 + 徽章链(订单查询→查询扩写→政策检索)+ 服务日志 trace 链
4. 不带单号问退款弹订单选择器、点选后走完 → 浏览器操作(order_selector 帧 → 点选 → resume 旁路 → 最终答复)

## 已知边界(记档不解决)

- 订单目录固定三单,mock 数据;未知单号走「未找到」话术
- 选择器卡片若被搁置、用户转而问别的,旧卡片仍可点:点选视作对该订单重新发起退款咨询(语义合理,不设过期)
- 服务重启丢 SSE 流(既有);resume 无服务端状态,重启无额外损失
- 意图识别三段 LLM(resolve/intent/expand)在同回合串联,退款轮延迟换准确率,符合本章定位
- escalation 降级路实现但默认关:准确率优先,成本吃紧时打开

## 实现期偏差

(预留:实现中发现的偏差回填此处,格式同 ch05 spec)
