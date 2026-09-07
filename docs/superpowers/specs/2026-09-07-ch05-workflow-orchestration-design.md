# ch05 设计:Workflow 确定性编排 + LangGraph 图骨架(2026-09-07)

## 背景与目标

客服系统升级为生产级编排:LangGraph Workflow 确定性图做骨架,主力 ReAct Agent 作为骨架中的核心节点。祛魅热身先手写裸 Agent 循环,看清"Agent = 带工具的循环",再用 LangGraph 重构。

- 意图识别、指代消解本章最简实现(简单 prompt 出七类 JSON / 原样透传),正式版后置。
- 不做:上下文管理升级、MCP、飞轮入库。

## 定稿决策(用户确认)

| 决策点 | 结论 |
|---|---|
| 裸循环落点 | `scripts/naive_agent.py` 演示脚本,不进生产路径 |
| 与旧 /api/chat 关系 | 直接替换,内部跑图;SSE 帧兼容 + 新增 actions 帧 |
| checkpointer | `InMemorySaver`(langgraph 自带,进程内) |
| 历史落库 | MySQL `ConversationStore` 保留双写,checkpointer 只承载图内运行态 |

## 选型理由

1. **Agent 节点手写 ReAct 循环,不用 `create_react_agent` 预制件**:本章主题是祛魅,预制件藏起循环;停止条件、token 控制、追问语义需要节点内可控。
2. **流式用 `astream(stream_mode=["custom"], version="v2")` + 节点内 `get_stream_writer()`(或 `StreamWriter` 参数注入)**:Agent 逐 token 发自定义帧,tool_status/citations/actions 也走 custom;SSE 映射只消费 custom,格式自控。`astream_events` 事件噪音大,弃。

版本基线:langgraph 1.2.11(新增依赖,pyproject 固定 `langgraph>=1.2,<2`)、langgraph-checkpoint 4.2.0、langchain 1.4.0。API 已对官方文档(streaming / memory 章节)与装好的 venv 源码双重核对。

## 图骨架

`app/graph/` 新包。

### State(一路贯穿)

```python
class ChatState(TypedDict):
    session_id: str
    user_message: str
    resolved_message: str      # 本章 = 原样透传
    intent: str                # 七类之一:物流/订单/商品咨询/退款退货/售后/投诉/闲聊
    evidence: list[dict]       # 检索证据(带全局引用编号 n),知识类才有
    gate_passed: bool
    agent_steps: int           # ReAct 实际步数(验收 5 观察)
    final_reply: str
    suggested_actions: list[str]   # "transfer_human" / "create_ticket" 子集
    trace: list[str]           # 每节点留一行,如 "node=retrieve strategy=hybrid_rerank top1=0.87"
    messages: list             # Agent 节点内的 LangChain 消息轨迹
```

### 节点与边

```
START → resolve → intent → route(写死分流,条件边)
  ├ 商品咨询/退款退货 → retrieve → gate ─(证据弱)→ fallback → log → END
  │                                  └(证据够)→ agent
  ├ 物流/订单/售后 ──────────────────────────→ agent
  ├ 投诉 → complaint → log → END
  └ 闲聊 → chitchat → log → END
agent → log → END
```

- **resolve**:原样透传(`resolved_message = user_message`)。
- **intent**:简单 prompt,输出 JSON `{"intent": "..."}`;解析失败或输出七类之外 → 一律归 `"商品咨询"`(走知识路径最安全,检索闸可拦)。
- **route**:纯代码条件边,七类 → 四出口:
  - 知识类 {商品咨询, 退款退货} → retrieve;
  - 业务数据类 {物流, 订单, 售后} → agent;
  - 投诉 → complaint;闲聊 → chitchat。
- **retrieve**:强制 `RetrievalService.retrieve(strategy="hybrid_rerank")`(复用 ch03/04,经 registry 同一套构造);证据格式沿用 ch04 citations 条目(`n/chunk_id/question/answer/category/section_path`),编号本回合从 1 起。
- **gate**:证据弱(`low_confidence=True`,即候选空或 Top-1 < `rerank_score_floor`)→ 兜底话术 + `pool.insert("retrieval_low_conf", ...)` 落池(飞轮素材)+ 不进 Agent;证据够 → 证据文本随问题注入 Agent prompt。业务数据类无检索证据,不过闸。
- **agent(主力 ReAct 节点)**:
  - 循环:bind_tools(registry 全五工具,退款退货可自主调订单/物流工具)→ astream 逐 token `writer` 发 token 帧(缓存攒完整文本),tool_call_chunks 静默拼装 → 有调用则逐个 `registry.execute`,发 tool_status 帧,结果回灌继续;无调用即收敛。
  - 知识类进入时,检索证据以 system/首条消息注入并要求带 [n] 角标;发 citations 帧(证据在答案流之前下发,沿用 ch04 前端契约)。
  - 停止条件:无 tool_calls 收敛 / `max_steps`(默认 6)/ 累计 token 超预算(默认复用 `history_token_budget`)强制收敛(输出已有文本或提示稍后再试)。信息不足时模型直接输出追问文本,即追问用户。
  - Agent 判断合适时(如处理不了、用户情绪强烈),在 `suggested_actions` 里带 `transfer_human` 和/或 `create_ticket`,后端不自动执行。
- **complaint**:不进 Agent、零模型调用,固定安抚话术 + `suggested_actions=["transfer_human","create_ticket"]`(两选项都给,用户自选)。
- **chitchat**:固定话术,零模型调用。
- **fallback**:固定兜底话术(证据弱)。
- **log**:整轮成功才落库 `ConversationStore`(沿用 ch04 语义);`logger.info` 输出 trace 每行(验收 1 看 `node=retrieve` 日志);投诉/Agent 的 suggested_actions 转成 actions 帧。

## 接口

- `POST /api/chat`:入参不变;内部 `graph.astream(state, config={"configurable": {"thread_id": session_id}}, stream_mode=["custom"])`,custom 事件逐一映射 SSE 帧:`session/token/tool_status/citations/actions/done/error`。actions 帧结构 `{"type":"actions","items":[{"action":"transfer_human","label":"转人工"},{"action":"create_ticket","label":"建工单"}]}`(可只含一个)。
- `POST /api/tickets`(新):`{"session_id", "description"}` → registry 的 `create_ticket`(conversation_id 解析、ticket_type 固定"投诉"场景可传参,默认"售后")→ 返回工单号。仅前端按钮触发,后端不自动建单。

## 前端(static/index.html)

- 收到 actions 帧在该条回复气泡下渲染独立按钮(转人工、建工单),互不绑定。
- 点「转人工」:本地展示「已转接人工客服」消息,再蹦「您好,我是客服小猫,请问有什么可以帮您的」;不调任何接口。
- 点「建工单」:确认交互(复用现有样式)→ `POST /api/tickets` → 展示工单号。
- 两按钮都不点、继续发消息:按普通对话走图,无任何动作。

## 错误处理

- 图内节点异常:与 ch04 一致,SSE 下发 error 帧、本轮不落库。
- intent JSON 解析失败 → 归商品咨询(检索闸兜底)。
- Agent 达到步数/预算上限未收敛 → 输出已有内容或引导语,不无限循环。

## 测试(TDD)

假模型替身(沿用现有 fake 模式)覆盖:
1. 分流四出口:知识类走了 retrieve 节点(trace 断言)、业务类直达 agent、投诉/闲聊不进 Agent;
2. gate:弱证据拦截 → 兜底话术 + 落池断言;强证据放行且证据入 Agent prompt;
3. Agent:单步收敛、两步 ReAct(查订单→查物流)、步数上限熔断、追问(纯文本输出);
4. SSE 帧兼容:session/token/tool_status/citations/done + actions 帧(投诉两选项、Agent 单选项);
5. `/api/tickets` 写 tickets 表;
6. intent 解析失败回落。

真机验收(五条验收标准):
1. 政策类问题日志见 retrieve 节点;
2. 「订单 1001 的物流到哪了」Agent 自调工具;
3. 「我要投诉」→ 两按钮,转人工本地展示、建工单写表,互不绑定;
4. 闲聊固定话术;
5. 「先查订单 1001 再告诉我物流」ReAct 多步。

## 实现期偏差记录(2026-09-07,Task 6)

**流式传输机制变更**:`get_stream_writer()`/StreamWriter 注入在 Python 3.10 async 上下文不可用——langgraph 源码 `config.py::get_config` 显式守卫"Python 3.11 or later required to use this in an async context"(已实测复现,且为官方已知问题 langchain-ai/langgraph#5927 同根因)。技术选型不变(仍是 LangGraph 图 + checkpointer + 确定性编排),仅节点向外发帧的通道改为:路由层建 `asyncio.Queue` 放入 `config["configurable"]["sink"]`,节点经注入的 writer 闭包 `put_nowait` 发帧,路由层并发排空转 SSE。节点单测接口(`writer` 可调用对象)不变。
