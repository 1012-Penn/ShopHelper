# ch07 设计:会话上下文管理——三层历史 / 后台异步摘要 / 多会话前端(2026-09-08)

## 背景与目标

ch03 起的历史管理是「简单裁剪」:`trim_history` 从最老开始整条丢弃,丢掉的信息永久消失,只留一个 `HISTORY_TOKEN_BUDGET` 常量。本章升级为正经的上下文管理:

- **三层结构**:按离当前轮的远近把历史切成层 1(最近,原文不压)/ 层 2(中间,渲染时截短)/ 摘要(最远,异步压成梗概)。边界是两个锚点 id(`summary_upto_msg_id`、`layer1_from_msg_id`),降级只挪 id、不搬数据。
- **摘要走后台异步**:层 2 攒到超预算,后台起一次摘要任务把这一批压成一段追加进摘要表,全程不阻塞当前回复;梗概一段段只追加、压完不回炉,旧梗概只作背景不参与合并。
- **拼装有序**:system(人设/红线,静态)打头,接层 2 半压、层 1 原文、当前用户消息;早期梗概与检索证据合成一条挂在当前用户消息之后,绝不单独占一条 system(上游模板会上提合并所有 system,前缀缓存整段作废)。
- **预算从窗口倒推**:扣固定开销(系统提示/证据/梗概预留/安全余量)与单轮 ReAct 峰值,历史预算取「想留轮数 × 每轮稳态」与「窗口匀得出的」较小者,层 1 七成、层 2 三成;启动自检,装不下一轮就报警。
- **可观测**:每轮把发给模型的上下文原样打进 `logs/app.log`(model_ctx / history_ctx),摘要任务触发/开始/完成/跳过/失败全留痕。
- **多会话前端**:侧栏列出全部会话(新在前/首问预览/已摘要标记),点击切换回载历史接着聊;后端两个只读接口。

不做:语义检索捞历史、主题重要度留关键事实、跨会话记忆、用户画像。

## 定稿决策

| 决策点 | 结论 | 依据 |
|---|---|---|
| 检查点持久化 | 沿用 `InMemorySaver`,不引 SqliteSaver。完整历史的**持久真源是 MySQL**;图启动后发现该 thread 的 checkpoint 无消息(进程重启/首touch)时,从 MySQL 回灌 State(`add_messages` 按 id 去重,回灌消息带 `id=str(row.id)` 防并发双灌重复)。「落盘」由 MySQL 承担,进程内完整轨迹由 checkpoint 承担 | MySQL 已是对话真源,再落一份 SQLite checkpoint 会造出第二个会漂移的持久真源 + 新依赖;InMemorySaver 边界 ch05 已记档 |
| State.messages 语义 | `Annotated[list[AnyMessage], add_messages]`;各节点只吐**本轮新增**的 LangChain 消息,框架按 id 并入;`initial_state` 记 `turn_start_len`,log 节点据此切出本轮新消息落库 | Context7/内省核对(langgraph 1.2.11):按 id 替换、缺 id 追加 |
| 分层视图的数据源 | **MySQL**(messages 行带自增 id + conversations 锚点 + conversation_summaries),不取自 checkpoint。锚点是 MySQL id,而 checkpoint 里的轮内工具消息没有(也不可能有)MySQL id,位置映射不可靠;MySQL 真源天然重启安全、切会话安全 | 两套真源各走各的:req 5 字面语义——checkpoint 完整轨迹(含工具消息)与模型输入精简版互不影响 |
| 工具结果落库契约 | **收窄**:messages 表只落 user 行与含文本的 assistant 行;assistant 纯工具调用行与 tool 行不再落库(ch02「全量落库」契约收窄)。工具轨迹活在当轮 State/checkpoint 内,跨轮事实靠摘要延续;层 2 渲染的「工具结果一行标识」规则仍实现(种子/旧数据可能含 tool 行) | 用户明示「工具结果压根不落消息表」;顺带让回载历史=干净的 user/assistant 文本对,OpenAI 协议合法性不再依赖工具行回灌 |
| 摘要触发时机 | 每轮开头(路由层,处理当前消息前)检查**上一轮落库后**的占用:层 1 超预算先同步降级(纯 id 移动,立刻生效),层 2 超预算再投后台摘要任务 | 触发看用量不数条数;后台任务与 SSE 流零竞争,「不阻塞当前回复」结构成立 |
| 梗概+证据的挂载 | 合并为一条 `HumanMessage`(【历史梗概】/【参考知识】/【订单数据】标注)挂在当前用户消息**之后**;agent 的 system 收敛为纯静态——证据、订单数据、窄化指令全部移出 system | 上游模板会上提合并 system、挤掉工具定义后的可变内容,前缀缓存整段作废;挂在用户句后每轮 system 逐字节一致 |
| RERANK_TOP_K | 新名 `rerank_top_k`(env `RERANK_TOP_K`,默认 10)顶替 `retrieval_final_top_k` 的全部职责,旧 env 名退役 | 验收演示配置按 `RERANK_TOP_K` 口径点名 |
| 闲聊/投诉轮的历史 | history_ctx 在路由层每轮必打(含闲聊兜底轮);resolve 的 {history} 槽位改喂「摘要行+滑窗」文本;intent 仍只看 resolved 消息(指代已在 resolve 折入,意图输入不变) | req 6「不进 Agent 的闲聊兜底轮也看得到」;意图四件套契约 ch06 已定稿不动 |
| 摘要模型 | `make_extract_model`(温度 0),裸文本输出;超 300 字硬截断兜底(prompt 软约束几十到一两百字) | 辅助抽取类同 resolve/expand 口径;不重问,成本可控 |

## 预算推导(验收数字可精确复现)

新设置(env 名):`MODEL_CONTEXT_WINDOW=65536`、`MAX_OUTPUT_TOKENS=2000`、`MAX_USER_INPUT_TOKENS=2000`、`MAX_AGENT_STEPS=6`(由 `agent_max_steps` 改名)、`TOOL_RESULT_MAX_TOKENS=2000`、`RERANK_TOP_K=10`,以及固定开销组件:`CTX_PROMPT_OVERHEAD_TOKENS=1800`(system 渲染 + 工具定义估)、`CTX_EVIDENCE_ITEM_TOKENS=250`、`CTX_SUMMARY_ALLOWANCE_TOKENS=200`、`CTX_SAFETY_MARGIN_TOKENS=1500`、`CTX_KEEP_ROUNDS=30`、`CTX_PER_ROUND_TOKENS=250`。

```
峰值   = MAX_USER_INPUT_TOKENS + MAX_AGENT_STEPS × TOOL_RESULT_MAX_TOKENS
固定   = CTX_PROMPT_OVERHEAD + RERANK_TOP_K × CTX_EVIDENCE_ITEM
         + CTX_SUMMARY_ALLOWANCE + CTX_SAFETY_MARGIN
滑窗   = max(MODEL_CONTEXT_WINDOW − MAX_OUTPUT_TOKENS − 峰值 − 固定, 0)
历史   = min(CTX_KEEP_ROUNDS × CTX_PER_ROUND, 滑窗)
层1预算 = int(历史 × 0.7)      # Python 浮点语义,5650×0.7=3954.99…→3954
层2预算 = int(历史 × 0.3)      # 各算各的再截断,5650×0.3→1695;两者之和 5649,1 token 余量
```

**演示配置复核**(`MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5`):峰值=2000+3×1200=5600,固定=1800+5×250+200+1500=4750,滑窗=18000−2000−5600−4750=**5650**,层1=int(3954.99…)=**3954**,层2=**1695**——与验收口径逐位一致。

**只调窗口的陷阱复核**(仅 `MODEL_CONTEXT_WINDOW=18000`,其余默认):峰值=2000+6×2000=14000,固定=1800+2500+200+1500=6000,滑窗=18000−2000−14000−6000<0→0,启动自检报「上下文预算不足」。默认配置滑窗=65536−2000−14000−6000=43536,历史=min(7500, 43536)=7500,层1=5250,层2=2250,二十轮闲聊量级(≈2-3k)远够,不降级不摘要(验收 3)。

启动自检:`create_app` 时算一遍,`滑窗 < CTX_PER_ROUND`(连一轮稳态都装不下)打 `ERROR 上下文预算不足:…`,正常时打一行预算明细。中文按字数折 token 的口径沿用 `estimate_tokens`(CJK 每字 1 token),预算推导与占用统计同一把尺子,改口径必须连预算一起校准。

## 三层结构与渲染规则

- **层 1(id > layer1_from,锚点 NULL 视为全部)**:原样,一字不压。装填用 `langchain_core.messages.trim_messages(strategy="last", start_on="human", token_counter=estimate_tokens 包装)`,保证后缀从 user 行开始、整轮边界切入;结果为空时至少保最后一轮。
- **层 2(summary_upto < id ≤ layer1_from)**:**渲染时**截短、存储保持原文——user 原文不动;assistant 只留开头 `CTX_LAYER2_ASSISTANT_HEAD_CHARS=60` 字加「…」;tool 行渲染成 `[工具结果已省略]` 一行;content 为空的 assistant 纯工具调用行跳过。
- **摘要(id ≤ summary_upto)**:`conversation_summaries` 一段一行只追加(seq 从 1 递增,uk_conv_seq),`conversations.summary` 列是全部段落按 seq 拼接的投影,压完不回炉重写;旧梗概只作背景给模型看、不参与合并。
- **降级(级联第一步)**:层 1 token 超 `层1预算` → `trim_messages` 重切层 1,新锚点 `layer1_from = 首条保留行 id − 1`(层 1 语义为 `id > layer1_from`,与 DDL 注释一致),`UPDATE conversations` 只挪 id;日志 `层1 降级 <旧>→<新> token X→Y`。至少保最后一轮。
- **摘要触发(级联第二步)**:降级后层 2 token 超 `层2预算` → 投后台摘要任务,批 = `(summary_upto, 触发时 layer1_from]` 的整段;完成时 `summary_upto` 追到批末行 id,期间层 2 以半压形态渲染顶住,若渲染后总量仍超滑窗,组装时从最老端丢弃层 2 渲染块(仅本次上下文,不动锚点,兜底日志)。

## 图与 State 变化

```python
class ChatState(TypedDict):
    # ...既有字段保留,以下变化...
    messages: Annotated[list[AnyMessage], add_messages]  # 完整历史+本轮轨迹,checkpoint 承载
    turn_start_len: int        # 本轮新消息在合并后列表中的起点(log 切片落库用)
    layered: dict              # 路由层预算好的分层上下文:
                               #   system/layer2/layer1(组装序)、background(梗概+证据文本)、
                               #   history_text(resolve 用)、tokens 统计
    history: dict              # 删除(旧 trim 产物),由 layered 取代
    messages(旧语义 dict pending): 删除
```

- **initial_state**:注入 `turn_start_len`、`layered`;thread 首touch(进程内 hydrated 集合判定)时把 MySQL 历史还原为带 `id=str(row.id)` 的 LC 消息并入 messages。
- **节点**:一律吐 LangChain 消息对象(无 id,由框架赋 uuid)——agent 吐 AI/Tool 消息,complaint/chitchat/ask_order 只吐 AI(或零吐);user 的 HumanMessage 由 initial_state 带入。
- **log 节点**:`messages[turn_start_len:]` 过滤落库——HumanMessage→user 行,含非空文本的 AIMessage→assistant 行;纯工具调用 AI 与 ToolMessage 不落。trace/actions/拒答落池逻辑不变。
- **agent 组装**:`[System(纯静态)] + render(层2) + render(层1) + HumanMessage(当前) + HumanMessage(背景块)`;ReAct 步内追加 AI(tool_calls)/ToolMessage;工具结果入库前按 `TOOL_RESULT_MAX_TOKENS` 截断(加「…(已截断)」标识);`agent_token_budget`(ch05 既有)继续兜单轮输出总量。背景块 = 【历史梗概】(摘要投影,非空才带)+【参考知识】(证据,带 [n] 角标规则文本)+【订单数据】+窄化指令(子流程轮);全空则省略该条。
- **resolve**:{history} 槽位改喂 `layered.history_text`(摘要行+滑窗文本);旁路/异常兜底不变。
- **组图**:节点与边零变化(ch06 骨架不动)。

## 数据与契约

- **DDL 两个文件**(用户提供内容落盘):`db/ch07-summary.sql`(conversations 加 `summary` TEXT、`summary_upto_msg_id` BIGINT)、`db/ch07-layers.sql`(加 `layer1_from_msg_id` BIGINT;建 `conversation_summaries(id, conversation_id, seq, from_msg_id, upto_msg_id, content, created_at)`,uk_conv_seq + idx_conv_upto)。`db/init.sql` 同步纳入(全新 docker 卷一遍建全)。
- **ORM**:`Conversation` 加三列;新增 `ConversationSummary` 模型(SQLite 测试库由 `Base.metadata.create_all` 覆盖)。
- **新端点(只读)**:
  - `GET /api/conversations` → `{"items": [{"id", "preview"(首条 user 消息截 100 字), "summarized"(summary 列非空), "updated_at"}]}`,按 updated_at DESC;
  - `GET /api/conversations/{id}/messages` → `{"conversation_id", "messages": [MessageItem…]}`(404 同既有语义);`GET /api/sessions/{id}/history` 保留兼容。
- **消息长度闸**:`chat.py` 400 阈值从 `history_token_budget`(退役)改为 `max_user_input_tokens`。
- **日志**:`create_app` 挂 `logs/app.log` FileHandler(目录自动建,logs/ 已 gitignore)。关键行口径(grep 锚点):
  - `[history_ctx] session=%s 摘要=%d段 滑窗=%d条 层2=%d 层1=%d tokens≈%d` + 摘要行与滑窗逐条原样(每轮必打,闲聊轮也有);
  - `[model_ctx] session=%s step=%d 条数=%d tokens≈%d` + 摘要全文 + 滑窗逐条原样(ReAct 每步打,system 只计 token 不落正文);
  - `[ctx] session=%s 层1 降级 %s→%s token %d→%d`;`[ctx] session=%s summary trigger 层2 约 %d token > 预算 %d(后台执行,不阻塞本轮回复)`;
  - `[summary] session=%s start 第%d段 覆盖=[%d,%d]` / `done 第%d段 覆盖=[%d,%d] 耗时=%dms 字数=%d` / `skip 已有任务在跑` / `fail err=…`。

## 摘要服务(app/summarizer.py + SUMMARY_PROMPT)

- 触发守卫:同会话在飞集合防重入;任务引用挂在 app.state 防 GC;uvicorn 关停时任务随进程结束(未摘要的层 2 下轮重触发,锚点单调,幂等)。
- 一次任务:取批内消息行(原文,非渲染态)→ 旧梗概作背景、本批对话作正文 → `SUMMARY_PROMPT`(温度 0)→ 只提炼事实与诉求:问过哪款商品、报过的订单号手机号等关键信息、明确诉求、还没解决的问题;对话里没出现的一个字不编;寒暄闲聊不留;输出一段几十到一两百字 → 超 300 字硬截断 → 事务内 `INSERT conversation_summaries(seq=max+1, from=batch首, upto=batch末)` + `UPDATE conversations SET summary=全部段落拼接, summary_upto_msg_id=batch末` → 全程 trigger/start/done/skip/fail 留痕带耗时。
- 非阻塞证据:trigger 行在轮前,done/fail 行晚于该轮 done 帧到达皆可,时间戳可查。

## 前端(static/index.html)

- 左侧会话侧栏:「新对话」按钮 + 会话列表项(首问预览单行省略、「摘要」小徽标=summarized、新在前);点项切换:设 session_id、`GET /api/conversations/{id}/messages` 回载渲染气泡、可继续聊(发送带该 session_id);当前项高亮。
- 「新对话」:清空当前 session_id 与聊天区,旧会话仍在侧栏可切回;每轮 done 后静默刷新侧栏。
- 侧栏加载失败:catch 后隐藏侧栏,不影响聊天区(静默降级)。
- 既有能力(工具徽章/citations/actions/订单选择器/退款表单/满意度)一律不动。

## 测试策略

单测全替身不出网(假模型/SQLite 内存库,沿用既有模式):预算推导(演示集逐位断言 5650/3954/1695、只调窗口→0+报警条件、默认不触发)、分层与渲染(锚点 NULL/降级锚点数学/层 2 三条截短规则/组装顺序/背景块合并/摘要不进 system/工具截断)、摘要服务(触发条件/段落追加与锚点推进/投影拼接/超长截断/失败不抛/在飞跳过/prompt 含批文与旧梗概)、log 契约(工具行不落库/turn_start_len 切片)、两个只读端点、e2e(caplog 见 history_ctx 每轮/model_ctx/级联+摘要链)。`trim_history` 退役,`tests/test_history.py` 迁移。真机验收走 docker MySQL + 真模型(脚本化多轮对话),浏览器侧栏终验尽力自动化、不可则留用户本地点验。

## 验收标准 → 验证方式映射

1. 默认配置连聊 20+ 轮 token 不爆系统不崩 → 真机脚本跑 20+ 轮,无 5xx/上游 400,app.log 无降级/摘要行(兼验收 3:不触发压缩,「装得下就不压」);
2. 演示配置级联全链 → 同脚本带演示 env 重跑:看到 `层1 降级`→`summary trigger`→`summary done 第N段`;问「最开始那个订单后来怎么说」答对订单号与诉求(靠梗概);
3. (=1 的不触发断言);
4. 后台摘要不阻塞 + 可观测 → app.log 时间戳证明 done 晚于本轮回复完成;`grep model_ctx / history_ctx` 直接看到每轮摘要与滑窗;
5. 侧栏多会话 → 开两个会话对照、切回旧会话历史完整、接着聊(浏览器自动化,不可自动化则 curl 证据 + 用户点验)。

## 已知边界(记档不解决)

- InMemorySaver 进程内无界增长、重启即清(ch05 记档延续):重启后 checkpoint 空,历史由 MySQL 回灌,轮内工具轨迹不跨重启存活(靠摘要延续事实)。
- 崩在 log 节点前的回合 checkpoint 里有而 MySQL 没有:模型输入以 MySQL 为准,该轮视作未发生(ch05「未落库即未发生」语义延续)。
- 摘要任务在 uvicorn 关停时可能被截断:锚点未推进即无副作用,下轮重触发。
- 层 2 超预算到摘要完成之间,当轮上下文以半压渲染顶住,必要时组装端从最老端丢弃层 2 渲染块(仅当轮,不动锚点)。
- 摘要质量依赖模型:prompt 禁编造 + 长度软约束 + 300 字硬截断,不做事后校验。
- 多会话无用户体系,guest 单用户(ch02 边界延续);同会话并发请求无锁(ch05 M8 记档延续)。

## 实现期偏差

(以下为实现期确认的偏差与契约演进,评审波回填)

1. **层 2 预算口径为 `int(历史×0.3)`**:七三开各自 int 截断(5650 → 3954 + 1695,和留 1 token 余量),非「层 2 = 历史 − 层 1」;初版推导有一处口算错(prompt 开销 1400→1800 校正),由预算单测首跑逮住。`MAX_AGENT_STEPS` 由 `agent_max_steps` 改名、`RERANK_TOP_K` 顶替 `retrieval_final_top_k`,`HISTORY_TOKEN_BUDGET` 退役。
2. **messages 表只落 user 行与最后一条含文本 assistant 行**:ReAct 中间文本碎片(「我先查一下」类)与熔断回溯的重复答复不落库——场景 S 真机验收逮住回载出现两条 assistant 后收窄;工具行/纯工具调用行不落库为需求原文。
3. **层 2 的「截短」是渲染规则、存储保持原文**(摘要批压的是原文);`split_layers` 的 NULL `layer1_from` 视为「从未降级」,层 1 从 summary_upto 之后起(防已摘要行回流层 1)。
4. **conversations.updated_at 随每轮落库推进**(评审 I-2):侧栏「新在前」按活跃序浮顶,而非创建序;`store.append` 顺带 touch 会话行。
5. **评审波修复(I-1/I-3)**:退款窄化指令的 `{order_id}` 占位符在拼入背景块前 format(ch06 重写时 format 丢失,字面量泄漏);`render_history_text` 的层 2 行补「用户:/客服:」角色前缀(与层 1 一致,护住指代消解)。
6. **退役偏差**:`trim_history`/`history_to_messages` 的删除从计划 Task 3 推迟到评审修复波执行(过渡期 chat.py/agent.py 仍引用,为保持每个提交全绿);已完成。
7. **记档**:层 2 渲染块因超滑窗被丢弃时,`history_text` 仍按未丢弃的行渲染(resolve 当轮看到的窗口略大于 agent,无害);model_ctx 的 tokens≈ 不计纯工具调用消息的参数(日志口径偏小);日志路径相对 CWD(约定仓库根启动);`TOOL_RESULT_MAX_TOKENS` 小于截断标记自身 token 数时截断反超(现实配置不可达)。
