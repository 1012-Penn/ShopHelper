# ch08 设计:即插即用工具系统——注册中心 / 校验 / 权限 / 执行引擎 / 审计 / MCP 接入 / 建工单确认流(2026-09-08)

## 背景与目标

ch02 起的工具层是写死的内置清单:加工具要动 definitions+registry 两处核心代码;参数校验靠 pydantic 异常兜;重试不分类(业务空结果也重试);无审计;外部能力为零;create_ticket 模型直调直执行(用户无确认环节)。本章升级为即插即用工具系统:

- **注册中心**:内置与 MCP 工具统一登记(工具名/用途描述/JSON Schema 参数三件套),新工具注册即可被主力 Agent 使用,不改核心代码;MCP 工具动态发现、现问现拿。
- **参数校验**:执行前统一按 JSON Schema 校验,错误包成工具结果回灌给模型(追问用户或重新组织调用),不抛异常。
- **权限控制**:工具分只读/写;写操作(create_ticket)由执行引擎把门——模型直调一律拒绝(落审计),真实执行必须走前端确认回传;MCP 工具的读写属性由我们这侧规则指定(全只读),不看 Server 声明、不让模型临场判断。
- **执行引擎**:所有调用走同一处,统一超时/重试/错误分诊(参数不合法/查询落空/真故障)/结果格式化;重试只给暂时性故障,业务空结果不重试,写操作默认不自动重试。
- **审计留痕**:tool_audit_logs 每次调用一条(含被拒/被拦),写审计失败不拦工具执行,不挂外键。
- **MCP 接入**:自建物流/售后两个业务 MCP Server(独立进程,FastMCP + Streamable HTTP,mock 数据);客服系统作 MCP Client(MultiServerMCPClient)接入,拿回工具与内置合成一份;内置 query_logistics 下线,物流查询由 MCP Server 接管。
- **建工单确认流**:明确要求建工单 → Agent 追问补齐必填 → 发起 create_ticket → 引擎拦截、推工单预览卡片 → 前端确认才落 tickets 表并回复工单号,取消则按「权限拒绝」落审计;ch05 投诉流程按钮建单路径不动。

不做:Skill 机制、仓储等更多外部系统。

## 定稿决策

| 决策点 | 结论 | 依据 |
|---|---|---|
| 建工单确认机制 | **无状态回传**(订单选择器同款):`ticket_preview` SSE 帧推预览 → 前端「确认提交/取消」→ `POST /api/tickets/confirm` / `cancel` 回传执行或记拒 | spike 实测(2026-09-08,/tmp/spike08,langgraph 1.2.11 + py3.10):`interrupt()` 在 async 上下文仍触发 `RuntimeError: Called get_config outside of a runnable context`,与 ch06 同款官方守卫,不可用;ch06 订单选择器的无状态回传已真机验证,语义等价(预览数据随帧下发,用户操作随请求带回,服务端零跨回合状态) |
| 工具统一形态 | 全部收编为 langchain `BaseTool`(内置已是 `@tool`;MCP adapters 产出即 BaseTool),注册中心存 `ToolRecord(tool, source, mcp_server, access, label)`,bind_tools 直接吃 record.tool | 内置零迁移成本;校验用 `tool.args_schema.model_json_schema()` 拿 JSON Schema,MCP 工具天然带 schema |
| 内置扩展点 | **插件目录自动装载**:`app/tools/plugins/*.py` 启动时按文件名序 import,模块暴露 `register(ctx)`(ctx 携带 session_factory/registry),调 `ctx.registry.register(...)` 即完成注册 | 验收 1「只做注册动作、不动核心代码」;新增工具 = 落一个插件文件,核心零改动 |
| MCP 动态发现 | 每轮聊天前 `McpService.sync()`(带 30s TTL 防抖):`get_tools()` 全量拉取 → 与注册中心 diff 同步(新增登记/消失下线);Server 重启+加工具,客户端不重启,下一问即用 | 验收 3「服务不重启、现问现拿」;本地 list_tools 开销毫秒级 |
| MCP 工具权限 | 我侧规则:所有 MCP 工具 `access=read`(工具用途声明不可信);与内置重名时**内置优先**、MCP 版跳过并 warn | req 3「只认我们这侧的权限规则」;query_logistics 已下线避重名 |
| 校验器 | `jsonschema.validate(args, schema)`:类型不对/必填缺失/越界统一 `校验拦下`,错误说明(jsonschema 路径+message)包成工具结果回灌 | jsonschema 4.26 已入依赖;统一入口在引擎 |
| 重试分类 | 读操作:仅 `TimeoutError`/网络类异常(ConnectError 等)重试 1 次;工具正常返回(含业务空结果)不重试;**写操作任何情况不自动重试**(超时未必没执行,重复执行更糟) | req 4;审计记实际 retry_count |
| 错误分诊 | 参数不合法(校验拦下)/查询落空(工具正常返回空/未找到,如实回灌不重试)/真故障(异常、超时)三档,坏消息原文回灌模型 | req 4 |
| create_ticket 审计语义 | 模型直调被拦截 → 即落一条 `权限拒绝`(说明:写入操作未获用户确认);用户确认后 resume 的真实执行 → 单独落 `成功`;用户取消 → 不新增(拦截行即该次调用终态) | req 5「每次工具调用落一条」+验收 4/5;拦截行即模型调用终态 |
| MCP Server 形态 | 每个独立进程:`mcp.server.fastmcp.FastMCP` + `streamable_http_app()` 挂 uvicorn;物流(logistics_tracker 查轨迹)/售后(query_warranty 查在保、query_return_progress 查退货进度),mock 随机数据不建表;端口 8001/8002,`MCP_LOGISTICS_URL`/`MCP_AFTERSALES_URL` 可配 | spike 全链实测通过(FastMCP→uvicorn→MultiServerMCPClient `{"transport":"http","url":…}`→get_tools→ainvoke) |
| debug 工具 | `TOOLS_DEBUG=true` 时注册 `debug_slow_query`(读,睡 N 秒)/`debug_slow_write`(写,睡 N 秒),供验收 6 人为制造超时 | 验收 6 需可控超时;默认关闭不污染生产清单 |

## 架构与数据流

### 组件(新/改)

```
app/tools/
  base.py        ToolRecord(dataclass)+ ToolRegistryV2:register/unregister/list/bind/labels
  builtin.py     register_builtin(registry, ctx):query_order/query_product/query_faq/create_ticket(由 definitions 迁入)
  plugins/       内置插件目录(autoload);__init__.py 扫 *.py 逐个 import 调 register(ctx)
  engine.py      ToolEngine:execute(name, args_json, *, tool_call_id, conversation_id, on_write_intercept, confirmed=False)
  mcp_client.py  McpService:MultiServerMCPClient 封装,sync()→registry diff 同步,connect 失败降级 warn
app/mcp_servers/
  logistics.py   FastMCP 物流 Server(mock 轨迹)
  aftersales.py  FastMCP 售后 Server(mock 在保/退货进度)
```

- `definitions.py` 退役(标签迁 base.py 的 label 入 record;TOOL_LABELS 静态字典删除)。
- `registry.py` 旧 ToolRegistry 删除,agent 改注入 ToolEngine。

### 执行引擎管线(engine.execute)

```
查 record →(未注册)回灌「未注册」
→ json.loads + jsonschema 校验 →(不合法)审计「校验拦下」+ 回灌错误说明
→ 权限:access=write 且未 confirmed →(引擎)审计「权限拒绝」+ on_write_intercept(record,args)发 ticket_preview 帧
  + 回灌「写入操作需用户在前端确认,已发起确认请求,请等待用户操作」
→ 执行:asyncio.wait_for(timeout) 包 record.tool.ainvoke(args)
   ├ 超时:读→重试1次后仍败→「超时」;写→不重试→「超时」
   ├ 连接类异常:读→重试1次;其余异常→「失败」
   └ 正常返回:MCP 内容块取 text / str 原样,ensure_ascii=False →「成功」(业务空结果如实回灌,不重试)
→ 每条出路写 tool_audit_logs(try/except 包裹,写失败仅 warning)
```

### 建工单确认流(无状态回传)

1. 用户明确要求建工单,信息不全 → 模型追问(纯 prompt 行为,问题描述/类型必填)。
2. 信息齐 → 模型调 `create_ticket` → 引擎拦截(权限门)→ agent 经 `on_write_intercept` 发 `{"type":"ticket_preview","ticket_type","description"}` 帧 → 回灌结果告知模型等待;本轮照常收敛落库。
3. 前端渲染预览卡(工单类型/问题描述 + 「确认提交」「取消」)。
4. 确认 → `POST /api/tickets/confirm {conversation_id, ticket_type, description}`(前端原样回显预览载荷,服务端零状态)→ 引擎 `confirmed=True` 执行 → 落 tickets 表 → 前端以机器人气泡展示「工单 {ticket_no} 已创建」。
5. 取消 → `POST /api/tickets/cancel {conversation_id, tool_name:"create_ticket", reason:"用户取消"}` → 落一条「权限拒绝」审计(用户取消),不建工单。
6. ch05 投诉流程的前端按钮建单(`POST /api/tickets`)保持不动——点按钮本身即用户确认。

### 审计表(DDL 已给定)

ORM `ToolAuditLog`(SQLite 方言兼容);`ToolAuditStore(factory).log(...)` 同步 INSERT,引擎 try/except 包裹;字段:conversation_id/tool_call_id/tool_name/tool_source/mcp_server/arguments(JSON)/result_summary(截 500)/status(五枚举)/error_message/retry_count/duration_ms/created_at;不挂外键,只有普通索引。

## 测试策略

单测全替身:引擎管线(校验拦下/权限拒绝/读超时重试/写不重试/业务空结果不重试/审计含被拒被拦/审计写失败不拦执行/格式化取 text)、registry diff 同步(stub client)、插件 autoload(tmp 目录)、confirm/cancel 端点;图 e2e(GraphChatModel + 替身引擎);MCP 真链路留给验收脚本。`scripts/acceptance_ch08.py` 跑六条验收(起双 MCP Server 进程+主服务)。前端预览卡 Vibe Coding,不走 TDD/评审。

## 验收标准 → 验证方式映射

1. 新工具即插即用 → plugins 目录落 `demo_time.py`(query_server_time),重启服务,对话问「现在几点」Agent 调用;
2. 双 MCP Server 独立进程,问物流轨迹 → MCP 工具应答;
3. MCP Server 侧加新工具只重启该 Server → 客户端 sync 现问现拿;
4. 缺信息追问 → 预览卡确认 → tickets 落行且回复带工单号;
5. 预览卡取消 → 工单未建,审计「权限拒绝」;
6. debug 工具人为超时 → 读:审计 超时+retry_count=1+耗时;写:超时+retry_count=0。

## 已知边界(记档不解决)

- 无状态回传:确认载荷由前端回显,服务端不持久待确认工单;重启丢失挂起卡片(与订单选择器一致)。
- MCP Server mock 随机数据,不接真实系统。
- 内置工具超时/重试为引擎级,工具内部已有 try/except 吞异常的(query_faq)按正常返回计,不重试(语义:服务可用但无结果)。
- 前端确认载荷未签名,本地演示系统不防篡改(ch02 无用户体系边界延续)。
- 多 worker 部署不在范围。

## 实现期偏差

(实现期回填)

1. **建工单确认机制为无状态回传而非 LangGraph interrupt**:spike 实测(2026-09-08,langgraph 1.2.11 + py3.10)async 上下文 `interrupt()` 仍触发 `RuntimeError: Called get_config outside of a runnable context`(ch06 同款官方守卫);功能语义(预览→确认→放行/取消)经订单选择器同款帧回传完整保留。
2. **未注册的工具名不落审计**:DDL 的 tool_source 枚举(builtin/mcp)无中立值,模型幻觉出的未注册名无工具实体可归属,只回灌错误文本不落行。
3. **MCP 型工具 args_schema 为原始 JSON Schema dict**(adapters 0.3.x),ToolRecord.json_schema 兼容 dict/pydantic 两形态;真机首验曾因对 dict 调 model_json_schema() 崩溃整轮(error 帧、审计缺行),回归测试覆盖。
4. **确认通道收口(I-4)**:POST /api/tickets/confirm 仅放行「已注册且 access=write」的工具(422 拒绝读工具/未注册名);非 create_ticket 工具执行失败映射 502;引擎最外层兜底的审计按注册表真实 source/mcp_server 归属(查不到才退 builtin)。
5. **McpService.sync 增强**:TTL 只在成功同步后起算(失败下轮立即重试);单次发现包 asyncio.wait_for 5s 上界(半死 Server 不拖死聊天轮);失败保留既有注册不做删除性同步;get_tools 连续 3 次失败才告警跳过。
6. **插件钩子签名为 register(registry, ctx)**(计划文本写作 register(ctx),以实现为准);插件 register() 抛异常时 warn 跳过,不炸启动。
7. **query_order 描述消歧**:与 logistics_tracker 并存后,描述明确「不含物流轨迹」,避免模型误选(验收 2 实测)。
