# ch08 即插即用工具系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把写死的内置工具升级为即插即用工具系统:统一注册中心(内置插件化 + MCP 动态发现)、JSON Schema 校验、读写权限把门、统一执行引擎(超时/分类重试/错误分诊/格式化)、tool_audit_logs 审计、双业务 MCP Server 接入、create_ticket 前端确认流。

**Architecture:** 全部工具收编为 langchain `BaseTool`,注册中心存 `ToolRecord(tool, source, access, mcp_server, label)`;执行引擎是唯一调用入口(校验→权限→执行→审计);MCP 工具每轮聊天前 diff 同步(现问现拿);建工单确认走订单选择器同款无状态回传(interrupt 被 py3.10 守卫禁,spike 证据在 spec)。

**Tech Stack:** mcp 1.30.0(FastMCP + Streamable HTTP)/ langchain-mcp-adapters 0.3.2(MultiServerMCPClient)/ jsonschema 4.26.0 / MySQL(tool_audit_logs);新增依赖三项已装。

**Spec:** `docs/superpowers/specs/2026-09-09-ch08-tool-system-design.md`

## Global Constraints

- 写操作只有 create_ticket:模型直调引擎一律拒(审计「权限拒绝」),真实执行必须前端确认回传(`confirmed=True`);ch05 按钮建单 `POST /api/tickets` 不动
- 重试:仅读操作 + TimeoutError/ConnectionError/OSError 网络类,1 次;业务空结果不重试;写操作任何情况不自动重试
- 审计:每次调用一条(含校验拦下/权限拒绝),写审计失败只 warning 不拦执行;不挂外键
- MCP 工具我侧全定只读;与内置重名内置优先、MCP 版跳过 warn;`query_logistics` 内置下线,物流走 MCP
- MCP Server:FastMCP + `streamable_http_app()`,独立进程,mock 随机数据不建表;日志 grep 口径见验收
- py3.10;测试替身不出网(不连真实 MCP);每任务全量 pytest 收尾;前端预览卡 Vibe 不走流程

---

### Task 1: 审计表 ORM/Store + 引擎配置项

**Files:**
- Create: `db/ch08-tool-audit.sql`(用户 DDL 原文);Modify: `db/init.sql`(尾追加该表+头注释 ch08 行)、`app/models.py`(ToolAuditLog)、`app/config.py`(tool_timeout_seconds=10.0 / tool_retry_attempts=1 / tools_debug=False)
- Create: `app/tools/audit.py`(ToolAuditStore)
- Test: `tests/test_ch08_audit.py`

**Interfaces:**
- `ToolAuditStore(factory).log(*, conversation_id=None, tool_call_id=None, tool_name, tool_source, mcp_server=None, arguments=None, result_summary=None, status, error_message=None, retry_count=0, duration_ms=None) -> None`(同步 INSERT,异常上抛由调用方兜)
- status 枚举:「成功」「失败」「超时」「校验拦下」「权限拒绝」
- 消费方:Task 3(引擎)

- [ ] **Step 1 失败测试** `tests/test_ch08_audit.py`:log 落行全字段回读(result_summary>500 截断;arguments dict 存 JSON);表在 Base.metadata.create_all 下可建
- [ ] **Step 2-4 实现+全量绿**:ORM 列与 DDL 一一对应(Enum→sqlite 兼容,照 LowConfidenceQuestion 模式);init.sql 追加
- [ ] **Step 5 Commit** `ch08: tool_audit_logs 表+审计 Store+引擎配置项`

### Task 2: 注册中心 ToolRecord/ToolRegistryV2 + 内置插件化 + 插件目录

**Files:**
- Create: `app/tools/base.py`(ToolRecord + ToolRegistryV2 + LABELS 常量)、`app/tools/builtin.py`(register_builtin(registry, ctx):四工具迁入,create_ticket access="write",query_logistics 不迁)、`app/tools/plugins/__init__.py`(load_plugins(ctx))、`app/tools/plugins/demo_time.py`(query_server_time:返回服务器时间,验收 1 的「新写工具」示范)
- Modify: `app/tools/definitions.py` → 删除(TOOL_LABELS/CreateTicketInput/build_tools 迁 base.py/builtin.py;ctx dataclass 放 base.py:`ToolContext(session_factory, service, kb, top_k, rewriter=None, reranker=None)`——build_tools 的注入逻辑原样搬进 builtin.register_builtin)
- Test: `tests/test_ch08_registry.py`

**Interfaces:**
- `ToolRecord(tool: BaseTool, source: str, access: str = "read", mcp_server: str | None = None, label: str = "")`;`.name` / `.json_schema`(args_schema.model_json_schema(),无 schema 给 `{"type":"object","properties":{}}`)
- `ToolRegistryV2`:`register(record)`(重名:若新记录 source="mcp" 则 warn 跳过返回 False;否则 ValueError)、`unregister(name)`、`get(name)`、`records()`、`bind_tools()`(tool 列表)、`labels()`、`names()`
- `load_plugins(ctx)`:glob `plugins/*.py` 跳 `_` 前缀,import 调 `register(ctx)`
- 消费方:Task 3(引擎吃 registry)、Task 4(MCP 同步)、Task 5(接线)

- [ ] **Step 1 失败测试**:register/重名 MCP 跳过内置 ValueError/json_schema 形状(labels 优先级)/load_plugins 用 tmp 插件文件
- [ ] **Step 2-4 实现+全量绿**(definitions.py 删除会红 test_tools.py/test_ch05_agent 等——本任务先改 import 面到新入口或临时 shim,Task 5 统一清理;记录清单)
- [ ] **Step 5 Commit** `ch08: ToolRecord 注册中心+内置插件化+插件目录 autoload`

### Task 3: 执行引擎 ToolEngine(校验/权限/超时/重试/分诊/格式化/审计)

**Files:**
- Create: `app/tools/engine.py`
- Test: `tests/test_ch08_engine.py`

**Interfaces:**
- `ToolEngine(registry, audit_store, timeout_seconds=10.0, retry_attempts=1)`
- `async execute(name, args_json, *, conversation_id=None, tool_call_id=None, on_write_intercept=None, confirmed=False) -> str`(回灌给模型的工具结果文本)
- `async execute_confirmed(name, args_json, *, conversation_id=None, tool_call_id=None) -> str`(= confirmed=True 快捷)
- 管线:未注册→回灌错误(不落审计?落「失败」);json.loads 失败→「校验拦下」;jsonschema 校验败→「校验拦下」(说明含路径+message+修正指引);write 且非 confirmed→审计「权限拒绝」+ `on_write_intercept(record, args)` + 回灌等待确认指引;执行 `record.tool.ainvoke(args)` 包 `asyncio.wait_for`;读+超时/网络类(ConnectionError/OSError)重试 1 次,写不重试;正常返回格式化(list 取 text 块拼接/str 原样)→「成功」;审计含 retry_count 与 duration_ms,truncate result_summary 500
- 消费方:Task 5(agent/端点/接线)

- [ ] **Step 1 失败测试**(每条一个用例):未注册/坏 JSON/必填缺失→校验拦下/类型错→校验拦下/写直调→权限拒绝+拦截回调收到 args+回灌指引/写 confirmed→执行成功且审计成功/读超时→重试 1 次审计 retry_count=1 超时/写超时→retry_count=0/业务空结果(工具正常返回空串)→不重试成功/连接异常→重试/MCP 块列表格式化/审计抛异常不拦执行(审计 store 打桩 raise)
- [ ] **Step 2-4 实现+全量绿**
- [ ] **Step 5 Commit** `ch08: ToolEngine 执行引擎——校验/权限把门/分类重试/三档分诊/审计留痕`

### Task 4: MCP 双 Server + McpService 客户端同步

**Files:**
- Create: `app/mcp_servers/__init__.py`、`app/mcp_servers/logistics.py`(FastMCP "logistics",tool `logistics_tracker(order_id)` mock 轨迹,env `AFTERSALES_STYLE_EXTRA` 不加——extra 在售后)、`app/mcp_servers/aftersales.py`(FastMCP "aftersales",tools `query_warranty(order_id, product)` / `query_return_progress(order_id)`,env `AFTERSALES_EXTRA_TOOLS=true` 时加注册 `query_repair_shop(city)` 供验收 3)
- Create: `app/tools/mcp_client.py`(McpService)
- Test: `tests/test_ch08_mcp.py`

**Interfaces:**
- `McpService(registry, urls: dict[str, str], ttl_seconds=30.0)`;`async sync(force=False)`(TTL 内跳过;`MultiServerMCPClient({n: {"transport": "http", "url": u}})` 逐 server `get_tools(server_name=n)`;diff:新名且不与内置重名→register ToolRecord(source="mcp", access="read", mcp_server=n, label=tool.name);消失的 mcp 工具 unregister;异常 warn 降级);`connected() -> bool`
- Server app 暴露 `app = mcp.streamable_http_app()`;mock 数据照 ch02 风格(随机承运商/节点/在保状态)
- 消费方:Task 5(接线)、Task 7(验收 2/3)

- [ ] **Step 1 失败测试**:stub client(假 get_tools 返回 BaseTool 列表——用 SimpleNamespace+langchain StructuredTool 构造)→ sync 注册 mcp 记录;TTL 内不重复拉;server 侧消失→unregister;与内置重名→跳过;client 抛异常→warn 不炸
- [ ] **Step 2-4 实现+全量绿**(两个 Server 文件的真实链路留验收;此处只测 McpService 逻辑)
- [ ] **Step 5 Commit** `ch08: 物流/售后双 MCP Server(独立进程)+ McpService 现问现拿同步`

### Task 5: 接线改造(agent/端点/main/conftest/旧测试迁移)

**Files:**
- Modify: `app/graph/agent.py`(registry→engine:`engine.registry.bind_tools()`;执行走 `engine.execute(..., conversation_id=state["session_id"], tool_call_id=tc["id"], on_write_intercept=...)`,拦截回调发 `{"type":"ticket_preview","ticket_type":...,"description":...}` 帧)、`app/main.py`(ctx 组装 builtin 注册+load_plugins+debug 工具+McpService+ToolEngine,替换旧 ToolRegistry/build_tools;`app.state.engine/registry/mcp_service`)、`app/routers/chat.py`(prepare_turn 里 `await request.app.state.mcp_service.sync()`)、`app/routers/tickets.py`(confirm/cancel 两端点)、`app/schemas.py`(TicketConfirmRequest/TicketCancelRequest)、`tests/conftest.py`(make_client 换新装配:真实 builtin 注册 + engine;不接 MCP)
- Delete: `app/tools/registry.py`
- Test: `tests/test_ch08_chat_api.py` + 旧测试迁移(test_tools.py 退役改写、test_ch05_agent/test_chat_toolflow 的 query_logistics 物流查询标签改走内置存根或换工具名、conftest 断言)

**Interfaces:**
- `POST /api/tickets/confirm {conversation_id, ticket_type, description}` → `engine.execute_confirmed("create_ticket", json.dumps({...}))` → 200 `{"ticket_no", "status"}`;失败 502
- `POST /api/tickets/cancel {conversation_id}` → 审计「权限拒绝」(tool_name=create_ticket,error_message="用户取消确认")→ `{"ok": true}`
- agent 对 `ticket_preview` 帧的生成:引擎拦截时回调;模型收到的回灌文本:「写入操作需要用户在前端确认:已推送工单预览卡片,请在本轮答复中引导用户在卡片上确认或取消,不要重复发起创建」
- 消费方:Task 6(前端)、Task 7(验收)

- [ ] **Step 1 失败测试** `tests/test_ch08_chat_api.py`:e2e 工单确认流(「帮我建个工单」→ 模型追问/直接带齐信息两用例;ticket_preview 帧断言;confirm 落 tickets 表且审计成功;cancel 审计权限拒绝);物流问句走「未注册」回灌(内置已下线,MCP 不在单测环境)
- [ ] **Step 2-4 实现+旧测试迁移**(grep `query_logistics|ToolRegistry|build_tools` 清单逐个迁移;物流标签断言改用 `debug_slow_query` 之外的内置存根名或 query_order)
- [ ] **Step 5 全量绿 + Commit** `ch08: 全链接线——agent/端点/main/conftest 切执行引擎,tickets confirm/cancel`

### Task 6: 前端工单预览卡(Vibe,不走 TDD/评审)

**Files:**
- Modify: `static/index.html`

**Interfaces:** `ticket_preview` 帧 → 预览卡(工单类型徽标+问题描述+两按钮);确认 → `POST /api/tickets/confirm`(session_id+预览载荷)→ 气泡显示「工单 {ticket_no} 已创建」;取消 → `POST /api/tickets/cancel` → 灰字「已取消,未创建工单」;两按钮点击后锁定。

- [ ] **Step 1 渲染与提交逻辑**;**Step 2 node --check + curl 端点**;**Step 3 Commit** `ch08: 前端工单预览卡(确认提交/取消)`

### Task 7: 验收脚本 + 真机六条 + README

**Files:**
- Create: `scripts/acceptance_ch08.py`、`reports/ch08-acceptance.md`;Modify: `README.md`(ch08 段/依赖/已知边界)、`.env.example`(MCP_URLS/TOOLS_DEBUG)

- [ ] **Step 1 脚本**:`--scenario mcp`(起双 Server 子进程+起主服务→问物流轨迹→断言审计 mcp 源)/`--plugin`(demo_time 已在插件目录→重启主服务→问服务器时间→断言审计含 query_server_time)/`--mcp-add`(重启 aftersales 带 AFTERSALES_EXTRA_TOOLS→不重启主服务→问维修网点→审计出现 query_repair_shop)/`--ticket`(缺信息追问→preview→confirm→tickets 落行;/cancel→审计权限拒绝)/`--timeout`(TOOLS_DEBUG=true+TOOL_TIMEOUT_SECONDS=2→debug_slow_query(seconds=5) 超时重试 1→debug_slow_write(seconds=5) retry 0)
- [ ] **Step 2 环境准备**(docker down -v 重建含八表+tool_audit_logs→seed→build_kb)
- [ ] **Step 3 逐场景跑**,证据落 `reports/ch08-acceptance.md`
- [ ] **Step 4 README 收口**;**Step 5 Commit** `ch08: 真机验收六条全过+README 收口`

### Task 8: Code Review(独立 subagent 评审+修复波)

- [ ] 评审 `git diff a75883c..<finish前>`;重点:引擎管线分类正确性(重试/分诊/写把门不可绕)、MCP 同步 diff 一致性、审计全覆盖与失败兜底、确认流无状态回传契约、旧工具下线无死引用、SSE/前端契约
- [ ] Critical/Important 必修带回归;Minor 修或记档;全量 pytest 全绿;dev-notes 补记

### Task 9: finish(完结交付)

- [ ] spec 实现期偏差回填;dev-notes finish 段;README/验收齐备;最终提交;`git push origin main`(ch07+ch08 一并推上去,用户要求 GitHub 留痕)
