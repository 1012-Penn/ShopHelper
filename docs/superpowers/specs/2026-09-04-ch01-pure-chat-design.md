# ShopHelper 第一章设计:纯对话(流式多轮客服 + 售后信息抽取)

日期:2026-09-04
状态:已定稿(用户逐节确认)
上一阶段:brainstorming(方案 A:ChatOpenAI 统一接入)

## 1. 背景与目标

电商智能客服系统的第一章:不接工具、不做 Agent 循环,先把"纯对话"链路完整跑通——SSE 流式多轮对话、模板化 Prompt、带 token 预算的历史裁剪、售后描述结构化抽取。全部功能有真实上游(DeepSeek)背书的验收命令。

## 2. 范围

**做:**

1. `POST /api/chat`:多轮对话,SSE 逐 token 流式输出,服务端内存会话。
2. Prompt 管理:客服角色 System Prompt 用 `PromptTemplate` 模板化,集中在一处。
3. `POST /api/extract`:售后描述 → 固定字段 JSON,`with_structured_output` 实现。
4. 历史裁剪 + token 预算(纯函数,可单测)。
5. 最小聊天页(单文件 HTML+JS,连 SSE 接口;按用户工作要求走 Vibe Coding,不套 brainstorm/TDD/review)。

**不做(本章明确排除):** 工具调用与 Agent 循环;会话持久化(重启即丢,可接受);鉴权;上游重试/降级策略(出错直接下发 error 事件);多 worker 部署(内存会话仅限单进程);心跳/ping 事件。

## 3. 架构与技术选型(已定)

- **模型接入(方案 A)**:`langchain-openai` 的 `ChatOpenAI`,`base_url`/`model`/`api_key` 全部从 `.env` 读取。应用侧统一 OpenAI 协议,换 GPT / Claude(OpenAI 兼容网关)/ DeepSeek / Ollama 只改 `.env`,不改代码。
- **Web 框架**:FastAPI + uvicorn。SSE 用 `StreamingResponse(media_type="text/event-stream")` 手写 `data: <json>\n\n` 帧,不引入 sse-starlette。
- **配置**:pydantic-settings 读 `.env`。
- **会话**:进程内 `dict[session_id, SessionData]`,模块级单例 + `asyncio.Lock` 保护。
- 具体库版本与 API 签名(如 `with_structured_output` 的 method 参数)在实现前用 Context7 MCP 查当前官方文档确认,不凭记忆写。

### 目录结构

```
ShopHelper/
├── .env                    # 密钥等,不进 git
├── .env.example
├── pyproject.toml          # fastapi/uvicorn/langchain/langchain-openai/pydantic-settings/pytest 等
├── app/
│   ├── main.py             # 应用工厂:挂 routers、静态页、注入模型实例
│   ├── config.py           # Settings(pydantic-settings)
│   ├── llm.py              # make_chat_model():按 env 建 ChatOpenAI
│   ├── prompts.py          # 客服 System Prompt + PromptTemplate
│   ├── schemas.py          # 请求/响应/SSE 事件/抽取结果 Pydantic 模型
│   ├── sessions.py         # 内存会话存储
│   ├── history.py          # estimate_tokens + trim_history(纯函数)
│   └── routers/
│       ├── chat.py         # POST /api/chat(SSE)
│       ├── sessions.py     # GET /api/sessions/{id}/history
│       └── extract.py      # POST /api/extract
├── static/index.html       # 最小聊天页(Vibe Coding)
├── scripts/eval_extract.py # 抽取评估脚本
├── tests/
│   ├── eval/extract_samples.jsonl   # 标注样例评估集
│   └── ...                           # pytest 单测
└── dev-notes/ch01.md       # 过程留痕
```

## 4. API 契约

### 4.1 POST /api/chat(流式对话)

请求体:`{"message": "string(必填)", "session_id": "string|可缺省"}`

响应:`text/event-stream`,每帧 `data: <JSON>\n\n`,事件按序:

| 序 | type | 字段 | 说明 |
|---|------|------|------|
| 1 | `session` | `session_id` | 首事件;缺省 session_id 时新建并下发,已带则原样确认 |
| 2 | `token` | `content` | 逐 token,可多帧 |
| 3 | `done` | — | 完整回复已写入会话历史 |
| 异常 | `error` | `message` | 上游/内部错误,之后收流 |

处理流程:取/建会话 → `trim_history(旧历史, 预算)` → 拼 System + 裁剪后历史 + 新 user 消息 → 链 `astream` 逐 token 下发 → **全部成功结束后**才把(user, assistant)一起写入会话;中途出错或客户端断开则本轮两条都不落库(会话不留半截状态)。新 user 消息本身超过预算 → 400 拒绝。

### 4.2 GET /api/sessions/{session_id}/history

`{"session_id": "...", "messages": [{"role": "user|assistant", "content": "..."}, ...]}`;会话不存在 → 404。供两轮上下文验收与调试。

### 4.3 POST /api/extract(结构化抽取)

请求体:`{"text": "售后描述"}` → `200 {"order_no": "string|null", "issue_type": "refund|exchange|repair|logistics|other", "expected_resolution": "string"}`

- 独立模型实例(temperature 0),不进会话、不影响对话模型。
- 字段描述用中文写进 Pydantic Field,辅助上游理解;订单号缺失为 null。
- 上游输出不符合 schema(解析失败)→ 422,`{"detail": "..."}`。

### 4.4 静态页

`GET /` 返回 `static/index.html`(单文件 HTML+JS,连 `/api/chat` SSE)。

## 5. Prompt 设计

- 角色"小帮":电商店铺智能客服。只接售前咨询、售后、订单问题;医疗/法律/投资等请求礼貌拒绝;不知道就坦白并建议转人工;语气友好、简洁、不超过 3 句优先。
- 用 `PromptTemplate` 承载,System Prompt 为模板主体;后续章节可扩展变量槽位。Prompt 文本属于纯 Prompt,不走 TDD,用对话冒烟验证。

## 6. 上下文裁剪与 token 预算

- `estimate_tokens(text)`(纯函数):CJK/全角字符每字记 1 token;其余字符每 4 字符记 1 token;两者相加,宁大勿小。
- `trim_history(messages, budget)`(纯函数):messages[0] 视为 System 永不裁剪;从最老的历史消息开始丢弃,直到预算内。返回裁剪后列表(保序、保最新)。
- 默认预算 `HISTORY_TOKEN_BUDGET=3000`,相对上游窗口极保守,抵消启发式估算误差。

## 7. 结构化输出

```python
class AfterSaleExtraction(BaseModel):
    order_no: str | None          # 订单号,描述里没有则 null
    issue_type: Literal["refund", "exchange", "repair", "logistics", "other"]
    expected_resolution: str      # 用户期望的处理方式,一句话
```

`llm.with_structured_output(AfterSaleExtraction)`;method 参数以 Context7 查到的当前版本推荐为准(DeepSeek 支持 function calling)。

## 8. 配置(.env)

```
OPENAI_API_KEY=...                 # DeepSeek key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
HISTORY_TOKEN_BUDGET=3000
CHAT_TEMPERATURE=0.7
EXTRACT_TEMPERATURE=0
```

换厂商 = 改这三个 OPENAI_* 值(如 Ollama:`http://localhost:11434/v1` + 本地模型名)。

## 9. 测试策略

| 产物 | 方式 |
|------|------|
| history.py(裁剪/估算)、sessions.py(存取/锁) | TDD,pytest 单测 |
| /api/chat SSE 事件序列、/api/extract 契约、错误分支 | TDD,httpx ASGI + 注入 `GenericFakeChatModel`(不打真上游) |
| System Prompt、抽取 Prompt(纯 Prompt) | 评估集替代 TDD:`tests/eval/extract_samples.jsonl` ≥8 条标注样例,`scripts/eval_extract.py` 跑真实 DeepSeek,报告逐字段准确率;目标 order_no 与 issue_type 8/8 全对,expected_resolution 语义相符(人工复核),不达标迭代 Prompt |
| 聊天页 | Vibe Coding,手工过,不进测试 |

模型注入点:`app.state.chat_model` / `app.state.extract_model`,测试直接替换为 fake。

## 10. 验收标准(最终交付)

1. `curl -N` 调 `/api/chat` 可见逐帧 SSE 流式回复。
2. 同一 session_id 连续两轮,第二轮回答接住第一轮信息;`GET history` 可核对。
3. `POST /api/extract` 对一段售后描述返回合规 JSON;评估集报告达标。
4. 聊天页可发起对话并看到流式打字效果。
5. 全量 pytest 通过。

验收命令模板随交付给出(见最终交付说明)。

## 11. 风险与对策

- **LangChain 1.x API 变动**(如 `with_structured_output`、prompt 导入路径):动手前 Context7 查当前文档;实现中报签名错误同样先查再改。
- **DeepSeek 对 JSON mode/function calling 的兼容性**:优先 function calling 路线;评估集不过关时查文档调整 method。
- **SSE 被缓冲**:开发环境直连 uvicorn 无代理;响应头加 `Cache-Control: no-cache`、`X-Accel-Buffering: no` 兜底。
- **中文 token 估算偏差**:估算宁大勿小 + 预算本身保守(3000)。

## 12. 决策记录

| 决策点 | 结论 | 理由 |
|--------|------|------|
| 模型接入 | 方案 A:ChatOpenAI 统一接入 | 契合"应用侧统一 OpenAI 协议、换厂商只改 .env";用户确认 |
| 会话形态 | 服务端内存会话(session_id) | curl 两轮验收最直观,聊天页接入最顺;用户确认 |
| 聊天页 | 本章交付,最小版 | 用户确认"本章就带最小聊天页" |
| 上游 | DeepSeek(key 已验证连通,服务端型号 deepseek-v4-flash) | 环境无其他可用上游;用户提供 key |
