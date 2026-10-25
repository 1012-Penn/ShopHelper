# ch14 后台管理鉴权设计

- 日期：2026-10-25
- 状态：已实施
- 范围：ShopHelper 后台管理页面与管理接口的访问控制
- 工作分支：`worktree-ch14-admin-auth`

## 1. 背景与目标

ch10 收口时，`/admin/review`、`/admin/evals`、`/admin/topics` 三个页面与 review queue / topics / eval-runs / usage 共 8 个接口全部无鉴权；README 明确标注为本地演示系统。ch14 总纲要求对管理、审核、知识写入及工具写操作加入身份与权限控制，并记录谁批准了什么。

本次目标是引入可验收的后台访问控制：未登录 401/跳登录、权限不足 403、合法管理员 200；写操作留审计。聊天、检索、知识库写入逻辑、评测集、LangGraph 状态与 ch11 文件不在本次范围内。

## 2. 方案选择

比较过三条路线：

| 方案 | 身份 | 登录 | 权限 | 取舍 |
|---|---|---|---|---|
| A | env 单管理员 | HTTP Basic | 无角色 | 最轻，但无法验证 403、无登出/审计 |
| **B（采用）** | **DB `admin_users`** | **登录页 + 签名 Cookie** | **viewer/operator/admin** | 可验收、可审计；需要迁移与 bootstrap |
| C | 反向代理/OIDC | SSO | IdP 分组 | 最强，但引入外部服务与 TLS/回调配置 |

选择 B：单机/多实例都可共享 MySQL 与同一会话密钥，测试可用 SQLite 替身；不新增第三方依赖。

## 3. 身份、登录与权限边界

### 3.1 管理员账号

`admin_users`：`username`（小写归一、唯一）、`password_hash`（PBKDF2-HMAC-SHA256，默认 600,000 次迭代）、`role`、`is_active`、`failed_login_count`、`locked_until`、`last_login_at`、`session_version`、时间戳。

角色：

- `viewer`：后台页面 + 全部只读 API。
- `operator`：viewer + review approve/reject + topics classify-batch。
- `admin`：operator + `GET /api/admin/audit`。

### 3.2 登录与会话

- `POST /api/admin/login`（JSON：`username`、`password`、`next`）。
- 口令校验使用常量时间比较；未知用户名做 dummy PBKDF2，降低枚举时序差。
- 连续失败达到阈值写 `locked_until`；未知用户/错误口令/锁定/禁用统一返回 401“用户名或密码错误”。
- 成功写 `last_login_at`，生成 CSRF，签发 HMAC-SHA256 会话。
- Cookie：`shophelper_admin_session`，`HttpOnly`、`SameSite=Lax`、`Path=/`、`Max-Age=TTL`、`Secure` 由 `ADMIN_COOKIE_SECURE` 控制。
- 每次请求验签、验过期、查库校验 `is_active` 和 `session_version`；角色只取 DB 值。
- `POST /api/admin/logout` 要求 CSRF，清 Cookie。
- 改密/启停增加 `session_version`，吊销全部旧会话；登出只清 Cookie（单会话撤销留作后续）。

### 3.3 CSRF 与来源校验

- 写接口要求 `X-CSRF-Token` 与 session 内 `csrf` 做 `hmac.compare_digest`。
- `Origin` 存在时比较 hostname/port（忽略 scheme，兼容 TLS 终止代理）；`Origin: null` 拒绝。
- 登录 POST 同样校验 `Origin`，防登录 CSRF。

### 3.4 页面与接口保护

- 页面未登录：`303 /admin/login?next=<path+query>`；角色不足：403；通过：`FileResponse`。
- API 未登录：401 `{"detail":"请先登录"}`；角色不足：403 `{"detail":"权限不足"}`；CSRF 不符：403。
- 公开路由：`/admin/login`、`/admin/assets/admin.js`。
- 前端共享 `static/admin-auth.js`：会话获取、CSRF 注入、401 跳登录、403 提示、退出、只读态。

## 4. 审计

`admin_audit_logs`（append-only，不挂外键）：操作人 id/username/role 快照、action、target_type/target_id、result（成功/失败）、detail JSON、ip、user_agent、created_at。

写入点：

- `admin.login` / `admin.login_failed` / `admin.logout`
- `review.approve`（detail 含 review id、chunk id、答案字符数、答案 SHA-256，不存全文）
- `review.reject`
- `topics.classify_batch`

审计写入是路由层 best-effort：失败只 warning，不阻断业务。与业务提交非同一事务，极端情况下会丢一条审计；强一致审计留作后续 outbox/hook 方案。

管理员查询：`GET /api/admin/audit?limit=50&offset=0&action=&username=&result=`。

## 5. 接口契约摘要

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| POST | `/api/admin/login` | 公开 | 登录，成功 Set-Cookie |
| POST | `/api/admin/logout` | 已登录 + CSRF | 清 Cookie |
| GET | `/api/admin/session` | viewer+ | username/role/csrf/can_write |
| GET | `/api/admin/audit` | admin | 审计查询 |
| GET | `/api/review-queue`、`/{id}` | viewer+ | 只读 |
| POST | `/api/review-queue/{id}/approve`、`/reject` | operator+ + CSRF | 写知识库/驳回 |
| GET | `/api/topics/distribution` | viewer+ | 只读 |
| POST | `/api/topics/classify-batch` | operator+ + CSRF | 批量归类 |
| GET | `/api/eval-runs` | viewer+ | 只读 |
| GET | `/api/usage/by-intent` | viewer+ | 只读 |

## 6. 数据模型与迁移

- `db/init.sql`：全新库直接建 `admin_users` / `admin_audit_logs`。
- `db/ch14-admin-auth.sql`：旧库增量，`SET NAMES utf8mb4`、`CREATE TABLE IF NOT EXISTS`、InnoDB/utf8mb4/中文 COMMENT。
- SQLite 测试由 `Base.metadata.create_all` 自动建表；id 使用 `BigInteger().with_variant(Integer, "sqlite")`。
- 首个管理员由 `python -m scripts.adminctl create --username ... --role ...` 创建；CLI 不提供明文 `--password`。

## 7. 安全取舍与已知边界

- 仅账号级失败锁定，无 IP/分布式限流；已知用户名可被恶意锁定。
- 无 MFA/SSO/管理员管理 UI/审计页面/口令找回 UI。
- 本地默认 `ADMIN_COOKIE_SECURE=false`；生产必须 true + HTTPS。
- `ADMIN_SESSION_SECRET` 未设置时后台登录 503；多 worker 必须共享同一 secret。
- IP 取 `request.client.host`，不信任 `X-Forwarded-For`。
- 审计表随登录事件增长，本次不做清理/归档。
- 终端用户 API 仍未鉴权；本任务只覆盖后台管理面。
- 不修改聊天、检索、知识库写入、评测集、LangGraph 与 ch11 文件。

## 8. 验收

- 未登录：3 个页面 303 到登录页；8 个管理 API 401。
- 权限不足：viewer 写操作 403；operator 访问审计 API 403。
- 合法管理员：登录后页面 200、只读/写 API 200；缺 CSRF 403。
- 会话：篡改/过期/`session_version` 变化/禁用用户旧 Cookie → 401。
- 审计：approve/reject/classify-batch 与登录事件落库；审计失败不阻断业务。
- 回归：`test_ch09_review_api.py`、`test_topic_classifier.py` 带管理员会话通过。

测试命令：

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_ch14_admin_auth.py -q
PYTHONPATH=. .venv/bin/pytest -q
```

不运行 ch09/ch10 真机验收（会连真实服务，可能产生费用）；只为验收脚本补管理员登录兼容。
