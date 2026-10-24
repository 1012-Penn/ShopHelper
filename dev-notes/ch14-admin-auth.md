# ch14 开发札记：后台管理鉴权

## 0. 工作区

- worktree：`/root/workplace/ShopHelper/.claude/worktrees/ch14-admin-auth`
- 分支：`worktree-ch14-admin-auth`
- 起始 HEAD：`a3060de`（ch10 收口）
- 起始工作区：干净
- ch11 完整工作在另一条线并行；本分支未改任何 ch11 文件，也未合并 ch11。

## 1. 决策记录

### 1.1 方案比较

| 方案 | 身份/登录 | 权限 | 结论 |
|---|---|---|---|
| A | env 单管理员 + HTTP Basic | 无角色 | 无法验证 403，无登出/审计 |
| **B（采用）** | DB `admin_users` + 登录页 + 签名 Cookie | viewer/operator/admin | 可验收、可审计；需迁移与 bootstrap |
| C | 反向代理/OIDC/SSO | IdP 分组 | 最强，但引入外部服务与 TLS/回调配置，本阶段不做 |

### 1.2 本次确定的细节

- 用户名统一 `strip().lower()`，规避 MySQL/SQLite 大小写差异。
- 登录失败统一 401“用户名或密码错误”，不区分未知用户/错误口令/锁定/禁用。
- 会话 8 小时；PBKDF2-HMAC-SHA256 默认 600,000 次迭代（测试注入 2,000）。
- Cookie：HttpOnly + SameSite=Lax + Path=/；`Secure` 由 `ADMIN_COOKIE_SECURE` 控制。
- 登出只清 Cookie；改密/启停增加 `session_version` 吊销全部旧会话。
- 写接口要求 `X-CSRF-Token`，并校验 Origin；登录 POST 也校验 Origin。
- 审计为路由层 best-effort：业务成功后写，失败只 warning；不改 `FlywheelService`/`TopicStore`。
- 第一个可验收交付只做“审计写入 + admin-only 查询 API”，不新增审计页面。
- ch09/ch10 验收脚本补 `ACCEPT_ADMIN_USER` / `ACCEPT_ADMIN_PASSWORD` 登录兼容。

## 2. 实现摘要

### 2.1 新增模块

- `app/admin_security.py`：PBKDF2 口令哈希/校验、HMAC-SHA256 会话签名、CSRF、`safe_next_path`；纯标准库。
- `app/admin_store.py`：`AdminStore`（登录/锁定/启停/改密/解锁/DTO）与 `AdminAuditStore`（审计写入/查询/截断）。
- `app/admin_auth.py`：`AdminPrincipal`、`read_principal`、`require_admin`、`require_role`、`require_csrf`、`page_guard`、Origin 校验、`audit_context`。
- `app/routers/admin.py`：`POST /api/admin/login`、`POST /api/admin/logout`、`GET /api/admin/session`、`GET /api/admin/audit`。
- `scripts/adminctl.py`：create/list/disable/enable/reset-password/unlock；禁止禁用最后一个启用中的管理员。
- `db/ch14-admin-auth.sql`：旧库增量；`db/init.sql` 同步新表。
- `static/admin-login.html`、`static/admin-auth.js`：登录页与共享前端鉴权。
- `tests/test_ch14_admin_auth.py`：安全原语/Store/HTTP 鉴权/审计测试。

### 2.2 修改

- `app/models.py`：`AdminUser`、`AdminAuditLog`。
- `app/config.py`、`.env.example`：ch14 配置。
- `app/main.py`：装配 admin stores、注册 admin router、页面守卫、公开登录页与 JS 路由。
- `app/routers/review_queue.py`、`topics.py`、`evaluation.py`、`usage.py`：角色依赖 + CSRF + 审计。
- `static/review-queue.html`、`topics.html`、`eval-runs.html`：会话、CSRF、401/403、只读态、退出。
- `tests/conftest.py`：测试 secret、低迭代次数、**重绑 admin stores 到 SQLite factory**。
- `tests/helpers.py`：`seed_admin` / `admin_login` / `csrf_headers`。
- `tests/test_ch09_review_api.py`、`tests/test_topic_classifier.py`：管理员会话 + CSRF。
- `scripts/acceptance_ch09.py`、`acceptance_ch10.py`：管理员登录兼容。
- `README.md`、spec/plan 文档。

### 2.3 明确未改

`app/flywheel.py`、`app/store.py`、`app/topic_classifier.py`、`app/kb.py`、聊天/检索/LangGraph/评测集/ch11 文件。

## 3. 验证记录

### 3.1 全量测试

```bash
PYTHONPATH=. /root/workplace/ShopHelper/.venv/bin/pytest -q
```

结果：**433 passed, 1 xfailed**（4 条既有 pytest 警告；退出时 Milvus Lite 关闭日志有 I/O 已关闭提示，非测试失败）。

### 3.2 本次目标测试

```bash
PYTHONPATH=. /root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_ch09_review_api.py tests/test_topic_classifier.py \
  tests/test_ch14_admin_auth.py -q
```

结果：**70 passed**。

### 3.3 语法与 CLI

- `python -m py_compile` 通过全部改动 Python 文件。
- `python -m scripts.adminctl --help` / `create --help` 正常。
- 临时 SQLite 上验证：create → list → disable 最后一个管理员被拒绝 → reset-password → unlock，均符合预期。

### 3.4 未运行的验证

- 未运行 `scripts/acceptance_ch09.py` / `acceptance_ch10.py`：会连真实服务，可能产生费用；只更新了登录兼容代码。
- 未在真实 MySQL 上 apply 迁移：测试使用 SQLite；部署前需手工执行 `db/ch14-admin-auth.sql`。
- 未做浏览器端到端点击：前端只通过页面状态码、脚本引用与接口契约验证；登录页/三个后台页需用户本地点开确认视觉与交互。

## 4. 已知边界

- 审计与业务提交非同一事务；极端情况下业务成功但审计写失败（只 warning）。强一致审计需后续 outbox/hook。
- 仅账号级失败锁定；无 IP/分布式限流、MFA、SSO、管理员管理 UI、审计页面、口令找回 UI。
- 登出只清 Cookie；被窃 token 到期前仍有效；改密/禁用可通过 `session_version` 吊销。
- 本地默认 `ADMIN_COOKIE_SECURE=false`；生产必须 `true` + HTTPS。
- `ADMIN_SESSION_SECRET` 为空时后台登录 503；多 worker 必须共享同一 secret。
- IP 取 `request.client.host`，不信任 `X-Forwarded-For`；审计表随登录事件增长，本次不做清理。
- 终端用户 API（聊天、会话、订单、反馈、工单）仍未鉴权，属于本任务边界之外。

## 5. 交付信息

- 提交信息：`ch14: 后台管理鉴权（DB 管理员/角色/会话/审计）`
- 提交号：由本次交付报告给出（不 push、不 merge）。
- 未运行付费评测；未自动重试失败任务。
