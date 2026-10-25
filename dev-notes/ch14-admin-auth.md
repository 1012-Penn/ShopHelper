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

## 6. 集成到最新 origin/main（2026-10-25）

### 6.1 起点与 worktree

- CH14 原分支：`worktree-ch14-admin-auth`，开始前核对 tip 为 `8ea3195`，worktree 干净；集成过程中未修改、未删除该 worktree 和分支。
- 集成前重新 fetch `origin/main`，基线为 `d586796`（Merge CH11 evaluation baseline）。
- 独立 worktree：`/root/.codex/worktrees/ch14-admin-auth-integration/ShopHelper`。
- 集成分支：`codex/ch14-admin-auth-integration`，从 `origin/main` 创建后合入 `worktree-ch14-admin-auth`。
- 主工作目录原有未提交文件未触碰。

### 6.2 冲突处理

Git 报出的内容冲突只有两处：

1. `app/routers/review_queue.py`
   - `main` 已加入 `_approval_result()`、`index_status` 响应和 `POST /{id}/retry-index` 可靠性修复；CH14 已加入角色/CSRF 依赖与审计。
   - 解决：保留可靠性接口契约；`approve` 先取得索引状态，再把 `knowledge_chunk_id`、`index_status`、答案长度与 SHA-256 写入审计并返回原契约字段。
   - `retry-index` 保留并补齐 `operator` 角色 + CSRF 校验，成功重试后写 `review.retry_index` 审计；没有放宽接口权限。
2. `static/review-queue.html`
   - `main` 已加入“待审/已通过”筛选、索引 pending/done 区分、重试按钮、异步切换旧表单防串写；CH14 已加入 `AdminAuth` 会话/CSRF/401/403/只读态。
   - 解决：两套行为合并；所有请求走 `AdminAuth.requestJSON()`，写请求自动带 `X-CSRF-Token`，保留状态筛选、索引重试、`viewVersion` 竞态保护与失败时答案不丢。
   - 页首仍加载公共 `/admin/assets/admin.js`，登录态初始化后才加载队列。

语义重点检查：

- `app/models.py`：`main` 的可靠性/CH11 模型字段全部保留，CH14 的 `AdminUser`、`AdminAuditLog` 追加在其后；未覆盖任何既有模型。
- `db/init.sql`：在 `main` 版基础上仅追加 `admin_users` / `admin_audit_logs`；`db/ch14-admin-auth.sql` 仍为旧库增量迁移。
- `tests/helpers.py` / `tests/conftest.py`：保留 CH11/reliability 测试辅助；CH14 的 `seed_admin`、`admin_login`、`csrf_headers` 与 SQLite admin store 重绑继续生效。
- 相关既有接口测试改为真实管理员登录并携带 CSRF；没有跳过或削弱鉴权断言。

### 6.3 设计取舍

- `retry-index` 属于写操作，因此要求 `operator`/`admin` + CSRF，并记录 `review.retry_index` 审计；这补上了可靠性新接口在 CH14 边界内的权限与审计缺口。
- 审计仍在业务成功后 best-effort 写入；不把业务事务与审计强耦合，保持原 CH14 设计，极端情况下可能丢一条审计。
- 审计 detail 只存答案长度和 SHA-256，不存核准答案全文。
- 前端不复制鉴权逻辑：CSRF 注入、401 跳登录、403 提示、只读态统一由 `static/admin-auth.js` 提供。

### 6.4 测试命令与结果

```bash
# 目标回归
PYTHONPATH=. /root/workplace/ShopHelper/.venv/bin/pytest \
  tests/test_ch14_admin_auth.py tests/test_review_reliability.py \
  tests/test_review_queue_ui.py tests/test_ch09_review_api.py \
  tests/test_topic_classifier.py -q
# 79 passed

# 全量
PYTHONPATH=. /root/workplace/ShopHelper/.venv/bin/pytest -q
# 659 passed, 2 skipped, 1 xfailed, 4 warnings in 77.50s
# 2 skipped 为未设置 SHOPHELPER_TEST_MYSQL_URL 时跳过的可选真实 MySQL 用例

# 审核页 DOM 行为
node --test tests/ui/review_queue.test.cjs
# 6 passed，包含 pending/重试/失败保留输入/旧表单防串写/
# AdminAuth 初始化/CSRF 头/只读角色禁用写按钮断言

# 真实 MySQL 回归（临时 shophelper_test_* 库，结束后删除）
SHOPHELPER_TEST_MYSQL_URL='mysql+pymysql://.../shophelper_test_*?charset=utf8mb4' \
  PYTHONPATH=. /root/workplace/ShopHelper/.venv/bin/pytest tests/test_review_mysql.py -q
# 2 passed
```

MySQL 增量迁移安全验证（MySQL 8.4，临时库，验证后删除）：

- 先应用 `origin/main:db/init.sql` 模拟旧库，再应用 `db/ch14-admin-auth.sql`，随后重复应用一次验证幂等。
- 校验 `admin_users` / `admin_audit_logs` 表、关键列类型、`JSON` detail、ENUM 中文结果值。
- 写入一条 operator 与一条 `review.retry_index` 审计，联表查询成功。
- 结论：迁移在本地 MySQL 8.4 可应用且可重复执行；未触碰现有 `shophelper` 数据库。

后台登录页实际浏览器交互（headless Chromium + DevTools Protocol，临时 MySQL）：

- 错误口令：页面显示“用户名或密码错误”。
- 正确口令：跳转 `/admin/review`，页面标题与列表正常，`/api/admin/session` 返回 `operator`、`can_write=true` 且 CSRF 存在，`/api/review-queue` 返回 200。
- 点击退出：CSRF 保护的 `POST /api/admin/logout` 成功并跳回 `/admin/login`。
- 本地完整启动需先应用 `db/reliability-review-index.sql`，再应用 `db/ch14-admin-auth.sql`；已按此顺序验证。
- 未点击通过/驳回/重试索引，因为真实 app 会调用嵌入/向量服务；这些写路径由 fake 替身测试覆盖。

### 6.5 剩余风险与未验证事项

- 未运行 `scripts/acceptance_ch09.py` / `acceptance_ch10.py`；未跑任何付费评测、未调用付费模型。
- 未在现有生产/开发 MySQL 数据卷上执行迁移，只验证了临时 MySQL 8.4 新库 + 增量路径；上线前仍需备份并具备回滚方案。
- 浏览器只做 headless Chromium 冒烟，未覆盖多浏览器、移动端和视觉像素级回归。
- 审核写操作的真实外部嵌入/向量化未在浏览器中点验；当前依赖 fake 替身与 MySQL/SQLite 契约测试。
- 审计为 best-effort、与业务提交非同一事务；登出只清 Cookie、无服务端单会话撤销；生产必须设置 `ADMIN_SESSION_SECRET`、HTTPS 与 `ADMIN_COOKIE_SECURE=true`。
- 集成分支按要求未 push、未合入 `main`、未删除原 CH14 分支或 worktree。
