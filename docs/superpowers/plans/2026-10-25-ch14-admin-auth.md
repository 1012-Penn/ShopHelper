# ch14 后台管理鉴权实施计划

- 日期：2026-10-25
- 设计稿：`docs/superpowers/specs/2026-10-25-ch14-admin-auth-design.md`
- 分支：`worktree-ch14-admin-auth`

## 交付范围

DB 管理员表 + 角色 + 签名 Cookie 会话 + CSRF + 审计；保护 3 个后台页面与 8 个管理接口；提供 admin-only 审计查询 API。不新增审计页面，不改聊天/检索/知识库写入/评测集/LangGraph/ch11。

## 实施步骤

### 1. 配置、模型、迁移

- [x] `app/config.py` 增加 `admin_*` 配置。
- [x] `app/models.py` 增加 `AdminUser`、`AdminAuditLog`。
- [x] `db/ch14-admin-auth.sql`、`db/init.sql` 增加两张表。
- [x] `.env.example` 增加 ch14 配置块。
- 验证：导入新模型；SQLite `create_all` 可建表。

### 2. 安全原语

- [x] `app/admin_security.py`：PBKDF2、HMAC 会话、CSRF、`safe_next_path`。
- 验证：哈希 roundtrip/损坏输入；会话篡改/过期/换 secret；`next` 白名单。

### 3. Store 与审计 Store

- [x] `app/admin_store.py`：`AdminStore`、`AdminAuditStore`、`AdminUserView`、`LoginResult`。
- [x] 登录失败计数/锁定、禁用、`session_version`、审计截断与过滤。
- 验证：SQLite Store 单测。

### 4. 鉴权依赖与管理员 API

- [x] `app/admin_auth.py`：`AdminPrincipal`、`read_principal`、`require_admin`、`require_role`、`require_csrf`、`page_guard`、`audit_context`、Origin 校验。
- [x] `app/routers/admin.py`：login/logout/session/audit。
- [x] `app/main.py` 装配 store/router/login 页/静态 JS 路由。
- 验证：登录/登出/session/audit/503/CSRF/过期/篡改。

### 5. 保护页面与 API，接审计

- [x] `/admin/review|evals|topics` 加 `page_guard`。
- [x] review queue、topics、eval-runs、usage 加角色依赖。
- [x] approve/reject/classify-batch 写审计。
- [x] `tests/conftest.py` 重绑 admin stores 到 SQLite；`tests/helpers.py` 增加登录/CSRF 助手。
- 验证：401/403/200 矩阵；审计行字段；审计失败不阻断业务。

### 6. 更新既有测试

- [x] `tests/test_ch09_review_api.py`：operator 登录 + CSRF。
- [x] `tests/test_topic_classifier.py`：重绑 admin store + operator 登录 + CSRF；503 用例保留。
- 验证：目标测试全绿。

### 7. 前端

- [x] `static/admin-login.html`。
- [x] `static/admin-auth.js` + `GET /admin/assets/admin.js`。
- [x] `review-queue.html`、`topics.html`、`eval-runs.html`：会话、CSRF、401/403、只读态、退出。
- 验证：页面 200、脚本引用、登录页公开。

### 8. CLI 与文档

- [x] `scripts/adminctl.py`：create/list/disable/enable/reset-password/unlock；禁止禁用最后一个 active admin。
- [x] README 配置/迁移/bootstrap/角色/审计/边界说明。
- 验证：`--help`；临时 SQLite 上 create/list/disable/reset/unlock。

### 9. 验收脚本与收口

- [x] `scripts/acceptance_ch09.py`、`acceptance_ch10.py`：`ACCEPT_ADMIN_USER` / `ACCEPT_ADMIN_PASSWORD` 登录；写请求带 CSRF；未提供凭据时明确失败提示。
- [x] spec/plan/dev-notes。
- [ ] 最终全量 pytest + 单次 commit（不 push/merge）。

## 测试矩阵

- 未登录：3 页 303；8 API 401。
- viewer：只读 200；写 403。
- operator：写 200（带 CSRF）；缺 CSRF 403；审计 API 403。
- admin：审计 API 200。
- 会话：篡改/过期/版本变化/禁用 → 401；登出清 Cookie。
- 审计：登录/审核/归类事件落库；审计异常不阻断业务。
- 回归：ch09 review API、ch10 topic API 带管理员会话通过；全量 pytest。

## 边界

- 审计 best-effort，非业务强一致。
- 仅账号级锁定；无 IP 限流/MFA/SSO。
- 登出只清 Cookie；改密/禁用靠 `session_version` 吊销。
- 终端用户 API 仍未鉴权；不修改 ch11 文件。
