-- =============================================================
-- ch14 · 后台管理鉴权增量迁移
-- 新建:admin_users(管理员账号) / admin_audit_logs(操作审计)
-- 说明:
--   * 旧库手工 apply 本文件;全新库由 db/init.sql 一次建好。
--   * 审计表 append-only,不挂外键;操作人字段保存快照,账号删除后仍可追溯。
--   * 角色:viewer 只读 / operator 可审核与归类 / admin 全部权限。
--   * 首次使用前用 scripts.adminctl 创建第一个管理员。
-- =============================================================

SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS admin_users (
  id                 BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '管理员主键',
  username           VARCHAR(64)     NOT NULL                COMMENT '登录名,小写归一后唯一',
  password_hash      VARCHAR(255)    NOT NULL                COMMENT 'pbkdf2_sha256$迭代次数$salt_b64$hash_b64',
  role               ENUM('viewer','operator','admin') NOT NULL DEFAULT 'viewer'
                                                             COMMENT 'viewer 只读 / operator 可写 / admin 全部',
  is_active          TINYINT(1)      NOT NULL DEFAULT 1      COMMENT '是否启用',
  failed_login_count INT UNSIGNED    NOT NULL DEFAULT 0      COMMENT '连续登录失败次数',
  locked_until       DATETIME        NULL                    COMMENT '锁定截止时间(UTC)',
  last_login_at      DATETIME        NULL                    COMMENT '最近成功登录时间(UTC)',
  session_version    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '会话版本;改密/启停时+1使旧 cookie 失效',
  created_at         DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at         DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                                                             COMMENT '更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_admin_username (username),
  KEY idx_admin_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='后台管理员账号';

CREATE TABLE IF NOT EXISTS admin_audit_logs (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '审计主键',
  admin_user_id BIGINT UNSIGNED NULL                    COMMENT '操作人 id;未知账号登录失败为空',
  username      VARCHAR(64)     NOT NULL                COMMENT '操作人登录名快照',
  role          VARCHAR(16)     NOT NULL                COMMENT '操作时角色快照',
  action        VARCHAR(64)     NOT NULL                COMMENT '动作,如 review.approve/admin.login',
  target_type   VARCHAR(32)     NOT NULL                COMMENT '目标类型,如 review_queue/topic_pool/admin_session',
  target_id     VARCHAR(64)     NULL                    COMMENT '目标 id,字符串以容纳非数字目标',
  result        ENUM('成功','失败') NOT NULL DEFAULT '成功' COMMENT '操作结果',
  detail        JSON            NULL                    COMMENT '补充信息;不存口令/核准答案全文',
  ip            VARCHAR(64)     NULL                    COMMENT '请求来源 IP',
  user_agent    VARCHAR(255)    NULL                    COMMENT 'User-Agent 截断',
  created_at    DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '发生时间',
  PRIMARY KEY (id),
  KEY idx_admin_audit_user (admin_user_id),
  KEY idx_admin_audit_action (action),
  KEY idx_admin_audit_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='后台管理操作审计,append-only';
