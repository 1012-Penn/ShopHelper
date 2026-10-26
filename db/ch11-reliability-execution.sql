-- CH11 同会话可靠执行：持久 run、SSE 事件、会话租约、工具尝试。
-- 独立迁移文件；集成时可将本文件内容追加到 db/init.sql。
SET NAMES utf8mb4;

CREATE TABLE chat_runs (
  run_id            VARCHAR(36)     NOT NULL,
  conversation_id   BIGINT UNSIGNED NOT NULL,
  client_request_id VARCHAR(128)    NULL,
  request_hash      CHAR(64)        NOT NULL,
  request_json      JSON            NOT NULL,
  status            VARCHAR(24)     NOT NULL DEFAULT 'queued',
  result_json       JSON            NULL,
  error_message     TEXT            NULL,
  write_state       VARCHAR(24)     NOT NULL DEFAULT 'none',
  owner_token       VARCHAR(64)     NOT NULL,
  attempt           INT UNSIGNED    NOT NULL DEFAULT 1,
  last_event_seq    INT UNSIGNED    NOT NULL DEFAULT 0,
  input_tokens      INT UNSIGNED    NOT NULL DEFAULT 0,
  output_tokens     INT UNSIGNED    NOT NULL DEFAULT 0,
  total_tokens      INT UNSIGNED    NOT NULL DEFAULT 0,
  duration_ms       INT UNSIGNED    NOT NULL DEFAULT 0,
  created_at        DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  started_at        DATETIME        NULL,
  finished_at       DATETIME        NULL,
  updated_at        DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP
                                     ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id),
  UNIQUE KEY uk_chat_run_client_request (conversation_id, client_request_id),
  KEY idx_chat_run_conversation (conversation_id, created_at),
  KEY idx_chat_run_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='聊天执行 run 的持久状态与用量';

CREATE TABLE chat_run_events (
  id         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  run_id     VARCHAR(36)     NOT NULL,
  seq        INT UNSIGNED    NOT NULL,
  event_type VARCHAR(64)     NOT NULL,
  payload    JSON            NOT NULL,
  created_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_chat_run_event_seq (run_id, seq),
  KEY idx_chat_event_replay (run_id, id),
  CONSTRAINT fk_chat_event_run FOREIGN KEY (run_id)
    REFERENCES chat_runs (run_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='SSE 持久事件；id 为 Last-Event-ID 游标';

CREATE TABLE chat_session_leases (
  conversation_id BIGINT UNSIGNED NOT NULL,
  run_id          VARCHAR(36)     NOT NULL,
  owner_token     VARCHAR(64)     NOT NULL,
  lease_expires_at DATETIME       NOT NULL,
  updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP
                                  ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (conversation_id),
  UNIQUE KEY uk_chat_lease_run (run_id),
  KEY idx_chat_lease_expiry (lease_expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='同会话单执行者租约；过期可恢复';

CREATE TABLE chat_tool_attempts (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  run_id          VARCHAR(36)     NOT NULL,
  tool_call_id    VARCHAR(64)     NOT NULL,
  tool_name       VARCHAR(128)    NOT NULL,
  access          VARCHAR(8)      NOT NULL,
  status          VARCHAR(24)     NOT NULL,
  idempotency_key VARCHAR(128)    NULL,
  arguments       JSON            NULL,
  result_text     TEXT            NULL,
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  finished_at     DATETIME        NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uk_chat_tool_call (run_id, tool_call_id),
  KEY idx_chat_tool_status (run_id, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工具开始/成功/未知状态，写前留痕';
