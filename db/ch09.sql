-- ch09 task 1：可独立应用的 request_usage 增量迁移。
CREATE TABLE request_usage (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NOT NULL,
  trace_id        VARCHAR(128)    NULL,
  intent          VARCHAR(64)     NOT NULL DEFAULT '',
  input_tokens    INT UNSIGNED    NOT NULL DEFAULT 0,
  output_tokens   INT UNSIGNED    NOT NULL DEFAULT 0,
  total_tokens    INT UNSIGNED    NOT NULL DEFAULT 0,
  duration_ms     INT UNSIGNED    NOT NULL DEFAULT 0,
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_request_usage_intent (intent),
  KEY idx_request_usage_created_at (created_at),
  CONSTRAINT fk_request_usage_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch09 每轮请求 token 与耗时统计';
