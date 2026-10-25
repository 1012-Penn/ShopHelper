-- =============================================================
-- ch12 · 受控 Agentic RAG · 存量库迁移(新库直接用 init.sql,不需要本文件)
-- 1) tickets 加 confirm_key 列 + 唯一约束(工单幂等,ADR-3)
-- 2) 新建 rag_loop_steps(检索步骤审计,仅可审计,ADR-1)
-- 幂等执行:先查 information_schema 再 ALTER/CREATE
-- =============================================================
SET NAMES utf8mb4;

SET @has_confirm_key = (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'tickets'
    AND COLUMN_NAME = 'confirm_key');
SET @sql = IF(@has_confirm_key = 0,
  'ALTER TABLE tickets ADD COLUMN confirm_key VARCHAR(64) NULL COMMENT ''一次用户确认的幂等键;同键重复提交返回同一工单'' AFTER conversation_id, ADD UNIQUE KEY uk_confirm_key (confirm_key)',
  'SELECT ''tickets.confirm_key 已存在'' AS info');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

CREATE TABLE IF NOT EXISTS rag_loop_steps (
  id                 BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '审计主键',
  conversation_id    BIGINT UNSIGNED NOT NULL                COMMENT '所属会话',
  turn_user_msg_id   VARCHAR(64)     NOT NULL                COMMENT '本轮 user 消息 id(initial_state 生成),一轮聊天内稳定',
  round_index        INT             NOT NULL                COMMENT '0=首轮检索,1..N=补检索轮',
  phase              VARCHAR(16)     NOT NULL                COMMENT 'retrieve / final(全部轮次后的终判)',
  query              TEXT            NOT NULL                COMMENT '本轮主查询(用户原问题)',
  queries            JSON            NULL                    COMMENT '本轮实际执行的查询(含子查询)',
  sub_queries        JSON            NULL                    COMMENT '缺口分析拆解出的子查询(未去重原始输出)',
  evidence_ids       JSON            NULL                    COMMENT '本轮各查询返回的 chunk id',
  new_evidence_ids   JSON            NULL                    COMMENT '本轮新并入合并集的 chunk id',
  merged_evidence_ids JSON           NULL                    COMMENT '本轮结束后的合并 top-k chunk id',
  judge_answered     TINYINT(1)      NULL                    COMMENT '蕴含判断器结论:1 回答了 / 0 没回答 / NULL 未判',
  analyzer_action    VARCHAR(24)     NULL                    COMMENT '缺口分析动作:retrieve_more/clarify/refuse/escalate/answer',
  conflicts          JSON            NULL                    COMMENT '识别到的证据冲突(逐条)',
  missing_aspects    JSON            NULL                    COMMENT '识别到的证据缺口(逐条)',
  stop_reason        VARCHAR(32)     NULL                    COMMENT '循环终止原因,仅本轮最后一行填',
  latency_ms         INT UNSIGNED    NULL                    COMMENT '本轮耗时毫秒',
  created_at         DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_rag_step_round (conversation_id, turn_user_msg_id, round_index, phase),
  KEY idx_rag_turn (conversation_id, turn_user_msg_id),
  KEY idx_rag_stop (stop_reason)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch12 受控 Agentic 检索步骤审计(仅可审计,不支持断点续跑)';
