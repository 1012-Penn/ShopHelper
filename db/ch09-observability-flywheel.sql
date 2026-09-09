-- ch09 · 可观测性与数据飞轮
-- 连接客户端必须以 utf8mb4 执行,否则中文 ENUM/COMMENT 可能双重编码。
SET NAMES utf8mb4;

CREATE TABLE review_queue (
  id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '缺口主键,也是查重命中要返回的 matched_question_id',
  normalized_question VARCHAR(512)    NOT NULL                COMMENT '标准化后的 FAQ 式问题',
  ai_suggested_answer TEXT            NULL                    COMMENT '模型生成的示例答案,备查',
  occurrence_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '出现次数,查重命中累加',
  review_status       ENUM('待审','通过','驳回') NOT NULL DEFAULT '待审' COMMENT '人工审核状态',
  approved_answer     TEXT            NULL                    COMMENT '审核通过时补的核准答案',
  created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_review_status (review_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='飞轮待审队列';

CREATE TABLE eval_runs (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '评估轮次主键',
  triggered_by ENUM('定时','手动') NOT NULL DEFAULT '定时' COMMENT '触发方式',
  dataset_size INT UNSIGNED    NOT NULL COMMENT '评估集条数',
  metrics      JSON            NOT NULL COMMENT 'Recall@K/MRR/Faithfulness 等指标 JSON',
  created_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自动化评估流水线轮次结果';

ALTER TABLE low_confidence_questions
  ADD COLUMN matched_review_id BIGINT UNSIGNED NULL COMMENT '查重后归并到 review_queue 的缺口行',
  ADD COLUMN retrieved_chunks JSON NULL COMMENT '落池时 Top-K 召回/精排片段快照',
  ADD KEY idx_matched_review_id (matched_review_id),
  ADD CONSTRAINT fk_lcq_review FOREIGN KEY (matched_review_id)
    REFERENCES review_queue (id) ON DELETE SET NULL;
