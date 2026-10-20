-- 审核幂等与可重试索引。先完成 ch09-observability-flywheel.sql,再应用本迁移。
-- 一次性增量迁移;升级前停止旧服务的审核写入,迁移后再启动新版。
SET NAMES utf8mb4;

ALTER TABLE knowledge_chunks
  ADD COLUMN review_id BIGINT UNSIGNED NULL COMMENT '来源审核;每条审核最多关联一个知识块',
  ADD UNIQUE KEY uk_knowledge_review (review_id),
  ADD CONSTRAINT fk_knowledge_review FOREIGN KEY (review_id)
    REFERENCES review_queue (id) ON DELETE SET NULL;

-- 旧版本失败重试可能已有重复;只关联最早的 id,其余块原样保留,不删除历史数据。
UPDATE knowledge_chunks AS target
JOIN (
  SELECT review.id AS review_id, MIN(chunk.id) AS chunk_id
  FROM review_queue AS review
  JOIN knowledge_chunks AS chunk
    ON chunk.section_path = CONCAT('review_queue:', review.id)
  GROUP BY review.id
) AS legacy ON target.id = legacy.chunk_id
SET target.review_id = legacy.review_id;
