-- =============================================================
-- ShopHelper · 建表 DDL
-- ch01:—
-- ch02 新建:faq / conversations / messages / tickets 四张表
-- ch03 新建:knowledge_chunks / qa_extraction_staging
-- ch04 新建:low_confidence_questions / faith_cases
-- ch07 新建:conversation_summaries;conversations 加三列(summary / summary_upto_msg_id / layer1_from_msg_id)
-- ch12 新建:rag_loop_steps;tickets 加一列 confirm_key(幂等键)
-- ch14 新建:admin_users / admin_audit_logs(后台管理员与操作审计)
-- CH11 reliability 新建:chat_runs / chat_run_events / chat_session_leases / chat_tool_attempts
-- 商品、订单、物流走工具内 mock,不建表
-- 全库统一 ENGINE=InnoDB、CHARSET=utf8mb4
-- 建表顺序:先 conversations,再依赖它的 messages / tickets / low_confidence_questions
-- =============================================================

-- 入口脚本默认用 latin1 连接执行本文件,必须先声明字符集,否则 ENUM 中文会双层编码存坏
SET NAMES utf8mb4;

-- 会话壳:一通对话的统一身份,messages / tickets 都引用它
-- ch07 三列:summary=分段梗概的拼接投影;summary_upto_msg_id=摘要覆盖到哪条;layer1_from_msg_id=层1(原文)起点
CREATE TABLE conversations (
  id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话主键',
  user_id             VARCHAR(64)     NOT NULL                COMMENT '用户标识',
  status              ENUM('进行中','已转人工','已结束') NOT NULL DEFAULT '进行中' COMMENT '处理状态',
  summary             TEXT            NULL                    COMMENT '最近几段梗概拼成的投影,拼装时跟证据一起挂在用户那句之后',
  summary_upto_msg_id BIGINT UNSIGNED NULL                    COMMENT '摘要已覆盖到哪条消息,滑窗从其后接原文',
  layer1_from_msg_id  BIGINT UNSIGNED NULL                    COMMENT '层1(原文)起点;此 id 之后原样,之前渲染成半压形态',
  created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开启时间',
  updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客服会话';

-- 消息流水:一通会话底下挂 N 条,role 对齐 Chat Completions 协议
CREATE TABLE messages (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '消息主键',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '所属会话',
  role            ENUM('user','assistant','tool') NOT NULL COMMENT '角色:用户/助手/工具结果',
  content         TEXT            NULL                     COMMENT '消息正文,assistant 纯工具调用时可为空',
  tool_calls      JSON            NULL                     COMMENT 'assistant 消息带的工具调用申请单',
  tool_call_id    VARCHAR(64)     NULL                     COMMENT 'tool 消息对应的申请单 id,回灌时对号入座',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '产生时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话消息流水';

-- FAQ 问答对:query_faq 的数据源;ch03 起检索改走向量库,这张表退居原始录入
CREATE TABLE faq (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'FAQ 主键',
  question    VARCHAR(512)    NOT NULL                COMMENT '问题',
  answer      TEXT            NOT NULL                COMMENT '答案',
  category    VARCHAR(64)     NOT NULL                COMMENT '分类',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='常见问答';

-- 人工工单:create_ticket 落地,工单号当业务主键
-- ch12:confirm_key = 「一次用户确认」幂等键(ADR-3),唯一约束;NULL=旧客户端无键,不参与去重
CREATE TABLE tickets (
  ticket_no       VARCHAR(32)     NOT NULL                COMMENT '工单号,如 T20260701008',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '关联会话,可倒查当时聊了什么',
  confirm_key     VARCHAR(64)     NULL                    COMMENT '一次用户确认的幂等键;同键重复提交(双击/超时重试)返回同一工单',
  description     TEXT            NOT NULL                COMMENT '问题描述',
  ticket_type     ENUM('售后','投诉','咨询') NOT NULL     COMMENT '工单类型',
  status          ENUM('待处理','已处理') NOT NULL DEFAULT '待处理' COMMENT '处理状态',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (ticket_no),
  UNIQUE KEY uk_confirm_key (confirm_key),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_tickets_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人工工单';

-- =============================================================
-- ch03 · RAG 基础 · 建表 DDL
-- 本章新建:knowledge_chunks(知识库原文权威源)
-- 向量落 Milvus 集合 knowledge(非 MySQL,DDL 不含);MySQL 存原文 + 双写状态
-- category + questions + answer 三格拼成向量化文本;其余字段是元数据,只存不进向量
-- =============================================================

CREATE TABLE knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',
  category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',
  questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',
  answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',
  section_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',
  content_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual 等,元数据',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,元数据',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,元数据',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',
  source_ref       VARCHAR(255)    NULL                    COMMENT '来源会话 / 导出文件标识,溯源用,不入最终知识库',
  question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',
  answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '抽取写入时间',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可清空';

-- =============================================================
-- ch04 · RAG 进阶 · 建表 DDL
-- 本章新建:low_confidence_questions(低置信度问题池)/ faith_cases(忠实度编造个案台账)
-- =============================================================

CREATE TABLE low_confidence_questions (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '来源会话',
  raw_question    TEXT            NOT NULL                COMMENT '用户原话,带情绪口语',
  source          ENUM('retrieval_low_conf','self_check','user_feedback') NOT NULL COMMENT '入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决',
  reason          TEXT            NULL                    COMMENT '判不能的原因,留作复盘',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入池时间',
  PRIMARY KEY (id),
  KEY idx_source (source),
  KEY idx_created_at (created_at),
  CONSTRAINT fk_lcq_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题池';

-- ch04 编造个案台账:忠实度裁判判出的每条编造,不只留在这一轮的报告里,进表长期管理。
--
-- 为什么要一张表:报告是产物,重跑一次就被覆盖,上一轮判出的个案连带它的处置状态一起没了。
-- 而这些个案的价值恰恰在跨轮追溯——同一道题反复被判编造,说明库里那一格一直没补对。
--
-- 一题一行(uk_eval_id):同一道题再次被判编造不新增行,只把答案/理由/角标快照更新成最近一次、
-- seen_count 加一。已经标「已解决」的题又被判出来,状态自动退回「未解决」——那是复发,
-- 不是新问题,得让它重新出现在待处理列表里。
--
-- citations 存的是这一轮喂给模型的 Top-K 证据**全集**(答案里的角标 [n] 就是这份列表的序号)。
-- 裁判说「证据里没有」,追溯时必须能当场看到当时喂进去的到底是什么,不能只留一句结论;
-- 而且要看得出「手里有哪几条、实际只引了哪几条」——没被引用的那些同样是判断依据。

CREATE TABLE faith_cases (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  eval_id       VARCHAR(16)     NOT NULL                COMMENT '评估集题号,如 A43;一题一行',
  bucket        VARCHAR(24)     NOT NULL                COMMENT '题目所属桶:A_policy / B_model / C_colloquial / E_multi',
  query         VARCHAR(512)    NOT NULL                COMMENT '用户问题原文',
  strategy      VARCHAR(24)     NOT NULL DEFAULT 'hybrid_rerank' COMMENT '产出这条答案的检索策略',
  answer        TEXT            NOT NULL                COMMENT '被判编造的那版生成答案原文',
  reason        TEXT            NOT NULL                COMMENT '裁判给的理由:编在哪一句',
  citations     JSON            NULL                    COMMENT '这一轮喂给模型的 Top-K 证据全集快照:[{n,chunk_id,section_path,question,answer}];答案里的角标 [n] 就是这份列表的序号,答案通常只引用其中两三条;老数据没记为 NULL',
  judge_model   VARCHAR(64)     NULL                    COMMENT '判这条的裁判模型',
  status        ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决' COMMENT '处置状态,人工点按钮改',
  seen_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '被判编造的累计次数(跨轮)',
  first_seen_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '第一次被判编造的时间',
  last_seen_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '最近一次被判编造的时间',
  resolution    VARCHAR(300)     NULL                    COMMENT '处置说明:标已解决要写清怎么解决的,标无需解决要写清为什么不用改;退回未解决时清空。空着的处置在台账上等于没有交代',
  resolved_at   DATETIME        NULL                    COMMENT '最近一次被标为已解决/无需解决的时间;复发后仍保留,用来标「复发」',
  PRIMARY KEY (id),
  UNIQUE KEY uk_eval_id (eval_id),
  KEY idx_status (status),
  KEY idx_last_seen_at (last_seen_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch04 忠实度编造个案台账';

-- =============================================================
-- ch07 · 会话上下文管理 · 分段摘要表
-- 摘要一段一行、只追加:压完的段落不再回炉重压,一个事实只经历一次有损压缩。
-- conversations.summary 是全部段落按 seq 拼接的投影,conversations.summary_upto_msg_id 是覆盖锚点。
-- 层边界靠消息 id 表达不搬数据:id ≤ summary_upto 已进摘要;summary_upto < id ≤ layer1_from 层2;
-- id > layer1_from 层1 原样。
-- =============================================================

CREATE TABLE conversation_summaries (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NOT NULL,
  seq             INT             NOT NULL COMMENT '第几段,从 1 开始',
  from_msg_id     BIGINT UNSIGNED NOT NULL COMMENT '这段覆盖的消息区间,闭区间',
  upto_msg_id     BIGINT UNSIGNED NOT NULL,
  content         TEXT            NOT NULL,
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv_upto (conversation_id, upto_msg_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='分段摘要,一段一行只追加';


-- =============================================================
-- ch08 · 工具系统 · 工具调用审计表
-- 统一执行引擎每次工具调用落一条:被权限拒、被校验拦的调用同样落;
-- 不挂外键,审计写入不能被引用约束拦住;写审计失败只记 warning 不拦执行。
-- =============================================================

CREATE TABLE tool_audit_logs (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '审计主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '所属会话,无会话上下文的调用为 NULL',
  tool_call_id    VARCHAR(64)     NULL                    COMMENT '模型申请单 id,可对回 messages 流水',
  tool_name       VARCHAR(128)    NOT NULL                COMMENT '工具名',
  tool_source     ENUM('builtin','mcp') NOT NULL          COMMENT '工具来源:内置 / MCP 接入',
  mcp_server      VARCHAR(64)     NULL                    COMMENT '来源 MCP Server 名,内置工具为 NULL',
  arguments       JSON            NULL                    COMMENT '调用参数',
  result_summary  TEXT            NULL                    COMMENT '返回结果,过长截断存摘要',
  status          ENUM('成功','失败','超时','校验拦下','权限拒绝') NOT NULL COMMENT '本次调用结局',
  error_message   VARCHAR(512)    NULL                    COMMENT '失败 / 拦下时的原因说明',
  retry_count     TINYINT UNSIGNED NOT NULL DEFAULT 0     COMMENT '实际重试次数,写操作默认不重试恒为 0',
  duration_ms     INT UNSIGNED    NULL                    COMMENT '耗时毫秒',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '调用时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  KEY idx_tool_name (tool_name),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工具调用审计留痕';

-- ch09：本地 token 统计与 Langfuse trace 对照；追踪服务不可用时仍可落本表。
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

-- ch09 · 数据飞轮审核队列与评估轮次。先建 review_queue,再扩展问题池外键。
CREATE TABLE review_queue (
  id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  normalized_question VARCHAR(512)    NOT NULL,
  ai_suggested_answer TEXT            NULL,
  occurrence_count    INT UNSIGNED    NOT NULL DEFAULT 1,
  review_status       ENUM('待审','通过','驳回') NOT NULL DEFAULT '待审',
  approved_answer     TEXT            NULL,
  created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_review_status (review_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='飞轮待审队列';

CREATE TABLE eval_runs (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  triggered_by ENUM('定时','手动') NOT NULL DEFAULT '定时',
  dataset_size INT UNSIGNED    NOT NULL,
  metrics      JSON            NOT NULL,
  created_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自动化评估流水线轮次结果';

ALTER TABLE low_confidence_questions
  ADD COLUMN matched_review_id BIGINT UNSIGNED NULL,
  ADD COLUMN retrieved_chunks JSON NULL,
  ADD KEY idx_matched_review_id (matched_review_id),
  ADD CONSTRAINT fk_lcq_review FOREIGN KEY (matched_review_id)
    REFERENCES review_queue (id) ON DELETE SET NULL;

-- ch10 · 微调分类器旁路批量归类结果。低置信度问题攒够一批归一次类,主链路不写这张表。
CREATE TABLE topic_classifications (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  question_id   BIGINT UNSIGNED NOT NULL                COMMENT '归类的问题,指向 low_confidence_questions.id',
  labels        JSON            NOT NULL                COMMENT '多标签,17 类权威类目里命中的若干个',
  classified_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '归类时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_question_id (question_id),
  CONSTRAINT fk_topic_question FOREIGN KEY (question_id) REFERENCES low_confidence_questions (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='主题分类结果';

-- ch14 · 后台管理员账号与操作审计。审计表 append-only,不挂外键。
CREATE TABLE admin_users (
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

CREATE TABLE admin_audit_logs (
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

-- =============================================================
-- ch12 · 受控 Agentic RAG · 检索步骤审计表
-- 仅可审计(ADR-1):一轮一行记录每轮查询/子问题/证据ID/判定/停止原因,
-- 供事后查询与运营回捞;不提供断点续跑/回放/跨进程恢复。
-- 不挂外键,审计写入不能被引用约束拦住;写入失败只 warning 不拦检索回合。
-- =============================================================

CREATE TABLE rag_loop_steps (
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

-- =============================================================
-- CH11 · 同会话可靠执行 · 持久 run、SSE 重放与租约
-- 与 db/ch11-reliability-execution.sql 保持一致；新表只追加，不修改业务表。
-- =============================================================

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
