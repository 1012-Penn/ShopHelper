# ShopHelper 前四项可靠性修复实施计划

用户已批准此计划。采用 TDD，最后全量测试和独立审查。不引入消息队列，不扩展到索引重建、工单、权限或 checkpoint。

### Task 1: 上下文和输出预算
裁剪后的消息与 history_text 共用保留内容，处理空助手行及工具行，不改变摘要与预算估算策略。聊天/抽取模型均传 max_tokens=settings.max_output_tokens。测试被丢弃历史不可见、角色顺序、无裁剪、跳过行，以及模型请求上限。

### Task 2: 检索异步卸载和故障闸
知识及退款图节点的同步 service.retrieve / kb.get_chunks 用 asyncio.to_thread，查询保持顺序，SSE在事件循环。普通query_faq已有线程卸载，勿重复包。重排缺失/异常保留候选供排查但设置degraded/low_confidence及原因；不拿RRF分数套重排阈值。知识/工具不使用低置信候选作为证据。退款仅合并可靠结果；全失败落池并走确定性fallback，不进入agent。补事件同步的非阻塞测试及三条消费路径故障/恢复回归，更新旧降级仍放行的断言。

### Task 3: 审核幂等与可重试索引
KnowledgeChunk新增可空唯一review_id外键。审核状态/核准答案/知识创建同事务，锁审核行并处理唯一约束竞争；同答案重复返回原id，不同答案409。提交后只索引当前chunk，失败保留pending并记录日志，审核仍成功。approve保留int返回值供现有调用方。API增加index_status(done/pending)，新增POST /api/review-queue/{id}/retry-index(404不存在/409未审核)。列表详情附知识id/索引状态。页面支持已通过视图及重试，检查response.ok、保留失败输入，不将pending显示为已入库。迁移按section_path关联最小id，不删旧重复。先迁移再新代码。测试重复/并发/冲突/嵌入和upsert失败/重启重试/定向索引/页面行为及迁移。

### Task 4: 集成验证
执行完整pytest，独立代码审查并修复重要发现。原基线375 passed, 1 xfailed。记录真实MySQL迁移/并发及浏览器可用的实际验证结果。最终交付迁移运行说明。
