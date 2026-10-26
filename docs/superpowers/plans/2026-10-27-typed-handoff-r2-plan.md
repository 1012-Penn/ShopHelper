# Typed Evidence Handoff R2 Implementation Plan

1. 新增合同模型和 run 级 ledger/cache；先写序列化、来源、supersede、过期和并发去重
   failing tests。
2. 实现工具结果到 claim 的保守映射、canonical scope 和最小角色投影；覆盖 unknown 与
   source validation。
3. 给 `MultiAgentRunner` 增加默认 `handoff_mode="raw"`，扩展 outcome/state 的默认字段；
   typed 分支接入 claim ledger、handoff 记录和 prompt projection，raw 分支保持原逻辑。
4. 接入只读缓存与调用统计；验证并发 in-flight 去重、写不缓存、动态状态过期和错误计数。
5. 跑定向测试，再跑接口兼容与多 Agent 回归。
6. 新增独立 fixtures、冻结 manifest、评分器和真实两臂 harness；先跑离线接线，再冻结。
7. 在 `$2.00` 隔离账本内跑 30 题 raw/typed 配对；保存原始 prompt/响应、逐题数据、
   provider usage、失败与成本。
8. 生成报告、dev-notes、integration-notes，执行全量回归并提交本分支。
