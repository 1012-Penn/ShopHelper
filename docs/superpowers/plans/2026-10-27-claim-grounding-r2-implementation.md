# 选择性 Claim Grounding R2 实施计划

1. 用单元测试固定 `ClaimInput`、三态结果、确定性冲突优先级、
   `only_uncertain` 闸门和 backend 失败降级。
2. 用单元测试固定 API JSON 协议、usage 提取、最大重试次数及
   `AnswerVerifier` semantic adapter 的解析/聚合。
3. 用并发测试固定 SQLite 预算账本的原子预留、多 pending、按实际 usage
   结算和 fail-closed 行为。
4. 从既有 uncertain trace 提取 dev claim；新建不与 dev 同事实/模板的
   held-out 对抗 claim，写来源、gold 和人工待审状态。
5. 冻结数据/评分/配置 SHA。先跑零调用规则基线，检查题面泄露与难度分布。
6. 以小规模真实 DeepSeek Flash 调用跑通 selective backend，保留原始 prompt、
   response、usage、延迟和逐题结果；预算不足时停在上限内。
7. 独立评分并生成 precision/recall、错误类型、调用率、token、费用和 P95；
   缓存重放单独计时。
8. 运行定向测试、本分支全量回归，写报告、dev notes 和 integration notes，
   仅提交本分支。
