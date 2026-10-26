# Typed Handoff R2 开发记录

## 观察

原 runtime 只按 task ID 过滤 `facts` dict。角色实际看到一个依赖任务的全部工具 JSON；
跨订单字段仍可能被 prompt 带入。evidence 只有 chunk ID，没有主体、predicate、版本或
时效。工具调用也没有 run 级等价去重记录。

## 选择

没有更换 Agent 框架。新增小型、本地、可序列化合同层：

- `EvidenceSource` 保留完整工具返回供审计；
- `EvidenceClaim` 将 direct tool observation 规范化为主体/predicate/value/scope；
- `EvidenceLedger` 处理 supersede、有效期和角色最小投影；
- `ToolResultCache` 只合并且发只读请求，写调用永远直接执行确认链。

raw 模式的 role system prompt、facts 投影和 API 输出保持原样；typed 模式只用显式参数
启用。role prompt 追加 typed 使用约束，不允许扩权。

## 测试先行

先看到 8 项合同测试失败，再实现模型、ledger 和 cache。新增测试覆盖：

- JSON 往返与 `unknown` 默认；
- 无工具来源 claim 被拒绝；
- order/policy source mapping；
- 字段级物流有效期；
- 依赖、主体与 scope 过滤；
- 改口 supersede；
- 并发等价读只并一次、写不缓存、TTL 归零重查；
- raw 默认投影兼容；
- 实际 action prompt 不含非依赖订单；
- 人类转接、角色 handoff、工具调用类型分离。

## 真实模型实验

冻结 10 dev + 20 held-out 后只跑一次完整两臂，共 312 次模型调用。初版评分错误地把
正确的 expired-policy 背景、例外条件和 Markdown 表格判错。保留 v1/v2 后，统一离线
重评为 v3：

- raw 25/30，typed 25/30；
- 配对 0/0/30；
- typed 输入 token +33.79%，每成功成本 +20.39%；
- source missing 2 -> 0，但缓存实际去重命中为 0；
- typed 的 action handoff 关键事实遗漏 0/3。

因此工程边界成立，但效果未成立。没有根据 held-out 修改角色策略或删减难例。

## 已知问题

- typed prompt 的 claim 元数据在当前实现中比 raw facts 更大；需要下一轮重新设计
  projection 格式，并作为新变量单独评测，不能把本轮负结果重解释成收益。
- 模拟 `create_ticket` 的参数 schema 比生产宽松。模型生成非三值枚举时工具接受，
  评分器按生产契约判错；两臂同口径，但这降低 harness 保真度。
- 动态 Planner 轨迹使少数工具调用次数不同；没有把 raw/typed 的请求数差全部归因于缓存。
- 当前 cache 仅 run 内，进程重启不保留。
