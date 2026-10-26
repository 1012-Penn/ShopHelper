# 后续技术改进评审

日期：2026-10-27

本轮只评最长三项，按面试收益、实现成本和可验证性排序。以下收益均为待验证假设，不能预填成
结果，也不因使用较新的名称就视为技术升级。

## 1. 成本感知的选择性单/多 Agent 路由

推荐顺序：第一。

当前缺口：

- held-out 多 Agent 17/20、单 Agent 15/20，但简单桶 11/12 对 12/12；
- 复杂桶 6/8 对 3/8，固定总成本却高约 6–7 倍；
- 当前没有自动决定“这题是否值得多 Agent”的机制。

建议机制：

- 以单 Agent 为默认，在请求进入主图前做一次低成本结构化判断；
- 路由特征只使用用户可见信息：订单/实体数、是否需要跨订单比较、政策+订单依赖、用户改口、
  写操作、缺失槽位和检索冲突迹象；
- 先实现可解释的规则/小型分类器，固定阈值和成本上限；
- 只有在无写到写、低风险到高风险的状态迁移处才升级 Agent，避免中途改变工具权限。

参考：

- [RouteLLM 论文](https://arxiv.org/abs/2406.18665) 学习在强/弱模型之间按查询选择；
- [ICLR 2025 页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html)；
- [官方仓库](https://github.com/lm-sys/routellm)。

解决的问题：

- 保留简单题的低延迟和低成本；
- 把多 Agent 的预算集中到比较、跨证据依赖和修正任务。

同题评测：

- 新建未被查看的冻结题集；
- 比较 fixed-single、fixed-multi、router 三臂的任务成功、配对得失、简单/复杂分桶、
  误升级/漏升级、调用数、token、P50/P95、每成功任务成本；
- 先固定路由特征、阈值、停止规则，再跑一次；
- 必须把“路由错误导致本可成功的任务失败”计入分母。

## 2. 仅对 uncertain 引用的轻量 claim-evidence 判断

推荐顺序：第二。

当前缺口：

- v3.1 frozen-v1 各 RAG 组仍有 44 个引用待复核，多个检索组为 54–64；
- 确定性规则能发现数字/实体冲突，但不能可靠识别同义支持；
- 全量再调用大模型裁判会增加成本，并已在 10 题 dev 中增加误拒。

建议机制：

- 只对确定性层返回 `uncertain` 的“一句 claim + 一组候选证据”运行小型 NLI/grounding 检查；
- 输入包含完整主体、型号、日期、政策版本和单位，不以字符相似度替代推理；
- 输出继续保留 supported/unsupported/uncertain，并单独保存模型、提示、版本和证据 ID；
- 先在人工标注的 uncertain dev 样本上校准，再冻结；不要用同一题同时调阈值和报结果。

参考：

- [MiniCheck 论文](https://arxiv.org/abs/2404.10774) 检查生成事实是否被 grounding documents 支持；
- [EMNLP 2024 页面](https://aclanthology.org/2024.emnlp-main.499/)；
- [MiniCheck 官方仓库](https://github.com/Liyan06/MiniCheck)。

解决的问题：

- 降低大量 uncertain 对质量报告可用性的影响；
- 比“每个答案都调用大模型裁判”更便宜，也比字符重叠更可信。

同题评测：

- 只在与 baseline 相同的检索证据和答案上判断；
- 指标包括 supported/unsupported/uncertain 的 precision/recall、弃权率、错误实体/条件漏检；
- 下游同时看严格正确、误拒和每成功任务成本；
- 若只减少 uncertain 但增加误拒，结论是不采用或只用于人工复核队列。

## 3. 类型化证据与结构化 handoff 投影

推荐顺序：第三。

当前缺口：

- 多 Agent 的 `facts`/`evidence` 是宽松 JSON dict；
- 订单号、商品、值、单位、政策版本、观察时间和证据 ID 没有统一类型；
- 跨角色 handoff 依赖自然语言和裸 dict，容易重复检索或丢条件。

建议机制：

- 定义稳定的 `EvidenceClaim`：subject IDs、predicate、value、unit、scope/version、
  observed_at、source chunk/order ID、validity 和 certainty；
- 组件间只传结构化事实与引用，不传无来源的摘要断言；
- Supervisor 的手工转接、内部角色 handoff 和“无状态工具调用”使用不同 schema；
- 对每个角色做最小上下文投影，明确哪些事实可继承、哪些必须重查；
- 工具写操作仍只走现有确认/幂等边界。

参考：

- [OpenAI Agents handoffs](https://openai.github.io/openai-agents-python/handoffs/) 说明显式转接与输入过滤；
- [OpenAI Agents orchestration](https://developers.openai.com/api/docs/guides/agents/orchestration) 区分 specialist handoff 与工具式委派；
- [LangGraph persistence](https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/persistence.mdx) 说明 thread checkpoint 与长期 store 的职责边界；
- [LangGraph Graph API](https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/graph-api.mdx) 说明共享状态和动态路由。

解决的问题：

- 减少跨订单实体串用和过期证据；
- 降低角色间重复调用与上下文膨胀；
- 让 handoff 完整性和来源支持可以被逐题检查。

同题评测：

- 在同一多 Agent 轨迹上比较裸 dict 对类型化投影；
- 指标：实体/条件错误、缺失 handoff 事实、重复工具调用、输入 token、任务成功和总成本；
- 增加“证据过期但语义相似”“两单同日签收不同商品”“v1/v2 边界日”等定向题；
- 不把 schema 数量、Pydantic 模型数或 trace 长度当作质量收益。

## 暂不建议本轮扩展

- 不为“技术栈新颖”强接 LightRAG；
- 不先扩建多租户/身份/HA 平台；
- 不把任意节点 checkpoint 当作可靠性必交付；它会带来状态迁移、版本和副作用边界，需由实际
  演示需求触发；
- 不在已查看的 20 题 held-out 上反复调路由或阈值。
