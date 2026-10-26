# 受控 RAG 与最终答案复核：短规格与实施计划

日期：2026-10-27。基线：`main@392e058d195bf731deff9ef6b4c2113da2a36a8b`。

## 目标与边界

复用 CH12 受控检索和 CH16 检索消融，不重做 Agentic RAG。先修评分尺子，再用 dev
错例定位证据触发和最终答案复核；最终以同题真实模型对照给出有限结论。默认线上路径继续
使用 `hybrid_rerank + baseline`，答案复核默认关闭。旧问题、金标、响应、聚合与历史报告
保持只读；修复评分只对旧响应做新版本重评。

本分支只修改：`app/config.py`、`app/graph/{builder,agent,agentic}.py`、`app/rag_loop.py`、
`app/prompts.py`、新增答案复核模块、`scripts/eval_checks.py`、相关评测与测试。聊天路由、
`app/main.py`、存储模型和可靠性模块的接线以 integration notes 交付。

## 评分修正

新评分版本为 `rag-quality-v3`，保留旧检查值和 `legacy_*` 诊断。

1. 引用归属：角标位于句号前或紧邻句号后时都归属前一个陈述句；多个角标逐个验证。
2. 数值与单位：统一全/半角、分隔符、中文数字、金额、时长；接受 `半年=6个月`、
   `一年=12个月`、`1天内=24小时` 等有确定换算关系的表达。
3. 条件：去掉空白和常见连接词差异；日期条件按规范化 ISO 年/月/日比较；窗口条件接受
   “第 N 天仍在窗口内”这类直接计算；显式别名仍可覆盖。
4. 必要事实与背景事实：新增只读 overlay。overlay 只能排除“问题未询问、答案无需引用”的
   背景 fact，或补等价表达；不能改值、不能删答案必需事实。每项修订记录 case、fact/condition、
   依据与评审状态。
5. 引用支持：确定性层只判数字/实体/明确词面是否冲突；可确认匹配为 `supported`，可确认冲突为
   `unsupported`，其余为 `uncertain`。不再用字符 4-gram 冒充语义支持。聚合同时报告
   supported/unsupported/uncertain 和人工待复核数，不把 uncertain 计入正例。

## 候选答案复核

新增独立 `app/answer_verifier.py`，不复用 EvidenceJudge 的 `sufficient` 作为放行。

确定性预检检查：引用编号存在；答案中的订单号、商品型号与引用证据实体绑定；数字和单位
与有效证据一致；v1/v2 政策条件与签收日期、窗口天数一致。特别检查“当前订单引用了另一订单
签收日/状态”的串单。结果有三态：`pass`、`fail`、`uncertain`。

需要语义判断时调用独立结构化裁判，检查事实支持、实体、必要条件和政策版本。仅 `fail` 允许
一次答案修订；修订后再次验证。仍失败或不确定时改为澄清/拒答。所有判分调用计入同一实验账本。

`answer_verification_enabled=False` 默认。关闭时保留现有 Agent 流式输出与 SSE 契约。开启时
只缓冲最终答案 token，工具状态和引用仍可实时发送；验证通过后才释放最终 token。Agent 节点
可注入 verifier，生产接线由 builder 完成。

## 受控检索触发

保留现有闸门、查询去重、轮数/时间/token 停止与 baseline。新增默认关闭的 dev 触发开关
`rag_gap_trigger_enabled`：首轮证据闸通过时，若查询结构显示至少两个必要维度且当前证据缺少
其中一个可检索维度，则进入一次有界缺口分析+补检索；绝不因问题本身已知 gold 而触发。
`entity mismatch`、跨订单日期和 v1/v2 条件优先作为可解释触发信号。触发只增加一次主要变量，
max rounds、时间、token 和查询去重均不放宽。若 dev 同题无净收益则默认路径保持不变。

## 实验与验收

1. 单元测试先覆盖引用在标点前后、单位等价、背景 fact、三态引用、订单日期串用、
   政策版本和输出缓冲。
2. 对 CH16 已有五组 dev 与六组 frozen 响应统一离线重评，新产物写入独立目录，不改旧文件。
3. dev 上用已观察失败样式预先固定小集，做 baseline/decompose_once/controlled_loop 的公平
   对照；各组的证据闸、回答模型、prompt 和上下文预算一致。答案复核作为独立开关变量。
4. 真实调用使用 `deepseek-flash` 与既有 embedding/reranker 配置，pre-call guard、零隐式重试、
   独立账本和总额硬上限 `$2.00`。优先复用精确缓存；缓存未命中才调用真实模型。
5. 报告检索覆盖、证据充分性、答案正确/条件/实体/引用、误拒、补查/修订次数、停止原因、
   P95、成本与人工待复核样本；失败和未测项目照实保留。

## 回滚

删除或关闭 `answer_verification_enabled`/`rag_gap_trigger_enabled` 即回到既有
`hybrid_rerank + baseline` 与流式契约。评分 overlay 和结果目录可单独丢弃，不影响旧产物。
