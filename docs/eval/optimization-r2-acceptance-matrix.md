# 第二轮并行优化验收矩阵

日期：2026-10-27

共同基线：`446fbb7f9972875b7649555cb9b4b19e402cea29`

集成分支：`codex/optimization-r2-integration-20261027`

这份文件只记录验收合同和前置检查，不把开发会话的自报结果当作已通过事实。

## 四支独立交付

| 方向 | 源分支 | 必须证明的效果边界 | 集成接口风险 |
|---|---|---|---|
| 成本感知路由 | `codex/opt-adaptive-routing-r2-20261027` | single、multi、adaptive 三臂同题；路由特征不含金标/题号/期望行为；决策成本计入 | dispatcher 可注入真实单/多 Agent；不得偷偷双跑；应用路径与默认关闭/回滚 |
| claim grounding | `codex/opt-claim-grounding-r2-20261027` | rules 对选择性 grounding；三态分母；`not_entailment` 不升级为 unsupported；真实 backend 证据 | 只接 uncertain；确定性冲突不可被覆盖；适配现有 verifier 但不改默认 |
| typed evidence/handoff | `codex/opt-typed-handoff-r2-20261027` | raw 对 typed；角色实际输入、来源、作用域、改口/过期和去重；质量与 token/成本代价 | 新参数默认 raw；现有多 Agent 响应/确认兼容；不得扩大工具权限 |
| 无标签端到端评测 | `codex/opt-e2e-quality-r2-20261027` | 真实应用/graph 与真实模型；输入不含 expected behavior、gold、金标证据或评分规则 | 独立 harness；区分生产路径、固定证据诊断、技术失败和模型失败 |

四支百分比不能相加。组合效果只能在集成分支用同题、同模型、同工具和统一评分另测。

## 输入与评分审计

每个方向合并前至少逐项确认：

1. 真实 prompt/轨迹可定位到模型实际收到的内容，而不只看配置名或 schema。
2. 路由只使用用户可见轮次、允许历史和可见预算；不读取 `gold`、`bucket`、`expected_behavior`、
   case ID 或评分规则。
3. grounding 的 `supported / unsupported / uncertain` 保留独立分母；模型无充分证据、超时或解析失败
   仍为 uncertain；确定性矛盾优先。
4. typed handoff 的每个角色实际投影有来源、主体作用域和时效；缺失字段显式 unknown。
5. E2E 数据按事实/模板分组切 dev/held-out；冻结 hash、配置、阈值和停止规则；失败留在分母。
6. 模型生成成本与裁判/修订/路由成本分开，再给总成本及每成功任务成本。
7. 缓存命中、冷执行、在线耗时分开；缓存耗时不能当线上 P95。
8. 新评分只重评旧响应时明确“评分修订”；不能表述成模型能力提升。

## 与 `446fbb7` 的语义兼容

| 现有契约 | 集成门槛 |
|---|---|
| `POST /api/chat` | 默认单 Agent、SSE、持久 run、租约、只读恢复和未知写阻断不倒退 |
| `POST /api/multi-agent/chat` | 默认响应结构、会话绑定、预览确认和取消语义保持兼容 |
| `ExecutionToolEngine` | 新路径仍经参数校验、写拦截、确认和幂等边界 |
| `AnswerVerifier` | 默认关闭；开启时最终答案验证完成后才释放 token |
| `RAG` 默认配置 | `hybrid_rerank + baseline`；新机制无证据时保持关闭 |
| 多 Agent recovery/usage | 未补齐就明确记录，不把多 Agent 写成与 `/api/chat` 同等持久语义 |

共享文件 `app/main.py`、`app/config.py`、`app/graph/*`、`app/answer_verifier.py`、
`scripts/eval_checks.py`、`db/*` 只由集成会话按各支 integration notes 接线。

## 合并顺序

先独立审查并 cherry-pick 不需要其他支线的提交，再处理共享入口：

1. claim grounding 与 typed handoff 的纯模块/兼容默认；
2. 多 Agent typed 模式与现有 API 回归；
3. adaptive router 的注入式 dispatcher 和应用可选入口；
4. E2E harness 保持独立，不改变生产路径；
5. 最后统一接入默认关闭的 config/factory 开关并跑组合回归。

若提交之间有文本冲突，按上述语义合同解决；无文本冲突也必须重新检查构造默认值、返回结构、
异常收口、usage 成本和写确认。

## 验收闸门

- 源分支已提交、工作区干净，报告中的关键数字能从逐题轨迹/聚合复算。
- 定向测试先通过；已有 reliability acceptance 重跑通过；应用工厂新路径有集成测试。
- 全量 pytest 只在集成完成后跑一次；仅新失败时再复跑。
- 组合效果必要时用独立账本和预先冻结的小规模 run；最多新增 $2，不继承源会话 ledger。
- 不迁移共享 MySQL/Milvus，不触碰旧 `/root/workplace/ShopHelper`，不 push。
- 未获可靠收益的机制默认关闭或只保留独立诊断入口；失败和不利成本写入报告。
