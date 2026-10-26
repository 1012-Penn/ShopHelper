# Schema 动作契约第三轮实验

日期：2026-10-27

共同基线：`59e233f5145298846f8870b74e351ac3db98f5e1`

分支：`codex/opt-action-contract-r3-20261027`

## 结论先行

本轮实现了一个默认关闭的 schema 驱动动作契约：真实预览、确认、取消和失败都由结构化
artifact 表示，模型文本不再算动作成功；`conversation_id` 由当前 run 覆盖，模型不能提交
`confirm_key` 或 `confirmed=True`；参数错误或“只说不做”最多获得一次明确反馈。

真实 dev 配对的中位结果没有证明整体任务完成率提升：baseline 与契约臂均为 `5/8`，
配对为 `2胜/2负/4平`。契约臂减少了错误额外预览 `1→0` 和创建写调用尝试 `8→2`，
代价是合法预览正确 `7/8→6/8`、取消正确 `7/8→6/8`；成本从
`$0.0131205` 降到 `$0.0115809`。因此不能写成“动作契约整体优于旧 runtime”。

首轮 8 dev + 20 held-out 全量运行发生了评测基础设施污染：运行中的 12,000-byte
请求包上限与两臂各 180 次组上限造成大量 `BudgetExceeded`。原始结果和账本已保留；
零调用重算后，每臂 20 道 held-out 中有 17 道受基础设施影响，不能用于模型质量结论。
held-out 没有重跑。

## 机制

只对显式工单动作开启，默认 `action_contracts_enabled=False`：

1. 从当前 registry 的 `record.json_schema` 读取允许参数、必填项和枚举。
2. 将当前 run 的 `conversation_id` 覆盖进参数；模型给出的其他会话号记录为
   `trusted_run_scope`，模型给出的 `confirm_key` 被移除。
3. 仅做语义无歧义的类型规范化：退款、退货、维修、返修、换货归为售后；模糊类型和
   空业务描述要求澄清，不默认猜成售后。
4. `ToolEngine` 未确认写仍被拦截。预览生成后进入
   `awaiting_confirmation`；客户端显式确认并实际落库后才可标记 `confirmed`。
5. 取消使用会话级 pending ledger；取消后的旧预览不会复活，修改描述会使旧预览变为
   `cancelled/superseded_by_revision` 并生成新预览。
6. 用户未要求写入时，action 任务不调用模型；模型声称有预览但没有 artifact 时，
   最多给一次缺口反馈，仍失败则返回真实的“未能生成”回复。
7. 所有修复调用都受原 runtime 的 step/token/timeout 上限约束，没有额外无限循环。

## 真实运行

### 首轮冻结运行（保留失败）

题集：8 dev + 20 held-out；场景 SHA-256
`91aeeff53af1201430731086117675b1febd2895888c8428f99f66e3070448e4`。
冻结源配置 SHA-256：`de990c0d57f210472780864e3f168ffc56bcc4a19b449677fe7c6e2a77d5f84e`。

两臂各记录 180 次 provider generation，合计新增
`$0.1048572`，输入 181,512 token、输出 41,503 token；无 pending 调用。

| 分组 | Arm | 原始任务成功 | v2 重算成功 | 预览正确 | 取消正确 | 额外预览 | 技术/基础设施失败 |
|---|---|---:|---:|---:|---:|---:|---:|
| dev | baseline | 2/8 | 1/8 | 3/8 | 7/8 | 2 | 5 |
| dev | action contract | 2/8 | 1/8 | 5/8 | 7/8 | 0 | 4 |
| held-out | baseline | 5/20 | 1/20 | 9/20 | 13/20 | 1 | 17 |
| held-out | action contract | 5/20 | 1/20 | 11/20 | 15/20 | 0 | 17 |

v1 的自动归类只看了 harness 层错误，低估了图内 `agent_error` /
`BudgetExceeded`。v2 只重新计算既有输出，没有模型调用，也没有删改原始 response。
原始 `summary.json` 与 `results.jsonl` 保留，重算结果写入
`summary-rescored.json` 与 `results-rescored.jsonl`。

首轮题目和预算在运行前冻结；manifest 记录原始场景和配置 hash。该次 held-out
不是有效质量对照，也不再重跑。

### 修正后 dev 诊断

为解决上述 harness 上限，使用新独立 SQLite 账本，将请求包上限调到 16,000 bytes、
提高 generation 上限，并按每次实际请求包做预调用原子预留。该诊断未运行 held-out。
该次源配置 SHA-256：`686cce74ba30e4c782fa7ad04fa1cac4ce1e19d5ea0e0644516aa5dc4809cba0`。
当前零调用重算代码 SHA-256：
`178c8038a5aa6bcde95a9d210642829e4ea0bf623374ba72fa3d628fe30d9c91`。

| Arm | 任务成功 | 配对 | 预览正确 | 取消正确 | 额外预览 | 创建调用尝试 | 未确认写 | 唯一重复写 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 5/8 | 基准 | 7/8 | 7/8 | 1 | 8 | 0 | 0 |
| action contract | 5/8 | `2胜/2负/4平` | 6/8 | 6/8 | 0 | 2 | 0 | 0 |

| Arm | Provider 调用 | 输入/输出 token | 新增费用 | P50 | P95 |
|---|---:|---:|---:|---:|---:|
| baseline | 42 | 24,723 / 4,753 | $0.0131205 | 6,263 ms | 7,167 ms |
| action contract | 40 | 22,915 / 3,922 | $0.0115809 | 4,643 ms | 6,819 ms |

“未确认写为 0”在本样本只说明没有观察到提前落库；baseline 仍有 8 次创建调用尝试，
不能把 0 写成安全率 100%。契约臂的收益主要是额外动作更少，不是任务成功更高。

该 dev 诊断发生在最后一组边界修复前：

- “预览，不要提交”原先因正则过宽被误判为取消；
- planner 未安排 action task 时，取消 artifact 原先可能不落账。
- 纯查询回复可以声称“预览已生成”，现在会被无 artifact guard 拦截；
- 没有客户端确认时，文字确认也会被 guard 拦截。

这些修复及在 action prompt 中加入 registry schema 上下文均在定向测试中通过，
但没有再跑真模型；因此不能把它们算进上表收益，也没有为了数字重跑同一组题。

## 工程验证

- 动作契约与 harness 定向测试：`23 passed`。
- 分支全量回归：`1034 passed, 2 skipped, 1 xfailed`，退出码 0。
- 全量结束后 Milvus Lite 在解释器关闭阶段输出了测试数据库清理日志 I/O 警告；
  测试断言与退出码不受影响，记录为测试依赖的清理期噪音。
- `git diff --check`：通过。
- 两份临时 SQLite 账本均检查未包含 API key 或凭据标记。

## 逐题结果与复跑

原始结果：

- `tests/eval/action_contract_r3/runs/live-20261027/results.jsonl`
- `tests/eval/action_contract_r3/runs/live-20261027/results-rescored.jsonl`
- `tests/eval/action_contract_r3/runs/live-20261027/summary.json`
- `tests/eval/action_contract_r3/runs/live-20261027/summary-rescored.json`

dev 诊断：

- `tests/eval/action_contract_r3/runs/dev-diagnostic-v2-20261027/results.jsonl`
- `tests/eval/action_contract_r3/runs/dev-diagnostic-v2-20261027/summary-rescored.json`

零调用重算：

```bash
/root/workplace/ShopHelper/.venv/bin/python \
  tests/eval/action_contract_r3/harness.py \
  --rescore-existing tests/eval/action_contract_r3/runs/live-20261027
```

新运行会拒绝覆盖已有结果或账本。付费入口必须显式传 `--allow-paid`。

## 成本与边界

- 第一轮独立账本：`$0.1048572`，上限 `$2`。
- 修正后 dev 独立账本：`$0.0247014`，上限 `$2`。
- 两次合计 `$0.1295586`；没有复用旧 pending，没有继承无上限授权。
- 每次调用前原子预留，使用 provider usage 结算；基础设施失败和最多两次退避重试均记账。
- 无缓存重放，延迟是本次端到端图执行时间。
- 所有写入只进入进程内 fake-write sink；没有写共享 MySQL 或 Milvus。
- 业务模型为 `deepseek-flash`，温度 0；Planner 走集中
  `make_structured_extractor`，没有重新引入强制 `tool_choice`。

## 未验证

- 首轮 held-out 受基础设施影响，不能作为质量提升证据；没有重跑。
- dev 诊断是在查看首轮 held-out 后运行的，不是独立 held-out。
- 最终两个语义/状态修复只有离线测试证据，没有新的真实模型配对结果。
- 当前分支通过 runtime 构造注入启用；HTTP 接线、持久 ledger 和确认路由联动留给统一集成。
- pending ledger 是进程内状态，没有跨进程恢复或正式 TTL；这不在本轮演示范围。

## 简历候选句

1. > 为客服 Agent 实现 schema 驱动的工单动作契约：从真实 registry 注入参数与枚举，
   > 用可验证 artifact 表达预览/确认/取消/失败，并限制一次缺口修复；23 个定向测试覆盖
   > 跨会话号、非法枚举、只说不做、同键重放和未确认写拦截。

2. > 在 8 道模拟 dev 任务上做同模型、同工具配对：动作契约臂把错误额外预览从 1 降到 0、
   > 创建调用尝试从 8 降到 2，任务成功保持 5/8 对 5/8；但合法动作完成未提升，
   > 不能表述为整体优于旧 runtime。
