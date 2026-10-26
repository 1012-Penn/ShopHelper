# ShopHelper 面试版本集成验收

日期：2026-10-27

集成分支：`codex/interview-integration-review-20261027`

集成基线：`392e058d195bf731deff9ef6b4c2113da2a36a8b`

工作区：`/root/.codex/worktrees/interview-integration-review-20261027/ShopHelper`

## 结论

可靠性、多 Agent 工程链路和 RAG 评分审计可以集成；两个质量增强实验没有获得足以默认启用的证据，
因此继续保持关闭。默认聊天仍是现有单 Agent 路径，多 Agent 通过独立端点显式调用。

| 模块 | 验收结论 | 可以对外说明 | 不能据此声称 |
|---|---|---|---|
| 同会话可靠性 | 通过 | 持久 run/SSE、会话租约、owner fencing、只读恢复、未知写阻断和摘要注入预算均有真实应用/子进程证据 | 任意节点断点续跑、生产 MySQL 压测、任意外部写 exactly-once |
| 多 Agent | 工程接入通过，效果保留为条件结论 | 现有 engine/model 已挂载 `POST /api/multi-agent/chat`；跨角色上下文、工具白名单、依赖并行、预览确认与幂等均有测试 | 多 Agent 应成为默认、所有任务都更强、与 `/api/chat` 具备相同恢复/usage 语义 |
| RAG 评分 | 通过 | `rag-quality-v3.1` 对 14 组、1050 条历史响应统一零调用重评；保留 supported/unsupported/uncertain 三态 | uncertain 是正确；修复评分等于提升线上质量 |
| 最终答案复核 | 负结果，默认关闭 | 可拦截串单并最多修订一次；固定 dev 中严格正确仍为 0/9，事实正确 1/9 降至 0/9，误拒 1 增至 2 | 复核带来净质量提升、三组是三次独立成功复现 |
| 结构缺口触发 | 负结果，默认关闭 | 仅命中 1 题，无 R@3/R@5 增益，条件均分下降且新增误拒 | 新触发改善检索或回答质量 |

本集成实现提交为 `fe8c18c`。远端未推送。

## 源分支与合并

按可靠性、多 Agent、RAG 顺序以 cherry-pick 保留完整历史：

| 模块 | 源分支 | 源 commit | 集成结果 |
|---|---|---|---|
| 可靠性 | `codex/interview-reliability-20261027` | `0c1d9cdccb0a401362b1e159a1461726c2f2bf00` | 合入 |
| 多 Agent | `codex/interview-multiagent-20261027` | `d2fa82fe6bffb979110418b4e989e6d812050bc5` | 合入并补应用接线 |
| RAG 质量 | `codex/interview-rag-quality-20261027` | `ff58a8b93553e241af2201b9f71a0f5a6b9a5bcc` | 合入并补评分版本化 |

没有合入 CH13 LightRAG 或其他工作区改动。

## 独立复核发现与修复

1. 多 Agent 源分支只有独立 app/目录，未挂到真实应用工厂。现已使用现有
   `app.state.engine`、`app.state.chat_model` 和 `app.state.store` 挂载
   `POST /api/multi-agent/chat`。`POST /api/chat` 仍是默认路径。
2. 多 Agent 模型生成的 `create_ticket` 参数可能不含 `conversation_id`。集成层现在从真实会话
   注入该字段，确保预览包含可被 `/api/tickets/confirm` 消费的同一会话 ID。
3. 默认单 Agent 在“已有 assistant 文本后触发工具熔断”的路径会把同一文本写两次。新增回归后
   修复为只保留一份，旧的多步工具轨迹不受影响。
4. RAG v3 严格分类已经正确处理句号后的引用，但声明级 support/uncertain 比率仍按未归属的
   原文切句，导致支持句被计入 unsupported。修复后以 `rag-quality-v3.1` 单独产出；v3 文件不改。
5. `db/init.sql` 已追加四张可靠性表，并与 `db/ch11-reliability-execution.sql` 的四段
   `CREATE TABLE` 逐字一致。

## 实际验证

### 全量与定向测试

```text
/root/workplace/ShopHelper/.venv/bin/python -m pytest -q
893 passed, 2 skipped, 1 xfailed, 4 warnings in 134.14s
```

4 条 warning 是既有测试函数的 asyncio marker 误用；Milvus Lite 在 pytest 退出后有日志流已关闭
的清理噪声，退出码仍为 0。

集成定向集合：

```text
166 passed in 15.48s
```

定向集合覆盖可靠性、多 Agent runtime/contract/eval、应用路由与确认、单 Agent 熔断、
答案复核、RAG 评分和配置。

### 可靠性故障注入

`scripts/acceptance_interview_reliability.py` 本次重跑结果：

| 场景 | 结果 |
|---|---|
| 同会话双请求 | 1 created / 1 busy / 2 线程结束 |
| 重复请求键 | 同 run；异载荷冲突 |
| 直接完成回放 | `session → token → done` |
| 事件前崩溃后恢复 | `interrupted → completed`，attempt=2 |
| 事件后崩溃 | 重放 `session → interrupted` |
| 写开始后崩溃 | `write_state=unknown`，自动 resume 被拒绝 |
| 响应丢失后的确认 | 同 ticket、deduplicated=true、1 张工单 |
| 隔离 SQLite 固定负载 | 32/32，墙钟 1.055s，30.32 run/s，run P95 679.98ms |

模型调用与付费调用均为 0。这里的吞吐只说明隔离 SQLite 持久/重放路径，不是线上首字节或
MySQL/真实模型性能。

### MySQL 隔离验证

本次启动了临时 MySQL 8.0.46，使用 Unix socket、`--skip-networking` 和独立临时数据目录：

- `db/init.sql` 成功创建 21 张表；
- 四张新表的主键、请求唯一键、事件 seq 唯一键、lease run 唯一键和工具调用唯一键均存在；
- `ExecutionStore` 成功验证同请求复用、同会话冲突、事件递增/断点读取和完成终态；
- 同 `confirm_key` 第二次工单插入被唯一约束拒绝。

测试实例已停止，未连接或写入共享业务 MySQL。未做 MySQL 并发压测。

### 多 Agent 离线重算

从已保存的 30 题 × 2 臂轨迹独立调用评分器，结果与报告一致：

| split | Multi Agent | Single Agent | 配对 W/L/T |
|---|---:|---:|---:|
| dev 10 | 10/10 | 8/10 | 2/0/8 |
| held-out 20 | 17/20 | 15/20 | 4/2/14 |
| 合并 30 | 27/30 | 23/30 | 6/2/22 |

分桶没有混用 split：

| split / bucket | Multi | Single |
|---|---:|---:|
| dev / simple 6 | 6/6 | 5/6 |
| dev / complex 4 | 4/4 | 3/4 |
| held-out / simple 12 | 11/12 | 12/12 |
| held-out / complex 8 | 6/8 | 3/8 |

held-out 更适合用于泛化叙述；合并 27/30 对 23/30 只作全部冻结轨迹的汇总。复杂桶合计
10/12 对 6/12，但 simple 合计 17/18 对 17/18，支持“按任务复杂度选择”，不支持无条件切到
多 Agent。

成本：

- 全部 30 题模型调用 178 对 66，成本 $0.060966 对 $0.008768；
- 每成功任务成本约 $0.002258 对 $0.000381，即 5.92 倍；
- held-out 每成功任务成本约 $0.002546 对 $0.000397，即 6.41 倍；
- held-out P50/P95 为 7213/16404ms 对 1906/3551ms；合并 P95 为 17303/4800ms。

`MA-FRZ-022` 的旧失败来自并行检索触发串行 CH15 账本的单 pending 假设。并发账本修复和回归已存在，
但旧失败仍保留在分母中。`MA-FRZ-028` 没有生成预览；`MA-FRZ-030` 返回澄清而非预期拒答。

报告中的 handoff N/A 指“人工转接”场景为 0，不是多 Agent 内部 Supervisor→角色 handoff 缺失。
内部 handoff 看 trace、role histories 和任务依赖，不使用该人工指标。

### RAG v3.1 统一重评

```text
/root/workplace/ShopHelper/.venv/bin/python -m scripts.eval_rag_quality \
  --offline-rescore \
  --snapshot /root/.codex/worktrees/interview-rag-quality-20261027/ShopHelper/data/eval-snapshots/ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba.db
```

结果：14 个比较组、1050 条历史响应评分、0 模型调用、0 美元。

- 严格答案分类和引用三态计数相对 v3 不变；
- 220 条轨迹的声明级 unsupported/uncertain 计数被修正；
- 产物在 `tests/eval/rag_quality/v3_1/`，v3 产物保持只读；
- frozen-v1 baseline/controlled_loop 事实正确为 43/64、44/64，未证明受控循环有净收益；
- RRF+rerank frozen-v1 R@5 为 50/65，这是检索覆盖，不是答案正确率。

固定 10 题答案复核统一重评后，三个历史模式结果相同：

| 指标 | 复核前 | 复核后 |
|---|---:|---:|
| 严格正确 | 0/9 | 0/9 |
| 事实正确 | 1/9 | 0/9 |
| 误拒 | 1 | 2 |
| 复核结果 | — | 8 pass / 2 refuse |
| 逻辑修订 | — | 每组 3 次 |

三组共享大量历史回答和精确缓存，所以相同的 8/2、3 次修订不是三个独立模型重复实验。

报告中的 61.992s P95 是 17 次在线复核/修订尝试的单请求耗时，不是完整 `/api/chat` 端到端 P95；
缓存重放不计为在线延迟。结构缺口触发只命中 1 题，`R@3/R@5` 均无增益，条件均值 0.45→0.40，
误拒 1→2，保持默认关闭。

历史回答 prompt 带 `expected_behavior` 提示，这是继承自既有评测口径；因此这些历史数字不能
外推成“无行为提示的线上准确率”。

## 接口、默认开关与语义边界

默认：

```text
retrieval_route=hybrid_rerank
answer_verification_enabled=false
answer_verification_max_revisions=1
rag_gap_trigger_enabled=false
```

单 Agent：

- `POST /api/chat` 保持兼容，增加可选 `client_request_id`、run ID 和持久事件；
- 同会话有租约互斥，完成 run 可回放；
- 只读 interrupted run 可显式 resume；写入结果未知时拒绝；
- run 表持久化 usage 和耗时，摘要投影计入 sliding 预算。

多 Agent：

- `POST /api/multi-agent/chat` 接受 `message`、可选 `session_id` 和 `task_id`；
- 使用现有 engine/model/session store，返回 `conversation_id`、facts、evidence、trace、
  errors 和 `write_previews`；
- 预览可交给现有 `/api/tickets/confirm`；重复确认同键返回同一工单，取消不落库；
- 当前没有与 `/api/chat` 相同的持久 run、SSE 重放、租约恢复或 `request_usage` 持久化；
- 返回的 `tokens` 是上下文估算，不是供应商 billing usage；多 Agent 子任务可并行，因此不能复用
  单 pending 的旧账本假设。

## 迁移与回滚

首次部署先执行 `db/init.sql` 的 CH11 段，或对既有库执行
`db/ch11-reliability-execution.sql`。四张表都是追加式：

`chat_runs → chat_run_events / chat_session_leases / chat_tool_attempts`。

应用回滚可先关闭三个默认关闭的 RAG 开关并移除多 Agent router。若确认不保留审计记录，按
`chat_run_events → chat_tool_attempts → chat_session_leases → chat_runs` 删除；
不要删除已确认工单，它们是已发生的业务事实。

## 仍有风险

- 多 Agent 仍是进程内 runtime，不是与默认聊天同等级的持久执行器；
- 可靠性只证明 run 级安全重启，不证明节点内 checkpoint；
- MySQL 仅做隔离 schema/store smoke，未做生产规模并发压测；
- 多 Agent 只有 10 dev + 20 held-out 模拟题，且 held-out 已被查看，不能继续据此调路由阈值；
- RAG 历史 prompt 的行为提示降低泛化解释力；三态中大量 uncertain 仍待语义或人工复核；
- 本轮集成本身没有新增付费模型调用。
