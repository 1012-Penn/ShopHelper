# CH16 RAG 消融与答案可信度对照实施计划

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定题集、语料、索引快照和候选上限下，复现四种检索策略与 CH12 三种 RAG 模式，并交付逐题、聚合、成本和失败归因证据。

**Architecture:** 复用 `scripts/eval_ch11.py` 的检索/回答流程、数据契约、价格守卫和 `ControlledRagLoop`，新增 CH16 配对指标、组别运行契约及报告编排。只在 dev 集检查流程；冻结集用于确认契约已锁定后的正式检索对照，任何缺失的模型输出保持未测并附预算，不发起新的付费调用。

**Tech Stack:** Python 3.11、pytest、SQLAlchemy/SQLite、Milvus Lite、本仓库 `scripts.eval_ch11` 与 `scripts.eval_report`。

**Spec:** `docs/superpowers/specs/2026-10-25-ch15-ch17-interview-evidence-roadmap.md` §CH16；用户请求给出的 CH16 验收边界。

## Global Constraints

- `origin/main` 必须为 `dde19f3420b6da00c8e28dd9f503e1de4b6cfde9`；工作只发生于 `codex/ch16-rag-ablation` 独立 worktree。
- 不改 `frozen-v1`、`regression-v1` 的题目、金标、manifest、CH11 历史报告或其他 worktree。
- 检索四组固定候选上限 50、回答上下文上限 10、同一改写缓存、同一快照 `ebfd49478b37ef387c8b124319a450ed73a86ae2a4b9f4fc28d6a52b2ccaa0ba`。
- CH12 三组固定 `hybrid_rerank` 检索及同一 CH11 提示词/裁判，差异限于 `baseline`、`decompose_once`、`controlled_loop` 的 `rag` 配置。
- 付费上限默认为 `$0`；不得执行任何新增付费请求。先交付逐组费用估算，等待统一安排。
- CH13 LightRAG 仅记录为未纳入、未验证；若复杂模式无净收益，推荐保留简单默认路由。
- 不 push、不合并。

---

## File Structure

Create:
- `scripts/ch16_analysis.py` — 配对检索/答案/延迟/成本指标与 Markdown/JSON 聚合。
- `scripts/eval_rag_ablation.py` — dev/frozen 分阶段运行、读取 CH11 逐题结果、写入 CH16 契约与配对产物；付费预算默认零。
- `tests/test_ch16_analysis.py` — 纯指标行为测试。
- `tests/test_ch16_runner.py` — 契约冻结、费用守卫和结果完整性测试。
- `tests/eval/rag_ablation/` — 版本化契约、非敏感逐题轨迹及机器可读聚合。
- `docs/eval/ch16-rag-ablation-report.md` — 口径、结果、失败归因、费用和限制。
- `dev-notes/ch16.md` — 可复现命令、证据等级、已知边界和后续安排。

Do not modify:
- `tests/eval/splits/{frozen-v1,dev}.jsonl`、registry、manifest、CH11/CH12 源实现及历史报告。

### Task 1: 纯指标与配对比较

**Interfaces:**
- `strict_answer_outcome(case: dict, trace: dict) -> bool | None`
- `summarize_group(traces: list[dict]) -> dict`
- `compare_groups(reference: list[dict], candidate: list[dict]) -> dict`
- `paired_bucket_outcomes(groups: dict[str, list[dict]], buckets: tuple[str, ...]) -> dict`

- [x] 先写测试：被 gate 拦下的可答题仍进入严格正确率分母并判错；不完整/拒答答案不能因零断言得满分；R@3/R@5/MRR 将未命中按 0 计；配对结果按相同 `case_id` 给出 gained/lost/tie 与桶分母。
- [x] 运行 `pytest tests/test_ch16_analysis.py -q`，确认因新 API 缺失而失败。
- [x] 实现纯函数，不复用会排除 gate-blocked 样本的 CH11 strict accuracy 分母。
- [x] 再运行该文件，确认通过；指标报告分别列分子、分母、未测数。

### Task 2: 实验契约与无付费运行器

**Interfaces:**
- `build_experiment_contract(split: str, experiment: str, strategy: str, rag_mode: str, source_contract: dict) -> dict`
- `validate_group_contracts(contracts: list[dict]) -> None`
- CLI `python -m scripts.eval_rag_ablation --experiment {retrieval,rag} --split {dev,frozen-v1} --out PATH [--allow-paid] [--max-cost-usd FLOAT]`

- [x] 先写测试：四种检索契约除 `retrieval.strategy` 外一致；三种 CH12 契约只允许 `rag.*` / `harness.rag_mode` 差异；`max_cost_usd` 缺省 0 且未传显式付费开关时不得启动组运行。
- [x] 运行 `pytest tests/test_ch16_runner.py -q`，确认缺少 API 的预期失败。
- [x] 实现 CLI 对 CH11 入口的复用和逐组轨迹保存；保留 `BudgetExceeded`，将缓存未命中记为该组未测，不继续调用。
- [x] 再运行该文件，确认契约、预算和缺失结果行为通过。

### Task 3: 开发集与冻结集产物

- [x] 先复制只读 CH11 Milvus 快照、CH12 私有 SQLite 评测库与 CH11 缓存到本 worktree 的 gitignored `data/`；核验 snapshot SHA 与 CH11 contract 相等。
- [x] 对 dev 运行四种检索策略，保持 strategy 参数之外的 contract 一致；核验 split hash、题数和桶计数。
- [x] dev 产物完成后固定 CH16 contract 和指标口径，再运行 frozen-v1 四组正式检索对照；不得根据 frozen 结果调阈值。
- [x] 按冻结报告的同题响应缓存做零成本答案重放；缺少精确缓存键的变体只标未测，不拿检索命中率替代答案正确率。
- [x] 分别生成 baseline / decompose_once / controlled_loop 的配对输入契约；因未授权付费，只有可精确缓存重放的组计结果，其他逐题轨迹与质量结果明确标未测。

### Task 4: 聚合报告、归因与最终验证

- [x] 报告 R@3/R@5/MRR、分桶 paired gains/losses、严格正确率、条件覆盖、引用支持、claim 级无证据断言、误拒、P50/P95、逻辑/API 调用和成本，并明确每个分母。
- [x] CH12 按 multi_hop/policy_conflict 输出逐题得失与可用延迟；列出失败标签和样例 ID。
- [x] 写逐组追加费用估算、CH13 边界与默认路由建议；刷新 `dev-notes/ch16.md` 和 CH16 报告。
- [x] 运行 CH16 定向测试、离线评测契约校验和 `git diff --check`；核对冻结文件 SHA、历史报告 SHA、根目录状态与其他 worktree 状态未变。
- [x] 形成一个本地提交；不 push、不合并。
