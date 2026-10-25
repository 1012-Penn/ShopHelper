# ch11 评测与语料基线 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch11 的“语料与评测基线”从设计稿落成可复现的语料 v1、冻结数据集 v1、L1–L5 评测 harness 与基线报告，为 ch12/ch13 提供同一套冻结评测契约。

**Architecture:** 以 `corpus/facts/*.yaml` 为结构化业务真源，`knowledge/*.md` 仍是唯一运行时文档源；新增 5 份文档到 `knowledge/`，由 `corpus/manifest.yaml` 注册并驱动 `build_kb`。数据集分 dev / frozen-v1 / regression-v1，冻结集 90 题全部进 git 并记录 SHA。评测在现有严格证据口径上扩展五层指标：检索/证据层用缓存保证确定性，回答/行为层用记录 SHA 的评测回答 prompt、确定性检查优先、可选裁判，过程层记录调用、token、延迟和费用。

**Tech Stack:** Python 3.10、pytest、PyYAML 6、Pydantic v2、Milvus Lite、MySQL 8、BAAI/bge-m3、BAAI/bge-reranker-v2-m3、DeepSeek OpenAI 兼容接口（`deepseek-chat`）。

**Spec:** `docs/superpowers/specs/2026-10-24-ch11-evaluation-baseline-design.md`（总纲：`docs/superpowers/specs/2026-10-24-ch11-ch14-agent-production-roadmap-design.md`）

## Global Constraints

- 本轮只做**语料与评测基线**；同会话并发、持久执行、崩溃恢复、写工具幂等不在本计划内。
- M1 规模：8 份文档、120–180 chunk、150–250 条结构化事实、210 题（dev 60 / frozen 90 / regression 60）。
- frozen-v1 桶分布固定：`single_hop 18`、`multi_hop 16`、`multi_constraint 14`、`entity_mismatch 10`、`policy_conflict 8`、`hard_negative 10`、`clarify 8`、`ood 6`；多轮/工具题 10 条（4 条工具依赖，6 条纯多轮）。
- 冻结集标签、证据、manifest 全部进 git；改题只能升 `frozen-v2` 并让所有方法重跑。
- 付费模型/嵌入/重排调用总预算 **≤ $10**；跨运行台账 `data/ch11_budget.json` 记录累计，调用前检查，超限即停。
- 证据 quote 必须逐字存在，并能被现有 `chunk_markdown` 映射到真实 chunk；旧 47 正 + 13 负文件不得覆盖。
- 阻断项：硬负例/OOD 误放行、无证据断言、`forbidden_claims`、`under_refusal_rate` 不得被其他指标抵消。
- 旧 3 份文档、旧 60 题、旧报告、旧 `evidence_calibration.json` 全部保留；新内容走新路径/新版本号。
- 新语料与新索引会使旧证据校准过期；本轮不覆盖生产 `config/evidence_calibration.json`，contract 标 `calibration.status=stale`，候选校准只在 dev/regression 上做。
- Milvus Lite 单进程：建库/评测/快照必须串行；快照必须在停写并确认无进程持有文件后创建。
- `reports/`、`data/`、`logs/`、`models/` 不入库；聚合报告与 manifest 放 `docs/eval/`，关键数字回写 `dev-notes/ch11.md`。
- 提交前缀 `ch11:`；提交信息末尾保留 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。

## Review Focus

1. **旧硬负例被新语料意外回答**：新增文档若含 `tests/test_ch09_hard_negatives.py` 的 `absent_terms`，或使旧负例变得可答，必须让测试失败；不能静默改旧标签。
2. **证据 quote 漂移**：文档改写后 quote 不再逐字存在，数据集校验必须报出 case/group，不能只报“映射失败”。
3. **冻结集泄漏**：查询、case id、fact_id、`forbidden_claims` 或评测标记出现在 `knowledge/`，泄漏扫描必须阻断冻结。
4. **快照不一致**：Milvus 文件仍在写入时复制，只比计数会漏掉内容错配；必须比 `build_id`、`pending=0`、逐 chunk 文本/类别哈希。
5. **预算或模型漂移**：模型名/价格变化、价格未核实或单次调用超预算时，harness 必须在调用前停止并记录，而不是跑完才发现超支。

---

## Context

现有 47 正 + 13 负、约 59 chunk 的基线适合回归，但已饱和（历史 `hybrid_rerank` 严格 R@10=1.000），无法证明 ch12/ch13 新架构的增益。ch11 的首要产出不是更高分，而是一套后续方案不能各造一套题的共同评测契约：语料可审计、题目冻结、五层指标分开计量、逐题轨迹和成本可复现。

设计稿已完成，Codex 复验还剩 P1/P2 返工项（快照校验细节、`dev-notes` 归因与第三次中断记录）。本轮用户决定：只做评测基线实施；预算 ≤ $10；冻结集标签全部进 git；冻结集预留 8–12 道多轮/工具题给 ch12。

## 已定技术与口径

- **文档源**：`knowledge/*.md` 是唯一运行时文档源；**不创建 `corpus/docs/`**。结构化事实放 `corpus/facts/*.yaml`，来源与证据单元放 `corpus/sources.yaml`，文档注册与规模放 `corpus/manifest.yaml`。这一偏离写入设计稿并说明理由（避免文档双份漂移）。
- **回归集**：不复制、不改写旧文件；`regression-v1` 通过 registry 加载旧两文件，再用 `data/eval/overlays/regression-v1.labels.json` 叠加默认字段与 C04/C05/高风险题的 `expected_behavior`/条件/禁用断言。
- **回答层**：`scripts/eval_prompts.py` 提供独立的评测回答 prompt，明确要求：有引用、缺槽位先问、需核验只给流程、证据不足/冲突时拒答或说明生效版本。它不是线上 system prompt；报告必须写明这是“评测回答策略”，并记录 prompt SHA。
- **多轮/工具**：多轮题用 `conversation_context` + 最终 `query`；工具题用 `requires_tools` + `tool_fixtures` 记录，ch11 只评检索/回答，L4 对工具题标 `not_applicable`，留给 ch12 跑完整工具链。
- **快照**：`scripts/eval_snapshot.py` 在停写后复制 Milvus 文件及其 sibling lock/wal 文件，重开快照并逐项校验。
- **模型与价格**：沿用 `.env` 的 `deepseek-chat`、`BAAI/bge-m3`、`BAAI/bge-reranker-v2-m3`；`config/pricing.yaml` 初始 `verified: false`，未核实官方价格前禁止付费运行。
- **校准**：旧 `config/evidence_calibration.json` 绑定旧语料与旧分布，本轮标记 stale；候选校准只用 dev/regression，不碰 frozen，不覆盖生产配置。

## File Structure

**Create**
- `corpus/manifest.yaml` — 8 份文档、content_type、规模上下限、corpus version
- `corpus/facts/{catalog,orders,policy,promotion,after_sales}.yaml` — 结构化事实
- `corpus/sources.yaml` — source 清单与 `evidence_units`
- `scripts/corpus_manifest.py` — manifest 加载、`doc_types()`、`corpus_sha256()`
- `scripts/corpus_facts.py` — Fact/EvidenceUnit/Source 模型、FactBook、引用完整性
- `scripts/eval_dataset.py` — v2 schema、split registry、overlay、校验
- `scripts/validate_eval_data.py` — 数据校验/冻结 CLI
- `scripts/leakage_scan.py` — 查询/标记/事实泄漏扫描
- `scripts/eval_checks.py` — L3/L4 确定性检查（数值、条件、禁用断言、引用、槽位）
- `scripts/eval_metrics.py` 追加 — `safe_div`、`evidence_precision`、`macro_average`
- `scripts/eval_cost.py` — 价格表、累计台账、调用前预算护栏
- `scripts/eval_cache.py` — 嵌入/重排/回答缓存
- `scripts/eval_contract.py` — `eval_contract.json`、失败枚举、contract diff
- `scripts/eval_snapshot.py` — 快照复制与五项一致性校验
- `scripts/eval_prompts.py` — 评测回答 prompt 与可选裁判 prompt
- `scripts/eval_stats.py` — bootstrap、McNemar
- `scripts/eval_report.py` — 分层聚合、阻断项、报告渲染
- `scripts/ch11_harness.py` — `HarnessConfig`、`run_case`、`run_split`
- `scripts/eval_ch11.py` — 模式化 CLI
- `scripts/acceptance_ch11.py` — 离线验收 + 可选付费基线
- `config/pricing.yaml`
- `tests/eval/splits/{registry.json,dev.jsonl,frozen-v1.jsonl}`
- `tests/eval/overlays/regression-v1.labels.json`
- `tests/eval/manifests/frozen-v1.json`
- `tests/eval/CHANGELOG.md`
- `tests/eval/caches/rewrite/{dev,frozen-v1,regression-v1}.json`
- `docs/eval/{ch11-historical-baseline.md,ch11-baseline-report.md,ch11-acceptance.md}`
- `docs/eval/baselines/2026-10-24-m1.json`
- `tests/test_corpus_manifest.py`、`test_corpus_facts.py`、`test_knowledge_docs_ch11.py`、
  `test_eval_dataset.py`、`test_ch11_regression_overlay.py`、`test_ch11_leakage.py`、
  `test_ch11_eval_metrics.py`、`test_ch11_checks.py`、`test_ch11_cost.py`、
  `test_ch11_cache.py`、`test_ch11_contract.py`、`test_ch11_snapshot.py`、
  `test_ch11_harness.py`、`test_ch11_report.py`

**Modify**
- `scripts/build_kb.py` — manifest 驱动、`--dry-run`、build manifest
- `scripts/rewrite_cache.py` — 支持显式 `eval_sha256` 与薄 CLI
- `scripts/gen_evidence_signals.py` — 修正改写缓存路径；正样本按非空 `required_evidence` 选
- `app/vector_store.py` — `all_rows()` 供逐 chunk 内容校验
- `tests/helpers.py` — `FakeVectorStore.all_rows()`
- `tests/test_build_kb.py` — 文档数由 manifest 决定
- `tests/test_ch04_eval_set.py` — `DOC_TYPES` 改从 manifest 导入
- `pyproject.toml` — 声明 `pyyaml==6.0.3`
- `knowledge/{商品目录,订单与支付,会员与优惠,政策版本说明,售后核验指引}.md`
- `dev-notes/ch11.md`、`README.md`、设计稿、`docs/agent-reviews/`
- `docs/superpowers/plans/2026-10-24-ch11-evaluation-baseline.md` — 本计划落库

**Not changing**
- `tests/eval/ch04_eval_set.jsonl`、`tests/eval/ch09_junk_set.jsonl` 的题目与标签
- `config/evidence_calibration.json` 的旧产物
- 聊天主链路、工具系统、MCP、飞轮、主题分类器

---

## Task 0：工作分支、设计稿复验返工

**Files:**
- Create (branch/worktree): `codex/ch11-eval-impl`
- Modify: `docs/superpowers/specs/2026-10-24-ch11-evaluation-baseline-design.md`
- Modify: `dev-notes/ch11.md`
- Create: `docs/agent-reviews/ch11-01-evaluation-design-review.md`（从 Codex worktree 复制）
- Create: `docs/superpowers/plans/2026-10-24-ch11-evaluation-baseline.md`（本计划）

**Interfaces:**
- Consumes: 分支 `codex/ch11-eval-reliability`（含任务卡提交 `0139ea8`）；Codex worktree `/root/.codex/worktrees/ch11-production-baseline/ShopHelper` 中未跟踪的设计稿、`dev-notes/ch11.md`、`docs/agent-reviews/`。
- Produces: 干净的 ch11 实施分支；设计稿覆盖 Codex P1/P2 返工项与文档源偏离。

- [ ] **Step 1: 建隔离分支**

```bash
cd /root/workplace/ShopHelper
git switch -c codex/ch11-eval-impl codex/ch11-eval-reliability
git status --short   # FinanceBench 改动不得进入提交
```

- [ ] **Step 2: 把设计产物带入分支**

从 `/root/.codex/worktrees/ch11-production-baseline/ShopHelper` 复制：
`docs/superpowers/specs/2026-10-24-ch11-evaluation-baseline-design.md`、`dev-notes/ch11.md`、`docs/agent-reviews/ch11-01-evaluation-design-review.md`。

- [ ] **Step 3: 修 P1 快照校验**

在设计稿 §6.6 与 §8 的 E2 中写死五项校验：`build_id`、`pending=0`、MySQL/Milvus id 集合相等、逐 chunk 文本/类别哈希相等、快照文件 SHA-256；并写明“创建前确认停写并检查持有 db 文件的进程”。

- [ ] **Step 4: 修 P2 留痕归因**

`dev-notes/ch11.md`：
- “用户关键原话”改为“Codex 委托指令”；
- 补第三次 CLI 轮次上限中断与报告值 `$0.599`；
- 状态改为“未运行付费评测、embedding、rerank 或建库；CLI 自身调用另记成本”。

- [ ] **Step 5: 记录两项设计偏离**

设计稿 §3.2：删除 `corpus/docs/*.md` 作为第二份文档源，改为“`knowledge/*.md` 唯一运行时源；事实与来源在 `corpus/`”。§4.3 补 `as_of_date`、`condition_aliases`、`slot_aliases`、`verification_markers`、`requires_tools`、`tool_fixtures`。§6.1 补 pricing/contract 字段与 `calibration.status=stale`。

- [ ] **Step 6: 提交**

```bash
git add docs/superpowers/specs/2026-10-24-ch11-evaluation-baseline-design.md \
  docs/agent-reviews/ch11-01-evaluation-design-review.md dev-notes/ch11.md \
  docs/superpowers/plans/2026-10-24-ch11-evaluation-baseline.md
git commit -m "ch11: 设计复验返工与评测基线实施计划"
```

- [ ] **Step 7: 验证**

Run: `git diff --check && git status --short`
Expected: 无空白错误；无业务代码/数据改动；FinanceBench 文件未暂存。

---

## Task 1：语料 manifest、build_kb 注册与 dry-run

**Files:**
- Create: `corpus/manifest.yaml`
- Create: `scripts/corpus_manifest.py`
- Modify: `scripts/build_kb.py`
- Modify: `tests/test_build_kb.py`、`tests/test_ch04_eval_set.py`
- Modify: `pyproject.toml`
- Test: `tests/test_corpus_manifest.py`

**Interfaces:**
```python
@dataclass(frozen=True)
class DocSpec:
    file: str; stem: str; content_type: str; role: str

@dataclass(frozen=True)
class CorpusManifest:
    corpus_version: str
    docs: tuple[DocSpec, ...]
    expected_docs: int | None
    expected_chunks_min: int | None
    expected_chunks_max: int | None
    expected_facts_min: int | None
    expected_facts_max: int | None

def load_manifest(path: Path = MANIFEST_PATH) -> CorpusManifest: ...
def doc_types(manifest: CorpusManifest | None = None) -> dict[str, str]: ...
def registered_docs(manifest=None) -> tuple[str, ...]: ...
def expected_docs(manifest=None) -> int: ...
def corpus_sha256(docs_dir: Path = DOCS_DIR, manifest=None) -> str: ...

# build_kb.py
def chunk_corpus(docs_dir=DOCS_DIR, *, max_chars=500, overlap_chars=80,
                 manifest=None) -> dict[str, list[Chunk]]: ...
def build(docs_dir, kb, vectors, embedder, *, max_chunks=None, max_chars=500,
          overlap_chars=80, rebuild_docs=True, manifest=None,
          embedding_model: str = "", manifest_out: Path | None = None) -> dict: ...
```
- 未注册文档抛 `ValueError`；`--dry-run` 只切分并校验规模，不建 Settings/DB/API；真实构建后写 `data/build_manifest.json`。
- `stats` 保留 `docs/chunks/vectorized`，新增 `doc_chunk_counts`、`corpus_sha256`。

- [ ] **Step 1: 写失败测试**

```python
def test_manifest_doc_types_match_registered_files(): ...
def test_unregistered_doc_rejected(): ...
def test_corpus_sha256_stable_across_order(): ...
def test_build_dry_run_matches_chunker(): ...
def test_ch04_eval_doc_types_come_from_manifest(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_corpus_manifest.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现 manifest 与 build 改造**

`corpus/manifest.yaml` 先登记现有 3 份文档，规模上下限暂设 50–80 chunk；Task 3 追加 5 份文档后改为 8 docs / 120–180 chunk。`build_kb` 仍只 glob `knowledge/*.md`，但用 manifest 决定 content_type 和期望规模。

- [ ] **Step 4: 运行并确认通过**

Run: `.venv/bin/pytest tests/test_corpus_manifest.py tests/test_build_kb.py tests/test_ch04_eval_set.py -q && .venv/bin/python -m scripts.build_kb --dry-run`
Expected: PASS；dry-run 输出 3 docs / 58 chunk。

- [ ] **Step 5: 提交**

```bash
git add corpus/manifest.yaml scripts/corpus_manifest.py scripts/build_kb.py \
  pyproject.toml tests/test_corpus_manifest.py tests/test_build_kb.py tests/test_ch04_eval_set.py
git commit -m "ch11: 语料 manifest 与 build_kb 文档类型注册"
```

---

## Task 2：结构化事实 schema 与校验器

**Files:**
- Create: `scripts/corpus_facts.py`
- Create: `corpus/sources.yaml`
- Create: `corpus/facts/{catalog,orders,policy,promotion,after_sales}.yaml`（先 2–3 条种子事实）
- Test: `tests/test_corpus_facts.py`

**Interfaces:**
```python
class CorpusDataError(ValueError): ...
FactValue = int | float | str | bool | list[str]

class EvidenceUnit(BaseModel):
    evidence_id: str          # ^EV-[A-Z0-9-]+$
    fact_ids: list[str]       # min_length=1
    doc: str
    section: str
    quotes: list[str]         # min_length=1

class Fact(BaseModel):
    fact_id: str              # ^[A-Z]+-[A-Z0-9-]+$
    entity_type: Literal["sku","order","policy","promotion","after_sales","region"]
    entity_id: str
    key: str
    value: FactValue
    unit: str = ""
    conditions: list[str] = []
    effective_from: date | None = None
    effective_to: date | None = None
    supersedes: list[str] = []
    source_ref: str
    evidence_ids: list[str]   # min_length=1
    notes: str = ""

class Source(BaseModel):
    source_id: str
    type: Literal["internal_rule","public_reference"]
    title: str
    rule_no: str = ""
    uri: str = ""
    retrieved_at: date | None = None
    license: str
    used_for: Literal["fact","wording_only"] = "fact"
    author: str = ""
    reviewer: str = ""
    confidence: Literal["high","medium","low"] = "high"

@dataclass(frozen=True)
class FactBook:
    facts: dict[str, Fact]
    evidence: dict[str, EvidenceUnit]
    sources: dict[str, Source]
    def fact(self, fact_id: str) -> Fact: ...
    def facts_for_entity(self, entity_type: str, entity_id: str) -> list[Fact]: ...
    def evidence_for_fact(self, fact_id: str) -> list[EvidenceUnit]: ...
    def sha256(self) -> str: ...
    def count(self) -> int: ...

def load_factbook(root: Path = ROOT / "corpus") -> FactBook: ...
def validate_factbook(book, docs_dir=DOCS_DIR, *, manifest=None) -> list[str]: ...
```
- 校验：id 唯一；每条 fact 有 source/evidence；source_ref 存在；evidence 的 doc 已注册、section 存在、quote 逐字存在且能落在同一 chunk；生效日期有序；`supersedes` 目标存在且无环；同一 `(entity_type, entity_id, key)` 不同值必须区间不重叠或有显式冲突组。

- [ ] **Step 1: 写失败测试**

```python
def test_loads_factbook_and_sha_is_stable(): ...
def test_duplicate_fact_id_rejected(): ...
def test_evidence_quote_must_exist_verbatim(): ...
def test_evidence_quote_must_land_in_one_chunk(): ...
def test_unregistered_evidence_doc_rejected(): ...
def test_missing_source_ref_rejected(): ...
def test_supersedes_cycle_rejected(): ...
def test_overlapping_conflicting_values_rejected(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_corpus_facts.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现模型与校验**

用 Pydantic 校验字段，用 `scripts.eval_metrics.normalize_span` 做 quote 归一化；quote 落块校验调用 `app.chunking.chunk_markdown`。

- [ ] **Step 4: 运行并确认通过**

Run: `.venv/bin/pytest tests/test_corpus_facts.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add scripts/corpus_facts.py corpus/sources.yaml corpus/facts tests/test_corpus_facts.py
git commit -m "ch11: 结构化事实 schema 与引用完整性校验"
```

---

## Task 3：新增五份知识文档

**Files:**
- Create: `knowledge/商品目录.md`、`订单与支付.md`、`会员与优惠.md`、`政策版本说明.md`、`售后核验指引.md`
- Modify: `corpus/manifest.yaml`
- Test: `tests/test_knowledge_docs_ch11.py`

**Content requirements:**
- `商品目录.md`：14–16 个 SKU；价格/参数/兼容/替代表；现有 SKU（`SH-E300`、`SH-E120`、`SH-K870`、`SH-K104`、`SH-B5`、`SH-W20`）与 `商品FAQ.md` 一致；**全篇不要出现 `无线充电` 这个字符串，也不写 SH-E300 的充电方式**。
- `订单与支付.md`：10–12 个合成订单，覆盖未支付、已支付未发货、已发货、已签收、已取消、退货中、已退款；带时间、金额、会员等级、地区；支付渠道与退款路径。
- `会员与优惠.md`：会员等级、95 折适用范围与排除、叠加/互斥、合并订单包邮门槛、偏远补运。可以提生日礼包存在，**不写礼包内容**。
- `政策版本说明.md`：v1 签收早于 2026-06-01 为 7 天，v2 自 2026-06-01 起 15 天；生效区间、supersedes、冲突处理。`退货政策.md` 保留为 v1 基础文档。
- `售后核验指引.md`：照片/视频凭证、检测流程、时限、责任划分、“是否质量问题以检测结果为准”；**不定义“开胶属于质量问题”**，**不写维修价格/收费金额**。

- [ ] **Step 1: 写失败测试**

```python
def test_manifest_covers_all_knowledge_docs(): ...
def test_m1_scale_docs_and_chunks(): ...          # 8 docs, chunks in [120,180]
def test_new_docs_have_required_sections(): ...
def test_new_tables_replicate_header(): ...
def test_new_docs_do_not_contain_hard_negative_absent_terms(): ...
def test_knowledge_contains_no_eval_markers(): ... # FROZEN-/DEV-/fact_id/required_evidence/本题应回答/标准答案
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_knowledge_docs_ch11.py -q`
Expected: FAIL。

- [ ] **Step 3: 写文档并登记 manifest**

`tests/test_knowledge_docs.py` 的表格断言已只针对 `售后手册.md`，无需修改；新文档表格只要每块带表头即可。新文档不得包含 `test_ch09_hard_negatives.py` 的 `absent_terms`。

- [ ] **Step 4: 跑相关测试**

Run: `.venv/bin/pytest tests/test_knowledge_docs_ch11.py tests/test_knowledge_docs.py tests/test_chunking.py tests/test_ch09_hard_negatives.py tests/test_build_kb.py -q && .venv/bin/python -m scripts.build_kb --dry-run`
Expected: 全绿；dry-run 8 docs / 120–180 chunk。

- [ ] **Step 5: 提交**

```bash
git add knowledge corpus/manifest.yaml tests/test_knowledge_docs_ch11.py
git commit -m "ch11: 新增五份知识文档与 M1 语料规模"
```

---

## Task 4：编写 150–250 条事实与证据单元

**Files:**
- Modify: `corpus/facts/{catalog,orders,policy,promotion,after_sales}.yaml`
- Modify: `corpus/sources.yaml`、`corpus/manifest.yaml`
- Test: `tests/test_corpus_facts.py`

**Coverage target:** 14–16 SKU × 5–7 facts；10–12 订单 × 5 facts；约 25 条政策；约 20 条优惠；约 20 条售后；约 10 条地区。目标 180–230 facts、90–130 evidence units。

- [ ] **Step 1: 写失败测试**

```python
def test_m1_fact_scale(): ...                   # 150 <= count <= 250
def test_every_fact_has_source_and_evidence(): ...
def test_policy_versions_have_ordered_ranges_and_supersedes(): ...
def test_orders_cover_seven_statuses(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_corpus_facts.py -q`
Expected: FAIL。

- [ ] **Step 3: 写事实与证据单元**

文档正文只放业务表述，不放 fact_id、题号、答案或“标准答案”。硬负例对应的缺失事实不得补进事实库。

- [ ] **Step 4: 跑校验并提交**

Run: `.venv/bin/pytest tests/test_corpus_facts.py -q`
Expected: PASS。

```bash
git add corpus/facts corpus/sources.yaml corpus/manifest.yaml tests/test_corpus_facts.py
git commit -m "ch11: 150+ 结构化业务事实与证据单元"
```

---

## Task 5：数据集 v2 schema、registry 与校验 CLI

**Files:**
- Create: `scripts/eval_dataset.py`
- Create: `scripts/validate_eval_data.py`
- Create: `tests/eval/splits/registry.json`（先只有 regression-v1）
- Test: `tests/test_eval_dataset.py`

**Interfaces:**
```python
SCHEMA_VERSION = 2
BUCKETS = ("single_hop","multi_hop","multi_constraint","entity_mismatch",
           "policy_conflict","hard_negative","clarify","ood")
EXPECTED_FROZEN = {"single_hop":18,"multi_hop":16,"multi_constraint":14,
                   "entity_mismatch":10,"policy_conflict":8,"hard_negative":10,
                   "clarify":8,"ood":6}

class EvidenceRef(BaseModel):
    doc: str; section: str | None = None; quotes: list[str]; fact_ids: list[str] = []

class GoldFact(BaseModel):
    fact_id: str; key: str = ""; value: FactValue; unit: str = ""
    conditions: list[str] = []; tolerance: float = 0.0
    effective_from: date | None = None; effective_to: date | None = None

class ConversationTurn(BaseModel):
    role: Literal["user","assistant"]; content: str

class ToolFixture(BaseModel):
    tool: str; arguments: dict = {}; result: dict

class EvalCase(BaseModel):
    schema_version: Literal[2] = 2
    id: str; split: str; bucket: str; label_version: str = "v2"
    query: str; conversation_context: list[ConversationTurn] = []
    expected_behavior: Literal["answer","answer_with_conditions",
                               "verify_then_answer","clarify","refuse"]
    answerable: bool
    evidence_state: Literal["sufficient","partial","insufficient","conflicting"]
    required_slots: list[str] = []; clarifying_slots: list[str] = []
    gold_facts: list[GoldFact] = []; required_evidence: list[EvidenceRef] = []
    required_conditions: list[str] = []
    condition_aliases: dict[str, list[str]] = {}
    slot_aliases: dict[str, list[str]] = {}
    verification_markers: list[str] = []
    forbidden_claims: list[str] = []
    expected_refusal_reason: str = ""
    as_of_date: date | None = None
    requires_tools: list[str] = []; tool_fixtures: list[ToolFixture] = []
    source: str = "synthetic"; source_ref: str = ""
    annotator: str = ""; reviewer: str = ""
    review_status: Literal["draft","reviewed","frozen"] = "draft"
    notes: str = ""

def load_registry(root=ROOT) -> dict: ...
def load_cases_from_path(path: Path) -> list[dict]: ...
def apply_overlay(cases: list[dict], overlay: dict) -> list[dict]: ...
def load_split(split: str, root=ROOT) -> list[dict]: ...
def split_sha256(cases: list[dict]) -> str: ...
def dataset_sha256(path: Path) -> str: ...
def bucket_counts(cases) -> dict[str, int]: ...
def behavior_counts(cases) -> dict[str, int]: ...
def validate_cases(cases, *, split: str, bucket_counts=None,
                   docs_dir: Path | None = None, factbook=None) -> None: ...
```
- 校验：id/query 唯一且跨 dev/frozen 不冲突；`refuse` → `answerable=false` + 原因 + `evidence_state=insufficient`；`clarify` → `clarifying_slots` 非空；`verify_then_answer` → 条件与禁用断言非空；`policy_conflict` → `evidence_state=conflicting` + `as_of_date`；`hard_negative/ood` → refuse、无 `required_evidence`；证据 doc 已注册、quote 逐字、能落块；`gold_facts` 存在且值与 FactBook 一致；`requires_tools` 有 `tool_fixtures`；新 split 的 answer* 题至少一条 gold fact。

- [ ] **Step 1: 写失败测试**

```python
def test_schema_rejects_refuse_without_reason(): ...
def test_schema_rejects_clarify_without_slots(): ...
def test_schema_rejects_policy_conflict_without_as_of_date(): ...
def test_overlay_defaults_and_explicit_entries_merge(): ...
def test_split_sha256_ignores_review_metadata(): ...
def test_duplicate_ids_and_queries_rejected(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_eval_dataset.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现 loader/validator/overlay 与 CLI**

`validate_eval_data.py --split all --check-evidence --check-facts --check-buckets`；数据错误退出 4。先让 `regression-v1` registry 可用。

- [ ] **Step 4: 运行并确认通过**

Run: `.venv/bin/pytest tests/test_eval_dataset.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add scripts/eval_dataset.py scripts/validate_eval_data.py \
  tests/eval/splits/registry.json tests/test_eval_dataset.py
git commit -m "ch11: 评测数据集 v2 schema 与 split 注册"
```

---

## Task 6：dev 60 题开发集

**Files:**
- Create: `tests/eval/splits/dev.jsonl`
- Modify: `tests/eval/splits/registry.json`
- Test: `tests/test_eval_dataset.py`

**Bucket distribution:** `single_hop=12`、`multi_hop=11`、`multi_constraint=9`、`entity_mismatch=7`、`policy_conflict=5`、`hard_negative=7`、`clarify=5`、`ood=4`。
**Multi-turn:** 6 条（3 `multi_hop`、2 `multi_constraint`、1 `clarify`）。

- [ ] **Step 1: 写失败测试**

```python
def test_dev_split_schema_and_exact_bucket_counts(): ...
def test_dev_queries_unique(): ...
def test_dev_every_doc_covered(): ...
def test_dev_evidence_reachable_by_chunker(): ...
def test_dev_gold_facts_exist(): ...
def test_dev_multi_turn_cases_have_context(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_eval_dataset.py -q`
Expected: FAIL。

- [ ] **Step 3: 编写 dev 题**

覆盖 8 份文档；`multi_hop` 至少两组跨文档证据；`multi_constraint` 至少两个条件；`policy_conflict` 带 `as_of_date`；问法变体只放 dev，不进 frozen。

- [ ] **Step 4: 跑校验并提交**

Run: `.venv/bin/python -m scripts.validate_eval_data --split dev --check-evidence --check-facts --check-buckets`
Expected: 60 题通过。

```bash
git add tests/eval/splits/dev.jsonl tests/eval/splits/registry.json tests/test_eval_dataset.py
git commit -m "ch11: dev 60 题开发集"
```

---

## Task 7：frozen-v1 90 题与冻结 manifest

**Files:**
- Create: `tests/eval/splits/frozen-v1.jsonl`
- Modify: `tests/eval/splits/registry.json`
- Create: `tests/eval/manifests/frozen-v1.json`
- Modify: `scripts/validate_eval_data.py`（`--freeze`）
- Test: `tests/test_eval_dataset.py`

**Exact buckets:** `single_hop=18`、`multi_hop=16`、`multi_constraint=14`、`entity_mismatch=10`、`policy_conflict=8`、`hard_negative=10`、`clarify=8`、`ood=6`。
**Behavior guideline:** `answer≈26`、`answer_with_conditions≈24`、`verify_then_answer≈8`、`clarify≈16`、`refuse≈16`。
**Multi-turn/tool:** exactly 10；`multi_hop=4`、`multi_constraint=4`、`clarify=2`；其中 4 条 `requires_tools` + `tool_fixtures`，6 条纯 `conversation_context`。

- [ ] **Step 1: 写失败测试**

```python
def test_frozen_exact_bucket_counts(): ...
def test_frozen_90_unique_ids_and_queries(): ...
def test_frozen_has_10_multi_turn_and_4_tool_cases(): ...
def test_frozen_no_near_duplicate_of_dev(): ...   # 12-gram containment < 0.8
def test_frozen_manifest_matches_file(): ...
def test_frozen_all_reviewed_and_frozen(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_eval_dataset.py -q`
Expected: FAIL。

- [ ] **Step 3: 编写冻结题**

全部 `review_status="frozen"`、`annotator="claude-code"`、`reviewer="codex"`；不得由文档标题直接生成，不得与 dev 共用模板；工具题只记录 fixture，不执行工具。

- [ ] **Step 4: 冻结并校验**

Run:
```bash
.venv/bin/python -m scripts.validate_eval_data --split frozen-v1 \
  --check-evidence --check-facts --check-buckets --freeze \
  --confirm-reviewer codex --write-manifest
```
Expected: 90 题通过；manifest 记录 `file_sha256/canonical_sha256/n/buckets/behaviors/multi_turn/tool_count/corpus_version/design_doc_sha/reviewer/frozen_at`。

- [ ] **Step 5: 提交**

```bash
git add tests/eval/splits/frozen-v1.jsonl tests/eval/splits/registry.json \
  tests/eval/manifests/frozen-v1.json scripts/validate_eval_data.py tests/test_eval_dataset.py
git commit -m "ch11: frozen-v1 90 题冻结集与冻结 manifest"
```

---

## Task 8：regression-v1 标签覆盖层与 C04/C05 审计

**Files:**
- Create: `tests/eval/overlays/regression-v1.labels.json`
- Create: `tests/eval/CHANGELOG.md`
- Modify: `tests/eval/splits/registry.json`
- Modify: `scripts/eval_dataset.py`
- Test: `tests/test_ch11_regression_overlay.py`

**Registry entry:**
```json
{
  "kind": "composed", "n": 60, "label_version": "v2-overlay",
  "cases": [
    {"path": "tests/eval/ch04_eval_set.jsonl",
     "overlay": "tests/eval/overlays/regression-v1.labels.json"},
    {"path": "tests/eval/ch09_junk_set.jsonl", "overlay": null}
  ]
}
```
- Overlay 格式 `{"defaults": {...}, "cases": {"C04": {...}, ...}}`；显式条目覆盖 `C01/C02/C04/C05/E01/E03/E04/E06/E08/F04`。
- 标签：`C01/C02/E01/E03/E06/E08/F04`、`C04` → `answer_with_conditions`；`C05/E04` → `verify_then_answer`。
- C04：`evidence_state=partial`、`required_slots=["sku","member_status"]`、条件「会员身份/商品参与活动/最终价格以页面为准」、禁用「可以便宜/给你打 95 折/所有商品 95 折」。
- C05：`evidence_state=partial`、`required_slots=["order_id","damage_photo"]`、条件「需售后检测/照片凭证」「是否属质量问题以检测结果为准」、禁用「算质量问题/一定可以免费退/开胶不属于质量问题」、`verification_markers=["照片","检测","核验","售后"]`。
- Changelog 记录旧标签、新标签、理由、影响题号、“旧报告不覆盖”。
- `forbidden_claims` 必须是具体断言，且不得逐字出现在任何知识文档；Task 9 的泄漏扫描会阻断。

- [ ] **Step 1: 写失败测试**

```python
def test_regression_has_60_unique_cases(): ...
def test_regression_preserves_required_evidence_and_answerable(): ...
def test_c04_and_c05_new_labels_exact(): ...
def test_ten_audited_cases_have_expected_behavior(): ...
def test_all_regression_cases_have_v2_defaults(): ...
def test_changelog_covers_every_overlay_id(): ...
def test_source_eval_files_not_modified(): ...   # 与测试常量中的 SHA 对比
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_ch11_regression_overlay.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现 overlay 并写 changelog**

旧 47/13 文件一个字节都不改；overlay 深拷贝后 merge。

- [ ] **Step 4: 跑校验并提交**

Run: `.venv/bin/python -m scripts.validate_eval_data --split regression-v1 --check-evidence --check-buckets && .venv/bin/pytest tests/test_ch11_regression_overlay.py tests/test_ch04_eval_set.py tests/test_ch09_hard_negatives.py -q`
Expected: 60 题通过；旧测试不变。

```bash
git add tests/eval/overlays/regression-v1.labels.json tests/eval/CHANGELOG.md \
  tests/eval/splits/registry.json scripts/eval_dataset.py tests/test_ch11_regression_overlay.py
git commit -m "ch11: regression-v1 标签覆盖层与 C04/C05 审计"
```

---

## Task 9：答案泄漏扫描

**Files:**
- Create: `scripts/leakage_scan.py`
- Test: `tests/test_ch11_leakage.py`
- Modify: `tests/eval/manifests/frozen-v1.json`（写入 leakage 结果）

**Interfaces:**
```python
@dataclass(frozen=True)
class LeakFinding:
    split: str; case_id: str
    kind: str       # exact_query | case_id | fact_id | forbidden_claim |
                    # eval_marker | ngram_overlap | eval_file_in_knowledge
    detail: str; severity: str  # blocking | warning
    score: float = 0.0

def scan_cases(cases, docs: dict[str,str], *, factbook=None,
               ngram: int = 8, warn_threshold: float = 0.6,
               block_threshold: float = 0.9) -> list[LeakFinding]: ...
def scan_all(root=ROOT) -> list[LeakFinding]: ...
def main() -> int: ...  # blocking -> 1; warning -> 0
```
- 阻断：query/case id/fact_id/`forbidden_claim` 逐字出现在知识文档；知识文档含评测标记；`knowledge/` 下出现评测文件。
- 警告：query 字符 8-gram 包含度 ≥0.6；≥0.9 升阻断。
- 不扫描 `evidence_units` 的 quote（它们本来就该在文档中）。

- [ ] **Step 1: 写失败测试**

```python
def test_exact_query_in_doc_is_blocking(): ...
def test_case_id_and_fact_id_in_doc_are_blocking(): ...
def test_forbidden_claim_in_doc_is_blocking(): ...
def test_paraphrase_is_not_blocking(): ...
def test_fact_evidence_quotes_are_not_findings(): ...
def test_real_splits_have_no_blocking_findings(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_ch11_leakage.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现扫描并跑真实数据**

```bash
.venv/bin/python -m scripts.leakage_scan --split all
```
Expected: 无 blocking；warning 记入 manifest 与报告。若有 blocking，先改题或改文档，再重新 freeze（Task 7 SHA 随之更新）。

- [ ] **Step 4: 提交**

```bash
git add scripts/leakage_scan.py tests/test_ch11_leakage.py tests/eval/manifests/frozen-v1.json
git commit -m "ch11: 评测题与知识文档泄漏扫描"
```

---

## Task 10：L1–L4 指标与确定性检查

**Files:**
- Modify: `scripts/eval_metrics.py`
- Create: `scripts/eval_checks.py`
- Test: `tests/test_ch11_eval_metrics.py`、`tests/test_ch11_checks.py`

**Interfaces:**
```python
# eval_metrics.py
def safe_div(num: float, den: float) -> float | None: ...
def evidence_precision(ranked_ids: list[int], support_ids: set[int], cap: int = 10) -> float | None: ...
def macro_average(by_bucket: dict[str, float]) -> float | None: ...

# eval_checks.py
@dataclass(frozen=True)
class CheckOutcome:
    name: str; passed: bool; score: float = 0.0; detail: str = ""

def normalize_answer(text: str) -> str: ...
def number_covered(answer: str, value, unit: str, tolerance: float = 0.0) -> bool: ...
def condition_coverage(answer: str, conditions: list[str], aliases=None) -> CheckOutcome: ...
def forbidden_claims_found(answer: str, forbidden: list[str]) -> list[str]: ...
def clarifying_slot_recall(answer: str, required_slots: list[str], aliases=None) -> CheckOutcome: ...
def clarifying_precision(answer: str, required_slots: list[str], aliases=None) -> CheckOutcome: ...
def gold_fact_accuracy(answer: str, gold_facts: list[dict]) -> CheckOutcome: ...
def parse_citations(answer: str) -> list[tuple[str,int]]: ...
def citation_support_rate(answer: str, evidence_by_n: dict[int,str], threshold: float = 0.30) -> CheckOutcome: ...
def classify_behavior(answer: str) -> Literal["answer","clarify","refuse"]: ...
def check_case_deterministic(case: dict, answer: str, evidence_by_n: dict[int,str]) -> dict: ...
```
- `check_case_deterministic` 返回 gold fact 准确率、条件覆盖、禁用断言、槽位召回/精确、引用支持、`refused`、`behavior_pred`、失败标签；**不合成单一总分**。
- `regression-v1` 旧题没有 `gold_facts` 时，`gold_facts_checked=0`，报告不得据此计算严格数值正确率。
- `unsupported_claim_rate` = 无角标或角标不支持的 substantive claim / 全部 substantive claim；确定性检查优先，LLM 裁判只处理同义转述。

- [ ] **Step 1: 写失败测试**

```python
def test_number_units(): ...
def test_forbidden_claim_normalization(): ...
def test_clarify_slot_recall_and_precision(): ...
def test_citation_support(): ...
def test_uncited_substantive_answer_is_unsupported(): ...
def test_no_composite_score_key(): ...
def test_evidence_precision_denominator(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_ch11_eval_metrics.py tests/test_ch11_checks.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现纯函数**

数值按 `gold_facts` 的 `unit`/`tolerance`；条件/槽位用 `condition_aliases`/`slot_aliases` 规范化匹配；中文数字与“1-3 个工作日”等区间要覆盖。

- [ ] **Step 4: 运行并确认通过**

Run: `.venv/bin/pytest tests/test_ch11_eval_metrics.py tests/test_ch11_checks.py tests/test_eval_retrieval_metrics.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add scripts/eval_metrics.py scripts/eval_checks.py \
  tests/test_ch11_eval_metrics.py tests/test_ch11_checks.py
git commit -m "ch11: L1-L4 确定性指标与答案检查"
```

---

## Task 11：成本台账、预算护栏与缓存

**Files:**
- Create: `config/pricing.yaml`、`scripts/eval_cost.py`、`scripts/eval_cache.py`
- Modify: `scripts/rewrite_cache.py`、`scripts/gen_evidence_signals.py`
- Test: `tests/test_ch11_cost.py`、`tests/test_ch11_cache.py`、`tests/test_rewrite_cache.py`

**Interfaces:**
```python
# eval_cost.py
class BudgetExceeded(RuntimeError): ...
class PricingTable:
    @classmethod
    def load(cls, path: Path = DEFAULT_PRICING_PATH) -> "PricingTable": ...
    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float: ...
class BudgetLedger:
    def __init__(self, path: Path = ROOT/"data/ch11_budget.json",
                 total_limit_usd: float = 10.0): ...
    def total_usd(self) -> float: ...
    def record(self, amount_usd: float, note: str) -> None: ...
    def remaining_usd(self) -> float: ...
class CostTracker:
    def __init__(self, pricing, ledger, max_cost_usd,
                 *, allow_paid: bool = False): ...
    def add(self, model, input_tokens, output_tokens, kind="answer") -> float: ...
    def guard(self, estimated_input_tokens: int, model: str) -> None: ...
    def snapshot(self) -> dict: ...

# eval_cache.py
class JsonCache: ...
class EmbeddingCache:
    def __init__(self, embedder, path: Path, model: str): ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
class RerankCache:
    def __init__(self, reranker, path: Path, model: str): ...
    def rerank(self, query, documents, top_n=None) -> list[tuple[int,float]]: ...
class ResponseCache(JsonCache):
    def response_key(self, *, model, prompt_sha256, temperature, query, evidence_ids) -> str: ...
```
- `config/pricing.yaml` 初始 `verified: false`；`verified=false` 时 `CostTracker` 拒绝任何付费调用。价格必须核对官方来源后写 `source_url` 与 `verified: true`。
- 台账跨运行累计，单次 `max_cost_usd` 与总 $10 双重检查；调用前 `guard`，超限抛 `BudgetExceeded`。
- 缓存：改写缓存进 git `tests/eval/caches/rewrite/`；嵌入/重排/回答缓存放 `data/eval-caches/`（gitignored），SHA 进 contract。
- `rewrite_cache.load_rewrites` 增加显式 `eval_sha256` 参数与薄 CLI；CLI 接受 `--allow-paid`/`--max-cost-usd`，用调用前估算写入台账，报告标注“改写费用为估算值”。`gen_evidence_signals` 修正缓存路径，并按非空 `required_evidence` 选正样本（clarify 不再被当检索正样本）。

- [ ] **Step 1: 写失败测试**

```python
def test_pricing_cost_math(): ...
def test_allow_paid_false_raises(): ...
def test_run_cap_and_total_ledger(): ...
def test_embedding_cache_hits_once(): ...
def test_rerank_cache_key_includes_documents(): ...
def test_response_cache_key_changes_with_model_or_prompt(): ...
def test_corrupt_cache_raises_with_guidance(): ...
def test_rewrite_cache_accepts_explicit_eval_sha(): ...
def test_evidence_signals_treats_clarify_as_non_positive(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_ch11_cost.py tests/test_ch11_cache.py tests/test_rewrite_cache.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现成本/缓存/修正**

缓存原子写；损坏视为 miss 并 warning（改写缓存仍按原语义报错）；价格表与台账都写版本/时间戳。

- [ ] **Step 4: 运行并确认通过**

Run: `.venv/bin/pytest tests/test_ch11_cost.py tests/test_ch11_cache.py tests/test_rewrite_cache.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add config/pricing.yaml scripts/eval_cost.py scripts/eval_cache.py \
  scripts/rewrite_cache.py scripts/gen_evidence_signals.py \
  tests/test_ch11_cost.py tests/test_ch11_cache.py tests/test_rewrite_cache.py
git commit -m "ch11: 成本上限、预算台账、评测缓存与正样本口径修正"
```

---

## Task 12：评测契约与快照一致性校验

**Files:**
- Modify: `app/vector_store.py`、`tests/helpers.py`、`tests/test_vector_store.py`
- Create: `scripts/eval_contract.py`、`scripts/eval_snapshot.py`
- Test: `tests/test_ch11_contract.py`、`tests/test_ch11_snapshot.py`

**Interfaces:**
```python
# vector_store.py
def all_rows(self, limit: int = 16384) -> list[dict]:
    """[{id, text, category}]，供逐 chunk 内容哈希校验。"""

# eval_contract.py
CONTRACT_VERSION = 1
FAILURE_LABELS = ("retrieval_miss","entity_confusion","version_conflict_missed",
                  "evidence_insufficient","gate_false_accept","gate_false_block",
                  "answer_wrong_fact","answer_missing_condition","forbidden_claim",
                  "citation_unsupported","over_refusal","under_refusal",
                  "clarify_missing_slot","clarify_over_asking","verification_overclaim",
                  "tool_error","timeout","cost_over_budget","data_error")
def build_contract(*, dataset, corpus, index, retrieval, rewrite, llm,
                   judge, pricing, run, harness) -> dict: ...
def contract_sha256(contract: dict) -> str: ...   # 排除 run.started_at/host
def contract_diff(a: dict, b: dict) -> list[str]: ...
def check_contract(contract: dict) -> None: ...
def write_contract(path: Path, contract: dict) -> None: ...

# eval_snapshot.py
@dataclass(frozen=True)
class ConsistencyReport:
    ok: bool; mysql_chunks: int; milvus_chunks: int; pending: int
    missing_ids: list[int]; extra_ids: list[int]
    text_mismatch_ids: list[int]; category_mismatch_ids: list[int]
    build_id: str; corpus_sha256: str; chunk_set_sha256: str
    def as_dict(self) -> dict: ...

def compute_build_id(rows, *, corpus_sha256, embedding_model, dim,
                     max_chars, overlap_chars) -> str: ...
def verify_consistency(kb, vectors, *, corpus_sha256, embedding_model, dim,
                       max_chars=500, overlap_chars=80) -> ConsistencyReport: ...
def acquire_eval_lock(path=ROOT/"data/.eval.lock"): ...
def copy_snapshot(source_db, dest_db, *, confirmed_writers_stopped) -> dict: ...
def verify_snapshot(snapshot_db, kb, *, corpus_sha256, embedding_model, dim) -> ConsistencyReport: ...
```
- Contract 必含：dataset split/SHA/n/label version/bucket counts；corpus manifest SHA/docs SHA/facts SHA/doc count/chunk count/params；index backend/build_id/collection/embedding model/dim/snapshot path+SHA/consistency；retrieval strategy/candidates/top_k/floors/calibration SHA+`status=stale`；rewrite model/mode/cache SHA；LLM/judge model/temperature/prompt SHA；pricing version/currency/verified/SHA；run git commit/dirty/seed/python/host；harness version/mode。
- `verify_consistency` 必须检查 `pending==0`、id 集合相等、逐 chunk `text`/`category` 相等（`vector_text(category, questions, answer)`）。
- `copy_snapshot` 仅在 `confirmed_writers_stopped=True` 时复制 db 与 sibling lock/wal 文件，记录 SHA，重开副本并校验。

- [ ] **Step 1: 写失败测试**

```python
def test_all_rows_returns_text_and_category(): ...
def test_pending_chunks_fail_consistency(): ...
def test_missing_and_extra_ids_fail(): ...
def test_same_ids_but_text_mismatch_fails(): ...
def test_happy_path_build_id_stable(): ...
def test_eval_lock_is_exclusive(): ...
def test_snapshot_requires_writers_stopped(): ...
def test_contract_hash_ignores_run_started_at(): ...
def test_contract_diff_detects_model_change(): ...
def test_failure_labels_are_stable(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_ch11_contract.py tests/test_ch11_snapshot.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现 contract 与快照**

快照目录 `data/eval-snapshots/`；不要在服务运行时直接复制；报告 JSON 写 build_id 与五项校验结果。

- [ ] **Step 4: 运行并确认通过**

Run: `.venv/bin/pytest tests/test_ch11_contract.py tests/test_ch11_snapshot.py tests/test_vector_store.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add app/vector_store.py tests/helpers.py tests/test_vector_store.py \
  scripts/eval_contract.py scripts/eval_snapshot.py \
  tests/test_ch11_contract.py tests/test_ch11_snapshot.py
git commit -m "ch11: 评测契约与快照逐 chunk 一致性校验"
```

---

## Task 13：分层评测 harness 与模式化 CLI

**Files:**
- Create: `scripts/eval_prompts.py`、`scripts/ch11_harness.py`、`scripts/eval_ch11.py`
- Test: `tests/test_ch11_harness.py`

**Modes:**
- `offline`：加载 registry/splits/facts/docs，校验 schema、分桶、证据可达、事实、泄漏，写 `offline_report.json`；不构造 Settings/DB/embedder/LLM。
- `retrieval`：L1；需要 MySQL + Milvus 一致性；首次可用 `--allow-paid` 填嵌入/重排缓存，之后可 `--max-cost-usd 0` 只读缓存。
- `evidence`：L1 + L2；分数闸 + 可选 `EvidenceJudge`，裁判调用走 `ResponseCache` 与预算。
- `full`：L1–L5；评测回答 prompt + 确定性检查 + 可选裁判；多轮拼 `conversation_context`；工具题 L4 标 `not_applicable`。

**CLI:**
```bash
.venv/bin/python -m scripts.eval_ch11 \
  --split regression-v1,frozen-v1 --mode full \
  --max-cost-usd 8 --allow-paid \
  --out reports/ch11/2026-10-24-m1
```
其他参数：`--limit`、`--strategy`、`--no-rewrite`、`--rewrite-cache`、`--rewrite-cache-mode`、`--cache-dir`、`--pricing`、`--judge/--no-judge`、`--snapshot`、`--strict-blocking`、`--seed`。

**Interfaces:**
```python
@dataclass(frozen=True)
class HarnessConfig:
    split: str
    mode: Literal["offline","retrieval","evidence","full"]
    strategy: str = "hybrid_rerank"
    limit: int = 0
    max_cost_usd: float = 0.0
    allow_paid: bool = False
    use_judge: bool = False
    include_tool_fixtures: bool = True

def run_retrieval_case(case, service, all_rows, gold_by_case, prepared, *, cap=10) -> dict: ...
def run_evidence_case(case, result, rows, judge=None, cache=None) -> dict: ...
def run_answer_case(case, result, rows, chat, cache, cost, *, use_judge=False, judge=None) -> dict: ...
def run_case(case, ctx, config) -> dict: ...
def run_split(cases, ctx, config) -> list[dict]: ...
```
- 逐题 trace 一行：`case_id/split/bucket/expected_behavior/query/retrieved/coverage_rank/group_recall/single_chunk_rank/evidence_precision/gate/evidence_block/answer/citations/deterministic_checks/judge/behavior_pred/llm_calls/tokens_in/tokens_out/latency_ms/cost_estimate/cached/failure_labels/trace_id`。`trace_id=sha256(contract_sha+case_id)[:16]`。
- 失败分类固定为 `FAILURE_LABELS`。
- `eval_prompts.py` 的 prompt 不写死任何答案；要求引用、缺槽位先问、需核验只给流程、证据不足/冲突时拒答或说明生效版本。

- [ ] **Step 1: 写失败测试**

```python
def test_offline_mode_never_constructs_paid_clients(monkeypatch): ...
def test_retrieval_trace_has_l1_fields_and_denominator_excludes_empty_evidence(): ...
def test_evidence_gate_false_accept_and_false_block(): ...
def test_forbidden_claim_and_under_refusal(): ...
def test_clarify_slot_trace(): ...
def test_verify_case_compliance(): ...
def test_full_mode_response_cache_avoids_second_paid_call(): ...
def test_budget_exceeded_stops_before_llm_call(): ...
def test_trace_jsonl_one_line_per_case(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_ch11_harness.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现 harness 与 CLI**

复用 `RetrievalService`、`eval_metrics`、`eval_checks`、`eval_cost`、`eval_cache`、`eval_contract`；报告写 `reports/ch11/...`，trace 与报告都不入库。

- [ ] **Step 4: 运行并确认通过**

Run: `.venv/bin/pytest tests/test_ch11_harness.py -q && .venv/bin/python -m scripts.eval_ch11 --mode offline --split all --out /tmp/ch11-offline`
Expected: PASS；offline 报告无网络调用。

- [ ] **Step 5: 提交**

```bash
git add scripts/eval_prompts.py scripts/ch11_harness.py scripts/eval_ch11.py tests/test_ch11_harness.py
git commit -m "ch11: 分层评测 harness 与模式化 CLI"
```

---

## Task 14：分层聚合、阻断项、统计与报告

**Files:**
- Create: `scripts/eval_stats.py`、`scripts/eval_report.py`
- Test: `tests/test_ch11_report.py`

**Interfaces:**
```python
# eval_stats.py
def bootstrap_ci(values: list[float], *, n_boot=2000, alpha=0.05,
                 seed=20261024) -> tuple[float, float]: ...
def mcnemar_exact(baseline: list[bool], candidate: list[bool]) -> dict: ...

# eval_report.py
def aggregate_traces(traces: list[dict]) -> dict: ...
def blocking_metrics(agg: dict) -> dict: ...
def render_report(agg: dict, contract: dict, *, historical_md: str = "") -> str: ...
def write_report(out_dir: Path, agg: dict, contract: dict, traces: list[dict]) -> Path: ...
```
- 聚合按 `split × bucket × expected_behavior`，给 n、微平均与宏平均。
- L1：R@3/5/10、MRR@10、group recall、single-chunk@1/3、evidence precision、hard-negative hit rate。
- L2：gate false accept/block、conflict detection、citation support、unsupported claims。
- L3：strict/partial accuracy、condition coverage、forbidden claim rate、clarify slot recall/precision。
- L4：under/over refusal、correct refusal、clarify appropriateness、conditional compliance、verification compliance。
- L5：调用、token、费用、P50/P95（只统计 live 调用）、缓存命中、工具调用、错误。
- `blocking_metrics` 分开给分母；**不得合成单一加权总分**。

- [ ] **Step 1: 写失败测试**

```python
def test_aggregate_denominators_for_clarify_and_refuse(): ...
def test_blocking_metrics_are_not_offset_by_recall(): ...
def test_no_total_score_key(): ...
def test_bootstrap_ci_deterministic_with_seed(): ...
def test_mcnemar_exact_simple_case(): ...   # b=5,c=0 -> p=0.0625
def test_report_keeps_historical_section_separate(): ...
def test_report_reproducible_for_same_traces(): ...
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv/bin/pytest tests/test_ch11_report.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现聚合与报告**

历史 3 文档/59 chunk 数字单独成节，不得与新 8 文档语料直接拼对比；没有同 contract 配对证据不写“提升”。

- [ ] **Step 4: 运行并确认通过**

Run: `.venv/bin/pytest tests/test_ch11_report.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add scripts/eval_stats.py scripts/eval_report.py tests/test_ch11_report.py
git commit -m "ch11: 分层聚合、阻断项与配对统计报告"
```

---

## Task 15：基线运行、验收脚本与留痕

**Files:**
- Create: `scripts/acceptance_ch11.py`
- Create: `docs/eval/{ch11-historical-baseline.md,ch11-baseline-report.md,ch11-acceptance.md}`
- Create: `docs/eval/baselines/2026-10-24-m1.json`
- Modify: `dev-notes/ch11.md`、`README.md`、`tests/eval/manifests/frozen-v1.json`
- Generated (gitignored): `reports/ch11/...` traces and per-case JSONL

- [ ] **Step 1: 写验收脚本与失败测试**

```python
def test_acceptance_reports_each_check(): ...
def test_acceptance_offline_does_not_require_api_keys(): ...
```

验收项：
1. 目标测试全绿；
2. 语料 8 docs / 120–180 chunk / 150–250 facts；
3. frozen-v1 n=90、桶分布精确、manifest SHA 匹配；
4. regression-v1 n=60、C04/C05 标签精确、旧文件 SHA 不变；
5. leakage 无 blocking；
6. 快照 `pending=0`、id 集合相等、逐 chunk 文本/类别一致、build_id 匹配；
7. 同 contract 两次确定性指标一致；
8. 阻断项实际值（硬负例/OOD 误放行、forbidden claims、under-refusal、unsupported claims）；
9. 累计费用 ≤ $10；
10. 报告含历史与新结果分节。

- [ ] **Step 2: 跑离线验收**

Run: `.venv/bin/python -m scripts.acceptance_ch11`
Expected: 十项全过；无网络调用。

- [ ] **Step 3: 建库与快照**

```bash
docker compose up -d mysql
.venv/bin/python -m db.seed
.venv/bin/python -m scripts.build_kb --manifest-out data/build_manifest.json
# 停 uvicorn 等所有持有 Milvus 的进程后：
.venv/bin/python -m scripts.eval_snapshot --copy \
  --source data/milvus_knowledge.db \
  --out data/eval-snapshots/<build_id>.db --confirmed-writers-stopped
.venv/bin/python -m scripts.eval_snapshot --verify-only
```
Expected: `pending=0`；五项一致；`docs/eval/baselines/2026-10-24-m1.json` 写 build_id。

- [ ] **Step 4: 核对价格再开付费**

检查 DeepSeek 与 SiliconFlow 官方价格，写 `config/pricing.yaml` 的 `source_url`、`verified: true`；无法核实则停止付费运行，先交付离线基线。

- [ ] **Step 5: 生成改写缓存与首次缓存填充**

```bash
.venv/bin/python -m scripts.rewrite_cache --set tests/eval/splits/dev.jsonl \
  --cache tests/eval/caches/rewrite/dev.json --mode write --allow-paid
.venv/bin/python -m scripts.rewrite_cache --set tests/eval/splits/frozen-v1.jsonl \
  --cache tests/eval/caches/rewrite/frozen-v1.json --mode write --allow-paid
.venv/bin/python -m scripts.eval_ch11 --split regression-v1,frozen-v1 --mode retrieval \
  --allow-paid --max-cost-usd 1 --out reports/ch11/2026-10-24-m1-retrieval
```
Expected: 改写缓存绑定 split SHA 与模型；检索缓存填充；累计台账更新。

- [ ] **Step 6: 跑 L2 与 L5 基线（预算 ≤ $10）**

```bash
.venv/bin/python -m scripts.eval_ch11 --split regression-v1,frozen-v1 --mode evidence \
  --allow-paid --max-cost-usd 2 --out reports/ch11/2026-10-24-m1-evidence
.venv/bin/python -m scripts.eval_ch11 --split regression-v1,frozen-v1 --mode full \
  --allow-paid --max-cost-usd 7 --out reports/ch11/2026-10-24-m1
```
Expected: 两次合计不超过 $10；子预算只是单步上限，实际按 `data/ch11_budget.json` 台账扣减，前一步少花可留给后一步。报告含 L1–L5、分桶、失败分类、逐题轨迹、P50/P95、token 与费用；历史数字与新结果分表。

- [ ] **Step 7: 回写留痕与 README**

`dev-notes/ch11.md` 记录：语料/数据集版本与 SHA、快照 build_id、基线数字及分母、费用、失败分类、legacy 校准 stale 的影响、遗留风险；`README.md` 增加最短复现命令与“合成语料/单机 Milvus/评测回答策略非线上 prompt”的边界。

- [ ] **Step 8: 最终验证并提交**

Run:
```bash
.venv/bin/pytest -q
.venv/bin/python -m scripts.acceptance_ch11
.venv/bin/python -m scripts.leakage_scan --split all
git diff --check
```
Expected: 全量测试通过（FinanceBench 缺依赖的既有失败单独说明）；验收十项通过；无 `.env`/`data/`/`reports/` 入库。

```bash
git add scripts/acceptance_ch11.py docs/eval dev-notes/ch11.md README.md \
  tests/eval/manifests/frozen-v1.json tests/eval/caches/rewrite
git commit -m "ch11: M1 基线报告、验收与开发留痕"
```

---

## Verification

**每个任务后**
- `.venv/bin/pytest <task tests> -q`
- `git diff --check`
- 只暂存本任务文件，禁止 `git add -A`；FinanceBench 未提交内容不得进入 ch11 提交。

**全量验收**
```bash
.venv/bin/pytest -q
.venv/bin/python -m scripts.validate_eval_data --split all \
  --check-evidence --check-facts --check-buckets
.venv/bin/python -m scripts.leakage_scan --split all
.venv/bin/python -m scripts.eval_ch11 --split regression-v1 --mode offline
.venv/bin/python -m scripts.eval_snapshot --verify-only
.venv/bin/python -m scripts.acceptance_ch11
```

**基线报告必须回答**
- 冻结集 90 题的分桶 n、L1 指标、L2 误放行/误拦、L3 严格正确率/条件覆盖/禁用断言、L4 误拒/误答/核验合规、L5 调用/token/P50/P95/费用。
- 历史 47/13 与旧 3 文档数字单独成节；同一 60 题在新 8 文档语料上也不与旧索引直接比“提升”。
- 阻断项实际值：硬负例/OOD 误放行、`forbidden_claim_rate`、`unsupported_claim_rate`、`under_refusal_rate`。

**不得作为验收**
- 只跑 `--mode offline` 就宣称回答层指标。
- 用缓存命中的离线延迟冒充线上延迟。
- 在不同 split/语料/索引/模型/预算之间拼对比表。
- 把合成语料、单机 Milvus Lite 或评测回答 prompt 称为生产系统。

## Risks & Rollback

| 风险 | 缓解 | 回滚 |
|---|---|---|
| 150 题标注工时超预期 | 按 split/桶分批提交；dev 先于 frozen；validator 做内循环 | 保留已完成批次；冻结前可改 |
| 新文档使旧硬负例可答 | `test_ch09_hard_negatives` 全量扫描；冲突题记入 changelog，不静默改 | 撤回对应文档并重新 dry-run |
| 冻结集泄漏 | 8-gram 扫描 + 人工复核；blocking 先改题再 freeze | 恢复上一版 frozen 与 manifest SHA |
| 快照不一致 | 五项校验；发现其他进程持有 db 即失败 | 停写后重建快照；不覆盖旧快照 |
| 付费超预算 | 价格未核实禁止付费；调用前 guard；台账跨运行累计 | 停跑，保留部分 trace；下轮从缓存续跑 |
| 旧校准过期 | contract 标 `stale`；候选校准只用 dev/regression | 不覆盖生产配置；需要时回退旧 artifact |
| 回答层非确定性 | 固定温度/seed；记录两次差异；确定性检查优先 | 只报区间与失败分类，不宣称提升 |
| 旧测试被新文档破坏 | 旧 47/13 文件不动；表格断言已限定售后手册 | 单独回滚测试改动，保留语料 |

**冻结后治理**：任何改题升 `frozen-v2`，并让 ch11 基线、ch12、ch13 全部重跑；不得只重跑新方法。

## Execution Handoff

用户已选择“评测基线实施”和 ≤$10 预算。计划批准后，优先使用 **subagent-driven-development**，按依赖关系 fan-out：

- 先串行完成 Task 0–2（分支/设计返工、manifest、事实 schema），它们定义接口。
- schema 稳定后，Task 3（5 份文档）与 Task 4（5 个事实域）可按文档/事实域拆给不同 agent 并行写，但必须共用同一 manifest 与 fact_id 命名段，避免跨域重复。
- Task 5（数据集 schema）先串行；Task 6（dev）与 Task 7（frozen）的 ID、证据组和冻结 SHA 必须单点分配，不能并行写同一个 JSONL。
- Task 8–12（overlay、泄漏、指标、成本、契约）可并行，但它们依赖 Task 3–5 的产物。
- Task 13–15 的 harness、报告、付费运行必须串行，避免重复计费和同一份 contract 被覆盖。

每个任务独立测试、独立提交；Task 15 的付费运行前再次确认价格、台账余额与用户授权。
