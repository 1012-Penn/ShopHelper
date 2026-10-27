# 检索静态调用缓存：实现与真实 API 对照

日期：2026-10-27

分支：`codex/retrieval-cache`

实验数据：[summary.json](../../tests/eval/retrieval_cache/results/live-20261027/summary.json)
与 [responses.jsonl](../../tests/eval/retrieval_cache/results/live-20261027/responses.jsonl)

## 结论

在 130 chunk 的知识库规模下，为嵌入与重排的精确输入增加进程内 LRU + TTL
缓存是值得的工程取舍：重复 FAQ 请求的检索阶段 P50 从 739.8 ms 降到 35.8 ms，
P95 从 2083.8 ms 降到 49.3 ms；缓存命中请求没有发出嵌入或重排 HTTP 请求。
同一缓存实例的首次真实响应与紧接着的缓存重放逐题完全一致。

这不是通用向量缓存平台，也不缓存订单动态状态、生成答案、低置信判断或证据闸结果。
实际生产接线尚未由本分支修改 `app/main.py`、`app/config.py` 或
`app/graph/builder.py`，最小接线建议见下文。

## 实际服务链路

线上 `RetrievalService.retrieve(strategy="hybrid_rerank")` 的静态调用点有两处：

1. 混合召回前对规范化查询调用一次 `embedder.embed([query])`；
2. RRF 候选确定后，对完整 query/document 列表调用一次 `reranker.rerank(...)`。

随后才执行证据置信度计算与可选的证据蕴含判断。本实现只包装前两个模型客户端，
因此不会改变检索顺序、分数、证据闸或拒答行为。`CachedEmbedder` 和
`CachedReranker` 是透明接口替身，接口分别保持：

```python
embed(texts: list[str]) -> list[list[float]]
rerank(query: str, documents: list[str], top_n: int | None) -> list[tuple[int, float]]
```

## 缓存语义

- 键包含缓存 schema、调用类型、模型名、语料版本、检索配置版本，以及完整实际输入。
  嵌入按单条文本作为一项；重排键包含 query、document 内容及顺序、`top_n`。
- 默认容量为嵌入 512 项、重排 256 项；默认 TTL 为 3600 秒。达到容量按 LRU 淘汰。
- 线程安全；同一键的并发 miss 只允许一个线程访问提供方，其余等待共享结果。
- 失败结果不写缓存；后续请求可以重试。返回向量和重排列表均做防御性复制。
- 语料重建或检索配置变化时生成新的 `corpus_version` / `config_version`，旧键立即不可达，
  再由容量和 TTL 清理。
- `stats()` 暴露 hits、misses、hit rate、coalesced、去重输入、提供方调用/条数、
  evictions、expired、entries、inflight、capacity 和 TTL。

订单、物流等动态业务数据不会经过这两个客户端；即使进入最终 prompt，也不会进入本缓存。

## 真实 API 对照

评测从仓库中的 8 份知识文档构建私有的 130 chunk SQLite + Milvus Lite 索引，未修改共享
MySQL/Milvus。查询使用真实 `BAAI/bge-m3` 嵌入与
`BAAI/bge-reranker-v2-m3` 重排接口，每次冷缓存请求都新建空缓存；热缓存请求复用已填充实例。
每类 8 轮，P95 使用 nearest-rank，样本较小时 P95 即最大值。

| 请求类型 | 样本数 | 冷 P50 / P95 | 热 P50 / P95 | P50 / P95 降幅 | 冷 HTTP 调用 | 热 HTTP 调用 |
|---|---:|---:|---:|---:|---:|---:|
| 完全重复 | 8 | 739.8 / 2083.8 ms | 35.8 / 49.3 ms | 95.2% / 97.6% | 16 | 0 |
| 不重复问题集 | 32 | 741.2 / 1139.6 ms | 27.8 / 45.7 ms | 96.2% / 96.0% | 64 | 0 |
| 资料更新后 | 8 | 761.1 / 1078.0 ms | 49.2 / 80.2 ms | 93.5% / 92.6% | 16 | 0 |

热缓存嵌入命中率在三个类别中均为 100%，重排命中率也均为 100%。缓存条目数分别只有
1、4、1，远低于容量上限。

热态时延仍包含应用内 SQLite/Milvus 本地检索和证据置信度计算，不是单独的字典访问时间；
它也不代表进程冷启动、模型加载、查询改写或回答生成耗时。这里没有拿缓存重放冒充冷启动速度。

## 一致性与失效

对同一缓存实例，先执行一次真实提供方请求，再立即执行完全相同的请求。三类工作负载的所有
查询均满足 `within_cache_replay_identical=true`：完整 chunk 顺序、分数、低置信结果、
证据置信度及蕴含标记经响应摘要比较全部一致。

资料更新夹具把 SH-E300 的价格从 299 元改为 329 元；语料哈希和 build id 均变化：

- v1 corpus SHA：`52d10484...`，130 chunks；
- v2 corpus SHA：`edc5ec83...`，130 chunks；
- v1 返回“售价 299 元”，v2 返回“售价 329 元”；
- v2 的首个请求为嵌入和重排双 miss，紧接着的请求为双 hit。

因此版本变化不会复用旧语料对应的重排结果。

## 调用、费用与失败

最终有效对照发出 110 个提供方 HTTP 请求（嵌入 55、重排 55），没有 API 失败或重试。
首次评测因输出一致性口径写错，额外产生 128 个请求；修正后复用私有索引重跑，没有重建索引。
两轮实际共 238 个请求，独立账本保守计费 $0.003698，单轮上限 $2。

本轮费用口径包括语料建库嵌入、冷查询嵌入、重排和失败重试；实际重试为 0。查询改写、
回答生成和证据裁判没有调用，因此这三项本轮费用为 $0，也未被计入任何时延指标。

价格表对 BAAI 标准版仍按 $0.01/M 的保守占位计费。该端点页面标注免费，所以本报告只报告
调用数和时延，不声称“节省了多少美元”。

## 适用边界

适合：中小知识库、FAQ/政策类高重复查询、单进程或少量进程的服务；尤其适合希望避免
Redis 和新基础设施的小型项目。

不适合直接作为：订单状态、库存、物流轨迹、用户个性化答案或生成结果的缓存。多进程部署时
每个进程有独立缓存，命中率和内存都要按进程计算；更大语料或跨实例共享需求才值得评估
持久化/集中式缓存。

## 验收方最小接线建议

由负责共享工厂的验收分支在 `make_retrieval_chain` 中包两层客户端即可：

```python
from app.retrieval_cache import CachedEmbedder, CachedReranker

embedder = CachedEmbedder(
    make_embedder(settings),
    model=settings.embedding_model,
    corpus_version=build_manifest["build_id"],
    config_version=retrieval_config_digest,
)
base_reranker = make_reranker(settings)
reranker = (
    CachedReranker(
        base_reranker,
        model=settings.rerank_model,
        corpus_version=build_manifest["build_id"],
        config_version=retrieval_config_digest,
    )
    if base_reranker is not None else None
)
```

`retrieval_config_digest` 应至少覆盖提供方 base URL、策略、候选数、最终 top-k、
重排阈值和切分参数。
缺失 build manifest 时应 fail closed 到无缓存客户端或拒绝启用缓存，不能用固定字符串。
容量/TTL/开关如需配置，由验收方添加 settings；本分支没有越界修改核心配置和工厂。
接入后可通过 `service._embedder.stats()` 与 `service._reranker.stats()` 导出计数。

## 简历候选

为 130 chunk RAG 服务实现带版本失效、LRU/TTL、并发单飞和命中统计的进程内嵌入/重排缓存；
真实 BGE API 同链路重复查询检索阶段 P50 739.8→35.8 ms、P95 2083.8→49.3 ms，
命中请求提供方 HTTP 调用 16→0，语料更新后首个请求重新填充且响应保持一致。
