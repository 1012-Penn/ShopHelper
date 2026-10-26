# Fact scope R3

三臂对照:

- `baseline`: 旧 `ORDER_CATALOG`,检索不带 scope;
- `identity`: namespace-qualified 订单事实,检索不带 scope;
- `scope`: 在 identity 基础上加入可解释的主体/政策版本/日期过滤。

题集固定为 8 dev + 20 held-out,输入与 gold 分文件保存,组名不交叉。冻结文件是
`frozen-manifest.json`;真实模型只从根 `.env` 在进程内读取凭证。

复跑:

```bash
/root/workplace/ShopHelper/.venv/bin/python -m tests.eval.fact_scope_r3.run_eval --preflight
/root/workplace/ShopHelper/.venv/bin/python -m tests.eval.fact_scope_r3.run_eval --run dev --allow-paid
/root/workplace/ShopHelper/.venv/bin/python -m tests.eval.fact_scope_r3.run_eval --run heldout --allow-paid
```

已有运行目录不会被覆盖。`heldout` 还要求同一冻结配置下已有 dev 结果。当前仓库保留:

- `live-20261027-dev-initial-scorer/`: 首次 dev 及评分误判修复前结果;
- `live-20261027-dev-preflight-ledger-failure/`: provider 调用前失败的预算配置记录;
- `live-20261027-dev/`: 最终 dev;
- `live-20261027-heldout/`: 首次且唯一的 held-out;
- `fact-scope-r3-paid-final.sqlite3`: 独立、原子预留并结算的累计费用账本。

真实比较只使用模型 provider;知识库建在临时 SQLite,向量/重排使用确定性测试替身。
因此 token/费用/延迟可复核,但不能解释为 Milvus/BGE 线上性能。
