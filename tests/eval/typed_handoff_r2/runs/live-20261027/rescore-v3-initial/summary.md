# Typed Handoff R2 结果

模型：`deepseek-flash`；题集：10 dev + 20 held-out；两臂只改变 `handoff_mode`。

| split | arm | success | entity errors | condition errors | source missing | duplicate calls | input tokens | cost/success | P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | raw | 8/10 | 0 | 0 | 0 | 1 | 43255 | 0.00229526 | 13443 |
| dev | typed | 8/10 | 0 | 0 | 0 | 0 | 56562 | 0.00277459 | 13187 |
| held-out | raw | 17/20 | 0 | 0 | 2 | 0 | 54251 | 0.00139056 | 9964 |
| held-out | typed | 17/20 | 0 | 0 | 0 | 0 | 73893 | 0.00166882 | 10293 |

配对结果（typed 对 raw）：

| split | W | L | T |
|---|---:|---:|---:|
| dev | 0 | 0 | 10 |
| held-out | 0 | 0 | 20 |
| all | 0 | 0 | 30 |

费用账本上限：$2.00；已结算/预留上限 $0.092568。

P95 是按完整案例耗时统计的真实 API 路径；未包含缓存重放冒充在线延迟。
