# Typed Handoff R2 结果

模型：`deepseek-flash`；题集：10 dev + 20 held-out；两臂只改变 `handoff_mode`。

| split | arm | success | entity errors | condition errors | source missing | duplicate calls | input tokens | cost/success | P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | raw | 8/10 | 0 | 0 | 0 | 1 | 43255 | 0.00229526 | 13443 |
| dev | typed | 7/10 | 2 | 0 | 0 | 0 | 56562 | 0.00317096 | 13187 |
| held-out | raw | 14/20 | 6 | 0 | 2 | 0 | 54251 | 0.00168854 | 9964 |
| held-out | typed | 15/20 | 4 | 0 | 0 | 0 | 73893 | 0.00189133 | 10293 |

配对结果（typed 对 raw）：

| split | W | L | T |
|---|---:|---:|---:|
| dev | 0 | 1 | 9 |
| held-out | 1 | 0 | 19 |
| all | 1 | 1 | 28 |

费用账本上限：$2.00；已结算/预留上限 $0.092568。

P95 是按完整案例耗时统计的真实 API 路径；未包含缓存重放冒充在线延迟。
