# Fact scope R3 开发记录

## 观察

`app.orders` 的裸 `1001` 是已签收无线耳机 299 元,corpus factbook 的 `ORD-1001` 是未支付
248 元。旧检索和回答路径没有稳定主体,同尾号很容易被当作同一单。

## 定位与修复

1. 先写失败测试固定两个 namespace、歧义尾号、未知 `ORD-*` 和旧三单兼容;
2. 新增 `app/business_facts`,用显式 manifest 读取不可变 corpus factbook;缺失商品不从相似
   商品补齐;
3. `get_order()` 返回主体、来源、版本、as-of 与缺失字段,但不把政策版本推导混入身份层;
4. `RetrievalService.retrieve(scope=...)` 在候选进入精排前过滤,只剔除能证明的主体/版本/
   日期冲突,泛化政策保留;过滤为空不回退;
5. `query_faq` 接受可选 `order_id/policy_version/as_of_date`;已知订单有签收时间时以工具事实
   推导版本,用户传入日期不能覆盖工具;
6. held-out 暴露“假设日期中的 2026 被裸 ID 捕获”,补测试后只跳过 ISO/中文日期中的年份。

## 实验

冻结 8 dev + 20 held-out,三臂只改事实身份和 scope。最终 dev identity 8/8、scope 7/8、
baseline 1/8;held-out 依次 17/20、17/20、5/20。scope 提高 R@K 和减少上下文延迟,
但任务成功与 identity 持平,所以不作为默认收益宣传。

## 取舍

- 裸数字旧契约必须保留,不能用“看起来更像当前数据”覆盖 legacy fixture;
- 缺失字段显式 unknown 比猜同一型号/尾号更安全;
- 作用域缺失时保留政策,只有显式冲突才删;
- 规则/检索修复不能包装为新模型能力。

## 兼容更新

第二轮 adaptive routing 的重跑清单把 `app/orders.py` 也作为依赖哈希。本轮有意改变该文件后,
只更新了 `tests/eval/adaptive_routing_r2/manifest.json` 中这一项 SHA;第二轮报告、题集、金标和
历史运行结果未覆盖。报告仍以 `446fbb7`/对应源 commit 作为当时实验证据。
