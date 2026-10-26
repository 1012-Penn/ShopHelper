# Fact scope R3 集成说明

## 最小接线

验收会话无需修改共享 `main.py/config.py/builder.py/agent.py/db/init.sql` 即可保留旧路径。
若要显式启用更强 scope:

1. Agent 调 `query_faq` 时传入已知 `order_id`;
2. 已知目标政策版本或假想签收日期时传 `policy_version`/`as_of_date`;
3. 订单签收字段存在时,`query_faq` 用结构化订单日期覆盖用户声称日期;
4. 直接调用 `RetrievalService.retrieve()` 的服务可传
   `scope=EvidenceScope(subject=..., policy_version=..., reference_date=...)`。

现有图只传 `keyword/category` 时仍走旧行为,响应额外带 `scope_filter.status=not_requested`。

## 默认与兼容

- `list_orders()` 默认 `namespace="legacy"`,旧三单顺序不变;
- 裸 3-6 位上下文订单和裸 4-6 位兜底契约保留;
- `get_order()` 只增加字段,不改旧字段名和值;
- `RetrievalService.retrieve()`、`query_faq()` 新参数均有兼容默认值;
- scope 为空不加载 factbook 政策窗口,旧检索不增加结果过滤。

## 回滚

- 清理新 `order_id/policy_version/as_of_date` 工具参数即可关闭自动 scope;
- `RetrievalService` 对 scope 的调用默认为 `None`,可单独回退检索过滤;
- `app/business_facts` 不迁移、不写主库;删除/停用适配层会恢复仅 legacy 行为,但完整
  `ORD-*` 查询也会随之失效;
- 未新增环境变量或数据库表。

## 冲突点

- `app/tools/builtin.py` 为共享接线交界文件;本分支只加可选参数和 scope 诊断,保留旧 service
  三参数调用兼容;
- 其他会话若也修改 `query_faq` 签名,合并时保留 `scope` 透传与 `scope_filter` 响应字段;
- 其他会话若修改 `RetrievalResult`,保留
  `scope_status/scope_excluded/scope_unknown_ids/scope_matched_ids` 默认值。
