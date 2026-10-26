# 第三轮最终集成开发记录

## 范围

只接收三个并行分支，完成接线、组合验证、最终 smoke 和本地 main 合入。
不创建第四轮，不修改旧脏根 `/root/workplace/ShopHelper`。

## 源分支

- fact scope：`40934f1` + `0d541c1`；
- action contract：`1b754d8`；
- answer plan：`05266eb` + `c7291dd`。

## 集成中发现并修复

1. answer plan 原节点只返回 `final_reply`，不返回 assistant message；若直接接入，
   log 节点会只保存 user 行。新增失败测试并返回 `AIMessage`。
2. answer plan 接线需要默认关闭配置、独立账本和 pricing 校验；应用工厂已通过真实
   `answer_plan` 节点测试。
3. action artifact 需要从 multi-Agent graph 透传到 HTTP；关闭功能时必须保持旧响应形状。
4. 客户端确认 action-contract 预览时，字段必须匹配已登记参数；取消后的旧 key 不能再建单。
5. action runtime 的 intercept 闭包未绑定 loop 变量 `argument_decision`；Ruff 暴露后改为显式
   默认参数，避免重复工具调用时捕获错误对象。

## 组合 smoke 失败与修正

第一次 smoke 在 4 个 chat 任务后由脚本自己的 `route_nodes` 层级错误中断。20 次调用、
`$0.01120026` 和全部响应均保留。

第二次使用新 manifest v2 和新账本：5 个 chat + 2 个 multi-Agent 任务、61 次调用、
`$0.042805572`、pending 0、技术传输失败 0。

- 3 个 chat 实际经过 answer_plan；
- legacy 订单工具成功；
- `ORD-1001` 工具主体和金额/状态正确，但缺商品字段被简单行为 scorer 判成拒答；
- 取消可行性没有生成 preview 或写入；
- 创建售后预览在 multi-Agent 聚合阶段失败，未生成 artifact，也未写入。

汇总脚本另有一次聚合字段错误，使用零调用 `--rescore` 生成
`summary-rescored.json`，原 summary 保留。

## 验证

- 组合定向：`180 passed`。
- 全量：`1110 passed, 2 skipped, 1 xfailed`。
- reliability fault acceptance：通过，0 模型调用。
- 本轮改动范围 Ruff：除冻结 fact harness 外通过。
- 冻结 fact harness 仍有 5 个 lint finding；修改会让冻结 manifest hash 失效，因此不动。
- 当前全仓 Ruff 仍有 485 项，不声称全绿。
- `compileall`、`git diff --check`、文本凭据扫描通过。

## 取舍

- 订单身份修复保留为旧契约兼容的真实工程改进。
- scope、动作契约和答案计划都保持非默认；scope 无任务成功收益，action held-out 无效，
  answer plan held-out 为负。
- preview→confirm 的真实模型组合路径未跑通，但 HTTP/确认/取消/篡改/旧 key 防护有应用级测试。
- 不再增加优化轮次或平台能力。
