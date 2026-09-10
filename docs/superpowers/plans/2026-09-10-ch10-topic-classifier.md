# ch10 实施计划 · 微调多标签主题分类器

规格:`docs/superpowers/specs/2026-09-10-ch10-topic-classifier-design.md`
执行方式:用户缺席代决策推进(授权见 spec 状态行),每任务完成即在 dev-notes/ch10.md 追记。
TDD 说明:后端可单测代码严格测试先行;数据/训练/评测类产出按用户要求改为「标注样例/评估集真跑验证」;前端 Vibe Coding。

## Task 1 · 数据层(术语表/清洗/分层/增强)+ 语料构建 ✅

- [x] `tests/test_topic_data.py` RED → `app/topic_taxonomy.py` + `app/topic_data.py` GREEN(10 用例)
- [x] `scripts/build_topic_dataset.py`:池内真实 5 条纠错预标全成功 + 17 类造数 1621 条
      → train 1558(含 260 增强)/ val 163 / test 163,`tests/eval/topic_dataset/` 落盘进 git
- [x] 样例验证:多标签样例抽查(「裤子买大了能换小一码不」→[尺码,退换货] 等口径正确)

## Task 2 · 训练(全参微调 + 早停)

- [x] `scripts/train_topic_classifier.py`(Context7 核对:problem_type=multi_label_classification →
      BCEWithLogitsLoss;EarlyStoppingCallback + load_best_model_at_end)
- [ ] 真跑:基座 hfl/chinese-roberta-wwm-ext 下载(HF 大文件被 Xet 通道卡死,已改 curl 断点续传
      hf-mirror)→ CPU 训练 ≤8 轮早停,产物 `models/topic_classifier/`(gitignore)+ 指标 JSON

## Task 3 · 评测

- [ ] `scripts/eval_topic_classifier.py`:各类目 P/R/F1 + 17 张二分类混淆矩阵 + 错例清单 +
      「买大了想退」多标签断言(退出码 3 = 验收 3 不过)→ reports/ 两份产物

## Task 4 · 旁路部署 ✅(除真机联跑)

- [x] TDD:tests/test_topic_classifier.py 8 用例(RED→GREEN):pending/upsert/分布聚合/
      清洗与标签闸/Top-1 兜底/503 降级/API
- [x] `app/models.py` TopicClassification;`db/ch10-topic-classifier.sql`(用户 DDL 原样)+
      init.sql 同步;真库已 apply(COMMENT utf8mb4 验证无误)
- [x] `app/topic_classifier.py`(ONNX 惰性加载)、`app/store.py` TopicStore、
      `app/routers/topics.py`、config topic_* 五项、main.py 接线 + /admin/topics
- [ ] `scripts/export_topic_onnx.py`(含 torch/ORT 对齐校验)+ `scripts/classify_topics.py` 真跑

## Task 5 · 前端 + 真机验收

- [x] `static/topics.html`(Vibe:17 类横向条形图 + 归类一批按钮,纯 CSS)
- [x] `scripts/acceptance_ch10.py`(report/classify/multilabel 三场景)
- [ ] 真机:起 uvicorn → 灌演示问题 → 批量归类 → 分布页有数 → 三场景全过

## 收口

- [ ] code review(requesting-code-review)+ verification-before-completion
- [ ] README/.env.example ch10 段落;dev-notes 六样留痕;合并 main;推 GitHub
