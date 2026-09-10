# ch10 设计文档 · 微调多标签主题分类器(旁路批量归类)

日期:2026-09-10
状态:已定稿(用户缺席,按「不需要跟进就自己干完」授权代为推进,代决策见文末)

## 背景与目标

用户原话:「我要用 Superpowers 模式给客服系统训练一个自己的模型:微调一个多标签主题分类器,
把飞轮攒下的低置信度问题按主题批量归类,好决定先补哪块知识。」

ch09 建好了低置信度问题池(三入口落池),但池子只回答「答不上」,不回答「答不上的都是哪类」。
本章训一个 17 类多标签分类器,挂在旁路按批量给池子里的真实问题打主题标签,结果落
`topic_classifications` 表,飞轮后台加主题分布页,让「先补哪块知识」有数据可依。

## 环境约束(2026-09-10 实测)

- 无 GPU(nvidia-smi 不存在),16 核 CPU / 7GB 内存(可用约 2-3GB)/ 磁盘充足。
- HuggingFace 与 hf-mirror 均可直连,`hfl/chinese-roberta-wwm-ext`(~400MB)可下载。
- 主仓库 `.venv`(Python 3.10.12)无 torch/transformers,需新装 CPU 版 torch。
- 低置信度问题池现状:仅 5 行(3 个去重缺口,多为 ch09 验收噪声题)——语料注定以
  「照术语表造模拟问题」为主,池内真实问题全量捞入,此为本章数据现实,记录在案。

## 技术选型(用户定死,不换)

- 基座:哈工大讯飞 `hfl/chinese-roberta-wwm-ext`,**全参微调**,不用 LoRA/QLoRA。
- 推理服务化:ONNX 导出 + onnxruntime CPU 推理(tokenizer 仍用 transformers)。
- 标注:大模型(deepseek-chat,走现有 `OPENAI_*` 配置)照术语表预标 + 人工抽审。

## 数据处理四步

### 1. 捞取与清洗

- 语料来源:`low_confidence_questions` 全量捞取(真实),不够由 LLM 照术语表造模拟问题补足。
- 清洗(纯函数,可单测,`app/topic_data.py`):
  - 脱敏:手机号/座机/邮箱/订单号后四位等替换为 `[PHONE]/[EMAIL]/[ORDER]` 占位符;
  - 修格式:全半角统一、多余空白折叠、首尾清理;
  - 修错别字:LLM 批量纠错(口语错字→正字),不改语义、不加戏;
  - 池内 5 条真实问题全部入语料;模拟问题目标规模 **约 2000 条**(17 类 × 各 100-140 条,
    多标签均值约 1.2-1.5),CPU 训练时间预估 20-40 分钟,内存峰值 < 3GB。

### 2. 归并术语表(权威类目)

- 全系统唯一权威定义:`app/topic_taxonomy.py` 常量 `TOPIC_TAXONOMY`(17 类,含边界说明),
  训练/预标/落库/前端分布共用这一份,任何一处不得另抄一份列表。
- 17 类:退换货、物流、尺码、发票、质量问题、运费、优惠活动、价保、支付、订单修改、
  库存补货、商品信息、保修维修、账号、会员积分、评价、其他。
- 近邻边界(用户指定):修归保修维修、退归退换货;运费管钱、物流管货;
  价保是补差价、优惠活动是券和满减。每类一句边界说明进 prompt 与文档。

### 3. 分层抽样(80/10/10)

- 多标签分层:按「首要标签(第一个标签)分层,每层内按 80/10/10 随机划分」,保证各类目
  三个集合都有份;实现为可单测的纯函数,测试断言每类三集合齐备且比例在容差内。

### 4. 数据增强(只扩训练集)

- 同义词替换(内置小词表,如 退货/退掉/退了、多少钱/什么价格/价格多少)+ 句式微调
  (加语气词/礼貌前缀等确定性模板);只对 train 集施加,增强比例约 15-25%;
  val/test 保持原样,测试断言 val/test 未被改动。

## 标注(多标签)

- 规则:一句话字面提到几个诉求就打几个标签,一个不多一个不少(宁可少打不脑补)。
- 生成即标注:模拟问题按「类目 → 诉求要点 → 生成问题(带标签)」生成,标签由生成目标给出;
- 池内真实问题:LLM 照术语表预标(结构化输出,标签必须 ⊆ 17 类,非法标签丢弃);
- 人工抽审:产出 `reports/ch10-label-audit-sample.md`(随机抽样 ≥60 条带标签清单),
  逐条复核,结论记入 dev-notes;用户回来可复查该文件。

## 训练

- `scripts/train_topic_classifier.py`:transformers 加载 `hfl/chinese-roberta-wwm-ext`,
  多标签头(17 维 sigmoid + BCEWithLogitsLoss),全参微调;
- 超参(CPU 约束下的保守档):max_len 96、batch 16、lr 2e-5、weight_decay 0.01(正则)、
  最多 8 轮、按验证集微 F1 早停(patience 2);
- 产物:`models/topic_classifier/`(checkpoint + tokenizer + label 配置)——**gitignore**,
  README 给重建命令;训练曲线与最佳轮次落 `reports/ch10-training-metrics.json`。

## 评测

- `scripts/eval_topic_classifier.py` 在留出测试集上:
  - 每类目精确率/召回率/F1(阈值 0.5,`reports/ch10-topic-classification-report.md`);
  - 每类目二分类混淆矩阵(17 张:该类正/负 × 预测正/负);
  - 错例清单落 `reports/ch10-error-samples.md` 供人工抽判复核,结论记入 dev-notes;
  - 验收样例内嵌:「买大了想退」须同时命中 ≥2 类目(尺码 + 退换货)。

## 部署(旁路批量,不碰实时主链路)

- 导出:训练/评测后 `scripts/export_topic_onnx.py` 导出 ONNX 至 `models/topic_classifier/model.onnx`;
- 服务模块 `app/topic_classifier.py`:onnxruntime 惰性加载,模型文件缺失时按 ch09 惯例
  优雅降级(接口返回明确错误,不影响聊天);
- 批量归类两条入口(均在旁路):
  1. `scripts/classify_topics.py`:捞池中尚未归类的问题(`LEFT JOIN` 未命中),清洗后
     批量推理,结果 upsert 进 `topic_classifications`(uk_question_id,重跑覆盖);
  2. `POST /api/topics/classify-batch`:后台页手动触发跑一批(便于演示),同样走旁路;
- 落库:用户给定的 DDL 原样采纳(`db/ch10-topic-classifier.sql`),并同步进 `db/init.sql`
  供全新部署;ORM 增加 `TopicClassification`(labels 存 JSON);
- **实时对话主链路不调它**:chat/graph 代码零改动。

## 前端(Vibe Coding,不走 brainstorm/TDD/code review)

- `/admin/topics` 主题分布页 `static/topics.html`:17 类问题量横向条形图(纯 CSS,与现有
  后台页同风格),哪类堆得多一眼可见;页面上给「归类一批」按钮触发旁路批量;
- API:`GET /api/topics/distribution`(各类目计数)、`POST /api/topics/classify-batch`。

## 配置新增(Settings / .env.example)

`topic_model_dir`(默认 `models/topic_classifier`)、`topic_threshold`(0.5)、
`topic_max_len`(96)、`topic_batch_size`(16)、`topic_classify_batch_limit`(200)。

## 不做(用户圈定)

- LoRA / QLoRA;embedding 微调、意图识别微调;实时对话主链路接入分类器;
- 池子仍未归类问题的自动定时任务(本章只做手动/脚本批量,定时化留待后续)。

## 验收标准映射

1. 测试集各类目 F1 + 混淆矩阵报告跑出来 → `reports/ch10-topic-classification-report.md`;
2. 攒下的真实问题跑归类、后台分布页出统计 → 验收脚本向池子灌一批**训练时未见过的**
   演示问题(`scripts/acceptance_ch10.py`),跑批量归类,断言 `/api/topics/distribution`
   有数;分布页肉眼可查;
3. 「买大了想退」同时命中多个类目 → 评测脚本内嵌断言 + 验收脚本复验。

## 代决策记录(用户缺席,回来可否决)

| 决策 | 理由 |
| --- | --- |
| 语料以 LLM 模拟为主(约 2000 条),池内 5 条真实全捞 | 池子只有 5 行,用户已预案「量不够就造」 |
| CPU 训练档位:max_len 96 / batch 16 / ≤8 轮早停 | 无 GPU,时间 20-40 分钟可接受 |
| ONNX+onnxruntime 服务化,而非 torch 常驻 | 用户点名「如 ONNX / transformers」,ONNX 运行时更轻 |
| 分布页触发批量归类按钮 | 「攒够一批归一次」在演示环境没有自然流量,给手动入口,仍属旁路 |
| 训练产物(models/)不进 git | 400MB+ 超 GitHub 限制,给重建命令;数据集/报告/指标进 git |
| 人工抽审由我代做并留痕 | 用户不在场;样本清单落 reports/,可复查 |

## 风险与对策

- 内存 7GB 偏紧:训练 batch 16 + max_len 96,峰值估 < 3GB;dataloader 不开多进程;
- deepseek 造数质量波动:生成后过「标签合法 + 非空 + 长度」校验,坏行丢弃重造;
- 真实问题与模拟问题分布偏差:验收灌池的演示问题单独造、不进训练,实测真数据表现。
