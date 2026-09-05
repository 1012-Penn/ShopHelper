from app.chunking import chunk_markdown

FAQ_DOC = """# 商品FAQ
## 售后
### 怎么申请退货?
在订单详情页点击「申请售后」,选择退货原因后提交。
- 其他问法:退货入口在哪 / 怎么退
### 质量问题怎么办?
凭照片凭证免费退货。
## 交易
### 支持哪些付款方式?
支持微信支付、支付宝、银联卡。
"""

POLICY_DOC = """# 退货政策
## 邮费与运费
普通订单邮费 8 元,满 99 元包邮。质量问题退货,运费由商家承担。
## 退货流程
第一步联系客服,第二步寄回商品,第三步等待退款。
"""


def test_heading_hierarchy_and_section_path():
    chunks = chunk_markdown(FAQ_DOC, "商品FAQ.md", "faq")
    entry = [c for c in chunks if "怎么申请退货" in c.questions][0]
    assert entry.section_path == "商品FAQ.md > 商品FAQ > 售后 > 怎么申请退货?"


def test_faq_category_and_questions_with_aliases():
    chunks = chunk_markdown(FAQ_DOC, "商品FAQ.md", "faq")
    entry = [c for c in chunks if "怎么申请退货" in c.questions][0]
    assert entry.category == "售后"
    assert entry.questions.splitlines() == ["怎么申请退货?", "退货入口在哪", "怎么退"]
    assert "其他问法" not in entry.answer  # 别名行从正文剥离
    # H2 直接是条目(无 H3)时 category 落到 H1
    pay = [c for c in chunks if "付款方式" in c.questions][0]
    assert pay.category == "交易"


def test_policy_questions_is_section_title_category_is_parent_path():
    chunks = chunk_markdown(POLICY_DOC, "退货政策.md", "policy")
    fee = [c for c in chunks if c.questions == "邮费与运费"][0]
    assert fee.category == "退货政策"
    assert fee.section_path == "退货政策.md > 退货政策 > 邮费与运费"
    assert "满 99 元包邮" in fee.answer


def test_long_section_recursively_split_with_sentence_overlap():
    body = "".join(f"第{i}句,这是一条用于撑长度的测试句子。" for i in range(60))
    doc = f"# 手册\n## 长章节\n{body}"
    chunks = chunk_markdown(doc, "手册.md", "manual", max_chars=200, overlap_chars=50)
    assert len(chunks) > 1
    # 每块不超限(留余量给重叠前缀)
    assert all(len(c.answer) <= 260 for c in chunks)
    # 相邻块有重叠,且重叠区从句号后开始(不留半截句)
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = prev.answer[-50:]
        start = min((i for i, ch in enumerate(tail) if ch in "。!?;;"), default=None)
        if start is not None:
            assert nxt.answer.startswith(tail[start + 1 :])


def test_table_split_replicates_header():
    rows = "\n".join(f"| 商品{i} | {i}9.9 元 | 48 小时 |" for i in range(30))
    doc = "# 手册\n## 运费标准\n| 商品 | 价格 | 发货时效 |\n|---|---|---|\n" + rows
    chunks = chunk_markdown(doc, "手册.md", "manual", max_chars=160, overlap_chars=40)
    table_chunks = [c for c in chunks if c.answer.startswith("|")]
    assert len(table_chunks) > 1
    for c in table_chunks:
        assert c.answer.splitlines()[0] == "| 商品 | 价格 | 发货时效 |"
        assert c.answer.splitlines()[1] == "|---|---|---|"


def test_key_clause_flag():
    doc = "# 政策\n## 退货约定\n商品必须保持完好,否则不予退货。\n## 温馨提示\n欢迎随时联系客服。\n"
    chunks = chunk_markdown(doc, "政策.md", "policy")
    by_q = {c.questions: c for c in chunks}
    assert by_q["退货约定"].is_key_clause is True
    assert by_q["温馨提示"].is_key_clause is False
