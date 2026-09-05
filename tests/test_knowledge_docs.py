"""知识文档种子验证:数据类任务以切分断言代替 TDD 红绿(工作约定 1)。"""
from pathlib import Path

from app.chunking import chunk_markdown

DOCS_DIR = Path(__file__).resolve().parent.parent / "knowledge"


def _chunks(name, ctype):
    return chunk_markdown((DOCS_DIR / name).read_text(encoding="utf-8"), name, ctype)


def test_all_docs_parse_into_chunks():
    assert len(_chunks("退货政策.md", "policy")) >= 4
    assert len(_chunks("商品FAQ.md", "faq")) >= 8
    assert len(_chunks("售后手册.md", "manual")) >= 6


def test_shipping_fee_section_is_retrieval_target():
    """验收 1 靶点:「邮费与运费」节必须包含金额与包邮门槛。"""
    fee = [c for c in _chunks("退货政策.md", "policy") if c.questions == "邮费与运费"]
    assert fee, "运费说明节必须存在"
    assert "8 元" in fee[0].answer and "满 99 元包邮" in fee[0].answer
    assert "商家承担" in fee[0].answer


def test_manual_table_chunks_all_carry_header():
    """大表格必须切成多块且每块带表头。"""
    chunks = _chunks("售后手册.md", "manual")
    table_chunks = [c for c in chunks if c.answer.lstrip().startswith("|")]
    assert len(table_chunks) >= 2, "运费标准表(28 行)必须被按行切成多块"
    for c in table_chunks:
        assert c.answer.splitlines()[0].startswith("| 区域 |")
        assert c.answer.splitlines()[1].startswith("|---|")


def test_long_section_triggers_recursive_split():
    chunks = _chunks("售后手册.md", "manual")
    detail = [c for c in chunks if c.questions == "退货处理细则"]
    assert len(detail) >= 2, "超长章节必须递归切分"
    assert all(len(c.answer) <= 500 + 80 for c in detail)  # 上限 + 重叠余量


def test_faq_entries_carry_aliases():
    chunks = _chunks("商品FAQ.md", "faq")
    entry = [c for c in chunks if c.questions.startswith("怎么申请退货")][0]
    assert "退货入口在哪" in entry.questions and "怎么退" in entry.questions
    assert "其他问法" not in entry.answer


def test_faq_covers_ch02_seed_equivalents():
    """ch02 八条 FAQ 的等价问法都要在场。"""
    questions = "\n".join(c.questions for c in _chunks("商品FAQ.md", "faq"))
    for key in ["退货政策", "怎么申请退货", "多久能发货", "物流一般几天到", "付款方式", "开发票", "会员", "质量"]:
        assert key in questions, f"缺 FAQ 等价条目:{key}"
