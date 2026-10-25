"""ch11 Task 3:五份新知识文档的规模、结构、表格与硬负例保护。

用真实 manifest + 真实 `chunk_markdown` 校验,不使用替身。
"""
import re
import unicodedata
from pathlib import Path

from app.chunking import chunk_markdown
from scripts.corpus_manifest import load_manifest, resolve_docs

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "knowledge"

NEW_DOCS = {
    "商品目录.md": "manual",
    "订单与支付.md": "manual",
    "会员与优惠.md": "policy",
    "政策版本说明.md": "policy",
    "售后核验指引.md": "manual",
}
EXISTING_DOCS = {"退货政策.md", "商品FAQ.md", "售后手册.md"}

# ch09 硬负例的缺失词:新文档不得把它们补成可答。
ABSENT_TERMS = [
    "飞行手机保修",
    "会飞的手机保修",
    "无线充电",
    "礼包内容",
    "生日礼包包含",
    "修改抬头",
    "抬头修改",
    "修改发票抬头",
    "安装微信",
    "微信应用",
    "具体收费",
    "收费金额",
    "维修价格",
]

EVAL_MARKERS = ["FROZEN-", "DEV-", "fact_id", "required_evidence", "本题应回答", "标准答案"]

REQUIRED_SECTIONS = {
    "商品目录.md": [
        "商品目录说明",
        "商品价格总览",
        "核心参数对比",
        "兼容配件",
        "替代型号",
        "在售与停产状态",
        "SH-E300 头戴式无线降噪耳机",
    ],
    "订单与支付.md": [
        "订单状态总览",
        "支付渠道",
        "退款路径",
        "支付失败与补付",
        "订单查询与凭证",
        "ORD-1001 未支付订单",
    ],
    "会员与优惠.md": [
        "会员等级",
        "95 折适用范围",
        "95 折排除范围",
        "优惠叠加规则",
        "优惠互斥规则",
        "包邮门槛与合并订单",
        "偏远地区补运",
        "生日礼包与会员日",
        "优惠活动日历",
        "优惠券使用限制",
    ],
    "政策版本说明.md": [
        "版本时间线",
        "v1 版七天无理由",
        "v2 版十五天无理由",
        "替代关系",
        "新旧版本适用判定",
        "冲突处理",
        "版本切换示例",
        "其他政策的版本",
    ],
    "售后核验指引.md": [
        "适用范围",
        "凭证要求",
        "照片拍摄规范",
        "视频拍摄规范",
        "检测流程",
        "处理时限",
        "责任划分",
        "质量问题认定原则",
        "人为损坏的判定",
        "物流破损核验",
        "争议复核",
    ],
}

SEP_RE = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+$")


def _text(name: str) -> str:
    return (DOCS_DIR / name).read_text(encoding="utf-8")


def _chunks(name: str):
    return chunk_markdown(_text(name), name, NEW_DOCS.get(name, "manual"))


def _titles(name: str) -> set[str]:
    return {chunk.questions for chunk in _chunks(name)}


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def test_manifest_covers_all_knowledge_docs():
    manifest = load_manifest()
    registered = {spec.file for spec in manifest.docs}
    actual = {path.name for path in DOCS_DIR.glob("*.md")}

    assert registered == actual
    assert EXISTING_DOCS <= registered
    assert set(NEW_DOCS) <= registered
    assert manifest.expected_docs == 8 == len(registered)

    types = {spec.file: spec.content_type for spec in manifest.docs}
    for doc, content_type in NEW_DOCS.items():
        assert types[doc] == content_type, f"{doc} content_type 应为 {content_type}"


def test_m1_scale_docs_and_chunks():
    manifest = load_manifest()
    types = {spec.file: spec.content_type for spec in manifest.docs}
    counts = {
        path.name: len(chunk_markdown(_text(path.name), path.name, types[path.name]))
        for path in resolve_docs(DOCS_DIR, manifest)
    }
    total = sum(counts.values())

    assert len(counts) == 8, counts
    assert 120 <= total <= 180, f"chunk 总数 {total} 不在 [120,180]:{counts}"
    assert manifest.expected_chunks_min == 120
    assert manifest.expected_chunks_max == 180
    for doc in NEW_DOCS:
        assert counts[doc] > 0, f"{doc} 没有切出任何 chunk"


def test_new_docs_have_required_sections():
    for doc, sections in REQUIRED_SECTIONS.items():
        titles = _titles(doc)
        for section in sections:
            assert section in titles, f"{doc} 缺少章节:{section}"

    catalog = _text("商品目录.md")
    skus = set(re.findall(r"\bSH-[A-Z]\d+\b", catalog))
    assert 14 <= len(skus) <= 16, sorted(skus)
    for sku in ("SH-E300", "SH-E120", "SH-K870", "SH-K104", "SH-B5", "SH-W20"):
        assert sku in skus

    # 从 FAQ 抽取 SKU→价格精确映射,再与目录价格表逐项比对。
    faq = _text("商品FAQ.md")
    faq_prices = dict(re.findall(r"(SH-[A-Z]\d+)[^\n]*?售价 (\d+) 元", faq))
    assert faq_prices == {
        "SH-E300": "299",
        "SH-E120": "99",
        "SH-K870": "199",
        "SH-K104": "219",
        "SH-B5": "129",
        "SH-W20": "399",
    }, faq_prices

    catalog_prices = {}
    for line in catalog.splitlines():
        match = re.match(r"\|\s*(SH-[A-Z]\d+)\s*\|[^|]*\|\s*(\d+)\s*元\s*\|", line)
        if match:
            catalog_prices[match.group(1)] = match.group(2)
    assert len(catalog_prices) == len(skus), (len(catalog_prices), len(skus))
    for sku, price in faq_prices.items():
        assert catalog_prices.get(sku) == price, (
            f"商品目录中 {sku} 价格 {catalog_prices.get(sku)!r} 与 FAQ 的 {price!r} 不一致"
        )
    assert "配件类 SH-C60 与 SH-T50" in catalog and "质保半年" in catalog
    assert "| SH-K104 | SH-K870 | 键位更少,轴体不同 |" in catalog

    orders = _text("订单与支付.md")
    status_by_order = {}
    for line in orders.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and re.fullmatch(r"ORD-\d{4}", cells[0]):
            status_by_order[cells[0]] = cells[-1]
    expected_orders = {
        "ORD-1001": "未支付",
        "ORD-1002": "已支付未发货",
        "ORD-1003": "已发货",
        "ORD-1004": "已签收",
        "ORD-1005": "已取消",
        "ORD-1006": "退货中",
        "ORD-1007": "已退款",
        "ORD-1008": "已签收",
        "ORD-1009": "已发货",
        "ORD-1010": "已支付未发货",
        "ORD-1011": "已退款",
    }
    assert status_by_order == expected_orders
    assert set(status_by_order.values()) == {
        "未支付",
        "已支付未发货",
        "已发货",
        "已签收",
        "已取消",
        "退货中",
        "已退款",
    }
    assert "超过 30 分钟未支付将自动取消" in orders
    assert "个人客户支持微信支付、支付宝、银联卡" in orders
    assert "对公转账仅面向企业客户" in orders

    versions = _text("政策版本说明.md")
    assert "2026-06-01" in versions
    assert "2026-05-31" in versions
    assert "自 2024-01-01 起至 2026-05-31" in versions
    assert "7 天" in versions and "15 天" in versions
    assert "supersedes" in versions

    member = _text("会员与优惠.md")
    for term in ("95 折", "合并", "包邮", "偏远", "生日礼包"):
        assert term in member, f"会员文档缺少:{term}"

    verification = _text("售后核验指引.md")
    for term in ("照片", "视频", "检测结果", "责任划分", "是否质量问题以检测结果为准"):
        assert term in verification, f"售后核验指引缺少:{term}"
    assert "开胶" not in verification, "不得定义或举例“开胶属于质量问题”"


def test_new_tables_replicate_header():
    table_sections = 0
    split_sections = 0

    for doc in NEW_DOCS:
        grouped: dict[str, list[list[str]]] = {}
        for chunk in _chunks(doc):
            if not chunk.answer.lstrip().startswith("|"):
                continue
            lines = [line.rstrip() for line in chunk.answer.strip().splitlines()]
            assert len(lines) >= 2, f"{doc} 表格块缺表头/分隔行:{chunk.section_path}"
            assert all(line.lstrip().startswith("|") for line in lines), (
                f"{doc} 表格块混入非表格行:{chunk.section_path}"
            )
            assert SEP_RE.match(lines[1].strip()), (
                f"{doc} 表格块第二行不是分隔行:{chunk.section_path}: {lines[1]!r}"
            )
            grouped.setdefault(chunk.section_path, []).append(lines)

        for section_path, tables in grouped.items():
            table_sections += 1
            header, separator = tables[0][0].strip(), tables[0][1].strip()
            for lines in tables:
                assert lines[0].strip() == header, f"{doc} {section_path} 分块表头不一致"
                assert lines[1].strip() == separator, f"{doc} {section_path} 分块分隔行不一致"
            if len(tables) >= 2:
                split_sections += 1

    assert table_sections >= 5, f"新文档表格节过少:{table_sections}"
    assert split_sections >= 2, "至少两个表格应被切成多块并各自复制表头"


def test_new_docs_do_not_contain_hard_negative_absent_terms():
    for doc in NEW_DOCS:
        text = _norm(_text(doc))
        for term in ABSENT_TERMS:
            assert _norm(term) not in text, f"{doc} 出现硬负例缺失词:{term}"

    # 商品目录全篇不得出现充电相关信息(尤其 SH-E300 的充电方式)。
    charging_terms = ("充电", "充电器", "充电接口", "无线充电")
    for chunk in _chunks("商品目录.md"):
        for term in charging_terms:
            assert term not in chunk.answer, (
                f"商品目录不得写充电信息({term}):{chunk.section_path}"
            )

    # 售后核验指引不得出现任何金额或计价字样。
    verification = _text("售后核验指引.md")
    assert not re.search(r"收费|价格|金额|元|块|角", verification), (
        "售后核验指引不得写收费金额"
    )
    assert not re.search(r"[一二三四五六七八九十百千万零两]+\s*(?:元|块|角)", verification), (
        "售后核验指引不得用中文数字写金额"
    )


def test_knowledge_contains_no_eval_markers():
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for marker in EVAL_MARKERS:
            assert marker not in text, f"{path.name} 含评测标记:{marker}"
