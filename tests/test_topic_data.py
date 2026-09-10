"""ch10 主题分类器数据层:术语表、清洗、多标签分层抽样、数据增强。"""
from app.topic_data import (
    augment_train,
    clean_question,
    desensitize,
    normalize_format,
    stratified_multilabel_split,
    synonym_replace,
    validate_sample,
)
from app.topic_taxonomy import (
    LABEL_COUNT,
    TOPIC_LABELS,
    TOPIC_TAXONOMY,
    validate_labels,
)


def test_taxonomy_exactly_17_labels_in_user_order():
    expected = ["退换货", "物流", "尺码", "发票", "质量问题", "运费", "优惠活动", "价保",
                "支付", "订单修改", "库存补货", "商品信息", "保修维修", "账号", "会员积分",
                "评价", "其他"]
    assert list(TOPIC_LABELS) == expected
    assert LABEL_COUNT == 17


def test_taxonomy_every_category_has_boundary_note_with_near_neighbours():
    assert set(TOPIC_TAXONOMY) == set(TOPIC_LABELS)
    for label, note in TOPIC_TAXONOMY.items():
        assert note.strip(), f"{label} 缺边界说明"
    # 用户点名的近邻边界必须写进双方的说明里
    assert "保修维修" in TOPIC_TAXONOMY["退换货"]
    assert "退换货" in TOPIC_TAXONOMY["保修维修"]
    assert "物流" in TOPIC_TAXONOMY["运费"]
    assert "运费" in TOPIC_TAXONOMY["物流"]
    assert "优惠活动" in TOPIC_TAXONOMY["价保"]
    assert "价保" in TOPIC_TAXONOMY["优惠活动"]


def test_validate_labels_keeps_order_dedupes_and_drops_invalid():
    assert validate_labels(["尺码", "退换货", "尺码", "不存在的类"]) == ["尺码", "退换货"]
    assert validate_labels([]) == []
    assert validate_labels(["其他"]) == ["其他"]


def test_desensitize_phone_email_and_order_number():
    text = "手机13812345678,邮箱a.b@shop.com,订单号20240910123456,麻烦查下"
    out = desensitize(text)
    assert "13812345678" not in out and "[PHONE]" in out
    assert "a.b@shop.com" not in out and "[EMAIL]" in out
    assert "20240910123456" not in out and "[ORDER]" in out
    # 普通短数字(数量、金额)不脱敏
    assert desensitize("买了3件一共299元") == "买了3件一共299元"


def test_normalize_format_fullwidth_and_whitespace():
    assert normalize_format("  我　买的ａｂｃ12   怎么　退 ") == "我买的abc12 怎么退"


def test_clean_question_combines_desensitize_and_format():
    assert clean_question("  手机１３８１２３４５６７８   怎么退　 ") == "手机[PHONE] 怎么退"


def _make_items(per_label: int = 12) -> list[dict]:
    items = []
    for i, label in enumerate(TOPIC_LABELS):
        for j in range(per_label):
            items.append({"question": f"样本{i}-{j}", "labels": [label, "其他"]})
    return items


def test_stratified_split_proportions_disjoint_and_coverage():
    items = _make_items(per_label=12)
    train, val, test = stratified_multilabel_split(items, ratios=(0.8, 0.1, 0.1), seed=42)
    # 总量守恒、两两不交
    assert len(train) + len(val) + len(test) == len(items)
    keys = [id(x) for x in train + val + test]
    assert len(keys) == len(set(keys))
    # 每个首要标签 12 条 → 10/1/1,三份都有
    for label in TOPIC_LABELS:
        assert sum(label in it["labels"] for it in train) >= 9
        assert sum(label in it["labels"] for it in val) >= 1
        assert sum(label in it["labels"] for it in test) >= 1
    # 全局比例大约 80/10/10
    total = len(items)
    assert abs(len(train) / total - 0.8) < 0.05
    assert abs(len(val) / total - 0.1) < 0.05


def test_stratified_split_tiny_groups_go_to_train():
    items = [{"question": "a", "labels": ["退换货"]},
             {"question": "b", "labels": ["退换货"]}]
    train, val, test = stratified_multilabel_split(items, seed=1)
    assert len(train) == 2 and val == [] and test == []


def test_synonym_replace_and_augment_train_only():
    text = "我想退货,怎么操作"
    assert synonym_replace(text) != text  # 退货 → 退掉/退了 等同义替换
    assert synonym_replace("这句话没有同义词") == "这句话没有同义词"

    train = [{"question": "我想退货", "labels": ["退换货"], "origin": "synthetic"}]
    val = [{"question": "验证书", "labels": ["其他"], "origin": "synthetic"}]
    augmented = augment_train(train)
    # 只扩训练集:输入不改、原样本保留、增强样本带 origin=augmented
    assert train[0]["origin"] == "synthetic"
    assert len(augmented) >= len(train)
    assert any(item["origin"] == "augmented" for item in augmented)
    assert all(item["labels"] == ["退换货"] for item in augmented)
    assert val[0]["origin"] == "synthetic"  # val/test 从不进增强函数


def test_validate_sample():
    assert validate_sample({"question": "怎么开发票", "labels": ["发票"]})
    assert not validate_sample({"question": "", "labels": ["发票"]})
    assert not validate_sample({"question": "怎么开发票", "labels": []})
    assert not validate_sample({"question": "怎么开发票", "labels": ["不存在的类"]})
