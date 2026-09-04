"""灌 FAQ 测试数据:python -m db.seed"""
from app.config import Settings
from app.db import make_engine, make_session_factory
from app.models import Faq

FAQ_ITEMS = [
    ("退货政策是什么?", "支持七天无理由退货,商品需保持完好,凭订单号联系客服即可发起退货。", "售后"),
    ("怎么申请退货?", "在订单详情页点击「申请售后」,选择退货原因后提交,客服会在 24 小时内审核。", "售后"),
    ("多久能发货?", "现货商品 48 小时内发出,预售商品按页面标注时间发货。", "物流"),
    ("物流一般几天到?", "普通快递 3-5 天送达,偏远地区 5-7 天。", "物流"),
    ("支持哪些付款方式?", "支持微信支付、支付宝、银联卡。", "交易"),
    ("可以开发票吗?", "支持电子发票,下单时备注抬头,发货后 24 小时内开出。", "交易"),
    ("会员有什么权益?", "会员享 95 折、生日礼包、优先客服通道。", "会员"),
    ("商品有质量问题怎么办?", "质量问题可凭照片凭证免费退货,退回的快递费由商家承担,退款原路返回。", "售后"),
]


def main() -> None:
    engine = make_engine(Settings())
    with make_session_factory(engine)() as session:
        if session.query(Faq).count() > 0:
            print("faq 表已有数据,跳过")
            return
        session.add_all(Faq(question=q, answer=a, category=c) for q, a, c in FAQ_ITEMS)
        session.commit()
    print(f"已灌入 {len(FAQ_ITEMS)} 条 FAQ")


if __name__ == "__main__":
    main()
