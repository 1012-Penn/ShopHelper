"""灌历史客服对话(挖知识演示料):python -m db.seed_conversations

对话内容与知识文档互补不重复:挖出来的问答对是新知识,能被换说法问题召回。
幂等:conversations 表已有数据则跳过。
"""
from app.config import Settings
from app.db import make_engine, make_session_factory
from app.models import Conversation, Message

CONVERSATIONS = [
    [("user", "国外的快递费怎么算?"),
     ("assistant", "海外件暂不支持直邮;港澳台地区首重 30 元,续重 15 元/kg,3-7 天送达。")],
    [("user", "大件商品寄过来邮费谁出?"),
     ("assistant", "大件商品若是质量问题退货,上门取件运费由商家承担;个人原因退货需自行承担取件费。")],
    [("user", "发票抬头写错了还能改吗?"),
     ("assistant", "发货前可在订单页自助修改抬头;已开出的电子发票可以红冲重开,1 个工作日内生效。")],
    [("user", "会员折扣和包邮活动能叠加吗?"),
     ("assistant", "可以叠加,会员 95 折之后的订单金额满 99 元依然享受包邮。")],
]


def main() -> None:
    engine = make_engine(Settings())
    factory = make_session_factory(engine)
    with factory() as session:
        if session.query(Conversation).count() > 0:
            print("conversations 表已有数据,跳过")
            return
        for pairs in CONVERSATIONS:
            conv = Conversation(user_id="guest")
            session.add(conv)
            session.flush()
            for role, content in pairs:
                session.add(Message(conversation_id=conv.id, role=role, content=content))
        session.commit()
    print(f"已灌入 {len(CONVERSATIONS)} 通历史对话")


if __name__ == "__main__":
    main()
