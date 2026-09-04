"""token 估算与历史裁剪——纯函数,宁大勿小。"""


def _is_wide(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF    # CJK 统一表意文字
        or 0x3400 <= code <= 0x4DBF  # 扩展 A
        or 0x3000 <= code <= 0x303F  # CJK 标点
        or 0xFF00 <= code <= 0xFFEF  # 全角形式
    )


def estimate_tokens(text: str) -> int:
    """CJK/全角字符每字记 1 token;其余字符每 4 个记 1 token(向上取整)。"""
    wide = sum(1 for ch in text if _is_wide(ch))
    narrow = len(text) - wide
    return wide + (narrow + 3) // 4


def trim_history(messages: list[dict], budget: int) -> list[dict]:
    """messages[0] 视为 System 永不裁剪;从最老的历史消息开始丢弃,直到预算内。

    保留部分是历史的"最新连续后缀",保序。预算只计历史部分,不含 System。
    """
    system, history = messages[:1], messages[1:]
    kept: list[dict] = []
    total = 0
    for msg in reversed(history):
        cost = estimate_tokens(msg.get("content") or "")  # 纯工具调用消息 content 为空,不计费
        if total + cost > budget:
            break
        kept.append(msg)
        total += cost
    kept.reverse()
    return system + kept
