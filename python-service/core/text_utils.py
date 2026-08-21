"""文本工具函数"""


def estimate_tokens(text: str) -> int:
    """估算文本 token 数（中英文混合的简单估算）

    中文约 1.5 字符/token，英文约 4 字符/token。
    对混合文本取折中值。
    """
    if not text:
        return 0
    chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other = len(text) - chinese
    return int(chinese / 1.5 + other / 4.0) + 1
