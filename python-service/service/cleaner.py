"""文档文本清洗：基础规则 + 跨页去页眉页脚

在「加载后、切分前」调用，去除会污染 embedding 的噪音：
- 控制字符、零宽字符、异常空白
- 页码、纯数字行、URL/邮箱/导航条
- 跨页重复出现的页眉/页脚
"""

import logging
import re
from collections import Counter
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 零宽字符
_ZERO_WIDTH = re.compile(r"[\u200b\ufeff\u200c\u200d\u200e\u200f]")
# 控制字符（保留 \n，\t 后续统一成空格）
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 页码行（整行匹配，避免误删"第一章/第一页内容"这类正文）
_PAGE_LINE_PATTERNS = [
    re.compile(r"^第\s*[一二三四五六七八九十百千0-9]+\s*页\s*$"),
    re.compile(r"^第\s*[一二三四五六七八九十百千0-9]+\s*页\s*[\/／共]\s*[一二三四五六七八九十百千0-9]+\s*页?\s*$"),
    re.compile(r"^page\s*\d+(\s*(of|/)\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\d+\s*[\/／]\s*\d+$"),
    re.compile(r"^[-—–]?\s*\d{1,3}\s*[-—–]?$"),
]

# URL / 邮箱 / 导航条（行首）
_URL_EMAIL = re.compile(r"^(https?://|www\.|[\w.+-]+@[\w-]+\.[\w.]+)", re.IGNORECASE)


def _is_page_number_line(line: str) -> bool:
    return any(p.match(line) for p in _PAGE_LINE_PATTERNS)


def clean_text(text: str) -> str:
    """单文本基础清洗"""
    if not text:
        return ""

    # 1. 换行统一
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 2. 去控制字符（保留 \n）
    text = _CONTROL_CHARS.sub("", text)
    # 3. 去零宽字符
    text = _ZERO_WIDTH.sub("", text)
    # 4. 全角空格 → 半角；多空格/tab → 单空格（保留换行）
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)

    cleaned_lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        # 6. 去页码行 + URL/邮箱行
        if _is_page_number_line(line):
            continue
        if _URL_EMAIL.match(line):
            continue
        # 7. 相邻重复行去重
        if cleaned_lines and cleaned_lines[-1] == line:
            continue
        cleaned_lines.append(line)

    # 5. 多空行 → 单空行（逐行 join，天然无连续空行）
    return "\n".join(cleaned_lines)


def remove_headers_footers(documents: List[Document]) -> List[Document]:
    """跨页启发式去页眉页脚：统计每页首/末行，重复出现 ≥2 次的删除"""
    if len(documents) < 2:
        return documents

    head_counter = Counter()
    foot_counter = Counter()
    per_doc_edges = []

    for doc in documents:
        lines = [l for l in doc.page_content.split("\n") if l.strip()]
        if not lines:
            per_doc_edges.append(((), ()))
            continue
        heads = tuple(lines[:2])
        foots = tuple(lines[-2:])
        per_doc_edges.append((heads, foots))
        for h in heads:
            head_counter[h] += 1
        for f in foots:
            foot_counter[f] += 1

    header_lines = {l for l, c in head_counter.items() if c >= 2}
    footer_lines = {l for l, c in foot_counter.items() if c >= 2}

    if not header_lines and not footer_lines:
        return documents

    cleaned = []
    for doc, (heads, foots) in zip(documents, per_doc_edges):
        lines = doc.page_content.split("\n")
        kept = []
        for i, line in enumerate(lines):
            s = line.strip()
            # 只在首部/尾部位置删除，避免误删正文中间的相同行
            if i < 2 and s in header_lines:
                continue
            if i >= len(lines) - 2 and s in footer_lines:
                continue
            kept.append(line)
        text = "\n".join(kept).strip()
        if text:
            doc.page_content = text
            cleaned.append(doc)
    return cleaned


def clean_documents(documents: List[Document]) -> List[Document]:
    """清洗 Document 列表：先 clean_text，再跨页去页眉页脚"""
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    documents = [d for d in documents if d.page_content and d.page_content.strip()]
    documents = remove_headers_footers(documents)
    return documents
