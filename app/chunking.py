"""结构感知 Markdown 切分:标题开节、超长递归、重叠裁到句号、表格按行切复制表头。"""
import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SENTENCE_END = "。!?;！？；…"  # 半角 + 全角 !?!;
KEY_CLAUSE_WORDS = ("必须", "不得", "禁止", "仅限", "不予", "免费", "七天无理由")
ALIAS_MARK = "其他问法:"
_SENTENCE_SEPS = ("。", "!", "?", ";", ";", "！", "？", "；", "…")  # 中英文句读全收(评审修复:此前漏全角 !?!;)


@dataclass
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool


def _parse_sections(text: str) -> list[tuple[list[str], str, list[str]]]:
    """→ [(祖先标题路径(不含文档名与自身), 自身标题, 正文行)];只有标题没有正文的节丢弃。"""
    sections: list[tuple[list[str], str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    current: tuple[list[str], str] | None = None
    body: list[str] = []
    for line in text.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            if current is not None:
                sections.append((current[0], current[1], body))
            level = len(m.group(1))
            stack = stack[: level - 1]
            stack.append((level, m.group(2).strip()))
            current = ([t for _, t in stack[:-1]], m.group(2).strip())
            body = []
        else:
            body.append(line)
    if current is not None:
        sections.append((current[0], current[1], body))
    return [s for s in sections if "\n".join(s[2]).strip()]


def _split_keep(text: str, sep: str) -> list[str]:
    """按分隔符切;句读类分隔符保留在段尾(不留半截句的前提出处)。"""
    if sep in _SENTENCE_SEPS:
        parts, buf = [], ""
        for ch in text:
            buf += ch
            if ch == sep:
                parts.append(buf)
                buf = ""
        if buf.strip():
            parts.append(buf)
        return [p for p in parts if p.strip()]
    return [p for p in text.split(sep) if p.strip()]


def _merge(pieces: list[str], max_chars: int) -> list[str]:
    blocks, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) > max_chars:
            blocks.append(buf)
            buf = p
        else:
            buf += p
    if buf.strip():
        blocks.append(buf)
    return blocks


def _recursive_split(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    for sep in ("\n\n", "\n", *_SENTENCE_SEPS):
        parts = _split_keep(text, sep)
        if len(parts) > 1:
            pieces: list[str] = []
            for p in parts:
                pieces.extend(_recursive_split(p, max_chars))
            return _merge(pieces, max_chars)
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _overlap_prefix(prev: str, overlap_chars: int) -> str:
    """取上一块尾部做下一块前缀;窗口内裁到最近句号之后,无句读则不重叠(不留半截话)。"""
    tail = prev[-overlap_chars:] if overlap_chars > 0 else ""
    cut = next((i for i, ch in enumerate(tail) if ch in SENTENCE_END), None)
    return tail[cut + 1 :].lstrip() if cut is not None else ""


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _split_table(lines: list[str], max_chars: int) -> list[str]:
    """超长表格按数据行分组,每块复制表头行 + 分隔行。"""
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) <= 3 or len("\n".join(lines)) <= max_chars:
        return ["\n".join(lines)]
    header, sep, data = lines[0], lines[1], lines[2:]
    budget = max_chars - len(header) - len(sep) - 2
    blocks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for row in data:
        if cur and cur_len + len(row) + 1 > budget:
            blocks.append("\n".join([header, sep, *cur]))
            cur, cur_len = [], 0
        cur.append(row)
        cur_len += len(row) + 1
    if cur:
        blocks.append("\n".join([header, sep, *cur]))
    return blocks


def _partition_tables(lines: list[str]) -> list[tuple[str, list[str]]]:
    """正文行 → [("text" | "table", 行)] 分段,表格与普通文本各自成段。"""
    segments: list[tuple[str, list[str]]] = []
    buf: list[str] = []
    mode: str | None = None
    for ln in lines:
        m = "table" if _is_table_line(ln) else "text"
        if m != mode and buf:
            segments.append((mode or "text", buf))
            buf = []
        mode = m
        buf.append(ln)
    if buf:
        segments.append((mode or "text", buf))
    return segments


def chunk_markdown(
    text: str, doc_name: str, content_type: str, *, max_chars: int = 500, overlap_chars: int = 80
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for ancestors, title, body_lines in _parse_sections(text):
        raw = "\n".join(body_lines).strip()
        is_key = any(w in title or w in raw for w in KEY_CLAUSE_WORDS)

        aliases: list[str] = []
        content_lines = body_lines
        if content_type == "faq":
            content_lines, aliases = [], []
            for ln in body_lines:
                if ALIAS_MARK in ln:
                    tail = ln.split(ALIAS_MARK, 1)[1]
                    aliases.extend(a.strip() for a in re.split(r"[/;]", tail) if a.strip())
                else:
                    content_lines.append(ln)

        if content_type == "faq":
            questions = "\n".join([title, *aliases])
            category = ancestors[-1] if ancestors else title
        else:
            questions = title
            category = " > ".join(ancestors) if ancestors else title
        section_path = " > ".join([doc_name, *ancestors, title])

        for seg_kind, seg_lines in _partition_tables(content_lines):
            if not "\n".join(seg_lines).strip():
                continue
            if seg_kind == "table":
                blocks = _split_table(seg_lines, max_chars)
            else:
                seg = "\n".join(seg_lines).strip()
                blocks = _recursive_split(seg, max_chars)
                for i in range(1, len(blocks)):
                    prefix = _overlap_prefix(blocks[i - 1], overlap_chars)
                    if prefix:
                        blocks[i] = prefix + blocks[i]
            for b in blocks:
                chunks.append(Chunk(category, questions, b, section_path, content_type, is_key))
    return chunks
