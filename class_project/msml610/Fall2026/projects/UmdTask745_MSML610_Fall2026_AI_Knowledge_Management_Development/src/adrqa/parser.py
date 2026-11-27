"""Parse Decision / Rationale / Approach records into structured fields (no LLM)."""
import re

from markdown_it import MarkdownIt

from .config import RECORD_ROOT

FIELDS = ("decision", "rationale", "approach")
_REF = re.compile(r"\.\./([\w.-]+)/README\.md")
_SUPERSEDES = re.compile(r"\bsupersedes\b|\bsuperseded\s+(?!by\b)", re.I)
_QUOTE = re.compile(r"[\"“]([^\"”]+)[\"”]")
_STATUS = re.compile(r"^\W*status\W*:\s*(\w+)", re.I | re.M)


def parse_record(record_id: str, text: str, corpus: set[str]) -> dict:
    tokens = MarkdownIt().parse(text)
    lines = text.splitlines()
    headings = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            headings.append((tok.tag, tok.map[0], tokens[i + 1].content.strip()))

    title = next((h[2] for h in headings if h[0] == "h1"), None)
    h2s = [h for h in headings if h[0] == "h2"]
    fields: dict[str, str | None] = dict.fromkeys(FIELDS)
    for pos, (_, start, name) in enumerate(h2s):
        key = name.lower()
        if key in fields:
            end = h2s[pos + 1][1] if pos + 1 < len(h2s) else len(lines)
            fields[key] = "\n".join(lines[start + 1:end]).strip() or None

    status = _STATUS.search(text)
    return {
        "id": record_id,
        "path": f"{RECORD_ROOT}{record_id}/README.md",
        "title": title,
        "date": record_id[:10],
        "status": status.group(1).lower() if status else None,
        "fields": fields,
        "references": sorted({m for m in _REF.findall(text) if m in corpus and m != record_id}),
        "supersedes": _find_supersedes(text, corpus, record_id),
        "missing_fields": [f for f in FIELDS if fields[f] is None],
    }


def _find_supersedes(text: str, corpus: set[str], record_id: str) -> list[dict]:
    flat = re.sub(r"\s+", " ", re.sub(r"^>\s?", "", text, flags=re.M))
    found = []
    for sentence in re.split(r"(?<=[.!?])\s+", flat):
        if not _SUPERSEDES.search(sentence):
            continue
        for target in _REF.findall(sentence):
            if target in corpus and target != record_id:
                quote = _QUOTE.search(sentence)
                found.append({"target": target, "scope": "partial" if quote else "full",
                              "statement": quote.group(1) if quote else None})
    return found


def section_spans(text: str) -> list[dict]:
    """Every heading with its line range, 0-based and end-exclusive. A section runs to the next heading
    of the same or a higher level, so Approach includes its ### subsections."""
    tokens = MarkdownIt().parse(text)
    heads = [(int(tok.tag[1]), tok.map[0], tokens[i + 1].content.strip())
             for i, tok in enumerate(tokens) if tok.type == "heading_open"]
    total = len(text.splitlines())
    return [{"name": name, "level": level, "start": start,
             "end": next((s for lv, s, _ in heads[i + 1:] if lv <= level), total),
             "field": name.lower() if level == 2 and name.lower() in FIELDS else None}
            for i, (level, start, name) in enumerate(heads)]
