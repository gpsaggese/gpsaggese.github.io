"""Find "this record replaces that one" relations with a language model, then check every claim.

A claim is kept only if its target is another record in the corpus and its evidence quote appears word for word
in the record. The rule-based parser still runs; its edges are merged in, so a model miss cannot lose them.
"""
import json
import re

SYSTEM = (
    "You read one architecture decision record (ADR) and list replacement relations between it and OTHER records "
    "in the same repository. Use only IDs from the candidate list. Types: 'supersedes' (this record replaces "
    "another) or 'superseded_by' (another record replaces this one). The record must SAY so, in any wording or "
    "format (replaces, supersedes, no longer applies, retired in favour of, obsolete, and so on). A record that "
    "builds on, extends, refines or depends on another record, or merely describes a limitation of it, is NOT a "
    "replacement, and neither is 'amends' or 'amended by' on its own: do not infer one. "
    "Scope is 'partial' when the record says only part or some aspects are replaced, otherwise 'full'. Every relation needs an "
    "'evidence' quote copied word for word from the record. For a partial replacement also give 'replaced' when the "
    "record quotes or names the statement that no longer holds, copied word for word, otherwise null. "
    "If the record states no replacement, return an empty list. Reply with JSON only: "
    '{"relations":[{"type":"supersedes","target":"<id>","scope":"full","evidence":"...","replaced":null}]}'
)
MAX_CHARS = 9000


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"(?m)^\s*>+ ?", "", s)).strip().lower()


def _parse(reply: str) -> list[dict]:
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return []
    try:
        relations = json.loads(match.group(0)).get("relations", [])
    except (ValueError, AttributeError):
        return []
    return [r for r in relations if isinstance(r, dict)]


def model_edges(record_id: str, text: str, corpus: set[str], chat) -> list[dict]:
    """Edges as {newer, older, scope, statement, evidence}, already checked against the record text."""
    candidates = sorted(corpus - {record_id})
    user = f"Candidate IDs: {json.dumps(candidates)}\nThis record's ID: {record_id}\n\n--- RECORD ---\n{text[:MAX_CHARS]}"
    reply = chat.complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    body, edges = _norm(text), []
    for rel in _parse(reply):
        target, kind, evidence = rel.get("target"), rel.get("type"), str(rel.get("evidence") or "")
        if target not in corpus or target == record_id or kind not in ("supersedes", "superseded_by"):
            continue
        if not evidence or _norm(evidence) not in body:
            continue
        replaced = rel.get("replaced")
        scope = "partial" if rel.get("scope") == "partial" else "full"
        statement = replaced if replaced and _norm(str(replaced)) in body else None
        newer, older = (record_id, target) if kind == "supersedes" else (target, record_id)
        edges.append({"newer": newer, "older": older, "scope": scope, "statement": statement, "evidence": evidence})
    return edges


def apply_model_relations(records: list[dict], text_of: dict[str, str], chat, log=print) -> None:
    """Replace each record's `supersedes` list with model edges plus the rule-based ones. Edited in place."""
    corpus = set(text_of)
    found: dict[tuple[str, str], dict] = {}
    for rid in text_of:
        for e in model_edges(rid, text_of[rid], corpus, chat):
            key = (e["newer"], e["older"])
            prior = found.get(key)
            # A partial reading from either side wins: calling a partial replacement full would hide a live statement.
            if prior is None or (e["scope"] == "partial" and prior["scope"] != "partial"):
                found[key] = {**e, "statement": e["statement"] or (prior or {}).get("statement")}
    for rec in records:
        merged = {s["target"]: {**s, "source": "rule"} for s in rec["supersedes"]}
        for (newer, older), e in found.items():
            if newer != rec["id"]:
                continue
            rule = merged.get(older)
            merged[older] = {"target": older, "scope": e["scope"] if not rule else rule["scope"],
                             "statement": (rule or {}).get("statement") or e["statement"],
                             "source": "rule+model" if rule else "model", "evidence": e["evidence"]}
        rec["supersedes"] = list(merged.values())
    log(f"  model relations: {len(found)} replacement edges found")
