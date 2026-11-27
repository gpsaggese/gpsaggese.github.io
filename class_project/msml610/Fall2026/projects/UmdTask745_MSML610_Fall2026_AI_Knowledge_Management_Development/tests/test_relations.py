import json

from adrqa.parser import parse_record
from adrqa.relations import apply_model_relations, model_edges

OLD, NEW = "2024-01-01-old", "2025-01-01-new"
TEXT = {OLD: "# Old\n\n## Decision\n\nUse queue A.\n",
        NEW: "# New\n\n## Decision\n\nWe move to Kafka. This replaces the queue choice in 2024-01-01-old in part.\n"}
CORPUS = set(TEXT)


class Fake:
    def __init__(self, relations):
        self.reply = json.dumps({"relations": relations})

    def complete(self, messages):
        return self.reply


def rel(**kw):
    return {"type": "supersedes", "target": OLD, "scope": "partial",
            "evidence": "This replaces the queue choice in 2024-01-01-old in part.", "replaced": None, **kw}


def test_keeps_verified_edge_and_scope():
    e = model_edges(NEW, TEXT[NEW], CORPUS, Fake([rel()]))
    assert [(x["newer"], x["older"], x["scope"]) for x in e] == [(NEW, OLD, "partial")]


def test_drops_invented_target_bad_quote_and_self():
    bad = [rel(target="nope"), rel(evidence="never written"), rel(target=NEW), rel(type="amends")]
    assert model_edges(NEW, TEXT[NEW], CORPUS, Fake(bad)) == []


def test_superseded_by_is_stored_on_the_newer_record():
    text = {OLD: "# Old\n\nThis was superseded by 2025-01-01-new.\n", NEW: TEXT[NEW]}
    e = model_edges(OLD, text[OLD], CORPUS, Fake([rel(type="superseded_by", target=NEW, scope="full",
                                                      evidence="This was superseded by 2025-01-01-new.")]))
    assert (e[0]["newer"], e[0]["older"]) == (NEW, OLD)


def test_garbage_reply_gives_no_edges():
    class Bad:
        def complete(self, m): return "sorry"
    assert model_edges(NEW, TEXT[NEW], CORPUS, Bad()) == []


def test_apply_merges_with_rule_edges_and_keeps_partial():
    recs = [parse_record(r, TEXT[r], CORPUS) for r in TEXT]
    apply_model_relations(recs, TEXT, Fake([rel()]), log=lambda *_: None)
    new = next(r for r in recs if r["id"] == NEW)
    assert new["supersedes"][0]["target"] == OLD and new["supersedes"][0]["scope"] == "partial"
    assert new["supersedes"][0]["source"] == "model"
