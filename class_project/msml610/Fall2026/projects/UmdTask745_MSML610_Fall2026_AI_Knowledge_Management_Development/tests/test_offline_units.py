"""Offline units on small fixtures: parser, linker token rules, graph, staleness rules."""
from adrqa.graph import build_graph, governing_records, linked_files, references, superseded_by
from adrqa.linker import SHARED_CONFIG, extract_tokens
from adrqa.parser import parse_record
from adrqa.staleness import warnings_for

RECORD = """# Some Title

## Decision

We do a thing.

```java
# not a heading
class Foo {}
```

## Rationale

Because of [other](../old-record/README.md).

> This changes things. It supersedes the statement of the
> [other](../old-record/README.md) decision record that "the store won't change".

## Approach

### Sub heading

Use `FooBar` and `some-module`.
"""


def parsed():
    return parse_record("2025-01-01-new", RECORD, {"2025-01-01-new", "old-record"})


def test_parser_fields_and_date():
    r = parsed()
    assert r["title"] == "Some Title" and r["date"] == "2025-01-01"
    assert r["fields"]["decision"].startswith("We do a thing.")
    assert "# not a heading" in r["fields"]["decision"]
    assert r["fields"]["approach"].startswith("### Sub heading")
    assert r["missing_fields"] == [] and r["status"] is None


def test_parser_finds_partial_supersede_and_references():
    r = parsed()
    assert r["references"] == ["old-record"]
    assert r["supersedes"] == [{"target": "old-record", "scope": "partial", "statement": "the store won't change"}]


def test_parser_records_missing_section_and_ignores_superseded_by():
    r = parse_record("x", "# T\n\n## Decision\n\nSuperseded by [n](../old-record/README.md).\n", {"x", "old-record"})
    assert r["missing_fields"] == ["rationale", "approach"]
    assert r["supersedes"] == []


def test_extract_tokens():
    t = extract_tokens("Uses `FooBar`, BareCamelCase, `some-module`, `a/b/c.java`, `settings.gradle.kts`, `edc.foo.id`.")
    assert t["camel"] == {"FooBar", "BareCamelCase"}
    assert t["kebab"] == {"some-module"}
    assert t["slashed"] == {"a/b/c.java"}
    assert t["named"] == {"settings.gradle.kts"}


def test_shared_config_matches_root_files_only():
    assert SHARED_CONFIG.match("gradle/libs.versions.toml") and SHARED_CONFIG.match("settings.gradle.kts")
    assert not SHARED_CONFIG.match("core/foo/build.gradle.kts")


def test_graph_queries():
    g = build_graph([parsed(), {"id": "old-record", "references": [], "supersedes": []}],
                    [{"record_id": "old-record", "file_id": "a/b/C.java", "signal": "explicit_class"}])
    assert references(g, "2025-01-01-new") == ["old-record"]
    assert superseded_by(g, "old-record")[0][0] == "2025-01-01-new"
    assert linked_files(g, "old-record") == ["a/b/C.java"]
    assert governing_records(g, "a/b/C.java") == ["old-record"]
    assert governing_records(g, "a/b") == ["old-record"]
    assert governing_records(g, "a/bb") == []


def base_score(**kw):
    return {"record_id": "r", "stale": 5.0, "churn_code": 5, "dead_identifiers": [], **kw}


def test_warnings_distinguish_null_from_zero():
    assert [w["kind"] for w in warnings_for(base_score(stale=None, churn_code=None), 15)] == ["no_linked_code"]
    assert warnings_for(base_score(stale=0.0, churn_code=0), 15) == []


def test_warnings_threshold_and_dead_identifiers():
    assert [w["kind"] for w in warnings_for(base_score(stale=15.0), 15)] == ["stale_score"]
    w = warnings_for(base_score(dead_identifiers=["GoneClass"]), 15)
    assert w[0]["kind"] == "dead_identifier" and "GoneClass" in w[0]["detail"]
