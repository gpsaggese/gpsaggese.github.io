# 02. Data model

**Status:** draft for review. The offline pipeline writes these as JSON under `prototype/data/`. Every online answer follows the `Answer` schema.

IDs: a record's `id` is its folder name. A file's `id` is its repo-relative path.

## DecisionRecord (`records.json`)

```json
{
  "title": "DecisionRecord",
  "type": "object",
  "required": ["id", "path", "title", "date", "status", "fields", "references", "supersedes", "missing_fields"],
  "properties": {
    "id": {"type": "string"},
    "path": {"type": "string"},
    "title": {"type": "string"},
    "date": {"type": "string", "description": "From the folder name"},
    "status": {"type": ["string", "null"], "description": "Null when the record has none. Never inferred"},
    "fields": {
      "type": "object",
      "required": ["decision", "rationale", "approach"],
      "properties": {
        "decision": {"type": ["string", "null"]},
        "rationale": {"type": ["string", "null"]},
        "approach": {"type": ["string", "null"]}
      }
    },
    "references": {"type": "array", "items": {"type": "string"}},
    "supersedes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["target", "scope", "statement"],
        "properties": {
          "target": {"type": "string"},
          "scope": {"enum": ["partial", "full"]},
          "statement": {"type": ["string", "null"], "description": "The replaced sentence, when quoted"}
        }
      }
    },
    "missing_fields": {"type": "array", "items": {"type": "string"}}
  }
}
```

## Link (`links.json`)

```json
{
  "title": "Link",
  "type": "object",
  "required": ["record_id", "file_id", "signal", "evidence", "shared_config"],
  "properties": {
    "record_id": {"type": "string"},
    "file_id": {"type": "string"},
    "signal": {"enum": ["explicit_class", "explicit_module", "explicit_path", "explicit_file", "adr_commit"]},
    "evidence": {"type": "string", "description": "The token in the record, or the commit SHA"},
    "shared_config": {"type": "boolean", "description": "If true, the file is excluded from churn"}
  }
}
```

## Graph (`graph.json`)

NetworkX node-link format, with these node and edge attributes:

```json
{
  "title": "GraphElement",
  "type": "object",
  "oneOf": [
    {"required": ["kind", "id"], "properties": {"kind": {"enum": ["record", "file"]}, "id": {"type": "string"}}},
    {"required": ["kind", "source", "target"], "properties": {
      "kind": {"enum": ["supersedes", "references", "governs"]},
      "source": {"type": "string"},
      "target": {"type": "string"},
      "scope": {"enum": ["partial", "full"]},
      "statement": {"type": ["string", "null"]},
      "signal": {"type": "string"}
    }}
  ]
}
```

## Chunk (held in the index)

```json
{
  "title": "Chunk",
  "type": "object",
  "required": ["chunk_id", "source_id", "source_type", "text"],
  "properties": {
    "chunk_id": {"type": "string"},
    "source_id": {"type": "string"},
    "source_type": {"enum": ["record", "code", "readme"]},
    "field": {"enum": ["decision", "rationale", "approach", "whole", null]},
    "start_line": {"type": ["integer", "null"]},
    "end_line": {"type": ["integer", "null"]},
    "text": {"type": "string"}
  }
}
```

## StalenessScore (`staleness.json`)

```json
{
  "title": "StalenessScore",
  "type": "object",
  "required": ["record_id", "stale", "reason", "churn_code", "edits_after_first", "n_linked_files", "dead_identifiers"],
  "properties": {
    "record_id": {"type": "string"},
    "stale": {"type": ["number", "null"]},
    "reason": {"enum": ["no_linked_files", null]},
    "churn_code": {"type": ["integer", "null"]},
    "edits_after_first": {"type": "integer"},
    "n_linked_files": {"type": "integer"},
    "dead_identifiers": {"type": "array", "items": {"type": "string"}}
  }
}
```

## GapEntry (`gap_report.json`)

```json
{
  "title": "GapEntry",
  "type": "object",
  "required": ["module", "complexity", "coverage", "gap"],
  "properties": {
    "module": {"type": "string"},
    "complexity": {"type": "number", "description": "Summed cyclomatic complexity of non-test Java"},
    "coverage": {"type": "number", "minimum": 0, "maximum": 1, "description": "Share of the module's files with a governs edge"},
    "gap": {"type": "number"}
  }
}
```

## Answer (every `ask` result)

```json
{
  "title": "Answer",
  "type": "object",
  "required": ["question", "config", "route", "outcome", "text", "claims", "warnings", "sources_checked"],
  "properties": {
    "question": {"type": "string"},
    "config": {"enum": ["B0", "B1", "B2", "B3", "B4"]},
    "route": {"enum": ["adr", "code", "graph", "none"], "description": "none for B0 and B1, which have no router"},
    "outcome": {"enum": ["answered", "abstained"]},
    "text": {"type": "string"},
    "claims": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["text", "citations", "in_context", "entailment", "supported"],
        "properties": {
          "text": {"type": "string"},
          "citations": {"type": "array", "items": {"type": "string"}},
          "in_context": {"type": "boolean"},
          "entailment": {"type": ["number", "null"]},
          "supported": {"type": "boolean"}
        }
      }
    },
    "warnings": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["record_id", "kind", "detail"],
        "properties": {
          "record_id": {"type": "string"},
          "kind": {"enum": ["stale_score", "dead_identifier", "no_linked_code", "superseded_statement"]},
          "detail": {"type": "string"}
        }
      }
    },
    "sources_checked": {"type": "array", "items": {"type": "string"}}
  }
}
```
