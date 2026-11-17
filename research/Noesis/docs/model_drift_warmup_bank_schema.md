# Controlled Warm-up Bank Schema

The $10 controlled warm-up consumes one JSON object per line. Blank lines are
ignored. The reviewed file must contain exactly 60 unique items: 12 in each of
`math`, `expert-science`, `code-tracing`, `instruction-following`, and
`multi-step-reasoning`.

## Row Format

```json
{"item_id":"math-001","family":"math","prompt":"What is 17 + 28? Return only the integer.","grader_id":"exact-integer-v1","grader_config_hash":"2bb80d537b1da3e38bd30361aa855686bde0ba72f8b25a25bfe97bf527a25b5b"}
```

| Field | Requirement |
| :---- | :---------- |
| `item_id` | Non-empty and unique across the bank |
| `family` | One of the five frozen family names; exactly 12 rows per family |
| `prompt` | Non-empty and no more than 20,000 characters |
| `grader_id` | Versioned identifier for the deterministic grader |
| `grader_config_hash` | SHA-256 identity of the grader's frozen expected answer and configuration |

The bank hash covers the ordered values of all five fields. Changing a prompt,
family, grader identity, grader configuration, or row order therefore creates a
new bank identity and a new paid plan.

The actual expected answers and grader configuration should live in a separate
reviewed file keyed by `item_id`. This keeps answer material out of the provider
request plan while allowing Lane 1 to grade the saved responses after
collection. Lane 2 uses the same saved prompt/response pairs without access to
answers or grades.
