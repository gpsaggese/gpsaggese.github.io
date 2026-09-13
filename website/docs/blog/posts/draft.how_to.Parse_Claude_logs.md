# Parsing Claude Code logs

When running Claude Code with `--print --output-format=stream-json`, every event
is written to stdout as a newline-delimited JSON stream (NDJSON). Each line is
a self-contained JSON record.

| Flag(s)                                                              | Output Format           | Notes                                                                 |
|------------------------------------------------------------------------|---------------------------|------------------------------------------------------------------------|
| `-p "..."`                                                            | Human-readable text      | Baseline: just Claude's final answer, then exits                      |
| `-p "..." --verbose`                                                  | Human-readable text      | No real effect alone; only matters combined with `--output-format stream-json` |
| `-p "..." --debug`                                                    | Text + debug logs        | Adds internal tracing (API calls, tool/hook activity, retries) on stdout |
| `-p --output-format=stream-json "..."`                                | Newline-delimited JSON   | Structured, machine-parseable event stream (assistant/user/system/result) |
| `-p --output-format=stream-json --verbose "..."`                      | Newline-delimited JSON   | `--verbose` is required for stream-json to work as intended            |
| `-p --output-format=stream-json --verbose --include-partial-messages "..."` | Newline-delimited JSON (token-level) | Adds `stream_event` messages with token-by-token deltas, tool calls, etc. — true real-time streaming |

PROMPT="Describe recursion in 100 words"

cc -p "Describe recursion in 100 words" 2>&1 | tee tmp1.txt
cc -p "Describe recursion in 100 words" --verbose 2>&1 | tee tmp2.txt
cc -p "Describe recursion in 100 words" --debug 2>&1 | tee tmp3.txt
cc -p "Describe recursion in 100 words" --output-format=stream-json 2>&1 | tee tmp4.txt
cc -p "Describe recursion in 100 words" --output-format=stream-json --include-partial-messages 2>&1 | tee tmp5.txt
cc -p "Describe recursion in 100 words" --verbose --output-format=stream-json --include-partial-messages 2>&1 | tee tmp6.txt

 25167 Jul 30 17:29 tmp1.txt
115069 Jul 30 17:30 tmp2.txt
 24455 Jul 30 17:31 tmp3.txt
 61436 Jul 30 17:59 tmp4.txt
 26337 Jul 30 18:04 tmp5.txt
122959 Jul 30 18:05 tmp6.txt

tmp1.txt (25KB) — baseline output with JSON events (init, status, stream messages, final result summary)

tmp2.txt (115KB) — --verbose adds extensive debug logging, ~4.5x larger. Contains detailed event traces and state at each step

tmp3.txt (24KB) — --debug similar to baseline size; appears to be alternate debug output format or filtered logging

tmp4.txt (61KB) — --output-format=stream-json structures output purely as JSON stream events, no plain text result summary at end

tmp5.txt (26KB) — stream-json + --include-partial-messages adds incomplete message fragments as they stream, only slightly larger than tmp1

tmp6.txt (123KB) — --verbose + stream-json + --include-partial-messages combines all three, largest output. Verbose + streaming partial messages = most detailed event log

Running extract_cc

JSON records parsed:
- tmp1 (baseline): 40
- tmp2 (--verbose): 209 ← 5.2x more
- tmp3 (--debug): 38
- tmp4 (--output-format=stream-json): 114 ← 2.8x more
- tmp5 (stream-json + include-partial): 43
- tmp6 (verbose + stream-json + include-partial): 214 ← 5.3x more

Key insight: All outputs contain same final answer text. Extraction differs in captured reasoning depth:

tmp1/tmp3 — minimal thinking shown. Quick, clean narrative.

tmp2 — thinking explodes with word-counting iterations. Shows verbose logging of internal decision loops (counting words, adjusting caveman style).

tmp4 — thinking inflates to 1,494 tokens (massively expanded). Stream-JSON captures incremental reasoning steps as separate events instead of aggregating them.

tmp5 — minimal overhead above baseline despite --include-partial-messages. Partial fragments don't bloat the internal reasoning capture.

tmp6 — combines verbose debug logging + stream-JSON granularity. 5,384 thinking tokens—dominant signal. Shows the full unfiltered thought process including multiple re-attempts at word counting and rephrasing.

Pattern:
- --verbose forces the model to emit all intermediate deliberation
- --output-format=stream-json fragments it into separate events, multiplying record count
- Combined, they capture the raw cognitive scaffolding rather than just the polished final output.

- Stream-JSON fragments events, multiplying record count without adding substance
- Together they create massive thinking token waste (~49x baseline)

## Record types

The `type` field classifies the top-level record:
- `system` records contain session metadata and status
- `stream_event` records carry intermediate LLM streaming data (model output,
  token usage)
- `assistant` records are complete LLM responses
- `user` records wrap tool results sent back to the model.

| Type | Subtype | What it contains |
|------|---------|------------------|
| `system` | `init` | Session metadata: model, tools, skills, config, version |
| `system` | `status` | Status transitions (e.g. `"requesting"`) |
| `system` | `thinking_tokens` | Estimated thinking token count (delta-based) |
| `stream_event` | `message_start` | Start of an LLM response: model name, provider (Alibaba/SiliconFlow), ttft_ms |
| `stream_event` | `content_block_start` | Start of a content block (thinking, tool_use, text, redacted_thinking) |
| `stream_event` | `content_block_delta` | Streaming text/thinking/input_json fragments within a block |
| `stream_event` | `content_block_stop` | End of a content block (signals a complete thinking or tool_use block) |
| `stream_event` | `message_delta` | End-of-request summary: input/output tokens, cost, speed tier |
| `stream_event` | `message_stop` | End of the message stream |
| `assistant` | | Complete assistant message with all content blocks (thinking, tool_use, text) |
| `user` | | User message containing tool_result blocks returned from tool execution |

## Key fields to extract

### Session metadata (`system/init`)

```json
{
  "model": "deepseek/deepseek-v4-flash",           // LLM used for this session
  "claude_code_version": "2.1.173",                // Claude Code CLI version
  "session_id": "fdac299b-...",                   // Unique session identifier
  "permissionMode": "bypassPermissions",          // Permission level for tool execution
  "tools": ["Bash", "Read", "WebFetch", ...],     // Available tools
  "skills": ["notebook.create_api_intro", ...]    // Available skills
}
```

### Request cost (`stream_event/message_delta`)

```json
{
  "event": {
    "type": "message_delta",
    "usage": {
      "input_tokens": 26451,
      "output_tokens": 320,
      "output_tokens_details": { "thinking_tokens": 62 },
      "cost": 0.003630,
      "speed": "standard"
    }
  }
}
```

### Tool calls (`assistant` > content blocks with `type: "tool_use"`)

```json
{
  "type": "tool_use",
  "name": "WebFetch",
  "input": { "url": "...", "prompt": "..." }
}
```

### Tool results (`user` > content blocks with `type: "tool_result"`)

```json
{
  "type": "tool_result",
  "tool_use_id": "call_...",
  "is_error": false,
  "content": "...",
  "url": "https://...",
  "durationMs": 22074
}
```

## Using `jq`

- One-liner to extract all tool calls with their names and inputs:
  ```bash
  jq -r 'select(.type == "assistant") |
    .message.content[] | select(.type == "tool_use") |
    "\(.name) \(.id | .[0:20])"'
  ```

- Example output (from a log with tool calls):
  ```
  WebFetch call_7k9x8c5m2q1a9b3
  Bash call_8p2k3j1l5q8w2b9
  Read call_6h8g2m4n9x5y1p7
  ```

- Extract per-request costs:
  ```bash
  jq -r 'select(.type == "stream_event" and .event.type == "message_delta") |
    [.event.usage.input_tokens, .event.usage.output_tokens, .event.usage.cost] | @tsv'
  ```

- Example output (tab-separated: input_tokens, output_tokens, cost):
  ```
  10	4097	0.003630
  26451	320	0.001845
  ```

- Reconstruct thinking content by collecting all `thinking_delta` fragments:

  ```bash
  jq -r 'select(.type == "stream_event" and .event.type == "content_block_delta"
    and .event.delta.type == "thinking_delta") | .event.delta.thinking' log.txt
  ```

- Example output (concatenated thinking fragments):
  ```
  User is asking for a 100-word description of recursion. This is straightforward.
  I should provide a clear explanation.

  Given the "caveman" style guidance (minimize tokens, be terse):
  - Drop articles (a/an/the)
  - Keep technical substance
  - Use fragments OK
  ```

## Using the extraction script

- The repo includes `extract_cc_log.py` which does all of the above in one pass:
  ```bash
  # Full narrative (thinking + costs + tools + text)
  ./extract_cc_log.py --file log.txt

  # Only the text the assistant showed the user
  ./extract_cc_log.py --file log.txt --text_only

  # Save to a file
  ./extract_cc_log.py --file log.txt --output_dir /tmp
  ```

## Log structure diagram

```
{"type":"system","subtype":"init",...}            # session init
{"type":"system","subtype":"status",...}          # requesting status
{"type":"stream_event","event":{"type":"message_start",...}}
{"type":"stream_event","event":{"type":"content_block_start",...}}
{"type":"stream_event","event":{"type":"content_block_delta",...}}  # ~200-2000x
{"type":"stream_event","event":{"type":"content_block_stop",...}}
... repeated per tool call ...
{"type":"assistant",...}                           # full message snapshot
{"type":"user",...}                                # tool results
{"type":"stream_event","event":{"type":"message_delta",...}}  # cost
{"type":"stream_event","event":{"type":"message_stop",...}}
```

Each request cycle is:
- `message_start`
- N × (`content_block_start` / `content_block_delta` / `content_block_stop`)
- `message_delta`
- `message_stop`
- `user` record carrying tool results back to the model
