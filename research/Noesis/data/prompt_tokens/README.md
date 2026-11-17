# Prompt-Token Calls

- `prompt_tokens.raw.jsonl`: the 150 calls of `scripts/experiment_prompt_tokens.py`
  run on 2026-11-12, one JSON line per call
  - Fields: `seller_id`, `provider_slug`, `text_type`, `n_chars`, `n_messages`,
    `status`, `served_by`, `prompt_tokens` (`null` on failure), `error`,
    `cost_usd`
  - 89 calls succeeded; Together and Novita failed every call with
    `gateway_error`
- Rewritten by each run of the script; `Test_build_plan` checks that the calls are
  the plan of the script's `CONFIG`
