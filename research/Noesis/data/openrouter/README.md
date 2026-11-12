# OpenRouter samples

- `endpoints_sample.json`: the response of
  `GET /api/v1/models/meta-llama/llama-3.3-70b-instruct/endpoints` captured on
  2026-10-18, trimmed to the fields the supply adapter reads (`tag`,
  `provider_name`, `quantization`, `pricing`)
- Used by `scripts/demo_supply_asks.py` so the demo runs without a key or network;
  pass `--live` to fetch the current catalog instead
