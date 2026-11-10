# OpenRouter provider spike

- Run: 2026-09-30 17:00
- Model: `meta-llama/llama-3.3-70b-instruct`
- Calls per provider: 20 (sequential), `max_tokens=16`, `temperature=0`, `provider.only=[tag]`, `allow_fallbacks=false`
- Total cost: $0.0026
- **Usable providers** (all calls ok and served by the pinned provider): **8** — deepinfra/turbo, novita/bf16, akashml/fp8, cloudflare/fp8, sambanova-turbo, groq, coreweave/fp16, together

| Provider | Tag | Quant | OK/calls | Attribution ok | Served as | p50 s | p95 s | Probe acc | Cost $ | Errors |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepInfra | `deepinfra/turbo` | fp8 | 20/20 | 20/20 | DeepInfra | 0.40 | 1.04 | 100% | 0.0001 | - |
| Novita | `novita/bf16` | bf16 | 20/20 | 20/20 | Novita | 0.99 | 2.39 | 95% | 0.0001 | - |
| AkashML | `akashml/fp8` | fp8 | 20/20 | 20/20 | AkashML | 0.51 | 0.94 | 100% | 0.0001 | - |
| Parasail | `parasail/fp8` | fp8 | 19/20 | 19/19 | Parasail | 0.48 | 2.21 | 100% | 0.0001 | http_429 |
| Cloudflare | `cloudflare/fp8` | fp8 | 20/20 | 20/20 | Cloudflare | 0.27 | 0.58 | 100% | 0.0004 | - |
| SambaNova | `sambanova-turbo` | unknown | 20/20 | 20/20 | SambaNova | 0.59 | 2.44 | 100% | 0.0002 | - |
| Groq | `groq` | unknown | 20/20 | 20/20 | Groq | 0.23 | 0.57 | 100% | 0.0006 | - |
| CoreWeave | `coreweave/fp16` | fp16 | 20/20 | 20/20 | CoreWeave | 0.18 | 0.26 | 100% | 0.0004 | - |
| Google | `google-vertex/us-central1` | unknown | 0/20 | 0/0 | - | nan | nan | nan% | 0.0000 | http_404 |
| Google | `google-vertex` | unknown | 0/20 | 0/0 | - | nan | nan | nan% | 0.0000 | http_404 |
| Together | `together` | unknown | 20/20 | 20/20 | Together | 0.57 | 1.30 | 100% | 0.0005 | - |

Decisions this informs: PRD D3 (model), D4 (pinning/attribution), `l_max` (p95), which providers become demo sellers.
