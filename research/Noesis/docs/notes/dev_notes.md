# Dev Notes

- Moved from the root `README.md` ("GP Notes"), unedited

## Build
```
cd /Users/saggese/src/umd_classes2/research/Noesis
i docker_build_local_image --version 1.0.0

cd /Users/saggese/src/umd_classes2
uvicorn research.Noesis.noesis_platform.main:app --reload

NOESIS_DB_BACKEND defaults to "memory" when unset — the postgres branch in main.py only triggers if you explicitly set NOESIS_DB_BACKEND=postgres. No Docker, no Postgres, no env vars needed at all for the default path.

Optional, to enable the write endpoints (POST /bids, /asks, /completions need X-API-Key):
export NOESIS_API_KEYS="key1:acct_1"
uvicorn research.Noesis.noesis_platform.main:app --reload

Verify:
curl http://127.0.0.1:8000/health
```

## Generate APIs

1. When server running, pull the live spec
  > curl http://127.0.0.1:8000/openapi.json > openapi.json

- Automatically generated
- http://127.0.0.1:8000/docs — Swagger UI
- http://127.0.0.1:8000/redoc — ReDoc

2. No server needed — dump spec straight from create_app()
cd /Users/saggese/src/umd_classes2

```
python3 -c "
import json
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.gateway.passthrough_proxy as rnopapro
import research.Noesis.noesis_platform.platform_api as rnoplapi

app = rnoplapi.create_app(rnbacaau.OrderBook(), rnopapro.Gateway(), {'key1': 'acct_1'})
with open('openapi.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
"
```

This is the schema built from the Pydantic models/routes in platform_api.py — no
HTTP round trip, no running process.

Once you have openapi.json, generate other representations from it:
- Markdown/HTML docs: npx @redocly/cli build-docs openapi.json
- Typed client SDK: openapi-python-client generate --path openapi.json
