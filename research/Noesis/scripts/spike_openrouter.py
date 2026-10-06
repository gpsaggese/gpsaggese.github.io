"""
OpenRouter provider spike (plan task T0.5).

For one model, call each OpenRouter provider endpoint N times with that provider
pinned and fallbacks off, then report per provider:
- whether the provider that actually served the request matches the pinned one
- latency p50 / p95, error rate
- tokens and real cost from OpenRouter's `usage`
- accuracy on a tiny known-answer probe set (a preview of canary grading)

Usage:
    # No key, no spend.
    > python research/Noesis/scripts/spike_openrouter.py --dry-run
    # Needs OPENROUTER_API_KEY.
    > python research/Noesis/scripts/spike_openrouter.py --n 20

The key is read from the environment or from `research/Noesis/.env`.
Writes `docs/findings/openrouter_spike.md` and a raw JSONL next to it.
"""

import argparse
import asyncio
import dataclasses
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional

import httpx

# `research/Noesis`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_API = "https://openrouter.ai/api/v1"
_OUT_DIR = os.path.join(_ROOT, "docs", "findings")

# Short, unambiguous questions every 70B model should get right.
_PROBES = [
    ("What is 17 * 23? Reply with only the number.", "391"),
    ("What is the capital of Australia? Reply with one word.", "canberra"),
    ("How many days are in a leap year? Reply with only the number.", "366"),
    ("What is 144 / 12? Reply with only the number.", "12"),
    ("What is the chemical symbol for gold? Reply with only the symbol.", "au"),
]
_MAX_TOKENS = 16


# #############################################################################
# Helpers
# #############################################################################


def _load_api_key() -> Optional[str]:
    key = os.environ.get("OPENROUTER_API_KEY")
    env_path = os.path.join(_ROOT, ".env")
    if not key and os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.read().splitlines()
        for line in lines:
            if line.strip().startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key or None


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


@dataclasses.dataclass
class Endpoint:
    provider_name: str
    tag: str
    quantization: Optional[str]
    prompt_price: float
    completion_price: float


@dataclasses.dataclass
class CallResult:
    tag: str
    pinned_provider: str
    served_provider: Optional[str]
    status: str
    latency_s: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cost_usd: Optional[float]
    answer: Optional[str]
    correct: Optional[bool]
    error: Optional[str]


# OpenRouter calls


async def _list_endpoints(
    client: httpx.AsyncClient, model: str
) -> List[Endpoint]:
    resp = await client.get(f"{_API}/models/{model}/endpoints")
    resp.raise_for_status()
    endpoints = []
    for e in resp.json()["data"]["endpoints"]:
        pricing = e.get("pricing", {})
        endpoints.append(
            Endpoint(
                provider_name=e["provider_name"],
                tag=e["tag"],
                quantization=e.get("quantization"),
                prompt_price=float(pricing.get("prompt", 0)),
                completion_price=float(pricing.get("completion", 0)),
            )
        )
    return endpoints


async def _call_once(
    client: httpx.AsyncClient,
    key: str,
    model: str,
    endpoint: Endpoint,
    prompt: str,
    expected: str,
    timeout_s: float,
) -> CallResult:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _MAX_TOKENS,
        "temperature": 0,
        "provider": {"only": [endpoint.tag], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    headers = {"Authorization": f"Bearer {key}"}
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{_API}/chat/completions",
            json=body,
            headers=headers,
            timeout=timeout_s,
        )
        latency = time.perf_counter() - start
    except httpx.TimeoutException:
        return CallResult(
            endpoint.tag,
            endpoint.provider_name,
            None,
            "timeout",
            time.perf_counter() - start,
            None,
            None,
            None,
            None,
            None,
            "timeout",
        )
    if resp.status_code != 200:
        return CallResult(
            endpoint.tag,
            endpoint.provider_name,
            None,
            f"http_{resp.status_code}",
            latency,
            None,
            None,
            None,
            None,
            None,
            resp.text[:200],
        )
    data = resp.json()
    if "error" in data:
        return CallResult(
            endpoint.tag,
            endpoint.provider_name,
            data.get("provider"),
            "upstream_error",
            latency,
            None,
            None,
            None,
            None,
            None,
            str(data["error"])[:200],
        )
    usage = data.get("usage") or {}
    answer = (data["choices"][0]["message"].get("content") or "").strip()
    return CallResult(
        tag=endpoint.tag,
        pinned_provider=endpoint.provider_name,
        served_provider=data.get("provider"),
        status="ok",
        latency_s=latency,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        cost_usd=usage.get("cost"),
        answer=answer,
        correct=_normalize(expected) in _normalize(answer),
        error=None,
    )


async def _probe_endpoint(
    client: httpx.AsyncClient,
    key: str,
    model: str,
    endpoint: Endpoint,
    n: int,
    timeout_s: float,
) -> List[CallResult]:
    # Sequential per provider so we measure latency, not our own queueing.
    results = []
    for i in range(n):
        prompt, expected = _PROBES[i % len(_PROBES)]
        results.append(
            await _call_once(
                client, key, model, endpoint, prompt, expected, timeout_s
            )
        )
    return results


# #############################################################################
# Report
# #############################################################################


def _estimate_max_cost(endpoints: List[Endpoint], n: int) -> float:
    # ~40 prompt tokens per probe incl. chat template; `_MAX_TOKENS` output.
    return sum(
        n * (40 * e.prompt_price + _MAX_TOKENS * e.completion_price)
        for e in endpoints
    )


def _summarize(
    endpoints: List[Endpoint], results: List[CallResult]
) -> List[Dict]:
    rows = []
    for e in endpoints:
        rs = [r for r in results if r.tag == e.tag]
        ok = [r for r in rs if r.status == "ok"]
        lat = [r.latency_s for r in ok]
        served = sorted({r.served_provider or "?" for r in ok})
        rows.append(
            {
                "provider": e.provider_name,
                "tag": e.tag,
                "quant": e.quantization or "unknown",
                "calls": len(rs),
                "ok": len(ok),
                "attribution_ok": sum(
                    r.served_provider == e.provider_name for r in ok
                ),
                "served_as": ", ".join(served) if served else "-",
                "p50_s": statistics.median(lat) if lat else float("nan"),
                "p95_s": _percentile(lat, 95),
                "accuracy": (
                    (sum(bool(r.correct) for r in ok) / len(ok))
                    if ok
                    else float("nan")
                ),
                "cost_usd": sum(r.cost_usd or 0 for r in ok),
                "errors": sorted({r.status for r in rs if r.status != "ok"}),
            }
        )
    return rows


def _write_report(
    model: str, n: int, rows: List[Dict], results: List[CallResult]
) -> str:
    os.makedirs(_OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    usable = [
        r
        for r in rows
        if r["ok"] == r["calls"] and r["attribution_ok"] == r["ok"]
    ]
    lines = [
        "# OpenRouter provider spike",
        "",
        f"- Run: {stamp}",
        f"- Model: `{model}`",
        (
            f"- Calls per provider: {n} (sequential), "
            f"`max_tokens={_MAX_TOKENS}`, "
            "`temperature=0`, `provider.only=[tag]`, `allow_fallbacks=false`"
        ),
        f"- Total cost: ${sum(r['cost_usd'] for r in rows):.4f}",
        (
            "- **Usable providers** (all calls ok and served by the pinned "
            "provider): "
            f"**{len(usable)}** — {', '.join(r['tag'] for r in usable) or 'none'}"
        ),
        "",
        (
            "| Provider | Tag | Quant | OK/calls | Attribution ok | Served as |"
            " p50 s | p95 s | Probe acc | Cost $ | Errors |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['provider']} | `{r['tag']}` | {r['quant']} | "
            f"{r['ok']}/{r['calls']} | "
            f"{r['attribution_ok']}/{r['ok']} | {r['served_as']} | "
            f"{r['p50_s']:.2f} | "
            f"{r['p95_s']:.2f} | {r['accuracy']:.0%} | {r['cost_usd']:.4f} | "
            f"{', '.join(r['errors']) or '-'} |"
        )
    lines += [
        "",
        (
            "Decisions this informs: PRD D3 (model), D4 (pinning/attribution), "
            "`l_max` "
            "(p95), which providers become demo sellers."
        ),
    ]
    report = os.path.join(_OUT_DIR, "openrouter_spike.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    raw = os.path.join(_OUT_DIR, "openrouter_spike.raw.jsonl")
    with open(raw, "w") as f:
        f.write(
            "\n".join(json.dumps(dataclasses.asdict(r)) for r in results) + "\n"
        )
    return report


# #############################################################################
# Main
# #############################################################################


async def _main(args: argparse.Namespace) -> int:
    async with httpx.AsyncClient() as client:
        endpoints = await _list_endpoints(client, args.model)
        if args.only:
            wanted = set(args.only.split(","))
            endpoints = [e for e in endpoints if e.tag in wanted]
        est = _estimate_max_cost(endpoints, args.n)
        print(
            f"{args.model}: {len(endpoints)} endpoints, "
            f"{args.n} calls each, est. max cost ${est:.4f}"
        )
        for e in endpoints:
            print(
                f"  {e.provider_name:<14} {e.tag:<28} "
                f"{e.quantization or 'unknown':<8} "
                f"${e.prompt_price * 1e6:.2f}/M in "
                f"${e.completion_price * 1e6:.2f}/M out"
            )
        if args.dry_run:
            return 0
        if est > args.max_usd:
            print(
                f"Estimated cost ${est:.4f} exceeds --max-usd {args.max_usd}; "
                "aborting"
            )
            return 1
        key = _load_api_key()
        if not key:
            print("OPENROUTER_API_KEY not set (env or .env); aborting.")
            return 1
        per_endpoint = await asyncio.gather(
            *[
                _probe_endpoint(client, key, args.model, e, args.n, args.timeout)
                for e in endpoints
            ]
        )
    results = [r for rs in per_endpoint for r in rs]
    rows = _summarize(endpoints, results)
    report = _write_report(args.model, args.n, rows, results)
    _print_report(report)
    return 0


def _print_report(report: str) -> None:
    with open(report, "r") as f:
        print(f.read())
    print(f"Wrote '{os.path.relpath(report, _ROOT)}'")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    p.add_argument("--n", type=int, default=20, help="calls per provider")
    p.add_argument("--only", default="", help="comma-separated endpoint tags")
    p.add_argument("--timeout", type=float, default=30.0, help="per-call seconds")
    p.add_argument(
        "--max-usd", type=float, default=1.0, help="abort above this estimate"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="list endpoints + cost, no calls"
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(_parse())))
