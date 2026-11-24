#!/usr/bin/env python
"""
Measure how many prompt tokens each seed seller bills per character (#720).

The script makes the paid calls and saves them; the notebook
`notebooks/prompt_tokens.ipynb` fits and plots them.

# Usage Example

- Print the plan and its maximum cost without calling anyone:
> experiment_prompt_tokens.py --dry_run

- Run it (needs `OPENROUTER_API_KEY`) and write
  `data/prompt_tokens/prompt_tokens.raw.jsonl`:
> experiment_prompt_tokens.py

Import as:

import research.Noesis.scripts.experiment_prompt_tokens as rnsexprto
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import math
import os
import random
from typing import Any, Dict, List, Optional

import tqdm

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hprint as hprint

import research.Noesis.common.noesis_settings as rnonoset
import research.Noesis.gateway.gateway_openrouter as rnogaope
import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.market.supply_adapter as rnmasuad
import research.Noesis.noesis_platform.supply_feed as rnplsufe

_LOG = logging.getLogger(__name__)

# The experiment's signature: the same config reproduces the same calls.
CONFIG: Dict[str, Any] = {
    "model": rnonoset.DEFAULT_MODEL_ID,
    "seed_path": rnonoset.DEFAULT_SEED_PATH,
    "catalog_path": os.path.join(
        rnonoset.NOESIS_DIR, "data", "openrouter", "endpoints_sample.json"
    ),
    "text_types": ["prose", "code", "numbers"],
    "n_chars": [50, 250, 1000, 3000, 5000],
    # The same text sent as one message and split over five.
    "n_messages": [1, 5],
    "seed": 1,
    "max_tokens": 1,
    "timeout_s": 30.0,
}
# Where the run saves its calls, and where the notebook reads them.
RAW_PATH = os.path.join(
    rnonoset.NOESIS_DIR, "data", "prompt_tokens", "prompt_tokens.raw.jsonl"
)
_WORDS = [
    "the", "market", "clears", "once", "a", "round", "and", "every", "seller",
    "posts", "one", "ask", "for", "model", "it", "serves", "while", "buyers",
    "bid", "tasks", "with", "latency", "limit", "price", "they", "will", "not",
    "exceed", "so", "checker", "can", "grade", "what", "was", "delivered",
]  # fmt: skip


@dataclasses.dataclass(frozen=True)
class Cell:
    """
    One planned call: a generated prompt sent to one seller.
    """

    seller_id: str
    provider_slug: str
    text_type: str
    n_chars: int
    n_messages: int
    messages: List[Dict[str, str]]


@dataclasses.dataclass(frozen=True)
class Measurement:
    """
    What one call billed; `prompt_tokens` is `None` on failure.
    """

    seller_id: str
    provider_slug: str
    text_type: str
    n_chars: int
    n_messages: int
    status: str
    served_by: Optional[str]
    prompt_tokens: Optional[int]
    error: Optional[str]
    cost_usd: float


# #############################################################################
# Plan
# #############################################################################


def generate_text(text_type: str, n_chars: int, seed: int) -> str:
    """
    Build exactly `n_chars` characters of prose, code, or numbers.
    """
    rng = random.Random(f"{text_type}-{n_chars}-{seed}")
    pieces = []
    length = 0
    while length < n_chars:
        if text_type == "prose":
            piece = rng.choice(_WORDS) + (". " if rng.random() < 0.1 else " ")
        elif text_type == "code":
            piece = f"def f{rng.randint(0, 99)}(x):\n    return x * "
            piece += f"{rng.randint(1, 999)} + {rng.randint(1, 999)}\n"
        elif text_type == "numbers":
            piece = f"{rng.uniform(-1e4, 1e4):.2f}, {rng.randint(0, 10**6)}; "
        else:
            raise ValueError(f"Unknown text_type='{text_type}'")
        pieces.append(piece)
        length += len(piece)
    text = "".join(pieces)[:n_chars]
    return text


def split_messages(text: str, n_messages: int) -> List[Dict[str, str]]:
    """
    Cut `text` into `n_messages` user messages of near-equal length.
    """
    step = math.ceil(len(text) / n_messages)
    messages = [
        {"role": "user", "content": text[i * step : (i + 1) * step]}
        for i in range(n_messages)
    ]
    return messages


def build_plan(config: Dict[str, Any]) -> List[Cell]:
    """
    Expand `config` into one cell per seller, text type, length and shape.
    """
    cells = []
    for seller in rnplsufe.load_seller_supply(config["seed_path"]):
        for text_type in config["text_types"]:
            for n_chars in config["n_chars"]:
                text = generate_text(text_type, n_chars, config["seed"])
                for n_messages in config["n_messages"]:
                    cells.append(
                        Cell(
                            seller.seller_id,
                            seller.provider_slug,
                            text_type,
                            n_chars,
                            n_messages,
                            split_messages(text, n_messages),
                        )
                    )
    return cells


def estimate_max_cost_usd(cells: List[Cell], config: Dict[str, Any]) -> float:
    """
    Bound the spend: one token per character plus 50 of framing, at the highest
    seed-seller prices in the catalog sample.
    """
    with open(config["catalog_path"], "r") as f:
        quotes = rnmasuad.parse_endpoint_quotes(json.load(f))
    slugs = {cell.provider_slug for cell in cells}
    quotes = [quote for quote in quotes if quote.provider_slug in slugs]
    hdbg.dassert_lt(0, len(quotes), "No seed seller in the catalog")
    prompt_tokens = sum(cell.n_chars + 50 for cell in cells)
    completion_tokens = len(cells) * config["max_tokens"]
    cost = prompt_tokens * max(q.prompt_price for q in quotes)
    cost += completion_tokens * max(q.completion_price for q in quotes)
    return cost


# #############################################################################
# Run
# #############################################################################


async def run(
    cells: List[Cell], provider: rnogapro.Provider, config: Dict[str, Any]
) -> List[Measurement]:
    """
    Send the cells one at a time and record what each provider billed.
    """
    measurements = []
    for cell in tqdm.tqdm(cells):
        call = rnogapro.ProviderCall(
            model=config["model"],
            messages=cell.messages,
            max_tokens=config["max_tokens"],
            timeout_s=config["timeout_s"],
            temperature=0.0,
        )
        result = await provider.call(cell.provider_slug, call)
        fields = dataclasses.asdict(cell)
        del fields["messages"]
        measurements.append(
            Measurement(
                **fields,
                status=result.status,
                served_by=result.served_by,
                prompt_tokens=result.prompt_tokens,
                error=result.error,
                cost_usd=result.cost_usd,
            )
        )
    return measurements


def save_measurements(measurements: List[Measurement], path: str) -> None:
    """
    Write one JSON line per call.
    """
    hio.create_enclosing_dir(path, incremental=True)
    with open(path, "w") as f:
        f.writelines(
            json.dumps(dataclasses.asdict(m)) + "\n" for m in measurements
        )


def load_measurements(path: str) -> List[Measurement]:
    """
    Read the calls written by `save_measurements()`.
    """
    with open(path, "r") as f:
        measurements = [Measurement(**json.loads(line)) for line in f]
    return measurements


# #############################################################################
# Script
# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=hparser.CustomHelpFormatter
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print the plan; call nobody"
    )
    parser.add_argument(
        "--max_usd", type=float, default=0.5, help="Abort above this cost"
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    _LOG.debug(hprint.to_str("args"))
    # Bound the cost before spending anything.
    cells = build_plan(CONFIG)
    max_cost = estimate_max_cost_usd(cells, CONFIG)
    print(f"{len(cells)} calls, cost at most ${max_cost:.4f}")
    if args.dry_run:
        return
    hdbg.dassert_lte(max_cost, args.max_usd, "Maximum cost above --max_usd")
    settings = rnonoset.load_settings()
    hdbg.dassert_ne(settings.openrouter_api_key, "", "Set OPENROUTER_API_KEY")
    provider = rnogaope.OpenRouterProvider(settings.openrouter_api_key)

    async def _go() -> List[Measurement]:
        try:
            return await run(cells, provider, CONFIG)
        finally:
            await provider.aclose()

    # Run and save the raw calls for the notebook.
    measurements = asyncio.run(_go())
    save_measurements(measurements, RAW_PATH)
    n_ok = sum(m.status == rnogapro.OK for m in measurements)
    spend = sum(m.cost_usd for m in measurements)
    print(
        f"{n_ok}/{len(measurements)} calls succeeded, ${spend:.4f}, {RAW_PATH}"
    )


if __name__ == "__main__":
    _main(_parse())
