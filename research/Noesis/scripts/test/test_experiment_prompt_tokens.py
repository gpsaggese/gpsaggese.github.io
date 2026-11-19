"""
Import as:

import research.Noesis.scripts.test.test_experiment_prompt_tokens as rnsttept
"""

import asyncio
import logging
import math
import os
from typing import Any, Dict, List

import helpers.hunit_test as hunitest

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.scripts.experiment_prompt_tokens as rnsexprto

_LOG = logging.getLogger(__name__)

# Two sellers whose slugs are in `data/openrouter/endpoints_sample.json`.
_SEED = """
sellers:
  - {seller_id: groq,      provider_slug: groq,            provider_name: Groq,      l_typical: 0.57, r_typical: 1.0}
  - {seller_id: deepinfra, provider_slug: deepinfra/turbo, provider_name: DeepInfra, l_typical: 1.04, r_typical: 1.0}
"""


# #############################################################################
# Helpers
# #############################################################################


class _ArithmeticProvider:
    """
    Bill `ceil(chars / 4) + 12` tokens per request plus 4 per extra message.
    """

    def __init__(self) -> None:
        self.calls: List[rnogapro.ProviderCall] = []

    async def call(
        self, slug: str, request: rnogapro.ProviderCall
    ) -> rnogapro.ProviderResult:
        """
        Record the call and return a successful result with the billed tokens.
        """
        self.calls.append(request)
        n_chars = sum(len(m["content"]) for m in request.messages)
        tokens = math.ceil(n_chars / 4) + 12 + 4 * (len(request.messages) - 1)
        result = rnogapro.ProviderResult(
            status=rnogapro.OK,
            text="",
            served_by=slug,
            model_served=request.model,
            prompt_tokens=tokens,
            completion_tokens=1,
            cost_usd=0.0,
            latency_s=0.1,
            http_status=200,
            error=None,
        )
        return result


def _get_config(
    self_: hunitest.TestCase, text_types: List[str], n_chars: List[int]
) -> Dict[str, Any]:
    """
    Build a config over the two sellers in `_SEED`, written to scratch space.
    """
    seed_path = os.path.join(self_.get_scratch_space(), "seed.yaml")
    with open(seed_path, "w") as f:
        f.write(_SEED)
    config = {
        **rnsexprto.CONFIG,
        "seed_path": seed_path,
        "text_types": text_types,
        "n_chars": n_chars,
    }
    return config


# #############################################################################
# Test_generate_text
# #############################################################################


class Test_generate_text(hunitest.TestCase):
    """
    Test `generate_text()`.
    """

    def test1(self) -> None:
        """
        Test that every text type hits the requested length and is
        reproducible.
        """
        # Prepare inputs.
        text_types = rnsexprto.CONFIG["text_types"]
        lengths = [50, 1000]
        seed = 1
        # Run test and check outputs.
        for text_type in text_types:
            for n_chars in lengths:
                actual = rnsexprto.generate_text(text_type, n_chars, seed)
                again = rnsexprto.generate_text(text_type, n_chars, seed)
                self.assertEqual(len(actual), n_chars)
                self.assertEqual(actual, again)


# #############################################################################
# Test_split_messages
# #############################################################################


class Test_split_messages(hunitest.TestCase):
    """
    Test `split_messages()`.
    """

    def test1(self) -> None:
        """
        Test that splitting into five messages keeps every character.
        """
        # Prepare inputs.
        text = rnsexprto.generate_text("prose", 1000, 1)
        n_messages = 5
        # Run test.
        actual = rnsexprto.split_messages(text, n_messages)
        # Check outputs.
        self.assertEqual(len(actual), n_messages)
        self.assertEqual("".join(m["content"] for m in actual), text)


# #############################################################################
# Test_run
# #############################################################################


class Test_run(hunitest.TestCase):
    """
    Test `run()`.
    """

    def test1(self) -> None:
        """
        Test that every cell is called once and billed as the provider says.
        """
        # Prepare inputs.
        text_types = ["prose"]
        n_chars = [400, 1000]
        config = _get_config(self, text_types, n_chars)
        provider = _ArithmeticProvider()
        cells = rnsexprto.build_plan(config)
        # Run test.
        actual = asyncio.run(rnsexprto.run(cells, provider, config))
        # Check outputs.
        self.assertEqual(len(provider.calls), len(cells))
        for cell, m in zip(cells, actual):
            expected = (
                math.ceil(cell.n_chars / 4) + 12 + 4 * (cell.n_messages - 1)
            )
            self.assertEqual(m.prompt_tokens, expected)
            self.assertEqual(m.status, rnogapro.OK)


# #############################################################################
# Test_save_measurements
# #############################################################################


class Test_save_measurements(hunitest.TestCase):
    """
    Test `save_measurements()` with `load_measurements()`.
    """

    def test1(self) -> None:
        """
        Test that saved calls load back unchanged.
        """
        # Prepare inputs.
        text_types = ["code"]
        n_chars = [400]
        config = _get_config(self, text_types, n_chars)
        cells = rnsexprto.build_plan(config)
        expected = asyncio.run(
            rnsexprto.run(cells, _ArithmeticProvider(), config)
        )
        path = os.path.join(self.get_scratch_space(), "raw.jsonl")
        # Run test.
        rnsexprto.save_measurements(expected, path)
        actual = rnsexprto.load_measurements(path)
        # Check outputs.
        self.assertEqual(actual, expected)


# #############################################################################
# Test_build_plan
# #############################################################################


class Test_build_plan(hunitest.TestCase):
    """
    Test `build_plan()` against the committed raw calls.
    """

    def test1(self) -> None:
        """
        Test that the committed calls are exactly the cells of `CONFIG`.
        """
        # Prepare inputs.
        cells = rnsexprto.build_plan(rnsexprto.CONFIG)
        # Prepare outputs.
        expected = [
            (c.seller_id, c.text_type, c.n_chars, c.n_messages) for c in cells
        ]
        # Run test.
        measurements = rnsexprto.load_measurements(rnsexprto.RAW_PATH)
        actual = [
            (m.seller_id, m.text_type, m.n_chars, m.n_messages)
            for m in measurements
        ]
        # Check outputs.
        self.assertEqual(actual, expected)


# #############################################################################
# Test_estimate_max_cost_usd
# #############################################################################


class Test_estimate_max_cost_usd(hunitest.TestCase):
    """
    Test `estimate_max_cost_usd()`.
    """

    def test1(self) -> None:
        """
        Test the bound at the highest prices of the two sellers in the
        catalog sample: groq, 5.9e-7 prompt and 7.9e-7 completion.
        """
        # Prepare inputs.
        text_types = ["prose"]
        n_chars = [1000]
        config = _get_config(self, text_types, n_chars)
        cells = rnsexprto.build_plan(config)
        # Prepare outputs.
        # 4 cells x (1000 chars + 50 framing) prompt tokens, 4 x 1 completion.
        expected = 4 * 1050 * 5.9e-7 + 4 * 1 * 7.9e-7
        # Run test.
        actual = rnsexprto.estimate_max_cost_usd(cells, config)
        # Check outputs.
        self.assertAlmostEqual(actual, expected, places=12)
