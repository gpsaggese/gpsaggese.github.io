"""
Import as:

import research.Noesis.notebooks.test.test_prompt_tokens_utils as rnnttptu
"""

import logging
import math
import os
from typing import List

import helpers.hsystem as hsystem
import helpers.hunit_test as hunitest

import research.Noesis.common.noesis_settings as rnonoset
import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.notebooks.prompt_tokens_utils as rnnoprtout
import research.Noesis.scripts.experiment_prompt_tokens as rnsexprto

_LOG = logging.getLogger(__name__)


# #############################################################################
# Helpers
# #############################################################################


def _get_measurements(
    n_chars: List[int], *, failed_seller: str = ""
) -> List[rnsexprto.Measurement]:
    """
    Bill `ceil(chars / 4) + 12` tokens per request plus 4 per extra message,
    for two sellers and one or five messages.
    """
    measurements = []
    for seller_id in ["deepinfra", "groq"]:
        for n in n_chars:
            for n_messages in [1, 5]:
                tokens = math.ceil(n / 4) + 12 + 4 * (n_messages - 1)
                failed = seller_id == failed_seller
                measurements.append(
                    rnsexprto.Measurement(
                        seller_id=seller_id,
                        provider_slug=seller_id,
                        text_type="prose",
                        n_chars=n,
                        n_messages=n_messages,
                        status=rnogapro.GATEWAY_ERROR if failed else rnogapro.OK,
                        served_by=None if failed else seller_id,
                        prompt_tokens=None if failed else tokens,
                        error="no route" if failed else None,
                        cost_usd=0.0 if failed else 0.001,
                    )
                )
    return measurements


# #############################################################################
# Test_fit
# #############################################################################


class Test_fit(hunitest.TestCase):
    """
    Test `fit()` on calls billed by a known rule.
    """

    def test1(self) -> None:
        """
        Test that the fit recovers 4 chars per token, 12 tokens per request
        and 4 per extra message.
        """
        # Prepare inputs.
        n_chars = [400, 1000, 3000]
        measurements = _get_measurements(n_chars)
        # Run test.
        actual = rnnoprtout.fit(measurements)
        # Check outputs.
        self.assertEqual(
            sorted(actual), [("deepinfra", "prose"), ("groq", "prose")]
        )
        for f in actual.values():
            self.assertAlmostEqual(f.chars_per_token, 4.0, delta=0.01)
            self.assertAlmostEqual(f.request_overhead, 12.0, delta=0.5)
            self.assertAlmostEqual(f.message_overhead, 4.0, delta=0.1)
            self.assertEqual((f.n_single, f.n_split), (3, 3))

    def test2(self) -> None:
        """
        Test that a seller whose calls all fail gets no fit.
        """
        # Prepare inputs.
        n_chars = [400, 1000]
        measurements = _get_measurements(n_chars, failed_seller="deepinfra")
        # Run test.
        actual = rnnoprtout.fit(measurements)
        # Check outputs.
        self.assertEqual(sorted(actual), [("groq", "prose")])


# #############################################################################
# Test_cell1_1_summarize_calls
# #############################################################################


class Test_cell1_1_summarize_calls(hunitest.TestCase):
    """
    Test `cell1_1_summarize_calls()`.
    """

    def test1(self) -> None:
        """
        Test the counts for one failing and one answering seller.
        """
        # Prepare inputs.
        n_chars = [400, 1000]
        measurements = _get_measurements(n_chars, failed_seller="deepinfra")
        # Prepare outputs.
        expected = """
              seller  calls  succeeded             failed  spend_usd
        0  deepinfra      4          0  4 x gateway_error      0.000
        1       groq      4          4                         0.004
        """
        # Run test.
        actual = rnnoprtout.cell1_1_summarize_calls(measurements).to_string()
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell1_2_fit_table
# #############################################################################


class Test_cell1_2_fit_table(hunitest.TestCase):
    """
    Test `cell1_2_fit_table()` on the committed run, freezing the result.
    """

    def test1(self) -> None:
        """
        Test that refitting the committed raw calls gives the published table.
        """
        # Prepare inputs.
        measurements = rnsexprto.load_measurements(rnsexprto.RAW_PATH)
        # Prepare outputs.
        expected = """
              seller     text  chars_per_token  request_overhead  per_message_overhead  one_message_calls  split_calls
        0  coreweave     code             2.44               8.9                   4.8                  5            5
        1  coreweave  numbers             1.75              10.5                   5.3                  5            5
        2  coreweave    prose             5.07              16.4                   5.9                  5            5
        3  deepinfra     code             2.44               8.9                   4.8                  5            5
        4  deepinfra  numbers             1.75              10.5                   5.3                  5            5
        5  deepinfra    prose             5.07              16.4                   5.9                  5            5
        6       groq     code             2.44              34.3                   5.4                  5            5
        7       groq  numbers             1.75              35.9                   5.4                  5            5
        8       groq    prose             5.10              40.3                   6.1                  4            4
        """
        # Run test.
        fits = rnnoprtout.fit(measurements)
        actual = rnnoprtout.cell1_2_fit_table(fits).to_string()
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)


# #############################################################################
# Test_cell2_1_plot_tokens_vs_chars
# #############################################################################


class Test_cell2_1_plot_tokens_vs_chars(hunitest.TestCase):
    """
    Test `cell2_1_plot_tokens_vs_chars()`.
    """

    def test1(self) -> None:
        """
        Test one panel per text type and one line per seller, with the billed
        tokens of the one-message calls.
        """
        # Prepare inputs.
        n_chars = [400, 1000]
        measurements = _get_measurements(n_chars)
        rnnoprtout.plt.switch_backend("Agg")
        # Run test.
        rnnoprtout.cell2_1_plot_tokens_vs_chars(measurements)
        # Check outputs.
        axes = rnnoprtout.plt.gcf().axes
        self.assertEqual([ax.get_title() for ax in axes], ["prose"])
        lines = axes[0].get_lines()
        self.assertEqual(
            [line.get_label() for line in lines], ["deepinfra", "groq"]
        )
        for line in lines:
            self.assertEqual(list(line.get_xdata()), [400, 1000])
            self.assertEqual(list(line.get_ydata()), [112, 262])
        rnnoprtout.plt.close("all")


# #############################################################################
# Test_notebook
# #############################################################################


class Test_notebook(hunitest.TestCase):
    """
    Smoke-test `notebooks/prompt_tokens.py`, the notebook's paired script.
    """

    def test1(self) -> None:
        """
        Test that `notebooks/prompt_tokens.py` runs top to bottom.
        """
        # Prepare inputs.
        script = os.path.join(
            rnonoset.NOESIS_DIR, "notebooks", "prompt_tokens.py"
        )
        # Run test and check outputs.
        hsystem.system(f"MPLBACKEND=Agg python {script}", suppress_output=True)
