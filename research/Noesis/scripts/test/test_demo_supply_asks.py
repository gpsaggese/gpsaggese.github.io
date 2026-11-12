"""
Import as:

import research.Noesis.scripts.test.test_demo_supply_asks as rnsttdsa
"""

import json
import logging

import helpers.hunit_test as hunitest
import research.Noesis.market.supply_adapter as rnmasuad
import research.Noesis.scripts.demo_supply_asks as rnscdesa

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_build_report
# #############################################################################


class Test_build_report(hunitest.TestCase):
    """
    Test `demo_supply_asks.build_report()`.
    """

    def test1(self) -> None:
        """
        Test the report on the checked-in catalog sample with two sellers.
        """
        # Prepare inputs.
        with open(rnscdesa._SAMPLE_CATALOG, "r") as f:
            catalog = json.load(f)
        sellers = [
            rnmasuad.SellerSupply("groq", "groq", 0.57, 1.0),
            rnmasuad.SellerSupply("deepinfra", "deepinfra/turbo", 1.04, 1.0),
        ]
        n_tasks = 60
        bid_n_tasks = 50
        bid_p_max = 0.001
        # Prepare outputs.
        expected = """
        1) OpenRouter quotes 11 endpoints for the model:
           deepinfra/turbo              prompt=1.000e-07 completion=3.200e-07 -> task=0.00018192 USD
           novita/bf16                  prompt=1.350e-07 completion=4.000e-07 -> task=0.00023740 USD
           akashml/fp8                  prompt=2.000e-07 completion=5.200e-07 -> task=0.00033312 USD
           parasail/fp8                 prompt=2.200e-07 completion=5.000e-07 -> task=0.00034800 USD
           cloudflare/fp8               prompt=2.930e-07 completion=2.253e-06 -> task=0.00086977 USD
           sambanova-turbo              prompt=4.500e-07 completion=9.000e-07 -> task=0.00068040 USD
           groq                         prompt=5.900e-07 completion=7.900e-07 -> task=0.00079224 USD
           coreweave/fp16               prompt=7.100e-07 completion=7.100e-07 -> task=0.00089176 USD
           google-vertex/us-central1    prompt=7.200e-07 completion=7.200e-07 -> task=0.00090432 USD
           google-vertex                prompt=7.200e-07 completion=7.200e-07 -> task=0.00090432 USD
           together                     prompt=1.040e-06 completion=1.040e-06 -> task=0.00130624 USD
        2) Seed sellers (2) matched by endpoint tag:
           groq       tag=groq             l_typical=0.57s r_typical=1.0  quoted
           deepinfra  tag=deepinfra/turbo  l_typical=1.04s r_typical=1.0  quoted
        3) Asks queued on the order book (2):
           groq       n_tasks=60 c_level=standard p_min=0.00079224
           deepinfra  n_tasks=60 c_level=standard p_min=0.00018192
           second refresh in the same round queued 0 asks
        4) Bid 50 tasks at p_max=0.001 -> clear_round():
           tier=standard clearing_price=0.00059096 unfilled_bid=0 unfilled_ask=70
           fill: buyer_demo <- deepinfra n_tasks=50 price=0.00059096
           pending asks after the round: 0
        """
        # Run test.
        actual = rnscdesa.build_report(
            catalog,
            sellers,
            n_tasks=n_tasks,
            bid_n_tasks=bid_n_tasks,
            bid_p_max=bid_p_max,
        )
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)
