import json
import logging
from typing import Optional

import helpers.hunit_test as hunitest
import research.Noesis.gateway_providers as rnogapro
import research.Noesis.gateway_request_log as rnogarelo
import research.Noesis.gateway_routing as rnogarou
import research.Noesis.test.gateway_test_utils as rnteteut

_LOG = logging.getLogger(__name__)

_MODEL = "meta-llama/llama-3.3-70b-instruct"
_MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "What is 17 * 23?"},
]
_CALL = rnogapro.ProviderCall(
    model=_MODEL, messages=_MESSAGES, max_tokens=16, timeout_s=4.0
)
_COLUMNS = (
    "contract_id, seller_id, pinned_provider, actual_provider,"
    " attribution_mismatch, model_requested, model_served, is_canary, canary_id,"
    " canary_correct, prompt, completion, prompt_tokens, completion_tokens,"
    " latency_ms, status, cost_usd, fault_injected"
)


def _route(contract_id: int) -> rnogarou.Route:
    """
    Return a route to Groq under `contract_id`.
    """
    route = rnogarou.Route(
        buyer_id="buyer_1",
        contract_id=contract_id,
        seller_id="groq",
        provider_slug="groq",
        provider_name="Groq",
        l_max=2.0,
    )
    return route


def _ok(served_by: Optional[str]) -> rnogapro.ProviderResult:
    """
    Return a successful result served by `served_by`.
    """
    result = rnogapro.ProviderResult(
        status=rnogapro.OK,
        text="391",
        served_by=served_by,
        model_served=_MODEL,
        prompt_tokens=20,
        completion_tokens=2,
        cost_usd=0.0000305,
        latency_s=0.29,
        http_status=200,
        error=None,
    )
    return result


def _failed(status: str) -> rnogapro.ProviderResult:
    """
    Return a failed result with `status`.
    """
    result = rnogapro.ProviderResult(
        status=status,
        text=None,
        served_by=None,
        model_served=None,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=0.0,
        latency_s=4.0,
        http_status=None,
        error="x",
    )
    return result


async def _setup(pool) -> rnogarou.Route:
    """
    Create a buyer, Groq, a canary, and an ACTIVE contract.
    """
    await rnteteut.add_account(pool, "buyer_1", "buyer", "k")
    await rnteteut.add_seller(pool, "groq", "groq", "Groq")
    contract_id = await rnteteut.add_contract(pool, "buyer_1", "groq")
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO canaries (canary_id, prompt, answer) "
            "VALUES ('c-17x23', 'q', '391')"
        )
    return _route(contract_id)


async def _row(pool, request_id: int) -> dict:
    """
    Return the `requests` row as a dict.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"SELECT {_COLUMNS} FROM requests WHERE request_id = %s",
            (request_id,),
        )
        values = await cursor.fetchone()
        names = [d.name for d in cursor.description]
    return dict(zip(names, values))


# #############################################################################
# Test_is_attribution_mismatch
# #############################################################################


class Test_is_attribution_mismatch(hunitest.TestCase):
    """
    Test when an answer can't be confirmed as the pinned provider's.
    """

    def test1(self) -> None:
        """
        Test the pinned provider, a different one, no name, and failures.
        """
        # Prepare inputs.
        cases = [
            (_ok("Groq"), False),
            (_ok("Together"), True),
            (_ok(None), True),
            (_failed(rnogapro.TIMEOUT), False),
            (_failed(rnogapro.UPSTREAM_ERROR), False),
        ]
        # Run test and check outputs.
        for result, expected in cases:
            actual = rnogarelo.is_attribution_mismatch(_route(1), result)
            self.assertEqual(actual, expected)


# #############################################################################
# Test_log_request
# #############################################################################


class Test_log_request(hunitest.TestCase):
    """
    Test that every provider call becomes one `requests` row.
    """

    def test1(self) -> None:
        """
        Test every column of a successful call.
        """

        async def _body(pool) -> None:
            # Prepare inputs.
            route = await _setup(pool)
            # Prepare outputs.
            expected = {
                "contract_id": route.contract_id,
                "seller_id": "groq",
                "pinned_provider": "Groq",
                "actual_provider": "Groq",
                "attribution_mismatch": False,
                "model_requested": _MODEL,
                "model_served": _MODEL,
                "is_canary": False,
                "canary_id": None,
                "canary_correct": None,
                "prompt": json.dumps(_MESSAGES),
                "completion": "391",
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "latency_ms": 290.0,
                "status": "ok",
                "cost_usd": 0.0000305,
                "fault_injected": False,
            }
            # Run test.
            request_id = await rnogarelo.log_request(
                pool, route, _CALL, _ok("Groq")
            )
            actual = await _row(pool, request_id)
            # Check outputs.
            actual["latency_ms"] = round(actual["latency_ms"], 6)
            self.assert_equal(str(actual), str(expected))

        rnteteut.run_with_clean_db(self, _body)

    def test2(self) -> None:
        """
        Test that each failure status is still logged, with no answer and $0.
        """

        async def _body(pool) -> None:
            # Prepare inputs.
            route = await _setup(pool)
            statuses = [
                rnogapro.TIMEOUT,
                rnogapro.UPSTREAM_ERROR,
                rnogapro.GATEWAY_ERROR,
                rnogapro.BUYER_ERROR,
            ]
            # Run test and check outputs.
            for status in statuses:
                request_id = await rnogarelo.log_request(
                    pool, route, _CALL, _failed(status)
                )
                row = await _row(pool, request_id)
                self.assertEqual(
                    (row["status"], row["completion"], row["cost_usd"]),
                    (status, None, 0.0),
                )
                self.assertFalse(row["attribution_mismatch"])

        rnteteut.run_with_clean_db(self, _body)

    def test3(self) -> None:
        """
        Test that answers from another or an unnamed provider are flagged.
        """

        async def _body(pool) -> None:
            # Prepare inputs.
            route = await _setup(pool)
            # Run test.
            wrong = await _row(
                pool,
                await rnogarelo.log_request(pool, route, _CALL, _ok("Together")),
            )
            unknown = await _row(
                pool, await rnogarelo.log_request(pool, route, _CALL, _ok(None))
            )
            # Check outputs.
            self.assertEqual(
                (wrong["pinned_provider"], wrong["actual_provider"]),
                ("Groq", "Together"),
            )
            self.assertTrue(wrong["attribution_mismatch"])
            self.assertTrue(unknown["attribution_mismatch"])

        rnteteut.run_with_clean_db(self, _body)

    def test4(self) -> None:
        """
        Test that canary and fault fields are stored as given.
        """

        async def _body(pool) -> None:
            # Prepare inputs.
            route = await _setup(pool)
            # Run test.
            request_id = await rnogarelo.log_request(
                pool,
                route,
                _CALL,
                _ok("Groq"),
                is_canary=True,
                canary_id="c-17x23",
                canary_correct=True,
                fault_injected=True,
            )
            row = await _row(pool, request_id)
            # Check outputs.
            self.assertEqual(
                (
                    row["is_canary"],
                    row["canary_id"],
                    row["canary_correct"],
                    row["fault_injected"],
                ),
                (True, "c-17x23", True, True),
            )

        rnteteut.run_with_clean_db(self, _body)
