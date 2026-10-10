"""
End-to-end tests for the gateway HTTP API.

The real `openai` SDK talks to the app in-process (no network), with
`FakeProvider` standing in for OpenRouter and the test Postgres behind it.
"""

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List

import pytest

# The SDK is only needed to prove OpenAI compatibility; skip where absent.
openai = pytest.importorskip("openai")

import httpx  # pylint: disable=wrong-import-position

import helpers.hunit_test as hunitest  # pylint: disable=wrong-import-position
import research.Noesis.gateway_fake_provider as rnogafapr  # pylint: disable=wrong-import-position
import research.Noesis.gateway_providers as rnogapro  # pylint: disable=wrong-import-position
import research.Noesis.noesis_settings as rnonoset  # pylint: disable=wrong-import-position
import research.Noesis.test.gateway_test_utils as rnteteut  # pylint: disable=wrong-import-position

_LOG = logging.getLogger(__name__)

_MODEL = rnonoset.DEFAULT_MODEL_ID
_8B = "meta-llama/llama-3.1-8b-instruct"
_QUESTION = [{"role": "user", "content": "What is 17 * 23?"}]
_OP = {"Authorization": f"Bearer {rnteteut.OPERATOR_KEY}"}


def _groq(**kwargs: Any) -> Dict[str, rnogafapr.FakeBehavior]:
    """
    Script Groq to answer "391".
    """
    return {
        "groq": rnogafapr.FakeBehavior("Groq", answer=lambda m: "391", **kwargs)
    }


def _buyer_headers(contract_id: Any) -> Dict[str, str]:
    """
    Return buyer auth + contract headers.
    """
    headers = {
        "Authorization": f"Bearer {rnteteut.BUYER_KEY}",
        "X-Noesis-Contract": str(contract_id),
    }
    return headers


def _sdk(gw: rnteteut.Gateway) -> Any:
    """
    Return an OpenAI SDK client pointed at the in-process gateway.
    """
    client = openai.AsyncOpenAI(
        api_key=rnteteut.BUYER_KEY,
        base_url="http://test/v1",
        http_client=httpx.AsyncClient(
            transport=gw.transport, base_url="http://test"
        ),
        max_retries=0,
    )
    return client


async def _rows(gw: rnteteut.Gateway) -> List[tuple]:
    """
    Return (status, actual_provider, mismatch, completion) of every request.
    """
    rows = await rnteteut.fetch_all(
        gw.pool,
        "SELECT status, actual_provider, attribution_mismatch, completion "
        "FROM requests ORDER BY request_id",
    )
    return rows


def _run_gateway(
    self_: hunitest.TestCase,
    behaviors: Dict[str, rnogafapr.FakeBehavior],
    body: Callable[[rnteteut.Gateway], Awaitable[None]],
) -> None:
    """
    Run `body` against a fresh gateway on a clean test DB.
    """

    async def _go(_pool) -> None:
        async with rnteteut.running_gateway(behaviors) as gw:
            await body(gw)

    rnteteut.run_with_clean_db(self_, _go)


# #############################################################################
# Test_chat_completions
# #############################################################################


class Test_chat_completions(hunitest.TestCase):
    """
    Test `POST /v1/chat/completions`.
    """

    def test1(self) -> None:
        """
        Test a round trip through the real OpenAI SDK.
        """

        async def _body(gw: rnteteut.Gateway) -> None:
            # Run test.
            raw = await _sdk(gw).chat.completions.with_raw_response.create(
                model=_MODEL,
                messages=_QUESTION,
                max_tokens=50,
                extra_headers={"X-Noesis-Contract": str(gw.contract_id)},
            )
            completion = raw.parse()
            slug, call = gw.fake.calls[-1]
            # Check outputs.
            self.assertEqual(completion.choices[0].message.content, "391")
            self.assertEqual(completion.choices[0].finish_reason, "stop")
            self.assertEqual(completion.usage.total_tokens, 15)
            self.assertEqual(raw.headers["X-Noesis-Provider"], "Groq")
            self.assertEqual(
                completion.id, f"noesis-{raw.headers['X-Noesis-Request-Id']}"
            )
            # Routed to the contract's seller with the contract's timeout.
            self.assertEqual(
                (slug, call.max_tokens, call.timeout_s), ("groq", 50, 4.0)
            )
            self.assertEqual(await _rows(gw), [("ok", "Groq", False, "391")])

        _run_gateway(self, _groq(), _body)

    def test2(self) -> None:
        """
        Test that `max_tokens` is clamped to the 256-token task cap.
        """

        async def _body(gw: rnteteut.Gateway) -> None:
            # Prepare inputs.
            cases = [(None, 256), (1000, 256), (10, 10)]
            # Run test and check outputs.
            for requested, expected in cases:
                payload: Dict[str, Any] = {"model": _MODEL, "messages": _QUESTION}
                if requested is not None:
                    payload["max_tokens"] = requested
                resp = await gw.http.post(
                    "/v1/chat/completions",
                    json=payload,
                    headers=_buyer_headers(gw.contract_id),
                )
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(gw.fake.calls[-1][1].max_tokens, expected)

        _run_gateway(self, _groq(), _body)

    def test3(self) -> None:
        """
        Test that provider failures are returned with the right HTTP status
        and still logged.
        """
        # Prepare inputs.
        cases = [
            (
                rnogafapr.FakeBehavior("Groq", latency_s=9.0),
                504,
                rnogapro.TIMEOUT,
            ),
            (
                rnogafapr.FakeBehavior("Groq", fail_rate=1.0),
                502,
                rnogapro.UPSTREAM_ERROR,
            ),
            (
                rnogafapr.FakeBehavior(
                    "Groq", fail_rate=1.0, fail_status=rnogapro.GATEWAY_ERROR
                ),
                503,
                rnogapro.GATEWAY_ERROR,
            ),
        ]
        # Run test and check outputs.
        for behavior, http_status, status in cases:

            async def _body(
                gw: rnteteut.Gateway,
                http_status: int = http_status,
                status: str = status,
            ) -> None:
                resp = await gw.http.post(
                    "/v1/chat/completions",
                    json={"model": _MODEL, "messages": _QUESTION},
                    headers=_buyer_headers(gw.contract_id),
                )
                rows = await _rows(gw)
                self.assertEqual(resp.status_code, http_status)
                self.assertEqual(resp.json()["error"]["code"], status)
                self.assertIn("X-Noesis-Request-Id", resp.headers)
                self.assertEqual([r[0] for r in rows], [status])

            _run_gateway(self, {"groq": behavior}, _body)

    def test4(self) -> None:
        """
        Test that an answer from the wrong provider is flagged in the log.
        """

        async def _body(gw: rnteteut.Gateway) -> None:
            # Run test.
            resp = await gw.http.post(
                "/v1/chat/completions",
                json={"model": _MODEL, "messages": _QUESTION},
                headers=_buyer_headers(gw.contract_id),
            )
            # Check outputs.
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers["X-Noesis-Provider"], "Together")
            self.assertEqual(await _rows(gw), [("ok", "Together", True, "391")])

        _run_gateway(self, _groq(impersonate="Together"), _body)

    def test5(self) -> None:
        """
        Test requests rejected before reaching a provider: right error, no
        provider call, nothing logged.
        """

        async def _body(gw: rnteteut.Gateway) -> None:
            # Prepare inputs.
            await rnteteut.add_account(gw.pool, "buyer_2", "buyer", "key-2")
            other = await rnteteut.add_contract(gw.pool, "buyer_2", "groq")
            done = await rnteteut.add_contract(
                gw.pool, "buyer_1", "groq", state="PASSED"
            )
            ok_body = {"model": _MODEL, "messages": _QUESTION}
            contract = str(gw.contract_id)
            cases = [
                (
                    {"X-Noesis-Contract": contract},
                    ok_body,
                    401,
                    "invalid_api_key",
                ),
                (
                    {
                        "Authorization": "Bearer nope",
                        "X-Noesis-Contract": contract,
                    },
                    ok_body,
                    401,
                    "invalid_api_key",
                ),
                (
                    {**_OP, "X-Noesis-Contract": contract},
                    ok_body,
                    403,
                    "wrong_account_kind",
                ),
                (
                    {"Authorization": f"Bearer {rnteteut.BUYER_KEY}"},
                    ok_body,
                    400,
                    "missing_contract",
                ),
                (_buyer_headers("abc"), ok_body, 400, "invalid_contract"),
                (_buyer_headers(other), ok_body, 404, "contract_not_found"),
                (_buyer_headers(done), ok_body, 409, "contract_not_active"),
                (
                    _buyer_headers(contract),
                    {**ok_body, "model": "gpt-4o"},
                    404,
                    "model_not_found",
                ),
                (
                    _buyer_headers(contract),
                    {**ok_body, "stream": True},
                    400,
                    "stream_not_supported",
                ),
                (
                    _buyer_headers(contract),
                    {"model": _MODEL, "messages": []},
                    400,
                    "invalid_request",
                ),
                (
                    _buyer_headers(contract),
                    {**ok_body, "max_tokens": 0},
                    400,
                    "invalid_request",
                ),
            ]
            # Run test and check outputs.
            for headers, payload, http_status, code in cases:
                resp = await gw.http.post(
                    "/v1/chat/completions", json=payload, headers=headers
                )
                self.assertEqual(
                    (resp.status_code, resp.json()["error"]["code"]),
                    (http_status, code),
                    msg=f"{headers} {payload} {resp.text}",
                )
            self.assertEqual(gw.fake.calls, [])
            self.assertEqual(await _rows(gw), [])

        _run_gateway(self, _groq(), _body)

    def test6(self) -> None:
        """
        Test that the prompt is logged exactly as sent.
        """

        async def _body(gw: rnteteut.Gateway) -> None:
            # Run test.
            await gw.http.post(
                "/v1/chat/completions",
                json={"model": _MODEL, "messages": _QUESTION, "temperature": 0},
                headers=_buyer_headers(gw.contract_id),
            )
            rows = await rnteteut.fetch_all(
                gw.pool, "SELECT prompt FROM requests"
            )
            # Check outputs.
            self.assertEqual(json.loads(rows[0][0]), _QUESTION)
            self.assertEqual(gw.fake.calls[-1][1].temperature, 0)

        _run_gateway(self, _groq(), _body)


# #############################################################################
# Test_models_and_health
# #############################################################################


class Test_models_and_health(hunitest.TestCase):
    """
    Test `GET /v1/models` (via the SDK) and `GET /health`.
    """

    def test1(self) -> None:
        """
        Test that the one served model is listed and health is ok.
        """

        async def _body(gw: rnteteut.Gateway) -> None:
            # Run test.
            models = await _sdk(gw).models.list()
            health = await gw.http.get("/health")
            # Check outputs.
            self.assertEqual([m.id for m in models.data], [_MODEL])
            self.assertEqual(
                (health.status_code, health.json()),
                (200, {"status": "ok", "db": "ok"}),
            )

        _run_gateway(self, _groq(), _body)


# #############################################################################
# Test_admin
# #############################################################################


class Test_admin(hunitest.TestCase):
    """
    Test the operator endpoints: dev contracts and fault switches.
    """

    def test1(self) -> None:
        """
        Test creating a dev contract and its permission/validation errors.
        """

        async def _body(gw: rnteteut.Gateway) -> None:
            # Run test.
            resp = await gw.http.post(
                "/admin/dev-contract",
                json={
                    "buyer_id": "buyer_1",
                    "seller_id": "coreweave",
                    "l_max": 1.0,
                },
                headers=_OP,
            )
            created = resp.json()
            state = await rnteteut.fetch_all(
                gw.pool,
                "SELECT state FROM contracts WHERE contract_id = %s",
                created["contract_id"],
            )
            # Check outputs.
            self.assertEqual(resp.status_code, 201)
            self.assertEqual(
                (created["seller_id"], created["l_max"], created["n_tasks"]),
                ("coreweave", 1.0, 50),
            )
            self.assertEqual(state, [("ACTIVE",)])
            cases = [
                (
                    {"Authorization": f"Bearer {rnteteut.BUYER_KEY}"},
                    {"buyer_id": "buyer_1", "seller_id": "groq"},
                    403,
                    "wrong_account_kind",
                ),
                (
                    _OP,
                    {"buyer_id": "buyer_1", "seller_id": "nope"},
                    404,
                    "seller_not_found",
                ),
                (
                    _OP,
                    {"buyer_id": "operator", "seller_id": "groq"},
                    404,
                    "buyer_not_found",
                ),
            ]
            for headers, payload, http_status, code in cases:
                resp = await gw.http.post(
                    "/admin/dev-contract", json=payload, headers=headers
                )
                self.assertEqual(
                    (resp.status_code, resp.json()["error"]["code"]),
                    (http_status, code),
                )

        _run_gateway(self, _groq(), _body)

    def test2(self) -> None:
        """
        Test setting, listing, and clearing faults, and their permissions.
        """

        async def _body(gw: rnteteut.Gateway) -> None:
            # Run test.
            put = await gw.http.put(
                "/admin/faults/groq", json={"extra_latency_s": 3}, headers=_OP
            )
            listed = await gw.http.get("/admin/faults", headers=_OP)
            deleted = await gw.http.delete("/admin/faults/groq", headers=_OP)
            after = await gw.http.get("/admin/faults", headers=_OP)
            # Check outputs.
            self.assertEqual(
                (put.json()["extra_latency_s"], put.json()["active"]), (3, True)
            )
            self.assert_equal(
                str(listed.json()),
                str(
                    {
                        "groq": {
                            "extra_latency_s": 3.0,
                            "model_override": "",
                            "slug_override": "",
                        }
                    }
                ),
            )
            self.assertEqual(
                deleted.json(), {"seller_id": "groq", "active": False}
            )
            self.assertEqual(after.json(), {})
            buyer = {"Authorization": f"Bearer {rnteteut.BUYER_KEY}"}
            cases = [
                (
                    gw.http.put(
                        "/admin/faults/groq",
                        json={"extra_latency_s": 1},
                        headers=buyer,
                    ),
                    403,
                    "wrong_account_kind",
                ),
                (gw.http.get("/admin/faults"), 401, "invalid_api_key"),
                (
                    gw.http.put(
                        "/admin/faults/nope",
                        json={"extra_latency_s": 1},
                        headers=_OP,
                    ),
                    404,
                    "seller_not_found",
                ),
                (
                    gw.http.put(
                        "/admin/faults/groq",
                        json={"slug_override": "x"},
                        headers=_OP,
                    ),
                    400,
                    "invalid_fault",
                ),
                (
                    gw.http.put(
                        "/admin/faults/groq",
                        json={"extra_latency_s": -1},
                        headers=_OP,
                    ),
                    400,
                    "invalid_request",
                ),
            ]
            for pending, http_status, code in cases:
                resp = await pending
                self.assertEqual(
                    (resp.status_code, resp.json()["error"]["code"]),
                    (http_status, code),
                )

        _run_gateway(self, _groq(), _body)

    def test3(self) -> None:
        """
        Test that a model swap is served and logged, and clearing restores it.
        """

        async def _body(gw: rnteteut.Gateway) -> None:
            # Run test.
            await gw.http.put(
                "/admin/faults/groq", json={"model_override": _8B}, headers=_OP
            )
            resp = await gw.http.post(
                "/v1/chat/completions",
                json={"model": _MODEL, "messages": _QUESTION},
                headers=_buyer_headers(gw.contract_id),
            )
            swapped_model = gw.fake.calls[-1][1].model
            row = await rnteteut.fetch_all(
                gw.pool,
                "SELECT model_requested, model_served, fault_injected,"
                " attribution_mismatch FROM requests",
            )
            await gw.http.delete("/admin/faults/groq", headers=_OP)
            await gw.http.post(
                "/v1/chat/completions",
                json={"model": _MODEL, "messages": _QUESTION},
                headers=_buyer_headers(gw.contract_id),
            )
            # Check outputs.
            self.assertEqual(resp.status_code, 200)
            # The buyer still sees the contracted model; the provider got 8B.
            self.assertEqual(resp.json()["model"], _MODEL)
            self.assertEqual(swapped_model, _8B)
            self.assertEqual(row, [(_MODEL, _8B, True, False)])
            self.assertEqual(gw.fake.calls[-1][1].model, _MODEL)

        _run_gateway(self, _groq(), _body)

    def test4(self) -> None:
        """
        Test that a fault on one seller doesn't affect another seller.
        """

        async def _body(gw: rnteteut.Gateway) -> None:
            # Run test.
            await gw.http.put(
                "/admin/faults/coreweave",
                json={"extra_latency_s": 5},
                headers=_OP,
            )
            resp = await gw.http.post(
                "/v1/chat/completions",
                json={"model": _MODEL, "messages": _QUESTION},
                headers=_buyer_headers(gw.contract_id),
            )
            # Check outputs.
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(await _rows(gw), [("ok", "Groq", False, "391")])

        _run_gateway(self, _groq(), _body)
