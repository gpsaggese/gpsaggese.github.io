"""
Test PostgreSQL persistence for checker state and final evidence.
"""

import datetime

import helpers.hunit_test as hunitest

import research.Noesis.checker_evidence as rnochevi
import research.Noesis.checker_models as rnochmod
import research.Noesis.checker_store as rnochsto
import research.Noesis.checker_verdicts as rnochver
import research.Noesis.test.gateway_test_utils as rnteteut


async def _setup(pool) -> int:
    await rnteteut.add_account(pool, "buyer_1", "buyer", "buyer-key")
    await rnteteut.add_seller(pool, "groq", "groq", "Groq")
    contract_id = await rnteteut.add_contract(pool, "buyer_1", "groq")
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE contracts SET window_start = now() WHERE contract_id = %s",
            (contract_id,),
        )
        await conn.execute(
            "INSERT INTO canaries (canary_id, prompt, answer, normalizer,"
            " bank_version, family, status) VALUES"
            " ('math-v1-001', 'What is 1+1?', '2', 'integer',"
            " 'v1', 'math', 'ACTIVE')"
        )
    return contract_id


async def _add_request(
    pool,
    contract_id: int,
    is_canary: bool = False,
    canary_correct=None,
) -> int:
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO requests (contract_id, seller_id, pinned_provider,"
            " model_requested, is_canary, canary_id, canary_correct, prompt,"
            " latency_ms, status) VALUES (%s, 'groq', 'Groq', 'test-model',"
            " %s, %s, %s, '[]', 100, 'ok') RETURNING request_id",
            (
                contract_id,
                is_canary,
                "math-v1-001" if is_canary else None,
                canary_correct,
            ),
        )
        (request_id,) = await cursor.fetchone()
    return request_id


# #############################################################################
# Test_claim_probe_slot
# #############################################################################


class Test_claim_probe_slot(hunitest.TestCase):
    """
    Test migration-backed loading and guarded probe transitions.
    """

    def test1(self) -> None:
        """
        Test contract/canary loading and unique guarded probe transitions.
        """

        async def _body(pool) -> None:
            contract_id = await _setup(pool)
            contracts = await rnochsto.list_active_contracts(pool)
            self.assertEqual(len(contracts), 1)
            self.assertEqual(contracts[0].terms.contract_id, contract_id)
            self.assertEqual(contracts[0].target_canaries, 10)
            self.assertEqual(contracts[0].maximum_seconds, 300)
            canaries = await rnochsto.load_active_canaries(pool, "v1")
            self.assertEqual([item.canary_id for item in canaries], ["math-v1-001"])
            probe_id = await rnochsto.claim_probe_slot(
                pool, contract_id, 1, "math-v1-001"
            )
            duplicate = await rnochsto.claim_probe_slot(
                pool, contract_id, 1, "math-v1-001"
            )
            self.assertIsNotNone(probe_id)
            self.assertIsNone(duplicate)
            self.assertTrue(await rnochsto.mark_probe_running(pool, probe_id))
            self.assertFalse(await rnochsto.mark_probe_running(pool, probe_id))
            request_id = await _add_request(pool, contract_id, True, True)
            self.assertTrue(
                await rnochsto.complete_probe(pool, probe_id, request_id, "m2-v1")
            )

        rnteteut.run_with_clean_db(self, _body)


# #############################################################################
# Test_finalize_contract
# #############################################################################


class Test_finalize_contract(hunitest.TestCase):
    """
    Test atomic finalization and frozen request-level evidence.
    """

    def test1(self) -> None:
        """
        Test finalization freezes evidence and is idempotent.
        """

        async def _body(pool) -> None:
            contract_id = await _setup(pool)
            for _ in range(20):
                await _add_request(pool, contract_id)
            for _ in range(8):
                await _add_request(pool, contract_id, True, True)
            observations = await rnochsto.list_request_observations(pool, contract_id)
            terms = rnochmod.ContractTerms(contract_id, 2.0, 0.9, 0.85)
            policy = rnochmod.WindowPolicy()
            evidence = rnochevi.build_evidence(observations, terms, policy)
            decision = rnochver.decide_verdict(evidence, terms, policy)
            ended_at = datetime.datetime.now(datetime.timezone.utc)
            async with pool.connection() as conn:
                await conn.execute(
                    "UPDATE contracts SET state = 'CLOSED' WHERE contract_id = %s",
                    (contract_id,),
                )
            kwargs = {
                "checker_version": "m2-v1",
                "bank_version": "v1",
                "window_end": ended_at,
            }
            self.assertTrue(
                await rnochsto.finalize_contract(
                    pool, contract_id, decision, evidence, **kwargs
                )
            )
            self.assertFalse(
                await rnochsto.finalize_contract(
                    pool, contract_id, decision, evidence, **kwargs
                )
            )
            rows = await rnteteut.fetch_all(
                pool,
                "SELECT verdict, n, latency_successes, n_canary,"
                " quality_successes, checker_disposition"
                " FROM contract_verdicts WHERE contract_id = %s",
                contract_id,
            )
            self.assertEqual(rows[0], ("PASSED", 20, 20, 8, 8, "VALID"))
            rows = await rnteteut.fetch_all(
                pool,
                "SELECT count(*) FROM verdict_evidence WHERE contract_id = %s",
                contract_id,
            )
            self.assertEqual(rows[0][0], 28)

        rnteteut.run_with_clean_db(self, _body)
