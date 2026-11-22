from concurrent.futures import ProcessPoolExecutor

import pytest

from m2_runtime.budget import (
    AccountingError, Budget, BudgetBusy, BudgetExhausted,
    MAX_REQUEST_NANO, NANO, price_usage,
)


def _competing_reservation(path):
    budget = Budget(path, NANO)
    try:
        return budget.reserve("job", "hash", maximum=300_000_000)
    except (BudgetBusy, BudgetExhausted):
        return None


def test_concurrent_processes_cannot_oversubscribe(tmp_path):
    path = tmp_path / "budget.sqlite"
    Budget(path, NANO)
    with ProcessPoolExecutor(max_workers=4) as pool:
        tokens = list(pool.map(_competing_reservation, [path] * 16))
    assert sum(t is not None for t in tokens) == 3
    assert Budget(path, NANO).summary()["accounted_plus_reserved_usd"] == .9


def test_unknown_call_and_restart_retain_maximum(tmp_path):
    path = tmp_path / "budget.sqlite"
    b = Budget(path, NANO)
    token = b.reserve("job", "hash")
    assert Budget(path, NANO).summary()["available_usd"] == .335
    assert b.settle(token, error_type="Timeout") == MAX_REQUEST_NANO
    with pytest.raises(BudgetExhausted):
        b.reserve("next", "hash")


def test_success_releases_unused_reservation_and_no_double_settlement(tmp_path):
    b = Budget(tmp_path / "b.sqlite", NANO)
    token = b.reserve("job", "hash")
    assert b.settle(token, usage={"prompt_tokens": 1000, "completion_tokens": 100}) == 150_000
    assert b.summary()["accounted_plus_reserved_usd"] == .00015
    with pytest.raises(AccountingError):
        b.settle(token, rejected=True)


def test_rejected_request_releases_reservation(tmp_path):
    b = Budget(tmp_path / "b.sqlite", NANO)
    token = b.reserve("job", "hash")
    assert b.settle(token, rejected=True) == 0
    assert b.summary()["available_usd"] == 1


def test_missing_usage_or_unsupported_tier_is_not_free(tmp_path):
    b = Budget(tmp_path / "b.sqlite", 3*NANO)
    token = b.reserve("job", "hash")
    with pytest.raises(AccountingError):
        b.settle(token, usage={"prompt_tokens": 10})
    assert b.summary()["accounted_plus_reserved_usd"] == .665
    with pytest.raises(AccountingError):
        price_usage({"prompt_tokens": 1000,"completion_tokens":1,"service_tier":"priority"})


def test_cache_reasoning_long_context_and_maximum():
    assert price_usage({"prompt_tokens":1000,"completion_tokens":200,"cache_read_input_tokens":600,"cache_creation_input_tokens":300}) == 153_500
    assert price_usage({"prompt_tokens":100001,"completion_tokens":16000}) == 90_000_500
    assert price_usage({"prompt_tokens":1000000,"completion_tokens":16000,"cache_creation_input_tokens":1000000}) == MAX_REQUEST_NANO


def test_changing_approved_cap_is_rejected(tmp_path):
    path=tmp_path/"b.sqlite"
    Budget(path, NANO)
    with pytest.raises(AccountingError):
        Budget(path, 3*NANO)
