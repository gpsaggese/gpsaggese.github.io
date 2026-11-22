import importlib.util
import os
from pathlib import Path
import threading

import pytest

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m2_validation", ROOT / "scripts/run_validation_stage.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)

from m2_runtime.budget import Budget, NANO  # noqa: E402
from minisweagent.exceptions import FormatError  # noqa: E402


def model():
    config = runtime.agent.load_config("chess_haiku_think.yaml")
    config["model"].pop("budget")
    return runtime.agent.make_model(config["model"])


def test_reservation_precedes_wire_call_and_format_error_is_charged(tmp_path):
    b = Budget(tmp_path / "b.sqlite", 3*NANO)
    def fake(**kwargs):
        assert b.summary()["accounted_plus_reserved_usd"] == .665
        assert kwargs["num_retries"] == kwargs["max_retries"] == 0
        assert kwargs["api_base"] == "https://api.anthropic.com"
        assert kwargs["max_tokens"] == 16000 and kwargs["service_tier"] == "standard"
        return runtime.litellm.ModelResponse(
            model="claude-haiku-5-5",
            choices=[{"index":0,"message":{"role":"assistant","content":"invalid format"},"finish_reason":"stop"}],
            usage={"prompt_tokens":1000,"completion_tokens":100,"total_tokens":1100},
        )
    query = runtime.make_guarded_query(b, threading.Event(), completion=fake)
    with pytest.raises(FormatError) as exc:
        query(model(), [{"role":"user","content":"test"}])
    assert exc.value.messages[0]["extra"]["cost"] == .00015
    assert b.summary()["requests"] == 1


def test_transport_failure_halts_and_retains_full_liability(tmp_path):
    b = Budget(tmp_path / "b.sqlite", 3*NANO)
    stop = threading.Event()
    calls = []
    def fail(**kwargs):
        calls.append(1)
        raise TimeoutError("not retained in output")
    query = runtime.make_guarded_query(b, stop, completion=fail)
    with pytest.raises(RuntimeError):
        query(model(), [{"role":"user","content":"test"}])
    assert stop.is_set() and b.summary()["accounted_plus_reserved_usd"] == .665
    with pytest.raises(RuntimeError):
        query(model(), [{"role":"user","content":"test"}])
    assert len(calls) == 1


def test_policy_change_refused_before_any_wire_call(tmp_path):
    b = Budget(tmp_path / "b.sqlite", 3*NANO)
    m = model();m.config.model_kwargs["max_tokens"] = 100
    query = runtime.make_guarded_query(b, threading.Event(), completion=lambda **kw: pytest.fail("wire called"))
    with pytest.raises(RuntimeError):
        query(m, [{"role":"user","content":"test"}])
    assert b.summary()["requests"] == 0
