import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('remaining_runtime',ROOT/'scripts/run_remaining_m2.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)


def test_rate_limit_retry_preserves_assignment_and_uses_shared_backoff(monkeypatch,tmp_path):
    class RateLimit(Exception):status_code=429
    clock=[1000.0];requests=[]
    monkeypatch.setattr(runtime,'OUT',tmp_path)
    monkeypatch.setattr(runtime.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(runtime.STOPPED,'wait',lambda delay:clock.__setitem__(0,clock[0]+66))
    def fake(**kwargs):
        requests.append(kwargs)
        if len(requests)==1:raise RateLimit()
        return 'response'
    monkeypatch.setattr(runtime.transport.litellm,'completion',fake)
    token=runtime.transport.CURRENT_JOB.set('fixed-job')
    gate=runtime.RateGate()
    try:
        assert gate.completion(model='same')=='response'
        assert requests==[{'model':'same'},{'model':'same'}]
        assert gate.retries['fixed-job']==1 and gate.interval==2
        assert gate.rate_events[0]['http_status']==429
    finally:runtime.transport.CURRENT_JOB.reset(token)


def test_rate_limit_attempts_are_bounded(monkeypatch,tmp_path):
    import pytest
    class RateLimit(Exception):status_code=429
    clock=[1000.0];requests=[]
    monkeypatch.setattr(runtime,'OUT',tmp_path)
    monkeypatch.setattr(runtime.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(runtime.STOPPED,'wait',lambda delay:clock.__setitem__(0,clock[0]+66))
    def fake(**kwargs):requests.append(1);raise RateLimit()
    monkeypatch.setattr(runtime.transport.litellm,'completion',fake)
    gate=runtime.RateGate();token=runtime.transport.CURRENT_JOB.set('bounded-job')
    try:
        with pytest.raises(RateLimit):gate.completion(model='same')
        assert len(requests)==3
    finally:runtime.transport.CURRENT_JOB.reset(token)
