"""
Unit tests for the CDD orchestrator's vision-feedback LOOP itself.

Everything in test_docker_all.py either avoids the LLM entirely (parsing,
metrics, renderer) or only checks orchestrator state right after __init__.
Nothing exercises process_message() actually running through multiple
iterations, so nothing catches a regression in the loop's control flow:
does it really stop when the critique accepts, does it really cap at
max_iterations, does a broken critique response get handled safely
instead of crashing or looping forever.

Since the LLM itself is nondeterministic, we can't unit test its output.
What we CAN unit test deterministically is everything the orchestrator
does in reaction to a given LLM response. So every test here stubs
cdd_llm.generate / cdd_llm.critique_image / cdd_llm.describe_and_suggest
with fixed, scripted responses, and asserts on the resulting trace,
revision_count, and call counts. No network, no API key, runs in
well under a second for the whole file.

Run with: pytest test/test_orchestrator_loop.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cdd_llm
import cdd_renderer
from cdd_orchestrator import CDDOrchestrator


VALID_DOT = "digraph { A -> B; }"
ACCEPT = {"is_acceptable": True, "issues": [], "suggested_changes": ""}
REJECT = {"is_acceptable": False, "issues": ["overlap"], "suggested_changes": "spread out"}
NO_DESCRIBE = {"description": "", "suggestions": []}


def _counting(fn):
    """Wrap fn so we can assert how many times it was called."""
    calls = []

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return fn(*args, **kwargs)

    wrapped.calls = calls
    return wrapped


class TestLoopStopsWhenAccepted:
    def test_stops_after_one_iteration_when_critique_accepts(self, monkeypatch):
        gen = _counting(lambda *a, **k: VALID_DOT)
        crit = _counting(lambda *a, **k: ACCEPT)
        monkeypatch.setattr(cdd_llm, "generate", gen)
        monkeypatch.setattr(cdd_llm, "critique_image", crit)
        monkeypatch.setattr(cdd_llm, "describe_and_suggest", lambda *a, **k: NO_DESCRIBE)

        orch = CDDOrchestrator(vision_feedback=True)
        orch.process_message("a simple diagram")

        assert len(gen.calls) == 1
        assert len(crit.calls) == 1
        steps = [t["step"] for t in orch.state.last_trace]
        assert steps.count("generate") == 1
        assert steps[-2] == "stop_accepted" or "stop_accepted" in steps
        assert orch.state.revision_count == 1


class TestLoopRegeneratesOnRejection:
    def test_regenerates_once_then_stops_on_accept(self, monkeypatch):
        # First critique rejects, second accepts.
        critique_responses = [REJECT, ACCEPT]
        crit = _counting(lambda *a, **k: critique_responses.pop(0))
        gen = _counting(lambda *a, **k: VALID_DOT)
        monkeypatch.setattr(cdd_llm, "generate", gen)
        monkeypatch.setattr(cdd_llm, "critique_image", crit)
        monkeypatch.setattr(cdd_llm, "describe_and_suggest", lambda *a, **k: NO_DESCRIBE)

        orch = CDDOrchestrator(vision_feedback=True)
        orch.process_message("a diagram that needs one fix")

        # Two full passes through generate, since the first was rejected.
        assert len(gen.calls) == 2
        assert len(crit.calls) == 2
        steps = [t["step"] for t in orch.state.last_trace]
        assert steps.count("generate") == 2
        assert steps.count("vision_critique") == 2
        assert "stop_accepted" in steps
        # process_message is one turn regardless of internal iterations.
        assert orch.state.revision_count == 1


class TestLoopCapsAtMaxIterations:
    def test_never_exceeds_max_iterations_even_if_always_rejected(self, monkeypatch):
        gen = _counting(lambda *a, **k: VALID_DOT)
        # Critique always rejects, loop must not run forever.
        crit = _counting(lambda *a, **k: REJECT)
        monkeypatch.setattr(cdd_llm, "generate", gen)
        monkeypatch.setattr(cdd_llm, "critique_image", crit)
        monkeypatch.setattr(cdd_llm, "describe_and_suggest", lambda *a, **k: NO_DESCRIBE)

        orch = CDDOrchestrator(vision_feedback=True, max_iterations=3)
        orch.process_message("a diagram that never satisfies the critic")

        assert len(gen.calls) == 3  # exactly the cap, never a 4th
        steps = [t["step"] for t in orch.state.last_trace]
        assert steps.count("generate") == 3
        assert "stop_max_iterations" in steps
        assert "stop_accepted" not in steps

    def test_custom_lower_cap_is_respected(self, monkeypatch):
        gen = _counting(lambda *a, **k: VALID_DOT)
        crit = _counting(lambda *a, **k: REJECT)
        monkeypatch.setattr(cdd_llm, "generate", gen)
        monkeypatch.setattr(cdd_llm, "critique_image", crit)
        monkeypatch.setattr(cdd_llm, "describe_and_suggest", lambda *a, **k: NO_DESCRIBE)

        orch = CDDOrchestrator(vision_feedback=True, max_iterations=1)
        orch.process_message("test")

        assert len(gen.calls) == 1
        steps = [t["step"] for t in orch.state.last_trace]
        assert "stop_max_iterations" in steps


class TestVisionFeedbackDisabled:
    def test_single_shot_never_calls_critique(self, monkeypatch):
        gen = _counting(lambda *a, **k: VALID_DOT)
        crit = _counting(lambda *a, **k: ACCEPT)
        monkeypatch.setattr(cdd_llm, "generate", gen)
        monkeypatch.setattr(cdd_llm, "critique_image", crit)
        monkeypatch.setattr(cdd_llm, "describe_and_suggest", lambda *a, **k: NO_DESCRIBE)

        orch = CDDOrchestrator(vision_feedback=False)
        orch.process_message("a diagram, single shot")

        assert len(gen.calls) == 1
        assert len(crit.calls) == 0  # critique should never fire
        steps = [t["step"] for t in orch.state.last_trace]
        assert "stop_vision_disabled" in steps
        assert "vision_critique" not in steps


class TestSafeCritiqueFallback:
    def test_critique_exception_is_swallowed_and_loop_stops(self, monkeypatch):
        """If the critique call itself throws (network, quota, bad JSON the
        parser couldn't save), the orchestrator must not crash or loop
        forever, it should accept the current diagram and stop.
        """
        def boom(*a, **k):
            raise RuntimeError("simulated network failure")

        gen = _counting(lambda *a, **k: VALID_DOT)
        monkeypatch.setattr(cdd_llm, "generate", gen)
        monkeypatch.setattr(cdd_llm, "critique_image", boom)
        monkeypatch.setattr(cdd_llm, "describe_and_suggest", lambda *a, **k: NO_DESCRIBE)

        orch = CDDOrchestrator(vision_feedback=True)
        # Must not raise.
        diagram_source, image_bytes = orch.process_message("test")

        assert diagram_source == VALID_DOT
        assert len(gen.calls) == 1  # stopped after first iteration, not retried
        steps = [t["step"] for t in orch.state.last_trace]
        assert "stop_accepted" in steps  # _safe_critique's safe default


class TestSyntaxRepairPass:
    def test_invalid_syntax_triggers_repair_then_succeeds(self, monkeypatch):
        # First generate call returns bad code, the repair call returns good code.
        responses = [
            "digraph { this is not valid syntax }}",  # initial generate
            VALID_DOT,                                  # repair pass
        ]
        gen = _counting(lambda *a, **k: responses.pop(0))
        crit = _counting(lambda *a, **k: ACCEPT)
        monkeypatch.setattr(cdd_llm, "generate", gen)
        monkeypatch.setattr(cdd_llm, "critique_image", crit)
        monkeypatch.setattr(cdd_llm, "describe_and_suggest", lambda *a, **k: NO_DESCRIBE)

        orch = CDDOrchestrator(vision_feedback=True, format="graphviz")
        diagram_source, _ = orch.process_message("test")

        assert diagram_source == VALID_DOT
        assert len(gen.calls) == 2  # original attempt + repair
        steps = [t["step"] for t in orch.state.last_trace]
        assert "syntax_error" in steps


class TestRenderFailureRetries:
    def test_render_failure_moves_to_next_iteration(self, monkeypatch):
        render_calls = {"n": 0}

        def flaky_render(source, format):
            render_calls["n"] += 1
            if render_calls["n"] == 1:
                raise RuntimeError("simulated render failure")
            return b"fake-png-bytes-0123456789"

        gen = _counting(lambda *a, **k: VALID_DOT)
        crit = _counting(lambda *a, **k: ACCEPT)
        monkeypatch.setattr(cdd_llm, "generate", gen)
        monkeypatch.setattr(cdd_llm, "critique_image", crit)
        monkeypatch.setattr(cdd_llm, "describe_and_suggest", lambda *a, **k: NO_DESCRIBE)
        monkeypatch.setattr(cdd_renderer, "validate", lambda source, format: (True, ""))
        monkeypatch.setattr(cdd_renderer, "render", flaky_render)

        orch = CDDOrchestrator(vision_feedback=True, max_iterations=3)
        diagram_source, image_bytes = orch.process_message("test")

        assert image_bytes == b"fake-png-bytes-0123456789"
        assert len(gen.calls) == 2  # first iteration failed to render, second succeeded
        steps = [t["step"] for t in orch.state.last_trace]
        assert "render_failed" in steps
        assert "rendered" in steps


class TestDescribeFailureIsNonFatal:
    def test_describe_exception_leaves_empty_fields_not_a_crash(self, monkeypatch):
        gen = _counting(lambda *a, **k: VALID_DOT)
        crit = _counting(lambda *a, **k: ACCEPT)

        def boom(*a, **k):
            raise RuntimeError("simulated describe failure")

        monkeypatch.setattr(cdd_llm, "generate", gen)
        monkeypatch.setattr(cdd_llm, "critique_image", crit)
        monkeypatch.setattr(cdd_llm, "describe_and_suggest", boom)

        orch = CDDOrchestrator(vision_feedback=True)
        orch.process_message("test")  # must not raise

        assert orch.state.last_description == ""
        assert orch.state.last_suggestions == []


class TestStateCommitOnEachTurn:
    def test_conversation_history_and_revision_count_update(self, monkeypatch):
        gen = _counting(lambda *a, **k: VALID_DOT)
        crit = _counting(lambda *a, **k: ACCEPT)
        monkeypatch.setattr(cdd_llm, "generate", gen)
        monkeypatch.setattr(cdd_llm, "critique_image", crit)
        monkeypatch.setattr(cdd_llm, "describe_and_suggest", lambda *a, **k: NO_DESCRIBE)

        orch = CDDOrchestrator(vision_feedback=True)
        orch.process_message("first request")
        assert orch.state.revision_count == 1
        assert len(orch.state.conversation_history) == 2  # user + assistant

        orch.process_message("second request, a follow-up edit")
        assert orch.state.revision_count == 2
        assert len(orch.state.conversation_history) == 4
