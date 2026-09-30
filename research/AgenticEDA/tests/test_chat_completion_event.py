"""Test that chat investigations push a completion event."""
import unittest
from unittest.mock import patch


class TestChatCompletionEvent(unittest.TestCase):
    @patch("backend.routers.stream.push_event")
    def test_finalize_pushes_completion_event(self, mock_push):
        # Avoid spawning kernels/processes in a unit test; just verify that the
        # completion event is emitted when an investigation is finalized.
        from src.agent.subagent import InvestigationResult
        from src.agent.hypothesis import Hypothesis
        hyp = Hypothesis(id="h1", title="Test", description="Test", priority=1)
        result = InvestigationResult(
            hypothesis_id="h1",
            hypothesis_title="Test",
            finding="Found something",
            confidence=0.8,
        )

        from backend.routers.chat import _finalize_chat_investigation
        _finalize_chat_investigation("test_session", hyp, "chat_test", result)

        pushed_events = [c[0][1] for c in mock_push.call_args_list]
        completion_events = [e for e in pushed_events if "complete" in e.get("type", "")]
        assert len(completion_events) >= 1, f"Expected completion event, got: {[e['type'] for e in pushed_events]}"
        assert completion_events[-1]["status"] == "complete"

if __name__ == "__main__":
    unittest.main()
