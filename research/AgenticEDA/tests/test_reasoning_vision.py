"""Test that interpret_output passes images to the LLM."""
import unittest
from unittest.mock import patch, MagicMock


class TestInterpretOutputVision(unittest.TestCase):
    @patch("src.config.config.get_chat_model")
    def test_images_passed_to_llm(self, mock_get_model):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Found bimodal distribution"
        mock_llm.invoke.return_value = mock_response
        mock_get_model.return_value = mock_llm

        from src.agent.reasoning import interpret_output
        result = interpret_output("some stats", "distributions", images=["base64img"])

        call_args = mock_llm.invoke.call_args[0][0]
        human_msg = call_args[1]
        assert isinstance(human_msg.content, list), f"Expected list, got {type(human_msg.content)}"
        assert len(human_msg.content) == 2
        assert human_msg.content[0]["type"] == "text"
        assert human_msg.content[1]["type"] == "image_url"
        assert "base64img" in human_msg.content[1]["image_url"]["url"]

    @patch("src.config.config.get_chat_model")
    def test_no_images_still_uses_content_parts(self, mock_get_model):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Found 12% missing"
        mock_llm.invoke.return_value = mock_response
        mock_get_model.return_value = mock_llm

        from src.agent.reasoning import interpret_output
        result = interpret_output("Missing: 12%", "check_missing", images=None)

        call_args = mock_llm.invoke.call_args[0][0]
        human_msg = call_args[1]
        assert isinstance(human_msg.content, list)
        assert len(human_msg.content) == 1
        assert result == "Found 12% missing"

if __name__ == "__main__":
    unittest.main()
