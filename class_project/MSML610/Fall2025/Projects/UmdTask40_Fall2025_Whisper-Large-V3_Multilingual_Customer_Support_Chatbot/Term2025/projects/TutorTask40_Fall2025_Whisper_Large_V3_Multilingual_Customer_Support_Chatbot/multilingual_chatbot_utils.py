"""
Utility and wrapper functions that adapt the *existing* multilingual
chatbot implementation (src/app/main_pipeline.py) to the clean API
defined in api_contract.py.

This is the file that notebooks should import from.

- It implements MultilingualSupportService using your MultilingualChatbot.
- It exposes a convenience factory: create_default_service().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from api_contract import (
    ChatRequest,
    ChatResponse,
    IntentResult,
    MultilingualSupportService,
    LanguageCode,
)

# Your existing high-level pipeline
from src.app.main_pipeline import MultilingualChatbot
# We re-use ResponseGenerator for the text-only path
from src.utils.response_utils import ResponseGenerator


class MultilingualChatbotService(MultilingualSupportService):
    """
    Concrete implementation of MultilingualSupportService that wraps
    your existing MultilingualChatbot class.

    This class is what API.ipynb and example.ipynb will use.
    """

    def __init__(self) -> None:
        # Initialize your real pipeline (loads Whisper + XLM-R once)
        self._bot = MultilingualChatbot()

    # ------------------------------------------------------------------
    # Internal helper: run your underlying logic for audio or text
    # ------------------------------------------------------------------
    def _run_underlying_pipeline(self, request: ChatRequest) -> Dict[str, Any]:
        """
        Internal helper to call your existing methods and normalize
        the result into a dict with keys:
        - audio_path
        - transcription
        - language
        - forced_language
        - intent
        - response
        """
        if request.text is None and request.audio_path is None:
            raise ValueError("ChatRequest must have either text or audio_path")

        # AUDIO PATH: use your existing process_audio pipeline directly
        if request.audio_path is not None:
            audio_str = str(request.audio_path)
            result = self._bot.process_audio(
                audio_path=audio_str,
                language=request.language_hint,
            )
            return result

        # TEXT-ONLY PATH: replicate what process_audio does, but skip Whisper
        assert request.text is not None  # for type checkers
        text = request.text
        lang_code: LanguageCode = request.language_hint or "unknown"

        # 1) Intent classification using the already-loaded IntentClassifier
        intent = self._bot.intent_classifier.predict_intent(text)

        # 2) Response generation (same as in process_audio)
        response_generator = ResponseGenerator()
        response = response_generator.generate(text, lang_code)

        # NOTE: We don't need Whisper here, so no audio or transcription.
        result = {
            "audio_path": None,
            "transcription": text,       # treat text as the "transcription"
            "language": lang_code,
            "forced_language": request.language_hint,
            "intent": intent,
            "response": response,
        }
        return result

    # ------------------------------------------------------------------
    # Public API method required by MultilingualSupportService
    # ------------------------------------------------------------------
    def handle(self, request: ChatRequest) -> ChatResponse:
        """
        Implementation of the high-level API:

        - Calls the underlying pipeline (audio or text).
        - Extracts reply text, language, and intent information.
        - Wraps everything into a ChatResponse dataclass.
        """
        raw = self._run_underlying_pipeline(request)

        # ---- Extract reply text ----
        reply_text: str = raw.get("response", "")

        # ---- Extract detected language ----
        detected_language: str = raw.get("language") or request.language_hint or "unknown"

        # ---- Extract intent info ----
        # Your IntentClassifier currently returns a single value (most likely a string label),
        # not scores or candidate labels, so we treat it as a label with score 1.0.
        intent_raw = raw.get("intent")
        intent_label = str(intent_raw) if intent_raw is not None else "UNKNOWN"

        intent = IntentResult(
            label=intent_label,
            score=1.0,          # we don't have a probability, so assume 1.0
            candidate_labels=[],  # not available from current classifier
        )

        # ---- Build debug info ----
        debug_info = {
            "audio_path": raw.get("audio_path"),
            "transcription": raw.get("transcription"),
            "forced_language": raw.get("forced_language"),
        }

        return ChatResponse(
            reply_text=reply_text,
            detected_language=detected_language,
            intent=intent,
            debug_info={k: str(v) for k, v in debug_info.items()},
        )


def create_default_service() -> MultilingualSupportService:
    """
    Convenience factory used by notebooks and examples.

    Usage:

        from api_contract import ChatRequest
        from multilingual_chatbot_utils import create_default_service

        service = create_default_service()
        resp = service.handle(ChatRequest(user_id="demo", text="Hello!"))
    """
    return MultilingualChatbotService()
