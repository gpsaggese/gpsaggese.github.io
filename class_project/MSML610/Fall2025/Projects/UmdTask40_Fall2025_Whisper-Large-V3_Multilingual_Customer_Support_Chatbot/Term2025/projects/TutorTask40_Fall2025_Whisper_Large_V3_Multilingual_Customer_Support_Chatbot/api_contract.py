# api_contract.py
"""
Stable API contract for the Multilingual Customer Support Chatbot.

This file defines:
- Dataclasses for inputs/outputs
- Protocol interfaces for the main services

No model / Whisper / transformer logic should live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol


# -------------------------------------------------------------------
# Basic type alias
# -------------------------------------------------------------------

LanguageCode = str  # e.g. "en", "es", "fr", "de"


# -------------------------------------------------------------------
# Core data models (inputs / outputs)
# -------------------------------------------------------------------

@dataclass
class ChatRequest:
    """
    One user query into the multilingual support system.

    Either `text` OR `audio_path` must be provided.

    - user_id: logical ID of the user (email, UUID, etc.)
    - text: text typed by the user (any language)
    - audio_path: path to an audio file to be transcribed
    - language_hint: optional ISO language code if the caller already knows it
    - metadata: free-form key/value, e.g. channel="web", app_version="1.2.0"
    """
    user_id: str
    text: Optional[str] = None
    audio_path: Optional[Path] = None
    language_hint: Optional[LanguageCode] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class TranscriptionResult:
    """
    Output of the speech-to-text engine.
    """
    text: str
    language: LanguageCode
    confidence: float  # 0.0–1.0 estimated confidence


@dataclass
class IntentResult:
    """
    Output of the intent classifier.

    - label: top predicted label (e.g. "ORDER_STATUS", "REFUND_REQUEST")
    - score: confidence of the top label
    - candidate_labels: optional ordered list of candidate labels
    """
    label: str
    score: float
    candidate_labels: List[str]


@dataclass
class ChatResponse:
    """
    Final response returned to the caller / UI.

    - reply_text: what the chatbot will show to the user
    - detected_language: language code of the user's message
    - intent: structured intent classification result (optional)
    - debug_info: any extra info for logs / debugging
    """
    reply_text: str
    detected_language: LanguageCode
    intent: Optional[IntentResult] = None
    debug_info: Optional[Dict[str, str]] = None


# -------------------------------------------------------------------
# Service interfaces (Protocols)
# -------------------------------------------------------------------

class SpeechToTextService(Protocol):
    """
    Abstract interface for speech-to-text.

    Implementations may use Whisper, Vosk, etc., but callers only rely
    on this method signature.
    """

    def transcribe(
        self,
        audio_path: Path,
        language_hint: Optional[LanguageCode] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio from `audio_path` and return text + language.

        Implementations are free to ignore `language_hint` if they do
        automatic language detection.
        """
        ...


class IntentClassifierService(Protocol):
    """
    Abstract interface for intent classification.

    Implementations may use XLM-R, DistilBERT, etc.
    """

    def classify(self, text: str) -> IntentResult:
        """
        Classify the user message into an intent label.
        """
        ...


class ResponseGeneratorService(Protocol):
    """
    Abstract interface for natural-language response generation.

    Implementations may use an LLM (GPT-style), rule-based templates,
    or any other mechanism.
    """

    def generate(
        self,
        user_text: str,
        intent: IntentResult,
        language: LanguageCode,
    ) -> str:
        """
        Generate a reply to the user, conditioned on:

        - user_text: original text
        - intent: structured intent prediction
        - language: language code for the reply
        """
        ...


class MultilingualSupportService(Protocol):
    """
    High-level facade used by UI layers (Streamlit, CLI, REST API).

    Callers do NOT need to know about Whisper, transformers, etc.
    They only construct a ChatRequest and receive a ChatResponse.
    """

    def handle(self, request: ChatRequest) -> ChatResponse:
        """
        Process a single user request (text or audio) end-to-end.

        Typical steps inside an implementation:
        1) If audio_path is provided, call SpeechToTextService.transcribe().
        2) Run IntentClassifierService.classify() on the text.
        3) Generate a reply via ResponseGeneratorService.generate().
        4) Wrap everything into ChatResponse.
        """
        ...
