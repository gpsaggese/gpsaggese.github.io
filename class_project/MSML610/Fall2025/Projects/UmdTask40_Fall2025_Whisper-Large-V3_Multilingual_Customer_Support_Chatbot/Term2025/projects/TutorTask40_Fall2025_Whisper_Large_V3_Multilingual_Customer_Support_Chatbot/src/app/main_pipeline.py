# src/app/main_pipeline.py

import whisper
import gc
import torch

from src.utils.audio_utils import load_audio
from src.utils.transcription_utils import transcribe_audio
from src.utils.intent_utils import IntentClassifier
from src.utils.response_utils import ResponseGenerator



class MultilingualChatbot:
    """
    End-to-end pipeline:
        Audio -> Whisper -> Text -> Intent -> Response (same language)
    """

    def __init__(self, whisper_model_name: str = "medium"):
        print("[Chatbot] Initializing Whisper model...")
        
        # Load Whisper ONCE (safe for your memory)
        self.whisper_model = whisper.load_model(whisper_model_name)

        # IntentClassifier loads XLM-R Large internally
        print("[Chatbot] Initializing Intent Classifier (XLM-R Large)...")
        self.intent_classifier = IntentClassifier()

        # We will instantiate ResponseGenerator ONLY when needed
        # to avoid RAM spikes
        print("[Chatbot] Ready.")

    def process_audio(self, audio_path: str, language: str | None = None):
        """
        Process a single audio file.
        """
        print(f"\n[Chatbot] Processing: {audio_path}")
        print(f"[Chatbot] Forced Language: {language if language else 'Auto-detect'}")

        # 1. Load audio
        waveform, sr = load_audio(audio_path)

        # 2. Transcription
        t_result = transcribe_audio(self.whisper_model, waveform, language=language)
        transcription_text = t_result["text"]
        lang_code = t_result["language"]

        print(f"[Chatbot] Transcription: {transcription_text}")
        print(f"[Chatbot] Detected Language: {lang_code}")

        # 3. Intent classification (XLM-R Large)
        print("[Chatbot] Classifying Intent...")
        intent = self.intent_classifier.predict_intent(transcription_text)
        print(f"[Chatbot] Intent: {intent}")

        # 4. Response generation (load model only here)
        print("[Chatbot] Generating Response...")
        response_generator = ResponseGenerator()
        response = response_generator.generate(transcription_text, lang_code)

        # Cleanup to keep RAM safe
        del response_generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[Chatbot] Response: {response}")

        # Build result
        result = {
            "audio_path": audio_path,
            "transcription": transcription_text,
            "language": lang_code,
            "forced_language": language,
            "intent": intent,
            "response": response,
        }

        return result
