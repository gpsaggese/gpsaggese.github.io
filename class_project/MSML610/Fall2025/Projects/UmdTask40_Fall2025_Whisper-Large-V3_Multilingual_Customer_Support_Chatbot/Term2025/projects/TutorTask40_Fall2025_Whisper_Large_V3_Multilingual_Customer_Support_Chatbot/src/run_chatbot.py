# src/test_chatbot.py

import json
from app.main_pipeline import MultilingualChatbot

# ==== CONFIG ====
# Put a test file at src/data/test_audio.mp3 (or .wav)
AUDIO_PATH = "/app/src/data/Recording (4).mp3"

# Optional: force language ("de", "ca", "fr", "en", etc.)
# Set to None to use Whisper's auto-detection.
FORCED_LANGUAGE = None
# FORCED_LANGUAGE = "de"
# FORCED_LANGUAGE = "ca"

# ==== RUN ====

if __name__ == "__main__":
    chatbot = MultilingualChatbot(whisper_model_name="medium")

    result = chatbot.process_audio(AUDIO_PATH, language=FORCED_LANGUAGE)

    print("\n=== FINAL RESULT (PRETTY) ===")
    print(f"Audio Path   : {result['audio_path']}")
    print(f"Language     : {result['language']} (forced={result['forced_language']})")
    print(f"Transcription: {result['transcription']}")
    print(f"Intent       : {result['intent']}")
    print(f"Response     : {result['response']}")

    print("\n=== FINAL RESULT (JSON) ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
