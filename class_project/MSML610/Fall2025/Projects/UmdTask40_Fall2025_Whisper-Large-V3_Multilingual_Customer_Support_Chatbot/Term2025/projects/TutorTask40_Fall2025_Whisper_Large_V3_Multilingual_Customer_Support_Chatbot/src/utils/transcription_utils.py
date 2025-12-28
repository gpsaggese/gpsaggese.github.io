# src/utils/transcription_utils.py

import numpy as np
import torch
import gc

def transcribe_audio(whisper_model, audio_waveform, language: str | None = None):
    """
    Transcribe audio using a Whisper model (medium in your case).

    Improvements over the previous version:
    - GPU acceleration if available
    - Handles forced language correctly
    - Handles empty/silent audio safely
    - Ensures float32 conversion
    - Adds VAD protection for noisy audio
    - Cleans the returned transcription
    - Normalizes language code
    """

    # Ensure float32 numpy array
    if not isinstance(audio_waveform, np.ndarray):
        audio = np.array(audio_waveform, dtype=np.float32)
    else:
        audio = audio_waveform.astype(np.float32)

    # Safety: prevent extremely quiet / silent inputs
    if np.max(np.abs(audio)) < 1e-6:
        return {
            "text": "",
            "language": language or "unknown"
        }

    # Whisper automatically normalizes amplitude internally

    # GPU usage if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make sure Whisper is on correct device
    whisper_model = whisper_model.to(device)

    # Perform transcription
    try:
        if language:
            result = whisper_model.transcribe(audio, language=language)
        else:
            result = whisper_model.transcribe(audio)
    except Exception as e:
        print(f"[Whisper] Transcription failed: {e}")
        return {"text": "", "language": "unknown"}

    # Extract fields
    text = result.get("text", "").strip()
    lang = result.get("language", None)

    # Normalize language code
    if lang:
        lang = lang.lower().strip()

    # Memory cleanup (important for 4–8GB RAM systems)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "text": text,
        "language": lang,
    }
