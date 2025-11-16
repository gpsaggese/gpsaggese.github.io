# src/transcribe.py
from transformers import pipeline
import torch
from . import config  # Import variables from our config file

# --- 1. Initialize the Transcription Pipeline ---
# This is slow the first time, as it downloads the model.
# We initialize it once here so it's ready to be used.
try:
    print(f"Loading Whisper model '{config.WHISPER_MODEL}' onto {config.DEVICE}...")
    
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=config.WHISPER_MODEL,
        torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32,
        device=config.DEVICE
    )
    print("Whisper model loaded successfully.")

except Exception as e:
    print(f"Error loading Whisper model: {e}")
    transcriber = None

# --- 2. Create the Main Function ---

def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """
    Transcribes an audio file using the Whisper model.

    Args:
        audio_path (str): The file path to the audio to transcribe.
        language (str, optional): The language of the audio (e.g., 'spanish'). 
                                  If None, Whisper will auto-detect.

    Returns:
        dict: A dictionary containing 'text' and 'language'.
    """
    if transcriber is None:
        return {"error": "Transcription model is not loaded."}

    print(f"Starting transcription for: {audio_path}")

    try:
        # Define generation arguments
        generate_kwargs = {}
        if language:
            # Force the language if provided
            generate_kwargs = {"language": language, "task": "transcribe"}

        # Perform the transcription
        # The pipeline can accept a file path directly
        output = transcriber(
            audio_path,
            chunk_length_s=30,  # Process in 30-second chunks
            batch_size=8,
            generate_kwargs=generate_kwargs
        )

        # The output format from pipeline can vary, 
        # but it should have a 'text' key.
        # Let's also try to get the detected language if not provided.
        detected_language = language or output.get("language", "unknown")

        print(f"Transcription successful. Detected language: {detected_language}")
        
        return {
            "text": output["text"].strip(),
            "language": detected_language
        }

    except Exception as e:
        print(f"Error during transcription: {e}")
        return {"error": f"Transcription failed: {e}"}

# --- 3. Example Usage (for testing this file directly) ---
if __name__ == "__main__":
    # This block runs ONLY when you execute `python src/transcribe.py`
    
    # Use a test file (make sure you have this file)
    test_audio = config.AUDIO_SAMPLES_DIR / "test_es_01.wav"
    
    # Check if the test file exists
    if not test_audio.exists():
        print(f"Test file not found at: {test_audio}")
        print("Please add a .wav file to data/sample_audio/ to run this test.")
    else:
        # Test 1: Auto-detect language
        print("\n--- Testing Auto-Detection ---")
        result_auto = transcribe_audio(str(test_audio))
        print(f"Result (Auto): {result_auto}")
        
        # Test 2: Force language
        print("\n--- Testing Forced Language (Spanish) ---")
        result_forced = transcribe_audio(str(test_audio), language="spanish")
        print(f"Result (Forced): {result_forced}")