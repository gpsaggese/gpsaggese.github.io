from faster_whisper import WhisperModel

def load_model(model_size="base"):
    print(f"Loading Whisper model ({model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("Model loaded successfully.")
    return model

def transcribe_audio(model, audio_data, sample_rate=16000):
    """
    Transcribes a short audio chunk using Whisper.
    audio_data: NumPy array (float32)
    Returns the transcribed text.
    """
    segments, _ = model.transcribe(audio_data, language="en", beam_size=5)
    text_output = " ".join([segment.text for segment in segments])
    return text_output.strip()
