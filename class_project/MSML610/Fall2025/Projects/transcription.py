import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

_MODEL_CACHE = {}


def load_model(model_id="openai/whisper-large-v3"):
    
    # Loads OpenAI Whisper Large V3 from Hugging Face.
    # Returns an ASR pipeline.
    

    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    print(f"Loading Whisper model on CPU: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=None,     
        low_cpu_mem_usage=True,
    )

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=-1,               
        return_timestamps=False,
    )

    _MODEL_CACHE[model_id] = asr
    print("Whisper model loaded (CPU only).")
    return asr


def transcribe_audio_chunk(asr, audio_data, sample_rate=16000):
    """
    Transcribes a short audio chunk (CPU).
    """
    if audio_data is None:
        return ""

    audio_data = np.asarray(audio_data)

    if audio_data.ndim > 1:
        audio_data = audio_data.reshape(-1)

    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    result = asr({"array": audio_data, "sampling_rate": sample_rate})
    return result.get("text", "").strip()


def transcribe_audio(asr, audio_data, sample_rate=16000):
    """
    Alias for full audio transcription.
    """
    return transcribe_audio_chunk(asr, audio_data, sample_rate)
