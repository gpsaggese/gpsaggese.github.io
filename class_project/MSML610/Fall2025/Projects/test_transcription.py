import sounddevice as sd
import numpy as np
from transcription import load_model, transcribe_audio

# Record 5 seconds of speech
SAMPLE_RATE = 16000
DURATION = 5
print("Speak now...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()
print("Recording complete")

# Flatten to 1D array (Whisper expects this format)
audio = audio.flatten().astype(np.float32)

# Load Whisper model
model = load_model("base")

# Transcribe
print("Transcribing...")
text = transcribe_audio(model, audio, sample_rate=SAMPLE_RATE)
print("\n Transcription result:\n", text)
