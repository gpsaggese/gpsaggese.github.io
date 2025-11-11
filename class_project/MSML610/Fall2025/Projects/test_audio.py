import sounddevice as sd
import numpy as np

# Audio parameters
SAMPLE_RATE = 16000  # 16 kHz
DURATION = 5         # seconds
CHANNELS = 1

print("Recording for 5 seconds... Speak now!")

# Record audio
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
sd.wait()  # Wait until recording is finished

print("Recording finished!")

# Show the shape of recorded audio
print("Audio shape:", audio.shape)

# Optional: play back the recording
print("Playing back...")
sd.play(audio, SAMPLE_RATE)
sd.wait()

print("Done!")
