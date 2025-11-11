import sounddevice as sd
import numpy as np
import queue

audio_queue = queue.Queue()

# Audio parameters
SAMPLE_RATE = 16000  # 16 kHz for Whisper
CHUNK_SIZE = 1024     # Number of frames per chunk

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status)
    # Copy the audio data to avoid issues with memory
    audio_queue.put(indata.copy())

def start_stream():
    """Start recording audio from the default microphone."""
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        print("Recording... Press Ctrl+C to stop.")
        try:
            while True:
                # Wait for audio chunks
                audio_chunk = audio_queue.get()
                yield audio_chunk
        except KeyboardInterrupt:
            print("Stopped recording.")
