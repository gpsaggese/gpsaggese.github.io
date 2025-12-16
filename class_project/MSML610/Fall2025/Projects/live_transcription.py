import sounddevice as sd
import numpy as np
import queue

SAMPLE_RATE = 16000
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print("Sounddevice status:", status)
    audio_queue.put(indata.copy())


def start_audio_stream():
    # Start a continuous InputStream and return the stream object.
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()
    return stream


def stop_audio_stream(stream):
    # Stop and close the given stream.
    if stream is not None:
        stream.stop()
        stream.close()


def drain_audio_queue():

    chunks = []

    while True:
        try:
            data = audio_queue.get_nowait()
        except queue.Empty:
            break

        if data is None:
            continue

        arr = np.asarray(data, dtype=np.float32)

        # Flatten (frames, 1) -> (frames,)
        if arr.ndim > 1:
            arr = arr.reshape(-1)

        # Skip invalid
        if arr.ndim != 1 or arr.size == 0:
            continue

        chunks.append(arr)

    if not chunks:
        return None

    return np.concatenate(chunks, axis=0)
