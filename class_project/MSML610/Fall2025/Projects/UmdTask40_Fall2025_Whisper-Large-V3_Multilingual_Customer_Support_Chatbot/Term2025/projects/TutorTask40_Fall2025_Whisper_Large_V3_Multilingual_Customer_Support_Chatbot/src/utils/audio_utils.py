# src/utils/audio_utils.py

import numpy as np
import soundfile as sf
import torchaudio
import torch
import os

def load_audio(path: str, max_duration_sec: int = 60):
    """
    Load audio robustly and return (waveform, sample_rate = 16000).

    Fixes:
    - Replace whisper.resample_audio (not available in pip version)
    - Use torchaudio.resample instead
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        # Load with soundfile
        waveform, sr = sf.read(path, dtype="float32")

        # Stereo -> mono
        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)

        # Convert to torch tensor for torchaudio
        waveform_t = torch.tensor(waveform)

        # Limit duration (prevents huge files breaking Whisper)
        max_samples = max_duration_sec * sr
        if len(waveform_t) > max_samples:
            waveform_t = waveform_t[:max_samples]

        # Resample to 16kHz if needed
        target_sr = 16000
        if sr != target_sr:
            waveform_t = torchaudio.functional.resample(
                waveform_t, orig_freq=sr, new_freq=target_sr
            )
            sr = target_sr

        # Back to numpy float32 for Whisper
        waveform = waveform_t.numpy().astype(np.float32)

        # Detect silence
        if np.max(np.abs(waveform)) < 1e-6:
            print("[AudioUtils] Warning: Audio is silent or too quiet.")

        return waveform, sr

    except Exception as e:
        raise RuntimeError(f"[AudioUtils] Failed to load audio {path}: {e}")
