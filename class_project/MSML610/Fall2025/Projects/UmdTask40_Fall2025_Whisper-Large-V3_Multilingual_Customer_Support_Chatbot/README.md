
# Whisper Large V3 Multilingual Customer Support Chatbot

This repository implements the **core engine** of a multilingual customer support chatbot.  

The system is built around four main capabilities:

1. **Robust audio loading & preprocessing** (for user voice messages)  
2. **Speech transcription** using a Whisper model  
3. **Multilingual intent classification** using XLM-R  
4. **Multilingual response generation** using Qwen

All of that logic lives in **core Python modules under `src/utils/`**.  
Assignment files like `XYZ.API.md`, `XYZ.API.ipynb`, `XYZ.example.md`, `XYZ.example.ipynb`, and `XYZ_utils.py` are **just thin wrappers**: they import these modules and demonstrate how to use them. The README below focuses on the *real* implementation.

---

## Core Architecture

At a high level, the system implements this pipeline:

1. **Audio ingestion**
   - Load a user audio file (e.g. `.wav`, `.mp3`) from disk.
   - Convert to a mono waveform at 16 kHz to provide high quaity audio files

2. **Transcription (Whisper)**
   - Feed the waveform into a Whisper model.
   - Get back **text + detected language code**.
   

3. **Intent classification (XLM-R)**
   - Take the transcribed text.
   - Use an XLM-R–based zero-shot classifier to decide which **intent label** best matches the text.

4. **Response generation (Qwen)**
   - Use an LLM (Qwen2.5-1.5B-Instruct) to:
     - Generate a response **in the user’s language** for all other languages.
     - Generate a reponse in english for a better understanding acts like subtitlte
     

This pipeline is typically orchestrated in a higher-level component (e.g. `MultilingualChatbot` in `src/app/main_pipeline.py`), which wires together the utility modules described below.

---

## Core Modules

### 1. `src/utils/audio_utils.py` – Audio Loading

The `load_audio(path, max_duration_sec=60)` function is responsible for **robustly loading audio** and preparing it for Whisper. It:


- Loads audio using `soundfile` as 32-bit float  
- Resamples to **16 kHz** using `torchaudio.functional.resample` if the original sample rate differs  
