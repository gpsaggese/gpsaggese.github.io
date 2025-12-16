import streamlit as st
import numpy as np

from transcription import load_model, transcribe_audio_chunk
from live_transcription import (
    start_audio_stream,
    stop_audio_stream,
    drain_audio_queue,
    SAMPLE_RATE,
)

from summarizer import summarize_text
from storage import (
    get_next_session_id,
    create_session_folder,
    save_transcript,
    save_summary,
)

CHUNK_DURATION = 10
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.02


# Page Config
st.set_page_config(page_title="Live Lecture Transcription", layout="wide")


# CSS
st.markdown(
    """
    <style>
    section.main > div.block-container {
        padding-top: 1.25rem !important;  /* ensures header not clipped */
        padding-bottom: 2rem !important;
    }

    .stApp {
        background-color: #020617;
    }

    h1, h2, h3, h4, p, span, div {
        color: #e5e7eb !important;
    }

    /* ---- HEADER ---- */
    .hero {
        padding-top: 1.5rem;        /* key: pushes header down */
        padding-bottom: 0.75rem;
        text-align: center;
    }

    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.2;
        margin: 0;
        color: #ffffff !important;
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
    }

    .app-subtitle {
        margin-top: 0.55rem;
        font-size: 1.05rem;
        color: #9ca3af !important;
    }

    /* ---- STATUS PILL ---- */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.3rem 0 1rem 0;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(2, 6, 23, 0.35);
        backdrop-filter: blur(10px);
    }
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
    }
    .status-recording .status-dot {
        background: #ef4444;
        box-shadow: 0 0 12px rgba(248, 113, 113, 0.9);
    }
    .status-idle .status-dot {
        background: #94a3b8;
        box-shadow: none;
    }

    /* ---- BUTTONS ---- */
    .stButton>button {
        border-radius: 999px;
        padding: 0.8rem 1.5rem;
        font-weight: 700;
        border: 0;
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.04);
        box-shadow: 0 0.6rem 1.2rem rgba(0, 0, 0, 0.25);
    }

    .start-btn button {
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: #000 !important;
    }
    .start-btn button:hover {
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: #ffffff !important;
    }

    .stop-btn button {
        background: linear-gradient(90deg, #ef4444, #b91c1c);
        color: #000 !important;
    }
    .stop-btn button:hover {
        background: linear-gradient(90deg, #ef4444, #b91c1c);
    }

    .stButton > button,
    .stDownloadButton > button {
        color: #000 !important;
    }

    .stButton > button * ,
    .stDownloadButton > button * {
        color: #000 !important;
    }

    /* Keep hover effect */
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: scale(1.04);
        box-shadow: 0 0.5rem 1.2rem rgba(0, 0, 0, 0.25);
    }

    section[data-testid="stSidebar"] * {
    color: #0b1220 !important;      /* dark text for light sidebar */
    }

    div[data-baseweb="select"] * {
    color: #0b1220 !important;
    }


    /* ---- CARDS ---- */
    .card {
        background: radial-gradient(circle at top left, #0f172a, #020617);
        border-radius: 1rem;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.7);
    }
    .card-title {
        font-size: 1rem;
        font-weight: 750;
        color: #ffffff !important;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .card-subtitle {
        font-size: 0.88rem;
        color: #cbd5e1 !important;
        margin-bottom: 0.8rem;
    }
    .card-body {
        font-size: 0.98rem;
        line-height: 1.6;
        color: #ffffff !important;
        max-height: 520px;
        overflow-y: auto;
        padding-right: 0.5rem;
        white-space: pre-wrap;
        opacity: 1 !important;
    }

    .badge {
        font-size: 0.72rem;
        padding: 0.18rem 0.65rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.55);
        color: #e5e7eb !important;
        background: rgba(2, 6, 23, 0.35);
        backdrop-filter: blur(10px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Header
st.markdown(
    """
    <div class="hero">
        <div class="app-title">🎧 Live Lecture Transcription</div>
        <div class="app-subtitle">
            Continuous Whisper Large V3 transcription with BART summarization.
            Speak naturally, stop when you're done, and we’ll save transcript + summary as a session.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Session State Init 
if "model" not in st.session_state:
    st.session_state.model = load_model("openai/whisper-large-v3")

if "recording" not in st.session_state:
    st.session_state.recording = False

if "audio_stream" not in st.session_state:
    st.session_state.audio_stream = None

if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = np.zeros(0, dtype=np.float32)

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "needs_summary" not in st.session_state:
    st.session_state.needs_summary = False

if "last_session_id" not in st.session_state:
    st.session_state.last_session_id = None


# Status Indicator
status_text = "Recording…" if st.session_state.recording else "Idle"
status_class = "status-recording" if st.session_state.recording else "status-idle"

st.markdown(
    f"""
    <div class="status-pill {status_class}">
        <div class="status-dot"></div>
        <span>{status_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# Controls
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="start-btn">', unsafe_allow_html=True)
    if st.button("Start Recording", use_container_width=True):
        if st.session_state.audio_stream is None:
            st.session_state.audio_stream = start_audio_stream()

        st.session_state.recording = True
        st.session_state.audio_buffer = np.zeros(0, dtype=np.float32)
        st.session_state.transcript = ""
        st.session_state.summary = ""
        st.session_state.needs_summary = False
        st.session_state.last_session_id = None
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    if st.button("Stop Recording", use_container_width=True):
        st.session_state.recording = False
        st.session_state.needs_summary = True

        if st.session_state.audio_stream is not None:
            stop_audio_stream(st.session_state.audio_stream)
            st.session_state.audio_stream = None
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)


# Transcript & Summary Layout
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">
                <span>Live Transcript</span>
                <span class="badge">{CHUNK_DURATION}s chunks @ {SAMPLE_RATE} Hz</span>
            </div>
            <div class="card-subtitle">
                Transcription is updated continuously while recording is active.
            </div>
        """,
        unsafe_allow_html=True,
    )
    transcript_placeholder = st.empty()
    transcript_placeholder.markdown(
        f"<div class='card-body'>{st.session_state.transcript or ''}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown(
        """
        <div class="card">
            <div class="card-title"><span>Summary</span></div>
            <div class="card-subtitle">
                A BART-generated summary of the full transcript after you stop recording.
            </div>
        """,
        unsafe_allow_html=True,
    )
    summary_placeholder = st.empty()
    summary_placeholder.markdown(
        f"<div class='card-body'>{st.session_state.summary or ''}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# Continuous Recording + Chunked Transcription
if st.session_state.recording:
    new_audio = drain_audio_queue()
    if new_audio is not None and len(new_audio) > 0:
        st.session_state.audio_buffer = np.concatenate([st.session_state.audio_buffer, new_audio])

    while len(st.session_state.audio_buffer) >= CHUNK_SAMPLES:
        chunk = st.session_state.audio_buffer[:CHUNK_SAMPLES]
        st.session_state.audio_buffer = st.session_state.audio_buffer[CHUNK_SAMPLES:]

        if np.max(np.abs(chunk)) < SILENCE_THRESHOLD:
            continue

        text = transcribe_audio_chunk(st.session_state.model, chunk, sample_rate = SAMPLE_RATE)

        if text.strip():
            st.session_state.transcript += " " + text
            transcript_placeholder.markdown(
                f"<div class='card-body'>{st.session_state.transcript}</div>",
                unsafe_allow_html=True,
            )

    st.rerun()


# Summary + Saving after Stop
if st.session_state.needs_summary and st.session_state.transcript.strip():
    summary = summarize_text(st.session_state.transcript)
    st.session_state.summary = summary
    st.session_state.needs_summary = False

    session_id = get_next_session_id()
    session_path = create_session_folder(session_id)
    save_transcript(session_path, st.session_state.transcript)
    save_summary(session_path, summary)
    st.session_state.last_session_id = session_id

    summary_placeholder.markdown(
        f"<div class='card-body'>{summary}</div>",
        unsafe_allow_html=True,
    )
    st.success(f"Session saved to: `sessions/session_{session_id}`")
