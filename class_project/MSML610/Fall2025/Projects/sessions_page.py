import os
import streamlit as st

# this pages file lives under app/pages/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")

st.set_page_config(page_title="Saved Sessions", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #020617; }
    h1, h2, h3, h4, p, span, div { color: #e5e7eb !important; }

    section[data-testid="stSidebar"] * {
    color: #0b1220 !important;
    }

    div[data-baseweb="select"] * {
    color: #0b1220 !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        color: #000 !important; 
    }

    .stButton > button * ,
    .stDownloadButton > button * {
        color: #000 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: scale(1.04);
        box-shadow: 0 0.5rem 1.2rem rgba(0, 0, 0, 0.25);
    }

    div[data-baseweb="select"] * {
    color: #000000 !important;
    }

    .card {
        background: radial-gradient(circle at top left, #0f172a, #020617);
        border-radius: 1rem;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.7);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1rem;
        font-weight: 750;
        color: #ffffff !important;
        margin-bottom: 0.45rem;
        display:flex;
        justify-content:space-between;
        align-items:center;
    }
    .card-body {
        font-size: 0.98rem;
        line-height: 1.6;
        color: #ffffff !important;
        white-space: pre-wrap;
    }
    .muted { color: #9ca3af !important; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📁 Saved Sessions")
st.caption("Browse previous sessions stored in the sessions/ folder and download transcript/summary.")

if not os.path.exists(SESSIONS_DIR):
    st.info("No sessions folder found yet. Record something on the main page first.")
    st.stop()

# List session folders like session_1, session_2...
folders = []
for name in os.listdir(SESSIONS_DIR):
    p = os.path.join(SESSIONS_DIR, name)
    if os.path.isdir(p) and name.startswith("session_"):
        folders.append(name)

def session_sort_key(name: str):
    try:
        return int(name.split("_")[1])
    except Exception:
        return 10**9

folders = sorted(folders, key=session_sort_key)

if not folders:
    st.info("No saved sessions found yet.")
    st.stop()

selected = st.selectbox("Select a session", folders, index=len(folders)-1)

session_path = os.path.join(SESSIONS_DIR, selected)
t_path = os.path.join(session_path, "transcript.txt")
s_path = os.path.join(session_path, "summary.txt")

transcript = ""
summary = ""

if os.path.exists(t_path):
    with open(t_path, "r", encoding="utf-8") as f:
        transcript = f.read()

if os.path.exists(s_path):
    with open(s_path, "r", encoding="utf-8") as f:
        summary = f.read()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">
                <span>Transcript</span>
                <span class="muted">{selected}</span>
            </div>
            <div class="card-body">{transcript if transcript else "<span class='muted'>(No transcript.txt found)</span>"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if transcript:
        st.download_button(
            label="⬇️ Download transcript.txt",
            data=transcript,
            file_name=f"{selected}_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

with col2:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">
                <span>Summary</span>
                <span class="muted">{selected}</span>
            </div>
            <div class="card-body">{summary if summary else "<span class='muted'>(No summary.txt found)</span>"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if summary:
        st.download_button(
            label="⬇️ Download summary.txt",
            data=summary,
            file_name=f"{selected}_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

#  download both in one zip
import io
import zipfile

if transcript or summary:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if transcript:
            zf.writestr("transcript.txt", transcript)
        if summary:
            zf.writestr("summary.txt", summary)
    zip_buffer.seek(0)

    st.download_button(
        "⬇️ Download transcript+summary (zip)",
        data=zip_buffer,
        file_name=f"{selected}.zip",
        mime="application/zip",
        use_container_width=True,
    )
