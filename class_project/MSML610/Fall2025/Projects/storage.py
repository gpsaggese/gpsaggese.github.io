import os

SESSIONS_DIR = "sessions"


def get_next_session_id():
    os.makedirs(SESSIONS_DIR, exist_ok=True)

    existing = []
    for name in os.listdir(SESSIONS_DIR):
        if name.startswith("session_"):
            try:
                existing.append(int(name.replace("session_", "")))
            except ValueError:
                pass

    return max(existing) + 1 if existing else 1


def create_session_folder(session_id):
    path = os.path.join(SESSIONS_DIR, f"session_{session_id}")
    os.makedirs(path, exist_ok=True)
    return path


def save_transcript(session_path, transcript):
    path = os.path.join(session_path, "transcript.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(transcript)
    return path


def save_summary(session_path, summary):
    path = os.path.join(session_path, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    return path
