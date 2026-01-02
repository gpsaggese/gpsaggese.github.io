import streamlit as st
import tempfile
import os
import json

# Import your chatbot class
from src.app.main_pipeline import MultilingualChatbot


# ==========================================================
# FULL WHISPER LANGUAGE MAP
# ==========================================================
LANG_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fr": "French", "fy": "Frisian", "ga": "Irish", "gd": "Scottish Gaelic",
    "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "haw": "Hawaiian",
    "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole",
    "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "jw": "Javanese", "ka": "Georgian",
    "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean",
    "la": "Latin", "lb": "Luxembourgish", "ln": "Lingala", "lo": "Lao",
    "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mi": "Maori",
    "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi",
    "ms": "Malay", "mt": "Maltese", "my": "Myanmar (Burmese)", "ne": "Nepali",
    "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian",
    "oc": "Occitan", "pa": "Punjabi", "pl": "Polish", "ps": "Pashto",
    "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit",
    "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian",
    "sn": "Shona", "so": "Somali", "sq": "Albanian", "sr": "Serbian",
    "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen",
    "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian",
    "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "yi": "Yiddish",
    "yo": "Yoruba", "zh": "Chinese"
}


def get_language_name(code):
    if code is None:
        return "Unknown"
    return LANG_NAMES.get(code, code)


# ==========================================================
# CACHED MODELS (Whisper + XLM-R + Qwen)
# ==========================================================
@st.cache_resource(show_spinner=True)
def get_chatbot():
    return MultilingualChatbot()

@st.cache_resource(show_spinner=True)
def get_translator():
    from src.utils.response_utils import ResponseGenerator
    return ResponseGenerator()


# ==========================================================
# CHATGPT-STYLE UI
# ==========================================================
def main():
    st.set_page_config(page_title="Multilingual Voice Assistant", layout="centered")

    # ------------------------------------------------------
    # LOAD PERSISTENT HISTORY
    # ------------------------------------------------------
    history_file = "chat_history.json"

    if "messages" not in st.session_state:
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                st.session_state.messages = json.load(f)
        else:
            st.session_state.messages = []

    st.title("🎙️ Multilingual Voice Assistant")

    st.write(
        "<div style='margin-bottom:20px; color:gray;'>"
        "Speak to your AI assistant by uploading audio messages."
        "</div>",
        unsafe_allow_html=True
    )

    chatbot = get_chatbot()
    translator = get_translator()

    # ==========================================================
    # DISPLAY CHAT HISTORY (ChatGPT-style bubbles)
    # ==========================================================
    for msg in st.session_state.messages:

        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style="
                    background-color:#1e1e1e;
                    padding:12px;
                    border-radius:8px;
                    margin-bottom:10px;">
                    <strong>👤 User</strong><br>{msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"""
                <div style="
                    background-color:#222831;
                    padding:12px;
                    border-radius:8px;
                    margin-bottom:10px;">
                    <strong>🤖 Assistant</strong><br>{msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )

    # CLEAR BUTTON
    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        if os.path.exists(history_file):
            os.remove(history_file)
        st.rerun()

    st.markdown("---")

    # ==========================================================
    # NEW MESSAGE INPUT (Upload audio at the bottom)
    # ==========================================================
    uploaded_audio = st.file_uploader(
        "▶ Upload an audio message (.mp3, .wav, .m4a)",
        type=["mp3", "wav", "m4a"]
    )

    if uploaded_audio:

        # NOTE: do NOT write temp here — write only inside "Send Message"
        st.audio(uploaded_audio)

        if st.button("Send Message"):

            # Create temp file ONLY when user sends
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(uploaded_audio.read())
                audio_path = tmp.name

            with st.spinner("Processing..."):
                result = chatbot.process_audio(audio_path)

            # SAFE: delete after processing (never before)
            try:
                os.remove(audio_path)
            except:
                pass

            # ------------------------------------------------------
            # SAVE USER MESSAGE
            # ------------------------------------------------------
            st.session_state.messages.append({
                "role": "user",
                "content": result["transcription"]
            })

            # ------------------------------------------------------
            # GENERATE ENGLISH TRANSLATION USING QWEN
            # ------------------------------------------------------
            with st.spinner("Translating to English..."):
                english_translation = translator.generate(result["response"], "en")

            # ------------------------------------------------------
            # ASSISTANT MESSAGE (formatted, with English)
            # ------------------------------------------------------
            assistant_msg = (
                f"🌐 <strong>Language:</strong> {get_language_name(result['language'])}<br>"
                f"🎯 <strong>Intent:</strong> {result['intent']}<br><br>"
                f"💬 <strong>Response:</strong><br>{result['response']}<br><br>"
                f"🔤 <strong>English Translation:</strong><br>{english_translation}"
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_msg
            })

            # Save to disk
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

            # Auto-refresh to show updated chat
            st.rerun()


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    main()
