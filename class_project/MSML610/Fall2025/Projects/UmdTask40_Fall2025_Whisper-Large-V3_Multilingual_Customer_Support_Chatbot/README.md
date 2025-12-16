===BEGIN_README===

Tutor Task 40 – Fall 2025
Whisper Large V3 Multilingual Customer Support Chatbot

This project implements a multilingual customer support chatbot powered by:

Whisper Large V3 for speech-to-text (audio → text)

XLM-R–based intent classifier for intent detection

An LLM-based response generator for answers in the user’s language

The project is structured to highlight a clean internal API layer, plus a runnable example application:

api_contract.py – stable API contract (dataclasses + interfaces)

multilingual_chatbot_utils.py – wrapper that adapts the real models to that API

API.md / API.ipynb – document & demonstrate the API

example.md / example.ipynb – end-to-end example, including logging & simple analytics

1. Folder Structure

Relative to the course layout, the project is located at:

MSML610/
└── Term2025/
    └── projects/
        └── TUTORTASK40_FALL2025_WHISPER_LARGE_V3_MULTILINGUAL_CUSTOMER_SUPPORT_CHATBOT/
            ├── src/
            │   ├── app/
            │   │   └── main_pipeline.py
            │   ├── data/
            │   │   └── ... (sample audio files, e.g. test_audio.mp3)
            │   └── utils/
            │       ├── audio_utils.py
            │       ├── transcription_utils.py
            │       ├── intent_utils.py
            │       ├── response_utils.py
            │       └── __init__.py
            ├── api_contract.py
            ├── multilingual_chatbot_utils.py
            ├── utils_data_io.py
            ├── utils_post_processing.py
            ├── API.md
            ├── API.ipynb
            ├── example.md
            ├── example.ipynb
            ├── Dockerfile
            ├── docker-compose.yml
            ├── requirements.txt
            ├── streamlit_app.py
            ├── test_chatbot.py
            └── README.md


High-level roles:

src/ – original project code (models, pipeline, utils, sample data).

api_contract.py – contract-only API layer.

multilingual_chatbot_utils.py – concrete implementation of the API using src/.

API.* – API documentation and minimal demo.

example.* – reference example application using the API.

utils_data_io.py / utils_post_processing.py – logging + analytics helpers.

Dockerfile / docker-compose.yml – containerized environment.

2. Conceptual Architecture
2.1 Processing pipeline

The chatbot supports text and audio inputs:

Audio path (ChatRequest.audio_path):

Whisper Large V3 (via transcribe_audio) → transcription text + language.

XLM-R-based intent classifier → intent label.

LLM-based response generator → reply in detected language.

Text input (ChatRequest.text):

XLM-R-based intent classifier → intent label.

LLM-based response generator → reply in the language_hint or default.

This logic is encapsulated in:

src/app/main_pipeline.py → MultilingualChatbot

Wrapped by MultilingualChatbotService in multilingual_chatbot_utils.py

Exposed through the high-level method:

from api_contract import ChatRequest
from multilingual_chatbot_utils import create_default_service

service = create_default_service()
response = service.handle(ChatRequest(user_id="demo", text="Hello!"))

2.2 API layer vs example layer

API layer (api_contract.py, API.md, API.ipynb)
Defines the stable programming interface:

Dataclasses: ChatRequest, TranscriptionResult, IntentResult, ChatResponse

Protocols (interfaces): SpeechToTextService, IntentClassifierService, ResponseGeneratorService, MultilingualSupportService

Concrete implementation: MultilingualChatbotService (via multilingual_chatbot_utils.py)

Example layer (example.md, example.ipynb)
Shows a realistic usage:

Simulate several user queries (EN/ES/FR, text + optional audio)

Log each interaction to logs/interactions.jsonl

Compute simple summaries: counts per intent, counts per language

3. Internal API – Key Files
3.1 api_contract.py

Defines the core types and interfaces:

LanguageCode = str – language codes (e.g., "en", "es", "fr").

ChatRequest – one user query (text or audio).

TranscriptionResult – result of STT.

IntentResult – result of intent classification.

ChatResponse – final bot response to the caller.

SpeechToTextService, IntentClassifierService, ResponseGeneratorService, MultilingualSupportService – Protocols describing service interfaces.

Any new implementation (e.g., different STT or LLM) can plug in as long as it implements these Protocols.

3.2 multilingual_chatbot_utils.py

Adapters that connect api_contract.py to the real models:

MultilingualChatbotService(MultilingualSupportService):

Holds an instance of MultilingualChatbot from src/app/main_pipeline.py.

For audio:

Calls self._bot.process_audio(audio_path, language=request.language_hint).

For text:

Uses self._bot.intent_classifier.predict_intent(text) to get intent.

Uses ResponseGenerator().generate(text, lang_code) to produce the reply.

Returns the result as a ChatResponse.

create_default_service():

Factory function for notebooks/clients:

service = create_default_service()

4. Logging & Post-Processing
4.1 utils_data_io.py

Provides helpers for logging interactions in JSON Lines (JSONL) format:

append_interaction(log_path, request, response)

Flattens ChatRequest + ChatResponse into a single dict.

Appends one JSON object per line to log_path.

Automatically creates directories if needed.

read_interactions(log_path)

Loads all entries from the log file into a list of dicts.

Each log record includes:

timestamp, user_id, request_type ("text" or "audio"), request_text, audio_path

language_hint, reply_text, detected_language

intent_label, intent_score, debug_info

4.2 utils_post_processing.py

Provides simple analytics:

summarize_by_intent(log_path) → {intent_label: count}

summarize_by_language(log_path) → {language_code: count}

load_records(log_path) → full list of logged records

Used inside example.ipynb to generate a quick summary at the end.

5. Notebooks
5.1 API.ipynb

Purpose: minimal demo of the API layer.

Imports ChatRequest and create_default_service().

Creates the service:

service = create_default_service()


Demonstrates:

A text query (e.g., Spanish customer issue).

An audio query (if src/data/test_audio.mp3 exists).

Usage pattern:

from api_contract import ChatRequest
from multilingual_chatbot_utils import create_default_service

service = create_default_service()

req = ChatRequest(
    user_id="demo-text",
    text="Hola, mi pedido no ha llegado todavía.",
    language_hint="es",
)
resp = service.handle(req)
print(resp.reply_text, resp.detected_language, resp.intent)

5.2 example.ipynb

Purpose: end-to-end example application, matching example.md.

Initializes the service and a log path (logs/interactions.jsonl).

Sends several text queries (EN/ES/FR) and optionally one audio query.

Logs each interaction via append_interaction.

Prints individual responses (reply text, language, intent).

Uses summarize_by_intent and summarize_by_language to show simple analytics.

Example snippet from the notebook:

from utils_data_io import append_interaction
from utils_post_processing import summarize_by_intent, summarize_by_language

log_path = "logs/interactions.jsonl"
service = create_default_service()

req_en = ChatRequest(user_id="user-1", text="My order is delayed.", language_hint="en")
resp_en = service.handle(req_en)
append_interaction(log_path, req_en, resp_en)

intent_counts = summarize_by_intent(log_path)
lang_counts = summarize_by_language(log_path)

6. Running the Project

You can run this project locally (Python 3.11 with all deps installed) or inside Docker. Since models like Whisper and XLM-R are heavy, Docker is recommended.

6.1 Prerequisites

Docker and Docker Compose installed

(Optional) Local Python 3.11 environment if you prefer to run without Docker

6.2 Running with Docker
6.2.1 Build the image

From the project root (TUTORTASK40_FALL2025_...):

docker build -t multilingual-chatbot .

6.2.2 Run an interactive container
docker run -it --rm \
    -p 8888:8888 \
    -v "$(pwd)":/workspace \
    --name multilingual-chatbot-container \
    multilingual-chatbot \
    bash


This:

Maps the current project into /workspace inside the container.

Gives you a shell to run notebooks or scripts.

6.2.3 Starting Jupyter inside the container (optional)

Inside the container:

cd /workspace

# If Jupyter is not already installed in the image:
pip install jupyterlab

jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root


Then, on the host, open the Jupyter URL with the token in your browser and:

Open API.ipynb to test the API.

Open example.ipynb to run the end-to-end example.

Expected terminal output when starting Jupyter:
A URL similar to http://127.0.0.1:8888/?token=...

6.3 Running locally (without Docker)

If you prefer to run directly on your machine:

Create and activate a Python 3.11 virtual environment.

Install dependencies:

pip install -r requirements.txt


Start Jupyter:

jupyter lab


Open and run:

API.ipynb – API demonstration.

example.ipynb – end-to-end example + logging and summaries.

Note: Running Whisper Large V3 and transformer models locally may be slow without a GPU.

6.4 Running the Streamlit demo (optional)

The repository also includes a streamlit_app.py that can be wired to the same API.

Inside the chosen environment (Docker container or local venv):

streamlit run streamlit_app.py


Expected behavior:

A web UI where a user can:

Enter text in different languages.

(Optionally) upload audio (if implemented in the app).

See a response generated by the chatbot.

7. Testing / Quick Sanity Checks

Internal API import test:

python -c "from api_contract import ChatRequest; from multilingual_chatbot_utils import create_default_service; print('OK')"


Notebook structure:

API.ipynb and example.ipynb should run top-to-bottom via “Restart & Run All” (given models and deps are installed).

Logs:

After running example.ipynb, logs/interactions.jsonl should exist.

utils_post_processing.summarize_by_intent("logs/interactions.jsonl") should return a non-empty dict after a few interactions.

8. Design Decisions (Summary)

Contract-first design:
api_contract.py separates interface from implementation. Other teams could re-implement the same contract with different models without touching the notebooks.

Wrapper vs core:
multilingual_chatbot_utils.py wraps the existing MultilingualChatbot pipeline instead of rewriting it, preserving the original project logic.

Thin notebooks:
Heavy logic (logging, analytics, service wiring) lives in reusable .py modules:

multilingual_chatbot_utils.py

utils_data_io.py

utils_post_processing.py
Notebooks only orchestrate calls and display results.

Reproducibility:
Dockerfile + requirements.txt allow the same environment to be recreated for grading or future runs.

This structure ensures the project is:

Well-documented

API-driven

Runnable end-to-end

Easy to integrate with other frontends or pipelines.
===END_README===