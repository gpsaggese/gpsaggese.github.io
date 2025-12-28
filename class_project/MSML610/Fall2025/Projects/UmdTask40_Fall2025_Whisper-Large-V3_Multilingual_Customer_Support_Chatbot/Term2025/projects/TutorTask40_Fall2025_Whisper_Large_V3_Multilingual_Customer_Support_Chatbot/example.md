===BEGIN_EXAMPLE_MD===

Example Application – Multilingual Customer Support Chatbot

Tutor Task #40 – Fall 2025 – Whisper Large V3 Multilingual Customer Support Chatbot

This document describes a complete example application built on top of the internal API layer:

API contract: api_contract.py

Wrapper implementation: multilingual_chatbot_utils.py

Logging helpers: utils_data_io.py

Post-processing helpers: utils_post_processing.py

The corresponding runnable notebook is example.ipynb, which demonstrates the same flow end-to-end.

1. Goal of the Example

The example simulates a small customer support scenario where:

Users ask questions in different languages (text or audio).

The system:

Transcribes audio (Whisper Large V3)

Classifies intent (XLM-R-based classifier)

Generates a reply in the same language (LLM-based)

Each interaction is logged to a JSONL file.

At the end, we compute simple analytics:

How many queries per intent

How many queries per language

This shows how a real application would integrate with the internal API layer without dealing directly with model loading or low-level details.

2. Architecture Overview

High-level components used in the example:

flowchart LR
    U[User / Notebook] --> API[MultilingualSupportService<br/> (create_default_service)]
    API -->|ChatRequest| Bot[MultilingualChatbot<br/>(Whisper + XLM-R + LLM)]
    Bot -->|ChatResponse| API
    API --> Logger[utils_data_io.append_interaction]
    Logger --> File[(logs/interactions.jsonl)]
    File --> Summary[utils_post_processing<br/>summaries]


The notebook constructs ChatRequest objects.

It calls service.handle(request) using create_default_service().

The returned ChatResponse is logged via append_interaction(...).

Later, utils_post_processing.py reads the log file and builds summaries.

3. Core API Usage

The example relies on the existing API layer:

api_contract.ChatRequest

api_contract.ChatResponse

multilingual_chatbot_utils.create_default_service()

Typical usage:

from pathlib import Path
from api_contract import ChatRequest
from multilingual_chatbot_utils import create_default_service

service = create_default_service()

# Text request
req_text = ChatRequest(
    user_id="demo-text",
    text="Hola, mi pedido no ha llegado todavía. ¿Puedes ayudarme?",
    language_hint="es",
)
resp_text = service.handle(req_text)

# Audio request
req_audio = ChatRequest(
    user_id="demo-audio",
    audio_path=Path("src/data/test_audio.mp3"),
)
resp_audio = service.handle(req_audio)


The caller only sees ChatRequest / ChatResponse objects and the service.handle(...) method.

4. Logging Interactions

To avoid embedding logging logic inside the notebook, the example uses:

utils_data_io.append_interaction(log_path, request, response)

Log format: JSON Lines (JSONL) – one JSON object per line.

Example usage (conceptually):

from utils_data_io import append_interaction

log_path = "logs/interactions.jsonl"

resp = service.handle(req_text)
append_interaction(log_path, req_text, resp)

resp2 = service.handle(req_audio)
append_interaction(log_path, req_audio, resp2)


Each record contains:

timestamp

user_id

request_type ("text" or "audio")

request_text or audio_path

language_hint

reply_text

detected_language

intent_label / intent_score

debug_info

This makes the example more realistic: the chatbot is not just answering once, but also leaving behind structured logs.

5. Post-Processing and Analytics

After a few interactions are logged, the example uses utils_post_processing.py to summarize the log:

summarize_by_intent(log_path) → {intent_label: count, ...}

summarize_by_language(log_path) → {language_code: count, ...}

load_records(log_path) → full list of logged dictionaries

In the notebook, this might look like:

from utils_post_processing import (
    summarize_by_intent,
    summarize_by_language,
    load_records,
)

log_path = "logs/interactions.jsonl"

intent_counts = summarize_by_intent(log_path)
lang_counts = summarize_by_language(log_path)
records = load_records(log_path)


These summaries can be displayed as plain dicts, tables, or simple charts in the notebook. The example keeps it minimal, but the same pattern could be used in a dashboard or monitoring tool.

6. Flow in example.ipynb

The notebook example.ipynb follows this structure:

Setup

Import ChatRequest, create_default_service

Import append_interaction and summary helpers

Initialize service = create_default_service()

Simulate 2–3 user queries

One or more text queries in different languages (e.g., English, Spanish)

Optionally one audio query using a file from src/data/

After each service.handle(request), log the interaction with append_interaction(...)

Inspect responses

Print reply text, detected language, and intent for each query

Post-process logs

Call summarize_by_intent(...) and summarize_by_language(...)

Optionally show the raw records (load_records(...)) in a table

The notebook is designed to be self-contained and runnable from top to bottom, relying only on:

api_contract.py

multilingual_chatbot_utils.py

utils_data_io.py

utils_post_processing.py

The existing models and logic under src/

7. How This Example Relates to the API Layer

The API layer (api_contract.py + multilingual_chatbot_utils.py) defines how the chatbot is called.

The example shows one complete application built on top of that API:

Multi-language support

Audio + text inputs

Logging and basic analytics

Anyone who wants to build a different frontend (web app, CLI tool, or REST API) could follow the same pattern:

Use create_default_service() to obtain a MultilingualSupportService.

Wrap user inputs into ChatRequest.

Call service.handle(request) and work with ChatResponse.

Optionally log interactions and analyze them with utilities similar to those in this example.
===END_EXAMPLE_MD===