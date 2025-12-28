
===BEGIN_API_MD===

# Multilingual Customer Support Chatbot – API Layer

Tutor Task #40 – Fall 2025 – Whisper Large V3 Multilingual Customer Support Chatbot

This document describes the **internal programming interface (API)** of the tool:

* The **contract layer** in `api_contract.py`
* The **wrapper implementation** in `multilingual_chatbot_utils.py`

“API” here means the tool’s own interface (classes, functions, dataclasses), not any external provider API.

---

## 1. API Contract – `api_contract.py`

This file defines the **stable interface** that other code should depend on.

### 1.1 Data Models

`LanguageCode` is a simple alias:

```python
LanguageCode = str  # e.g. "en", "es", "fr", "de"
```

#### `ChatRequest`

Represents one user query into the system.

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

@dataclass
class ChatRequest:
    user_id: str
    text: Optional[str] = None
    audio_path: Optional[Path] = None
    language_hint: Optional[LanguageCode] = None
    metadata: Optional[Dict[str, str]] = None
```

* Either `text` or `audio_path` must be provided.
* `language_hint` is optional (e.g. `"en"`, `"es"`).

#### `TranscriptionResult`

```python
from dataclasses import dataclass

@dataclass
class TranscriptionResult:
    text: str
    language: LanguageCode
    confidence: float
```

Output of speech-to-text (STT).

#### `IntentResult`

```python
from dataclasses import dataclass
from typing import List

@dataclass
class IntentResult:
    label: str
    score: float
    candidate_labels: List[str]
```

Output of the intent classifier.

#### `ChatResponse`

```python
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ChatResponse:
    reply_text: str
    detected_language: LanguageCode
    intent: Optional[IntentResult] = None
    debug_info: Optional[Dict[str, str]] = None
```

Final result returned to the UI or caller.

---

### 1.2 Service Interfaces (Protocols)

These are abstract interfaces that can be implemented with any models.

#### `SpeechToTextService`

```python
from typing import Protocol, Optional
from pathlib import Path

class SpeechToTextService(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        language_hint: Optional[LanguageCode] = None,
    ) -> TranscriptionResult:
        ...
```

#### `IntentClassifierService`

```python
class IntentClassifierService(Protocol):
    def classify(self, text: str) -> IntentResult:
        ...
```

#### `ResponseGeneratorService`

```python
class ResponseGeneratorService(Protocol):
    def generate(
        self,
        user_text: str,
        intent: IntentResult,
        language: LanguageCode,
    ) -> str:
        ...
```

#### `MultilingualSupportService`

```python
class MultilingualSupportService(Protocol):
    def handle(self, request: ChatRequest) -> ChatResponse:
        ...
```

This is the **main entry point** other code should use.

---

## 2. Wrapper Implementation – `multilingual_chatbot_utils.py`

This file connects the contract above to the **real project code** in `src/`.

It provides:

* `MultilingualChatbotService` – a concrete implementation of `MultilingualSupportService` that:

  * Uses `MultilingualChatbot` from `src/app/main_pipeline.py`
  * For audio: calls `process_audio(audio_path, language=language_hint)`
  * For text: reuses the existing `IntentClassifier` and `ResponseGenerator`
* `create_default_service()` – a factory that returns a ready-to-use service.

Example usage:

```python
from pathlib import Path
from api_contract import ChatRequest
from multilingual_chatbot_utils import create_default_service

service = create_default_service()

# Text query
req_text = ChatRequest(
    user_id="demo-text",
    text="Hola, mi pedido no ha llegado todavía.",
    language_hint="es",
)
resp_text = service.handle(req_text)

# Audio query
req_audio = ChatRequest(
    user_id="demo-audio",
    audio_path=Path("src/data/spanish_sample.wav"),
)
resp_audio = service.handle(req_audio)
```

Callers only need to work with:

* `ChatRequest`
* `ChatResponse`
* `create_default_service()`

They do **not** need to know about Whisper, XLM-R, or GPT internals.

---

## 3. Relationship to Other Submission Files

* `API.md` (this file): explains the design of the API.
* `API.ipynb`: will demonstrate basic usage of the API and wrapper.
* `example.md` / `example.ipynb`: will show a complete end-to-end example app using this API.
  ===END_API_MD===

