"""Chat access with an on-disk cache, so re-running costs nothing. Used only by the relation extractor, and only
when an API key and a chat model are set. Tests inject a fake instead."""
import hashlib
import json
import sqlite3
from pathlib import Path

from .config import CACHE_DIR, Settings


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key: str):
        row = self.db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, value))
        self.db.commit()


class ChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache = _Cache(CACHE_DIR / "llm.sqlite")
        self._api = None

    def complete(self, messages: list[dict]) -> str:
        model = self.settings.chat_model
        if not model or not self.settings.openai_api_key:
            raise RuntimeError("Set OPENAI_API_KEY and OPENAI_CHAT_MODEL in .env to use the model extractor.")
        key = _key(model, json.dumps(messages, sort_keys=True))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self._api is None:
            from openai import OpenAI

            self._api = OpenAI(api_key=self.settings.openai_api_key, max_retries=5)
        try:
            response = self._api.chat.completions.create(model=model, messages=messages, temperature=0)
        except Exception as exc:
            # Some models only accept their default temperature.
            if "temperature" not in str(exc).lower():
                raise
            response = self._api.chat.completions.create(model=model, messages=messages)
        text = response.choices[0].message.content or ""
        self.cache.put(key, text)
        return text
