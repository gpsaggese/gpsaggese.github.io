"""
Configurable chat-completion service for the housing analysis agent.

The service is built with the Semantic Kernel OpenAI connector, which also
works with any OpenAI-compatible endpoint. Three providers are supported:

- `ollama` (default): a local Ollama server through its OpenAI-compatible
  endpoint; no API key or internet access is required
- `groq`: GroqCloud through its OpenAI-compatible endpoint; requires
  `GROQ_API_KEY`
- `openai`: the OpenAI API; requires `OPENAI_API_KEY` and an explicit model

Configuration is read from environment variables, optionally loaded from a
`.env` file next to this module (the file is ignored by Git):

- `HOUSING_LLM_PROVIDER`: `ollama`, `groq`, or `openai` (default `ollama`)
- `HOUSING_LLM_MODEL`: model name (default `qwen3:4b` for Ollama and
  `openai/gpt-oss-120b` for Groq; required for OpenAI)
- `HOUSING_LLM_BASE_URL`: endpoint URL (defaults to the provider endpoint)
- `HOUSING_LLM_TIMEOUT`: request timeout in seconds (default 180)
- `HOUSING_LLM_MAX_RETRIES`: retries for failed requests such as rate-limit
  errors (default 2)
- `GROQ_API_KEY`: API key, required for the `groq` provider
- `OPENAI_API_KEY`: API key, required for the `openai` provider

Import as:

import housing_llm
"""

import dataclasses
import logging
import os
import pathlib
from collections.abc import Mapping

from typing import Any

import dotenv
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory

_LOG = logging.getLogger(__name__)

ENV_FILE = pathlib.Path(__file__).resolve().parent / ".env"
SUPPORTED_PROVIDERS = ("ollama", "groq", "openai")
DEFAULT_PROVIDER = "ollama"
# `host.docker.internal` resolves to the host machine from inside a Docker
# Desktop container, where the Ollama server runs.
DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
# Ollama ignores the API key, but the OpenAI client requires a non-empty one.
OLLAMA_PLACEHOLDER_API_KEY = "ollama"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_MAX_RETRIES = 2
SERVICE_ID = "housing-llm"


@dataclasses.dataclass(frozen=True)
class LLMConfig:
    """
    Settings for the chat-completion service.

    The API key is excluded from `repr()` so that it is never printed.
    """

    provider: str
    model: str
    base_url: str | None
    timeout_s: float
    max_retries: int
    api_key: str = dataclasses.field(repr=False)


def load_llm_config(
    env: Mapping[str, str] | None = None,
    *,
    env_file: pathlib.Path | None = ENV_FILE,
) -> LLMConfig:
    """
    Build the LLM configuration from environment variables.

    :param env: variables to read; defaults to `os.environ`, after loading
        `env_file` if it exists (existing variables are not overridden)
    :param env_file: optional `.env` file; only used when `env` is None
    :return: validated configuration
    :raises ValueError: if the provider is unknown, a required setting is
        missing, or the timeout is invalid
    """
    if env is None:
        if env_file is not None and env_file.exists():
            dotenv.load_dotenv(env_file, override=False)
        env = os.environ
    provider = env.get("HOUSING_LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported HOUSING_LLM_PROVIDER '{provider}'; "
            f"expected one of {SUPPORTED_PROVIDERS}."
        )
    model = env.get("HOUSING_LLM_MODEL", "").strip()
    base_url = env.get("HOUSING_LLM_BASE_URL", "").strip() or None
    timeout_text = env.get("HOUSING_LLM_TIMEOUT", "").strip()
    try:
        timeout_s = float(timeout_text) if timeout_text else DEFAULT_TIMEOUT_S
    except ValueError as exc:
        raise ValueError(
            f"HOUSING_LLM_TIMEOUT must be a number, got '{timeout_text}'."
        ) from exc
    if timeout_s <= 0:
        raise ValueError("HOUSING_LLM_TIMEOUT must be positive.")
    retries_text = env.get("HOUSING_LLM_MAX_RETRIES", "").strip()
    try:
        max_retries = int(retries_text) if retries_text else DEFAULT_MAX_RETRIES
    except ValueError as exc:
        raise ValueError(
            f"HOUSING_LLM_MAX_RETRIES must be an integer, got '{retries_text}'."
        ) from exc
    if max_retries < 0:
        raise ValueError("HOUSING_LLM_MAX_RETRIES must not be negative.")
    if provider == "ollama":
        return LLMConfig(
            provider=provider,
            model=model or DEFAULT_OLLAMA_MODEL,
            base_url=base_url or DEFAULT_OLLAMA_BASE_URL,
            timeout_s=timeout_s,
            max_retries=max_retries,
            api_key=OLLAMA_PLACEHOLDER_API_KEY,
        )
    if provider == "groq":
        api_key = env.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for the 'groq' provider.")
        return LLMConfig(
            provider=provider,
            model=model or DEFAULT_GROQ_MODEL,
            base_url=base_url or DEFAULT_GROQ_BASE_URL,
            timeout_s=timeout_s,
            max_retries=max_retries,
            api_key=api_key,
        )
    # OpenAI: require an explicit key and model rather than guessing.
    api_key = env.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for the 'openai' provider.")
    if not model:
        raise ValueError(
            "HOUSING_LLM_MODEL is required for the 'openai' provider."
        )
    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        timeout_s=timeout_s,
        max_retries=max_retries,
        api_key=api_key,
    )


class GroqChatCompletion(OpenAIChatCompletion):
    """
    OpenAI connector adapted to GroqCloud's OpenAI-compatible API.

    Groq rejects requests whose messages contain a `name` field, which the
    `ChatCompletionAgent` adds to its instruction message. This service
    removes that field; all other request fields are unchanged.
    """

    def _prepare_chat_history_for_request(
        self,
        chat_history: ChatHistory,
        role_key: str = "role",
        content_key: str = "content",
    ) -> Any:
        messages = super()._prepare_chat_history_for_request(
            chat_history, role_key=role_key, content_key=content_key
        )
        return [
            {key: value for key, value in message.items() if key != "name"}
            for message in messages
        ]


def create_chat_service(config: LLMConfig) -> OpenAIChatCompletion:
    """
    Create the Semantic Kernel chat-completion service for a configuration.

    No request is sent when the service is created.

    :param config: LLM configuration
    :return: chat-completion service usable by a `ChatCompletionAgent`
    """
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_s,
        max_retries=config.max_retries,
    )
    _LOG.debug("Creating chat service: %s", config)
    service_class = (
        GroqChatCompletion if config.provider == "groq" else OpenAIChatCompletion
    )
    return service_class(
        ai_model_id=config.model,
        service_id=SERVICE_ID,
        async_client=client,
    )
