"""Tests for the configurable LLM service (no network access required)."""

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory

import housing_llm

GROQ_ENV = {"HOUSING_LLM_PROVIDER": "groq", "GROQ_API_KEY": "test-groq-key"}


def test_default_config_uses_local_ollama():
    """With no settings, the configuration targets the local Ollama model."""
    config = housing_llm.load_llm_config({})

    assert config.provider == "ollama"
    assert config.model == housing_llm.DEFAULT_OLLAMA_MODEL
    assert config.base_url == housing_llm.DEFAULT_OLLAMA_BASE_URL
    assert config.timeout_s == housing_llm.DEFAULT_TIMEOUT_S
    assert config.max_retries == housing_llm.DEFAULT_MAX_RETRIES


def test_groq_config_defaults():
    config = housing_llm.load_llm_config(GROQ_ENV)

    assert config.provider == "groq"
    assert config.model == "openai/gpt-oss-120b"
    assert config.base_url == "https://api.groq.com/openai/v1"
    assert config.api_key == "test-groq-key"


def test_groq_requires_api_key():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        housing_llm.load_llm_config({"HOUSING_LLM_PROVIDER": "groq"})


def test_groq_api_key_is_not_shown_in_repr():
    config = housing_llm.load_llm_config(GROQ_ENV)

    assert "test-groq-key" not in repr(config)


def test_groq_service_uses_groq_connector():
    config = housing_llm.load_llm_config({
        **GROQ_ENV,
        "HOUSING_LLM_MAX_RETRIES": "3",
    })

    service = housing_llm.create_chat_service(config)

    assert isinstance(service, housing_llm.GroqChatCompletion)
    assert service.ai_model_id == "openai/gpt-oss-120b"
    assert str(service.client.base_url).startswith(
        "https://api.groq.com/openai/v1"
    )
    assert service.client.max_retries == 3


def test_groq_request_messages_have_no_name_field():
    """
    Groq rejects `messages[].name`; the Groq service must not send it.

    The request is captured by a mock HTTP transport, so nothing is sent.
    """
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "openai/gpt-oss-120b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
        })

    client = AsyncOpenAI(
        api_key="test-groq-key",
        base_url=housing_llm.DEFAULT_GROQ_BASE_URL,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    service = housing_llm.GroqChatCompletion(
        ai_model_id="openai/gpt-oss-120b", async_client=client
    )
    history = ChatHistory()
    # The ChatCompletionAgent names its instruction message like this.
    history.add_system_message("Instructions.")
    history.messages[0].name = "HousingAnalyst"
    history.add_user_message("Hello")

    asyncio.run(service.get_chat_message_contents(
        history, OpenAIChatPromptExecutionSettings()
    ))

    messages = bodies[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert all("name" not in m for m in messages)


def test_ollama_settings_can_be_overridden():
    config = housing_llm.load_llm_config({
        "HOUSING_LLM_PROVIDER": " Ollama ",
        "HOUSING_LLM_MODEL": "other-model",
        "HOUSING_LLM_BASE_URL": "http://localhost:11434/v1",
        "HOUSING_LLM_TIMEOUT": "30",
    })

    assert config.provider == "ollama"
    assert config.model == "other-model"
    assert config.base_url == "http://localhost:11434/v1"
    assert config.timeout_s == 30.0


def test_openai_requires_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        housing_llm.load_llm_config({
            "HOUSING_LLM_PROVIDER": "openai",
            "HOUSING_LLM_MODEL": "some-model",
        })


def test_openai_requires_model():
    with pytest.raises(ValueError, match="HOUSING_LLM_MODEL"):
        housing_llm.load_llm_config({
            "HOUSING_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
        })


def test_openai_config():
    config = housing_llm.load_llm_config({
        "HOUSING_LLM_PROVIDER": "openai",
        "HOUSING_LLM_MODEL": "some-model",
        "OPENAI_API_KEY": "test-key",
    })

    assert config.provider == "openai"
    assert config.model == "some-model"
    # The official OpenAI endpoint is used when no base URL is given.
    assert config.base_url is None


def test_api_key_is_not_shown_in_repr():
    config = housing_llm.load_llm_config({
        "HOUSING_LLM_PROVIDER": "openai",
        "HOUSING_LLM_MODEL": "some-model",
        "OPENAI_API_KEY": "secret-test-key",
    })

    assert "secret-test-key" not in repr(config)


@pytest.mark.parametrize(
    "env, message",
    [
        ({"HOUSING_LLM_PROVIDER": "unknown"}, "Unsupported"),
        ({"HOUSING_LLM_TIMEOUT": "abc"}, "must be a number"),
        ({"HOUSING_LLM_TIMEOUT": "0"}, "must be positive"),
        ({"HOUSING_LLM_MAX_RETRIES": "x"}, "must be an integer"),
        ({"HOUSING_LLM_MAX_RETRIES": "-1"}, "must not be negative"),
    ],
)
def test_invalid_config(env, message):
    with pytest.raises(ValueError, match=message):
        housing_llm.load_llm_config(env)


def test_create_chat_service_without_network():
    """Creating the service configures the client but sends no request."""
    config = housing_llm.load_llm_config({
        "HOUSING_LLM_BASE_URL": "http://localhost:9/v1",
    })

    service = housing_llm.create_chat_service(config)

    assert isinstance(service, OpenAIChatCompletion)
    assert service.ai_model_id == housing_llm.DEFAULT_OLLAMA_MODEL
    assert service.service_id == housing_llm.SERVICE_ID
    assert str(service.client.base_url).startswith("http://localhost:9/v1")
