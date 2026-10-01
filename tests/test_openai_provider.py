"""Контрактные тесты OpenAI-провайдера через замоканный HTTP."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx

from app.core.config import Settings
from app.providers.base import LLMResponse
from app.providers.factory import create_providers
from app.providers.openai_provider import OpenAIProvider


@pytest.fixture
def settings() -> Settings:
    """Возвращает настройки, подходящие для HTTP-тестов."""
    return Settings(
        openai_api_key="test-key",
        llm_model="test-chat-model",
        embed_model="test-embedding-model",
        openai_timeout_seconds=2,
        openai_max_retries=2,
        openai_retry_backoff_seconds=0.001,
    )


@pytest.fixture
async def provider(settings: Settings) -> AsyncIterator[OpenAIProvider]:
    """Создаёт отдельный клиент для теста и гарантирует его закрытие."""
    instance = OpenAIProvider(settings)
    yield instance
    await instance.aclose()


def _response_payload(output: list[dict[str, Any]]) -> dict[str, Any]:
    """Создаёт минимальный валидный объект Responses API."""
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1_750_000_000,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "model": "test-chat-model",
        "output": output,
        "parallel_tool_calls": True,
        "temperature": 1,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1,
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 16,
        },
    }


@pytest.mark.asyncio
@respx.mock
async def test_chat_builds_responses_request_and_parses_tool_call(
    provider: OpenAIProvider,
) -> None:
    """Запрос использует настроенную модель и нормализует function call."""
    route = respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json=_response_payload(
                [
                    {
                        "type": "function_call",
                        "id": "fc_test",
                        "call_id": "call_test",
                        "name": "search_kb",
                        "arguments": '{"query":"аванс"}',
                        "status": "completed",
                    }
                ]
            ),
        )
    )

    result = await provider.chat(
        [
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "Найди порядок авансирования."},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_kb",
                    "description": "Ищет в базе знаний",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
    )

    assert isinstance(result, LLMResponse)
    assert result.tool_calls[0].call_id == "call_test"
    assert result.tool_calls[0].name == "search_kb"
    assert result.tool_calls[0].arguments == {"query": "аванс"}
    assert result.response_id == "resp_test"
    assert result.input_tokens == 12
    assert route.called
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer test-key"
    payload = await request.aread()
    body = json.loads(payload)
    assert body["model"] == "test-chat-model"
    assert body["input"][1]["content"] == "Найди порядок авансирования."
    assert body["tools"] == [
        {
            "type": "function",
            "name": "search_kb",
            "description": "Ищет в базе знаний",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "strict": False,
        }
    ]


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("status_code", [429, 503])
async def test_chat_retries_rate_limit_and_server_errors(
    provider: OpenAIProvider,
    status_code: int,
) -> None:
    """После временной ошибки повторяет запрос и возвращает успешный ответ."""
    route = respx.post("https://api.openai.com/v1/responses").mock(
        side_effect=[
            httpx.Response(
                status_code,
                json={
                    "error": {
                        "message": "temporary failure",
                        "type": "server_error",
                        "param": None,
                        "code": None,
                    }
                },
            ),
            httpx.Response(
                200,
                json=_response_payload(
                    [
                        {
                            "type": "message",
                            "id": "msg_test",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": "Готово", "annotations": []}
                            ],
                        }
                    ]
                ),
            ),
        ]
    )

    result = await provider.chat([{"role": "user", "content": "Привет"}])

    assert isinstance(result, LLMResponse)
    assert result.content == "Готово"
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_embeddings_use_configured_model(provider: OpenAIProvider) -> None:
    """Запрос эмбеддингов использует модель из настроек и сохраняет порядок."""
    route = respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.4, 0.5]},
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                ],
                "model": "test-embedding-model",
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )
    )

    vectors = await provider.embed(["первый", "второй"])

    assert vectors == [[0.1, 0.2], [0.4, 0.5]]
    body = json.loads(await route.calls[0].request.aread())
    assert body["model"] == "test-embedding-model"
    assert body["input"] == ["первый", "второй"]


@pytest.mark.asyncio
@respx.mock
async def test_chat_streams_text_deltas(provider: OpenAIProvider) -> None:
    """Потоковый режим отдаёт только текстовые delta-события."""
    route = respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'event: response.output_text.delta\ndata: '
                '{"type":"response.output_text.delta","delta":"Ответ"}\n\n'
                'event: response.output_text.delta\ndata: '
                '{"type":"response.output_text.delta","delta":" готов"}\n\n'
                'event: response.completed\ndata: '
                '{"type":"response.completed","response":{},"sequence_number":2}\n\n'
            ),
        )
    )

    result = await provider.chat([{"role": "user", "content": "Привет"}], stream=True)

    assert not isinstance(result, LLMResponse)
    chunks = [chunk async for chunk in result]
    assert chunks == ["Ответ", " готов"]
    assert route.called


def test_factory_rejects_unknown_provider() -> None:
    """Фабрика сообщает о провайдере, для которого пока нет реализации."""
    settings = Settings(llm_provider="unknown", openai_api_key="test-key")

    with pytest.raises(ValueError, match="Неподдерживаемый провайдер"):
        create_providers(settings)
