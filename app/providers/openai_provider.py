"""Реализация текстового и embedding-провайдера через OpenAI Responses API."""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openai.types.responses import Response

from app.core.config import Settings, get_settings
from app.providers.base import (
    ChatResult,
    JsonValue,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)


class OpenAIProvider:
    """OpenAI-провайдер с настраиваемыми моделями и повтором временных ошибок."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            raise ValueError("Не задан OPENAI_API_KEY для провайдера OpenAI")

        self._client = AsyncOpenAI(
            api_key=self._settings.openai_api_key,
            timeout=self._settings.openai_timeout_seconds,
            max_retries=0,
        )

    async def chat(
        self,
        messages: Sequence[Mapping[str, JsonValue]],
        *,
        tools: Sequence[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> ChatResult:
        """Вызывает Responses API и нормализует текст и function calls."""
        request: dict[str, object] = {
            "model": self._settings.llm_model,
            "input": [dict(message) for message in messages],
        }
        if tools:
            request["tools"] = [self._normalize_tool(tool) for tool in tools]

        if stream:
            request["stream"] = True
            return self._stream(request)

        response: Response = await self._retry(lambda: self._create_response(request))
        return self._normalize_response(response)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Создаёт эмбеддинги, сохраняя порядок входных текстов."""
        if not texts:
            return []

        response: Any = await self._retry(
            lambda: self._create_embeddings(list(texts))
        )
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]

    async def aclose(self) -> None:
        """Закрывает HTTP-клиент провайдера."""
        await self._client.close()

    async def _create_response(self, request: dict[str, object]) -> Response:
        response = await cast(Any, self._client.responses).create(**request)
        return cast(Response, response)

    async def _create_embeddings(self, texts: list[str]) -> Any:
        return await cast(Any, self._client.embeddings).create(
            model=self._settings.embed_model,
            input=texts,
        )

    async def _stream(self, request: dict[str, object]) -> AsyncIterator[str]:
        """Отдаёт текстовые delta-события Responses API."""
        stream: Any = await self._retry(lambda: self._create_response_stream(request))
        async for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta

    async def _create_response_stream(self, request: dict[str, object]) -> Any:
        return await cast(Any, self._client.responses).create(**request)

    async def _retry[T](self, operation: Any) -> T:
        """Повторяет сетевые ошибки, 429 и серверные ошибки с экспоненциальной задержкой."""
        retries = self._settings.openai_max_retries
        for attempt in range(retries + 1):
            try:
                return cast(T, await operation())
            except APIStatusError as error:
                retryable = error.status_code == 429 or error.status_code >= 500
                if not retryable or attempt == retries:
                    raise
            except APIConnectionError:
                if attempt == retries:
                    raise

            delay = self._settings.openai_retry_backoff_seconds * (2**attempt)
            if delay:
                await asyncio.sleep(delay)

        raise RuntimeError("Повтор OpenAI-запроса завершился неожиданно")

    @staticmethod
    def _normalize_tool(tool: ToolDefinition) -> dict[str, object]:
        """Преобразует декларацию функции к формату Responses API."""
        function = tool.get("function")
        if tool.get("type") == "function" and function is None:
            return dict(tool)
        if not isinstance(function, Mapping):
            raise ValueError("Инструмент должен содержать описание function")
        return {
            "type": "function",
            "name": function.get("name"),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {"type": "object", "properties": {}}),
            "strict": function.get("strict", False),
        }

    @staticmethod
    def _normalize_response(response: Response) -> LLMResponse:
        """Выбирает текст и приводит function call к общему типу."""
        calls: list[ToolCall] = []
        for output_item in response.output:
            if output_item.type != "function_call":
                continue
            arguments = json.loads(output_item.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Аргументы function call должны быть JSON-объектом")
            calls.append(
                ToolCall(
                    call_id=output_item.call_id,
                    name=output_item.name,
                    arguments=cast(dict[str, JsonValue], arguments),
                )
            )

        input_tokens = response.usage.input_tokens if response.usage else None
        output_tokens = response.usage.output_tokens if response.usage else None
        return LLMResponse(
            content=response.output_text or "",
            tool_calls=tuple(calls),
            response_id=response.id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
