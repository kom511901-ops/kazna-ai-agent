"""Контракты для языковых моделей и моделей эмбеддингов."""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type ToolDefinition = Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Нормализованный вызов инструмента, возвращённый языковой моделью."""

    call_id: str
    name: str
    arguments: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Ответ языковой модели и, если есть, вызовы инструментов."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


type ChatResult = LLMResponse | AsyncIterator[str]


class LLMProvider(Protocol):
    """Интерфейс асинхронного текстового провайдера."""

    async def chat(
        self,
        messages: Sequence[Mapping[str, JsonValue]],
        *,
        tools: Sequence[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> ChatResult:
        """Генерирует ответ или возвращает поток текстовых фрагментов."""

    async def aclose(self) -> None:
        """Закрывает ресурсы провайдера."""


class EmbeddingProvider(Protocol):
    """Интерфейс асинхронного провайдера векторных представлений."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Преобразует тексты в упорядоченный список векторов."""
