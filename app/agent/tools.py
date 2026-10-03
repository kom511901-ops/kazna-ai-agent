"""Function-calling инструменты агента и их безопасная диспетчеризация."""

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from app.providers.base import JsonValue, ToolDefinition
from app.rag.retriever import HybridRetriever, SearchHit


class Searcher(Protocol):
    """Минимальный интерфейс поиска, используемый инструментами агента."""

    async def search(
        self, query: str, *, top_k: int = 5, year: int | None = None, active_only: bool = True
    ) -> list[SearchHit]:
        """Находит нормативные фрагменты."""


LeadHandler = Callable[[Mapping[str, JsonValue]], Awaitable[Mapping[str, JsonValue]]]

TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Найти действующие нормативные источники в базе знаний.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый вопрос"},
                    "year": {"type": "integer", "description": "Год применимости"},
                    "top_k": {"type": "integer", "description": "Число фрагментов"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "qualify_ks",
            "description": "Проверить, подпадает ли ситуация под казначейское сопровождение.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fund_type": {"type": "string", "description": "Вид средств"},
                    "year": {"type": "integer", "description": "Год"},
                    "recipient_type": {"type": "string", "description": "Тип получателя"},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_lead",
            "description": "Передать запрос на связь со специалистом (пока заглушка).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "contact": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["contact", "summary"],
                "additionalProperties": False,
            },
        },
    },
)


class AgentTools:
    """Реализует доступные функции и возвращает сериализуемые результаты."""

    def __init__(
        self, searcher: Searcher | HybridRetriever, lead_handler: LeadHandler | None = None
    ):
        self._searcher = searcher
        self._lead_handler = lead_handler

    async def execute(self, name: str, arguments: Mapping[str, JsonValue]) -> str:
        """Проверяет аргументы и выполняет инструмент с JSON-результатом."""
        if name == "search_kb":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return self._json({"error": "Для поиска нужен непустой query"})
            year = arguments.get("year")
            top_k = arguments.get("top_k", 5)
            hits = await self._searcher.search(
                query,
                year=year if isinstance(year, int) and not isinstance(year, bool) else None,
                top_k=top_k if isinstance(top_k, int) and not isinstance(top_k, bool) else 5,
            )
            return self._json({"sources": [self._serialize_hit(hit) for hit in hits]})

        if name == "qualify_ks":
            required = {
                "fund_type": (
                    "вид средств (например, госконтракт, субсидия или бюджетные инвестиции)"
                ),
                "year": "год",
                "recipient_type": "тип получателя",
            }
            missing: list[JsonValue] = [
                label for key, label in required.items() if not arguments.get(key)
            ]
            if missing:
                return self._json({"status": "needs_clarification", "missing": missing})
            return self._json(
                {
                    "status": "not_implemented",
                    "message": "Квалификатор будет подключён в следующей задаче.",
                    "provided": dict(arguments),
                }
            )

        if name == "create_lead":
            if not isinstance(arguments.get("contact"), str) or not isinstance(
                arguments.get("summary"), str
            ):
                return self._json(
                    {"status": "not_created", "reason": "Нужны контакт и краткое описание"}
                )
            if self._lead_handler is None:
                return self._json(
                    {"status": "not_created", "reason": "Передача специалисту пока не подключена"}
                )
            result = await self._lead_handler(arguments)
            return self._json(dict(result))

        return self._json({"error": f"Неизвестный инструмент: {name}"})

    @staticmethod
    def _serialize_hit(hit: SearchHit) -> dict[str, JsonValue]:
        metadata = hit.metadata
        return {
            "chunk_id": str(hit.chunk_id),
            "document_id": str(hit.document_id),
            "text": hit.text,
            "score": hit.score,
            "citation": {
                "title": metadata.title,
                "type": metadata.document_type,
                "number": metadata.number,
                "date": metadata.date.isoformat() if metadata.date else None,
                "article": metadata.article,
                "budget_level": metadata.budget_level,
                "valid_year": list(metadata.valid_year),
                "status": metadata.status,
                "source_url": metadata.source_url,
                "loaded_at": metadata.loaded_at.isoformat(),
            },
        }

    @staticmethod
    def _json(value: Mapping[str, JsonValue]) -> str:
        return json.dumps(value, ensure_ascii=False)
