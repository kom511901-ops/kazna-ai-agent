"""Проверки ядра агента с подменённой моделью и поиском."""

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any
from uuid import UUID

import pytest

from app.agent.orchestrator import NO_SOURCE_RESPONSE, AgentOrchestrator, UserProfile
from app.agent.tools import TOOL_DEFINITIONS, AgentTools
from app.providers.base import ChatResult, JsonValue, LLMResponse, ToolCall, ToolDefinition
from app.rag.retriever import CitationMetadata, SearchHit


def _hit() -> SearchHit:
    return SearchHit(
        chunk_id=UUID(int=1),
        document_id=UUID(int=2),
        text="Фрагмент с подтверждённым правилом.",
        metadata=CitationMetadata(
            title="Нормативный акт",
            document_type="Постановление",
            number="123",
            date=date(2025, 1, 1),
            article="Статья 4",
            budget_level="федеральный",
            valid_year=(2025, 2026),
            status="действует",
            source_url="https://example.test/act",
            loaded_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    )


class FakeSearcher:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    async def search(
        self, query: str, *, top_k: int = 5, year: int | None = None, active_only: bool = True
    ) -> list[SearchHit]:
        self.queries.append(query)
        return self.hits[:top_k]


class FakeLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[
            tuple[Sequence[Mapping[str, JsonValue]], Sequence[ToolDefinition] | None]
        ] = []

    async def chat(
        self,
        messages: Sequence[Mapping[str, JsonValue]],
        *,
        tools: Sequence[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> ChatResult:
        assert not stream
        self.calls.append((messages, tools))
        return self.responses.pop(0)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_no_sources_forces_honest_refusal_even_if_model_guesses() -> None:
    llm = FakeLLM([LLMResponse(content="Вымышленное нормативное утверждение")])
    orchestrator = AgentOrchestrator(llm, AgentTools(FakeSearcher([])))

    reply = await orchestrator.respond("Какова неизвестная норма?")

    assert reply.answer == NO_SOURCE_RESPONSE
    assert not reply.sources


@pytest.mark.asyncio
async def test_answer_is_returned_with_only_passed_sources_and_role_context() -> None:
    hit = _hit()
    llm = FakeLLM(
        [
            LLMResponse(
                content="В источнике указано правило.",
                tool_calls=(ToolCall("call-1", "search_kb", {"query": "правило", "year": 2026}),),
            ),
            LLMResponse(content="Правило подтверждается переданным фрагментом."),
        ]
    )
    searcher = FakeSearcher([hit])
    orchestrator = AgentOrchestrator(llm, AgentTools(searcher))

    reply = await orchestrator.respond(
        "Расскажите о правиле", profile=UserProfile(role="бухгалтер заказчика")
    )

    assert searcher.queries == ["правило"]
    assert "https://example.test/act" in reply.answer
    assert "Нормативный акт № 123" in reply.answer
    assert len(reply.sources) == 1
    initial_messages, offered_tools = llm.calls[0]
    assert "бухгалтер заказчика" in str(initial_messages[0]["content"])
    assert offered_tools == TOOL_DEFINITIONS
    assert any(message.get("type") == "function_call_output" for message in llm.calls[1][0])


@pytest.mark.asyncio
async def test_qualify_tool_returns_question_when_required_details_are_missing() -> None:
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=(ToolCall("call-2", "qualify_ks", {}),),
            )
        ]
    )
    orchestrator = AgentOrchestrator(llm, AgentTools(FakeSearcher([])))

    reply = await orchestrator.respond("Подпадает ли мой случай под КС?")

    assert reply.needs_clarification
    assert "вид средств" in reply.answer
    assert "год" in reply.answer
    assert "тип получателя" in reply.answer


@pytest.mark.asyncio
async def test_create_lead_is_dispatched_only_to_injected_handler() -> None:
    received: dict[str, Any] = {}

    async def save_lead(arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        received.update(arguments)
        return {"status": "created", "lead_id": "lead-1"}

    tools = AgentTools(FakeSearcher([]), lead_handler=save_lead)
    import json

    result = json.loads(
        await tools.execute(
            "create_lead", {"contact": "person@example.test", "summary": "Нужна консультация"}
        )
    )

    assert result == {"status": "created", "lead_id": "lead-1"}
    assert received["contact"] == "person@example.test"
