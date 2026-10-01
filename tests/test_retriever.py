"""Тесты гибридного поиска без подключения к PostgreSQL."""

from collections.abc import Sequence
from datetime import UTC, date, datetime
from uuid import UUID

import pytest

from app.core.config import Settings
from app.knowledge.metadata import ACTIVE_STATUS, REPEALED_STATUS
from app.providers.base import EmbeddingProvider
from app.rag.retriever import (
    CitationMetadata,
    HybridRetriever,
    SearchBackend,
    SearchHit,
)


def _hit(
    chunk_id: int,
    document_id: int,
    *,
    title: str,
    text: str,
    status: str = ACTIVE_STATUS,
    years: tuple[int, ...] = (2026,),
) -> SearchHit:
    return SearchHit(
        chunk_id=UUID(int=chunk_id),
        document_id=UUID(int=document_id),
        text=text,
        metadata=CitationMetadata(
            title=title,
            document_type="Постановление",
            number="123",
            date=date(2025, 1, 1),
            article="Статья 4",
            budget_level="федеральный",
            valid_year=years,
            status=status,
            source_url="https://example.test/source",
            loaded_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    )


class FakeEmbeddings(EmbeddingProvider):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        assert len(texts) == 1
        return [[0.1, 0.2, 0.3]]


class FakeBackend(SearchBackend):
    def __init__(self, semantic: Sequence[SearchHit], keyword: Sequence[SearchHit]) -> None:
        self.semantic = semantic
        self.keyword = keyword
        self.years: list[int] = []

    async def semantic_search(
        self, vector: Sequence[float], *, year: int, limit: int
    ) -> Sequence[SearchHit]:
        assert vector == [0.1, 0.2, 0.3]
        assert limit > 0
        self.years.append(year)
        return self.semantic

    async def keyword_search(self, query: str, *, year: int, limit: int) -> Sequence[SearchHit]:
        assert query
        assert limit > 0
        self.years.append(year)
        return self.keyword


@pytest.mark.asyncio
async def test_hybrid_search_recall_at_k_and_citation_metadata() -> None:
    expected_a = _hit(1, 101, title="Казначейское сопровождение", text="условия расчетов")
    expected_b = _hit(2, 102, title="Правила бюджета", text="казначейское сопровождение")
    distractor = _hit(3, 103, title="Другая тема", text="общая информация")
    backend = FakeBackend(
        semantic=[expected_a, distractor],
        keyword=[expected_b, expected_a],
    )
    retriever = HybridRetriever(FakeEmbeddings(), backend, Settings(current_fiscal_year=2026))

    results = await retriever.search("казначейское сопровождение", top_k=2)

    expected_documents = {expected_a.document_id, expected_b.document_id}
    returned_documents = {result.document_id for result in results}
    recall_at_k = len(expected_documents & returned_documents) / len(expected_documents)
    assert recall_at_k == 1.0
    assert results[0].metadata.title == "Казначейское сопровождение"
    assert results[0].metadata.number == "123"
    assert results[0].metadata.source_url == "https://example.test/source"
    assert backend.years == [2026, 2026]


@pytest.mark.asyncio
async def test_current_filter_excludes_repealed_and_wrong_year_documents() -> None:
    current = _hit(10, 110, title="Действующий", text="актуальный фрагмент")
    repealed = _hit(
        11,
        111,
        title="Утративший силу",
        text="старый фрагмент",
        status=REPEALED_STATUS,
    )
    old_year = _hit(12, 112, title="Старый год", text="фрагмент", years=(2024,))
    backend = FakeBackend(
        semantic=[current, repealed, old_year],
        keyword=[repealed, current, old_year],
    )
    retriever = HybridRetriever(FakeEmbeddings(), backend, Settings(current_fiscal_year=2026))

    results = await retriever.search("фрагмент", top_k=5)

    assert [result.document_id for result in results] == [current.document_id]
    assert all(result.metadata.status == ACTIVE_STATUS for result in results)
