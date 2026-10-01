"""Гибридный поиск фрагментов базы знаний с метаданными для цитирования."""

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol
from typing import cast as typing_cast
from uuid import UUID

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import cast, func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from app.core.config import Settings, get_settings
from app.knowledge.metadata import ACTIVE_STATUS
from app.knowledge.models import EMBEDDING_DIMENSIONS, Chunk, Document
from app.providers.base import EmbeddingProvider

_RRF_CONSTANT = 60
_TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class CitationMetadata:
    """Метаданные нормативного документа для построения цитаты."""

    title: str
    document_type: str
    number: str | None
    date: date | None
    article: str | None
    budget_level: str | None
    valid_year: tuple[int, ...]
    status: str
    source_url: str | None
    loaded_at: datetime


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Найденный фрагмент и сведения о его источнике."""

    chunk_id: UUID
    document_id: UUID
    text: str
    metadata: CitationMetadata
    score: float = 0.0


class SearchBackend(Protocol):
    """Контракт независимого от БД источника результатов поиска."""

    async def semantic_search(
        self, vector: Sequence[float], *, year: int, limit: int
    ) -> Sequence[SearchHit]:
        """Ищет ближайшие векторные фрагменты."""

    async def keyword_search(self, query: str, *, year: int, limit: int) -> Sequence[SearchHit]:
        """Ищет совпадения полнотекстового индекса."""


def _citation(document: Document) -> CitationMetadata:
    return CitationMetadata(
        title=document.title,
        document_type=document.document_type,
        number=document.number,
        date=document.date,
        article=document.article,
        budget_level=document.budget_level,
        valid_year=tuple(document.valid_year),
        status=document.status,
        source_url=document.source_url,
        loaded_at=document.loaded_at,
    )


class SQLAlchemySearchBackend:
    """PostgreSQL backend с использованием HNSW и GIN индексов схемы T2."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def semantic_search(
        self, vector: Sequence[float], *, year: int, limit: int
    ) -> Sequence[SearchHit]:
        distance = cast(Chunk.embedding, HALFVEC(EMBEDDING_DIMENSIONS)).cosine_distance(
            list(vector)
        )
        return await self._run_search(
            distance.label("rank"),
            Document.valid_year.contains([year]),
            order_by=distance.asc(),
            limit=limit,
        )

    async def keyword_search(self, query: str, *, year: int, limit: int) -> Sequence[SearchHit]:
        config: ColumnElement[Any] = literal_column("'simple'::regconfig")
        document_vector = func.to_tsvector(config, Chunk.text)
        query_vector = func.plainto_tsquery(config, query)
        rank = func.ts_rank_cd(document_vector, query_vector)
        return await self._run_search(
            rank.label("rank"),
            Document.valid_year.contains([year]),
            document_vector.op("@@")(query_vector),
            order_by=rank.desc(),
            limit=limit,
        )

    async def _run_search(
        self,
        score_expression: ColumnElement[Any],
        *filters: ColumnElement[bool],
        order_by: ColumnElement[Any],
        limit: int,
    ) -> list[SearchHit]:
        statement = (
            select(Chunk, Document, score_expression)
            .join(Document, Chunk.document_id == Document.id)
            .where(*filters, Document.status == ACTIVE_STATUS)
            .order_by(order_by)
            .limit(limit)
        )
        async with self._sessions() as session:
            rows = typing_cast(
                list[tuple[Chunk, Document, float]],
                (await session.execute(statement)).all(),
            )
        return [
            SearchHit(
                chunk_id=chunk.id,
                document_id=document.id,
                text=chunk.text,
                metadata=_citation(document),
                score=float(score),
            )
            for chunk, document, score in rows
        ]


class HybridRetriever:
    """Объединяет семантическую и полнотекстовую выдачи и сортирует их повторно."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        backend: SearchBackend,
        settings: Settings | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._backend = backend
        self._settings = settings or get_settings()

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        year: int | None = None,
        active_only: bool = True,
    ) -> list[SearchHit]:
        """Возвращает top-k фрагментов, актуальных для заданного года по умолчанию."""
        if not query.strip():
            raise ValueError("Поисковый запрос не должен быть пустым")
        if top_k < 1:
            raise ValueError("top_k должен быть положительным")

        requested_year = year if year is not None else self._settings.current_fiscal_year
        vectors = await self._embeddings.embed([query])
        if len(vectors) != 1:
            raise ValueError("Embedding provider должен вернуть ровно один вектор")
        candidate_count = max(top_k * 4, top_k)
        semantic, keyword = await asyncio.gather(
            self._backend.semantic_search(vectors[0], year=requested_year, limit=candidate_count),
            self._backend.keyword_search(query, year=requested_year, limit=candidate_count),
        )

        fused: dict[UUID, tuple[SearchHit, float]] = {}
        for results in (semantic, keyword):
            for position, hit in enumerate(results, start=1):
                if requested_year not in hit.metadata.valid_year:
                    continue
                if active_only and hit.metadata.status != ACTIVE_STATUS:
                    continue
                prior = fused.get(hit.chunk_id)
                score = (prior[1] if prior else 0.0) + 1.0 / (_RRF_CONSTANT + position)
                fused[hit.chunk_id] = (hit, score)

        query_terms = set(_TOKEN_RE.findall(query.casefold()))
        reranked = [
            SearchHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                text=hit.text,
                metadata=hit.metadata,
                score=score + self._text_match_bonus(query_terms, hit),
            )
            for hit, score in fused.values()
        ]
        reranked.sort(key=lambda hit: (-hit.score, str(hit.chunk_id)))
        return reranked[:top_k]

    @staticmethod
    def _text_match_bonus(query_terms: set[str], hit: SearchHit) -> float:
        if not query_terms:
            return 0.0
        searchable = f"{hit.metadata.title} {hit.metadata.article or ''} {hit.text}".casefold()
        matched = sum(term in searchable for term in query_terms)
        return 0.05 * matched / len(query_terms)
