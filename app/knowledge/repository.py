"""Слой хранения документов и чанков в PostgreSQL."""

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.knowledge.metadata import DocumentMetadata
from app.knowledge.models import Chunk, Document, current_chunks_statement


class KnowledgeRepository(Protocol):
    """Минимальный контракт постоянного хранилища базы знаний."""

    async def contains_hash(self, content_hash: str) -> bool:
        """Проверяет, хранится ли уже документ с этим содержимым."""

    async def save_document(
        self,
        source_path: Path,
        content_hash: str,
        metadata: DocumentMetadata,
        chunk_texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> bool:
        """Сохраняет документ и чанки атомарно; False означает дубликат."""


class SQLAlchemyKnowledgeRepository:
    """Реализация репозитория на SQLAlchemy async sessions."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def contains_hash(self, content_hash: str) -> bool:
        """Проверяет уникальный хэш документа."""
        async with self._sessions() as session:
            result = await session.scalar(
                select(Document.id).where(Document.content_hash == content_hash)
            )
            return result is not None

    async def save_document(
        self,
        source_path: Path,
        content_hash: str,
        metadata: DocumentMetadata,
        chunk_texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> bool:
        """Сохраняет документ и его чанки в одной транзакции."""
        if len(chunk_texts) != len(embeddings):
            raise ValueError("Каждому чанку должен соответствовать один embedding")

        async with self._sessions() as session:
            existing = await session.scalar(
                select(Document.id).where(Document.content_hash == content_hash)
            )
            if existing is not None:
                return False

            document = Document(
                content_hash=content_hash,
                source_path=str(source_path),
                title=metadata.title,
                document_type=metadata.document_type,
                number=metadata.number,
                date=metadata.date,
                article=metadata.article,
                budget_level=metadata.budget_level,
                valid_year=metadata.valid_year,
                status=metadata.status,
                source_url=metadata.source_url,
            )
            try:
                session.add(document)
                await session.flush()

                chunk_rows = [
                    Chunk(
                        document_id=document.id,
                        chunk_index=index,
                        text=chunk_text,
                        embedding=list(embedding),
                        title=metadata.title,
                        document_type=metadata.document_type,
                        number=metadata.number,
                        date=metadata.date,
                        article=metadata.article,
                        budget_level=metadata.budget_level,
                        valid_year=metadata.valid_year,
                        status=metadata.status,
                        source_url=metadata.source_url,
                    )
                    for index, (chunk_text, embedding) in enumerate(
                        zip(chunk_texts, embeddings, strict=True)
                    )
                ]
                session.add_all(chunk_rows)
                await session.commit()
            except IntegrityError:
                await session.rollback()
                if await session.scalar(
                    select(Document.id).where(Document.content_hash == content_hash)
                ) is not None:
                    return False
                raise
        return True

    async def current_chunks(self, year: int) -> list[Chunk]:
        """Возвращает только чанки действующих документов текущего года."""
        async with self._sessions() as session:
            result = await session.scalars(current_chunks_statement(year))
            return list(result.all())


def content_hash(text: str) -> str:
    """Вычисляет SHA-256 по нормализованному содержимому документа."""
    return sha256(text.encode("utf-8")).hexdigest()
