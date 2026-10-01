"""ORM-схема документов и индексируемых фрагментов базы знаний."""

from datetime import date as PythonDate
from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    select,
)
from sqlalchemy import (
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import Select
from sqlalchemy.types import Date

from app.knowledge.metadata import ACTIVE_STATUS

EMBEDDING_DIMENSIONS = 3072


class Base(DeclarativeBase):
    """Базовый класс ORM-моделей приложения."""


class Document(Base):
    """Исходный документ и его общие нормативные метаданные."""

    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_status_valid_year", "status", "valid_year"),)

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column("type", String(100), nullable=False)
    number: Mapped[str | None] = mapped_column(String(200))
    date: Mapped[PythonDate | None] = mapped_column(Date)
    article: Mapped[str | None] = mapped_column(Text)
    budget_level: Mapped[str | None] = mapped_column(String(100))
    valid_year: Mapped[list[int]] = mapped_column(
        PG_ARRAY(Integer),
        nullable=False,
        default=list,
        server_default=sql_text("'{}'::integer[]"),
    )
    status: Mapped[str] = mapped_column(
        String(100), nullable=False, default=ACTIVE_STATUS, server_default=ACTIVE_STATUS
    )
    source_url: Mapped[str | None] = mapped_column(Text)
    loaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """Фрагмент текста с вектором и копией метаданных для выдачи и цитирования."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            sql_text(f"(embedding::halfvec({EMBEDDING_DIMENSIONS})) halfvec_cosine_ops"),
            postgresql_using="hnsw",
        ),
        Index(
            "ix_chunks_text_gin",
            sql_text("to_tsvector('simple', text)"),
            postgresql_using="gin",
        ),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column("type", String(100), nullable=False)
    number: Mapped[str | None] = mapped_column(String(200))
    date: Mapped[PythonDate | None] = mapped_column(Date)
    article: Mapped[str | None] = mapped_column(Text)
    budget_level: Mapped[str | None] = mapped_column(String(100))
    valid_year: Mapped[list[int]] = mapped_column(
        PG_ARRAY(Integer),
        nullable=False,
        default=list,
        server_default=sql_text("'{}'::integer[]"),
    )
    status: Mapped[str] = mapped_column(
        String(100), nullable=False, default=ACTIVE_STATUS, server_default=ACTIVE_STATUS
    )
    source_url: Mapped[str | None] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")


def current_chunks_statement(year: int) -> Select[Chunk]:
    """Строит запрос только к фрагментам действующих в указанном году документов."""
    return (
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.status == ACTIVE_STATUS, Document.valid_year.contains([year]))
    )


def is_document_current(status: str, valid_years: list[int], year: int) -> bool:
    """Проверяет пригодность метаданных документа для актуальной выдачи."""
    return status == ACTIVE_STATUS and year in valid_years
