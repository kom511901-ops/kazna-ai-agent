"""Создание таблиц документов и фрагментов базы знаний."""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "20261001_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Создаёт расширение pgvector, таблицы и поисковые индексы."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("type", sa.String(length=100), nullable=False),
        sa.Column("number", sa.String(length=200), nullable=True),
        sa.Column("date", sa.Date(), nullable=True),
        sa.Column("article", sa.Text(), nullable=True),
        sa.Column("budget_level", sa.String(length=100), nullable=True),
        sa.Column(
            "valid_year",
            sa.ARRAY(sa.Integer()),
            server_default=sa.text("'{}'::integer[]"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=100), server_default="действует", nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column(
            "loaded_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("content_hash", name="uq_documents_content_hash"),
    )
    op.create_index(
        "ix_documents_status_valid_year", "documents", ["status", "valid_year"], unique=False
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(3072), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("type", sa.String(length=100), nullable=False),
        sa.Column("number", sa.String(length=200), nullable=True),
        sa.Column("date", sa.Date(), nullable=True),
        sa.Column("article", sa.Text(), nullable=True),
        sa.Column("budget_level", sa.String(length=100), nullable=True),
        sa.Column(
            "valid_year",
            sa.ARRAY(sa.Integer()),
            server_default=sa.text("'{}'::integer[]"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=100), server_default="действует", nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        [sa.text("(embedding::halfvec(3072)) halfvec_cosine_ops")],
        postgresql_using="hnsw",
    )
    op.create_index(
        "ix_chunks_text_gin",
        "chunks",
        [sa.text("to_tsvector('simple', text)")],
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Удаляет таблицы и расширение, если оно больше не используется."""
    op.drop_index("ix_chunks_text_gin", table_name="chunks")
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_documents_status_valid_year", table_name="documents")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
