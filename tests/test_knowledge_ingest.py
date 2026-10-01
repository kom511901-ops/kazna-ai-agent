"""Проверки парсинга, чанкинга, метаданных и идемпотентности ингеста."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from sqlalchemy.dialects import postgresql

from app.knowledge import loaders
from app.knowledge.ingest import KnowledgeIngestor, chunk_text
from app.knowledge.loaders import load_document
from app.knowledge.metadata import REPEALED_STATUS, DocumentMetadata
from app.knowledge.models import current_chunks_statement, is_document_current


class FakeEmbeddingProvider:
    """Простой embedding double для тестов без сетевых запросов."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index)] for index, _ in enumerate(texts)]


class FakeKnowledgeRepository:
    """In-memory репозиторий, имитирующий уникальность content hash."""

    def __init__(self) -> None:
        self.documents: dict[str, dict[str, object]] = {}

    async def contains_hash(self, content_hash: str) -> bool:
        return content_hash in self.documents

    async def save_document(
        self,
        source_path: Path,
        content_hash: str,
        metadata: DocumentMetadata,
        chunk_texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> bool:
        if content_hash in self.documents:
            return False
        self.documents[content_hash] = {
            "source_path": source_path,
            "metadata": metadata,
            "chunks": [
                {
                    "text": text,
                    "embedding": list(vector),
                    "metadata": metadata.model_dump(by_alias=True),
                }
                for text, vector in zip(chunk_texts, embeddings, strict=True)
            ],
        }
        return True


@pytest.mark.asyncio
async def test_txt_and_html_loaders_clean_extracted_text(tmp_path: Path) -> None:
    """TXT и HTML загружаются, лишние пробелы и скрипты убираются."""
    txt_path = tmp_path / "plain.txt"
    txt_path.write_text("Первая   строка\r\n\r\nВторая строка", encoding="utf-8")
    html_path = tmp_path / "page.html"
    html_path.write_text(
        "<html><script>ignored()</script><p>Текст&nbsp; страницы</p></html>",
        encoding="utf-8",
    )

    plain = await load_document(txt_path)
    page = await load_document(html_path)

    assert plain.text == "Первая строка\n\nВторая строка"
    assert "ignored" not in page.text
    assert "Текст" in page.text


@pytest.mark.asyncio
async def test_docx_loader_extracts_paragraphs(tmp_path: Path) -> None:
    """DOCX-парсер извлекает текст абзацев."""
    from docx import Document as WordDocument

    document = WordDocument()
    document.add_paragraph("Параграф для теста")
    path = tmp_path / "source.docx"
    document.save(path)

    loaded = await load_document(path)

    assert loaded.text == "Параграф для теста"


@pytest.mark.asyncio
async def test_pdf_loader_extracts_page_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PDF-ветка загрузчика извлекает текст страниц."""

    class Page:
        def extract_text(self) -> str:
            return "Содержимое PDF"

    class Reader:
        def __init__(self, _: object) -> None:
            self.pages = [Page()]

    monkeypatch.setattr(loaders, "PdfReader", Reader)
    path = tmp_path / "source.pdf"
    path.write_bytes(b"fixture")

    loaded = await load_document(path)

    assert loaded.text == "Содержимое PDF"


@pytest.mark.asyncio
async def test_ingest_stores_chunk_metadata_and_skips_repeated_hash(tmp_path: Path) -> None:
    """Чанки сохраняют метаданные, повторный импорт не создаёт дубли."""
    path = tmp_path / "order.txt"
    path.write_text("Текст документа для загрузки.", encoding="utf-8")
    metadata = DocumentMetadata(
        title="Порядок казначейского сопровождения",
        type="приказ",
        number="пример",
        date="2026-01-15",
        article="раздел I",
        budget_level="федеральный",
        valid_year=[2026],
        source_url="https://example.invalid/source",
    )
    embeddings = FakeEmbeddingProvider()
    repository = FakeKnowledgeRepository()
    ingestor = KnowledgeIngestor(embeddings, repository, chunk_tokens=20, chunk_overlap_tokens=2)

    first = await ingestor.ingest_file(path, metadata)
    second = await ingestor.ingest_file(path, metadata)

    assert first.skipped_as_duplicate is False
    assert first.chunk_count == 1
    assert second.skipped_as_duplicate is True
    assert len(repository.documents) == 1
    assert len(embeddings.calls) == 1
    saved = next(iter(repository.documents.values()))
    assert saved["metadata"] == metadata
    assert saved["chunks"]
    first_chunk = saved["chunks"][0]
    assert first_chunk["metadata"]["title"] == metadata.title
    assert first_chunk["metadata"]["type"] == metadata.document_type
    assert first_chunk["metadata"]["valid_year"] == [2026]
    assert first_chunk["metadata"]["status"] == "действует"
    assert first_chunk["embedding"]


@pytest.mark.asyncio
async def test_sidecar_metadata_is_loaded(tmp_path: Path) -> None:
    """Sidecar позволяет задать нормативные метаданные явно."""
    path = tmp_path / "order.txt"
    path.write_text("Текст.", encoding="utf-8")
    sidecar = tmp_path / "order.metadata.json"
    sidecar.write_text(
        json.dumps(
            {
                "title": "Приказ",
                "type": "приказ",
                "number": "45",
                "date": "2026-02-01",
                "valid_year": [2026],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    repository = FakeKnowledgeRepository()
    ingestor = KnowledgeIngestor(FakeEmbeddingProvider(), repository)

    await ingestor.ingest_file(path)

    stored = next(iter(repository.documents.values()))
    metadata = stored["metadata"]
    assert isinstance(metadata, DocumentMetadata)
    assert metadata.document_type == "приказ"
    assert metadata.number == "45"
    assert metadata.valid_year == [2026]


def test_token_chunking_overlaps_neighboring_chunks() -> None:
    """Токеновые фрагменты создаются с заданным перекрытием."""
    text = "This sentence has a number of words that should form multiple chunks."

    chunks = chunk_text(text, chunk_tokens=5, overlap_tokens=2)

    assert len(chunks) > 1
    assert all(chunks)


def test_repealed_document_is_excluded_from_current_chunks_query() -> None:
    """Текущий поиск фильтрует утратившие силу документы и другой год."""
    statement = current_chunks_statement(2026).compile(dialect=postgresql.dialect())
    sql = str(statement)

    assert "documents.status =" in sql
    assert "documents.valid_year @>" in sql
    assert REPEALED_STATUS != statement.params["status_1"]
    assert is_document_current(REPEALED_STATUS, [2026], 2026) is False
    assert is_document_current("действует", [2026], 2026) is True
    assert is_document_current("действует", [2025], 2026) is False
