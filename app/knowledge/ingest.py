"""Ингест документов: извлечение, чанкинг, эмбеддинг и запись в PostgreSQL."""

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from app.core.config import Settings, get_settings
from app.knowledge.database import create_session_factory
from app.knowledge.loaders import LoadedDocument, discover_documents, load_document
from app.knowledge.metadata import DocumentMetadata
from app.knowledge.repository import (
    KnowledgeRepository,
    SQLAlchemyKnowledgeRepository,
    content_hash,
)
from app.providers.base import EmbeddingProvider
from app.providers.factory import ProviderBundle, create_providers


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Результат обработки одного файла."""

    source_path: Path
    content_hash: str
    chunk_count: int
    skipped_as_duplicate: bool


class KnowledgeIngestor:
    """Обрабатывает файлы и передаёт результаты в репозиторий."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        repository: KnowledgeRepository,
        *,
        chunk_tokens: int = 700,
        chunk_overlap_tokens: int = 100,
    ) -> None:
        if chunk_tokens < 1:
            raise ValueError("Размер чанка должен быть положительным")
        if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_tokens:
            raise ValueError("Перекрытие должно быть неотрицательным и меньше размера чанка")
        self._embedding_provider = embedding_provider
        self._repository = repository
        self._chunk_tokens = chunk_tokens
        self._chunk_overlap_tokens = chunk_overlap_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    async def ingest_file(
        self,
        path: Path,
        metadata: DocumentMetadata | None = None,
    ) -> IngestResult:
        """Ингестит один документ с идемпотентностью по хэшу очищенного текста."""
        document = await load_document(path)
        resolved_metadata = metadata or await load_metadata(document.source_path)
        return await self.ingest_loaded(document, resolved_metadata)

    async def ingest_loaded(
        self,
        document: LoadedDocument,
        metadata: DocumentMetadata,
    ) -> IngestResult:
        """Вычисляет хэш, разбивает текст, получает эмбеддинги и сохраняет записи."""
        document_hash = content_hash(document.text)
        if await self._repository.contains_hash(document_hash):
            return IngestResult(document.source_path, document_hash, 0, True)

        chunks = chunk_text(
            document.text,
            encoding=self._encoding,
            chunk_tokens=self._chunk_tokens,
            overlap_tokens=self._chunk_overlap_tokens,
        )
        embeddings = await self._embedding_provider.embed(chunks)
        if len(embeddings) != len(chunks):
            raise ValueError("Провайдер вернул число embeddings, не совпадающее с числом чанков")

        saved = await self._repository.save_document(
            document.source_path,
            document_hash,
            metadata,
            chunks,
            embeddings,
        )
        stored_chunk_count = len(chunks) if saved else 0
        return IngestResult(document.source_path, document_hash, stored_chunk_count, not saved)


def chunk_text(
    text: str,
    *,
    encoding: tiktoken.Encoding | None = None,
    chunk_tokens: int = 700,
    overlap_tokens: int = 100,
) -> list[str]:
    """Делит текст по токенам с перекрытием соседних фрагментов."""
    if chunk_tokens < 1:
        raise ValueError("Размер чанка должен быть положительным")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("Перекрытие должно быть неотрицательным и меньше размера чанка")
    if not text.strip():
        return []

    tokenizer = encoding or tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    step = chunk_tokens - overlap_tokens
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        end = min(start + chunk_tokens, len(tokens))
        part = tokenizer.decode(tokens[start:end]).strip()
        if part:
            chunks.append(part)
        if end == len(tokens):
            break
    return chunks


async def load_metadata(path: Path) -> DocumentMetadata:
    """Читает соседний JSON sidecar или формирует пустые метаданные по умолчанию."""
    sidecar = path.with_suffix(".metadata.json")

    def read_sidecar() -> dict[str, object] | None:
        if not sidecar.is_file():
            return None
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Ожидался JSON-объект метаданных: {sidecar}")
        return payload

    payload = await asyncio.to_thread(read_sidecar)
    if payload is not None:
        return DocumentMetadata.model_validate(payload)
    return DocumentMetadata.model_validate({"title": path.stem, "type": "не указан"})


async def ingest_directory(
    source: Path,
    *,
    settings: Settings | None = None,
    providers: ProviderBundle | None = None,
    repository: KnowledgeRepository | None = None,
) -> list[IngestResult]:
    """Ингестит поддерживаемые файлы из каталога или единственного пути."""
    current_settings = settings or get_settings()
    owned_providers = providers is None
    provider_bundle = providers or create_providers(current_settings)
    engine = None
    if repository is None:
        sessions, engine = create_session_factory(current_settings.database_url)
        current_repository: KnowledgeRepository = SQLAlchemyKnowledgeRepository(sessions)
    else:
        current_repository = repository

    ingestor = KnowledgeIngestor(
        provider_bundle.embeddings,
        current_repository,
        chunk_tokens=current_settings.ingest_chunk_tokens,
        chunk_overlap_tokens=current_settings.ingest_chunk_overlap_tokens,
    )
    try:
        paths = await discover_documents(source)
        results: list[IngestResult] = []
        for path in paths:
            results.append(await ingestor.ingest_file(path))
        return results
    finally:
        if engine is not None:
            await engine.dispose()
        if owned_providers:
            await provider_bundle.llm.aclose()


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Загрузка документов в базу знаний")
    parser.add_argument("source", type=Path, help="Файл или каталог с исходными документами")
    return parser.parse_args(arguments)


async def _run_cli(source: Path) -> None:
    results = await ingest_directory(source)
    added = sum(not item.skipped_as_duplicate for item in results)
    duplicates = len(results) - added
    print(f"Обработано: {len(results)}; добавлено: {added}; пропущено-дубликатов: {duplicates}")


def main() -> None:
    """Точка входа CLI `python -m app.knowledge.ingest`."""
    arguments = _parse_args()
    asyncio.run(_run_cli(arguments.source))


if __name__ == "__main__":
    main()
