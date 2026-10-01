"""Опциональный короткий smoke test реального OpenAI API."""

import pytest

from app.core.config import get_settings
from app.providers.base import LLMResponse
from app.providers.openai_provider import OpenAIProvider

pytestmark = pytest.mark.skipif(
    not get_settings().run_live_tests,
    reason="Живые проверки включаются переменной RUN_LIVE_TESTS=true",
)


@pytest.mark.asyncio
async def test_openai_chat_and_embedding_smoke() -> None:
    """Проверяет короткий запрос к модели и создание эмбеддинга."""
    provider = OpenAIProvider(get_settings())
    try:
        response = await provider.chat([{"role": "user", "content": "Ответь одним словом: OK"}])
        vectors = await provider.embed(["казначейское сопровождение"])
    finally:
        await provider.aclose()

    assert isinstance(response, LLMResponse)
    assert response.content
    assert len(vectors) == 1
    assert vectors[0]
