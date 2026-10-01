"""Фабрика провайдеров на основе настроек приложения."""

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.providers.base import EmbeddingProvider, LLMProvider
from app.providers.openai_provider import OpenAIProvider


@dataclass(frozen=True, slots=True)
class ProviderBundle:
    """Набор провайдеров, совместно использующих HTTP-клиент."""

    llm: LLMProvider
    embeddings: EmbeddingProvider


def create_providers(settings: Settings | None = None) -> ProviderBundle:
    """Создаёт настроенные провайдеры или сообщает о неподдержанном backend."""
    current_settings = settings or get_settings()
    provider_name = current_settings.llm_provider.casefold()
    if provider_name != "openai":
        raise ValueError(f"Неподдерживаемый провайдер моделей: {current_settings.llm_provider}")

    provider = OpenAIProvider(current_settings)
    return ProviderBundle(llm=provider, embeddings=provider)
