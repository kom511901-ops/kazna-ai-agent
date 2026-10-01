"""Настройки приложения из переменных окружения."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Параметры среды выполнения приложения."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    llm_provider: str = "openai"
    llm_model: str = "gpt-5.6-terra"
    embed_model: str = "text-embedding-3-large"
    openai_timeout_seconds: float = Field(default=30.0, gt=0)
    openai_max_retries: int = Field(default=2, ge=0)
    openai_retry_backoff_seconds: float = Field(default=0.5, ge=0)
    database_url: str = "postgresql+asyncpg://user:pass@db:5432/ks"
    telegram_bot_token: str = ""
    current_fiscal_year: int = Field(default=2026, ge=2000)
    kb_actual_as_of: str = "2026-09-30"
    run_live_tests: bool = False


@lru_cache
def get_settings() -> Settings:
    """Возвращает кэшированный экземпляр настроек."""
    return Settings()
