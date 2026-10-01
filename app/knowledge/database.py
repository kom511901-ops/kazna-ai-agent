"""Создание async engine и фабрики сессий PostgreSQL."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


def create_session_factory(
    database_url: str | None = None,
) -> tuple[async_sessionmaker[AsyncSession], AsyncEngine]:
    """Создаёт фабрику сессий и engine для выбранной БД."""
    resolved_url = database_url or get_settings().database_url
    engine = create_async_engine(resolved_url, pool_pre_ping=True)
    return async_sessionmaker(engine, expire_on_commit=False), engine
