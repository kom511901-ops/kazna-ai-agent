"""Точка входа FastAPI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Настраивает инфраструктуру при запуске приложения."""
    configure_logging()
    yield


app = FastAPI(title="Консультант по казначейскому сопровождению", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Возвращает состояние процесса приложения."""
    return {"status": "ok"}
