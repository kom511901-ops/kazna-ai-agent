"""Схема метаданных документов базы знаний."""

from datetime import date as Date

from pydantic import BaseModel, ConfigDict, Field

ACTIVE_STATUS = "действует"
REPEALED_STATUS = "утратил силу"


class DocumentMetadata(BaseModel):
    """Метаданные НПА, доступные для цитирования и фильтрации."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    title: str
    document_type: str = Field(alias="type")
    number: str | None = None
    date: Date | None = None
    article: str | None = None
    budget_level: str | None = None
    valid_year: list[int] = Field(default_factory=list)
    status: str = ACTIVE_STATUS
    source_url: str | None = None
