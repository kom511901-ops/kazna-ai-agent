"""Базовая настройка логирования без записи содержимого диалогов."""

import logging


def configure_logging() -> None:
    """Настраивает стандартный вывод логов; PII и тексты запросов сюда не передаются."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
