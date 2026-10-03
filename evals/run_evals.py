"""Проверка схемы eval-кейсов; оценка ответов требует живой среды приложения."""

import json
import sys
from pathlib import Path
from typing import Any

CASES_PATH = Path(__file__).with_name("test_cases.yaml")


def load_cases() -> list[dict[str, Any]]:
    """Загружает JSON-совместимый YAML без дополнительной зависимости."""
    parsed = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("Eval-набор должен быть списком кейсов")
    required = {"id", "question", "expected_behavior", "citation_required", "refusal_required"}
    for index, case in enumerate(parsed):
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError(f"Некорректный eval-кейс с индексом {index}")
    return parsed


def main() -> None:
    """Проверяет набор и поясняет, где выполняется фактическая оценка."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    cases = load_cases()
    report = {
        "status": "not_run",
        "case_count": len(cases),
        "cases": [case["id"] for case in cases],
        "message": (
            "Фактический eval не запускался: он выполняется в живой среде с наполненной "
            "базой знаний, LLM-провайдером и адаптером сценариев."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
