"""Цикл взаимодействия модели с инструментами и оформление источников ответа."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOL_DEFINITIONS, AgentTools
from app.providers.base import JsonValue, LLMProvider, LLMResponse

NO_SOURCE_RESPONSE = (
    "В базе знаний нет подтверждённого источника для ответа на этот вопрос. "
    "Пожалуйста, уточните информацию в территориальном органе Федерального казначейства (ТОФК)."
)
MAX_TOOL_ROUNDS = 8


@dataclass(frozen=True, slots=True)
class UserProfile:
    """Минимальный профиль пользователя для настройки подачи ответа."""

    role: str | None = None


@dataclass(frozen=True, slots=True)
class AgentReply:
    """Текст агента и источники, на которых он основан."""

    answer: str
    sources: tuple[Mapping[str, JsonValue], ...] = ()
    needs_clarification: bool = False


class AgentOrchestrator:
    """Исполняет function calls провайдера и возвращает ответ с собранными цитатами."""

    def __init__(self, llm: LLMProvider, tools: AgentTools) -> None:
        self._llm = llm
        self._tools = tools

    async def respond(
        self,
        message: str,
        *,
        profile: UserProfile | None = None,
        history: Sequence[Mapping[str, JsonValue]] = (),
    ) -> AgentReply:
        """Отвечает по источникам, полученным только в ходе этой беседы."""
        if not message.strip():
            raise ValueError("Сообщение пользователя не должно быть пустым")
        system_prompt = SYSTEM_PROMPT
        if profile and profile.role:
            system_prompt += f"\n\nРоль пользователя для адаптации ответа: {profile.role}."

        messages: list[Mapping[str, JsonValue]] = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": message},
        ]
        sources: dict[str, Mapping[str, JsonValue]] = {}
        lead_result: Mapping[str, object] | None = None

        for _ in range(MAX_TOOL_ROUNDS):
            result = await self._llm.chat(messages, tools=TOOL_DEFINITIONS)
            if not isinstance(result, LLMResponse):
                raise TypeError("Оркестратор ожидает обычный ответ, не потоковый режим")
            if not result.tool_calls:
                if not sources:
                    if lead_result is not None:
                        if lead_result.get("status") == "created":
                            return AgentReply("Запрос на связь со специалистом передан.")
                        if lead_result.get("status") == "not_created":
                            reason = lead_result.get("reason", "причина не указана")
                            return AgentReply(f"Запрос не передан: {reason}.")
                    return AgentReply(NO_SOURCE_RESPONSE)
                answer = result.content.strip() or "Сведения приведены в найденных источниках."
                answer = self._append_citations(answer, tuple(sources.values()))
                return AgentReply(answer, tuple(sources.values()))

            messages.extend(
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                }
                for call in result.tool_calls
            )
            for call in result.tool_calls:
                tool_result = await self._tools.execute(call.name, call.arguments)
                try:
                    parsed = json.loads(tool_result)
                except json.JSONDecodeError:
                    parsed = {"error": "Инструмент вернул неверный JSON"}
                if isinstance(parsed, dict):
                    self._collect_sources(parsed, sources)
                    if call.name == "create_lead":
                        lead_result = parsed
                    missing = parsed.get("missing")
                    if parsed.get("status") == "needs_clarification" and isinstance(missing, list):
                        question = self._clarification_question(missing)
                        return AgentReply(question, tuple(sources.values()), True)
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": tool_result,
                    }
                )

        raise RuntimeError("Модель превысила допустимое число циклов вызова инструментов")

    @staticmethod
    def _collect_sources(
        result: Mapping[str, object], sources: dict[str, Mapping[str, JsonValue]]
    ) -> None:
        found = result.get("sources")
        if not isinstance(found, list):
            return
        for item in found:
            if not isinstance(item, dict):
                continue
            chunk_id = item.get("chunk_id")
            if isinstance(chunk_id, str):
                sources[chunk_id] = item

    @staticmethod
    def _clarification_question(missing: list[object]) -> str:
        details = ", ".join(item for item in missing if isinstance(item, str))
        return (
            "Чтобы проверить применимость казначейского сопровождения, уточните, "
            f"пожалуйста: {details}."
        )

    @staticmethod
    def _append_citations(answer: str, sources: tuple[Mapping[str, JsonValue], ...]) -> str:
        citations: list[str] = []
        for source in sources:
            citation = source.get("citation")
            if not isinstance(citation, dict):
                continue
            title = citation.get("title")
            number = citation.get("number")
            date = citation.get("date")
            years = citation.get("valid_year")
            url = citation.get("source_url")
            label = str(title or "Источник")
            if number:
                label += f" № {number}"
            edition = str(
                date or (", ".join(map(str, years)) if isinstance(years, list) else "год не указан")
            )
            if url:
                label += f" ({edition}; {url})"
            else:
                label += f" ({edition})"
            citations.append(f"- {label}")
        if not citations:
            return answer
        return f"{answer}\n\nИсточники:\n" + "\n".join(citations)
