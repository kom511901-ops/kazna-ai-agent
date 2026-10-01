# Консультант по казначейскому сопровождению

Каркас проекта. Для локального запуска скопируйте `.env.example` в `.env`, заполните необходимые значения и выполните:

```shell
docker compose up --build
```

Проверка доступности: `GET http://localhost:8000/health`.

Локально без Docker:

```shell
python -m pip install -e '.[dev]'
uvicorn app.api.main:app --reload
```

Проверки качества:

```shell
ruff check .
mypy app
pytest
```

## Загрузка документов базы знаний

Примените миграции перед первой загрузкой:

```shell
alembic upgrade head
```

Загрузите поддерживаемые файлы из каталога (включая вложенные каталоги):

```shell
python -m app.knowledge.ingest /data/sources
```

Поддерживаются PDF, DOCX, HTML и TXT. Рядом с файлом можно положить JSON sidecar с теми же базовыми именем и расширением `.metadata.json`, например `document.metadata.json` для `document.pdf`. В sidecar можно указать `title`, `type`, `number`, `date`, `article`, `budget_level`, `valid_year`, `status` и `source_url`. Если sidecar отсутствует, название берётся из имени файла, а остальные поля остаются неуказанными. Повторная загрузка такого же очищенного текста пропускается по SHA-256.
