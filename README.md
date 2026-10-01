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
