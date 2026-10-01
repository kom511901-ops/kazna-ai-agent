"""Проверки минимального HTTP API."""

from fastapi.testclient import TestClient

from app.api.main import app


def test_health_returns_ok() -> None:
    """Health endpoint отвечает успешным статусом."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
