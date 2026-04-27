from fastapi.testclient import TestClient

from app.main import app


def test_chat_endpoint_returns_grounded_answer(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    if hasattr(app.state, "rag_pipeline"):
        delattr(app.state, "rag_pipeline")
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"question": "Can SleepPilot help with jet lag?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["in_scope"] is True
    assert payload["answer"]
    assert payload["sources"]
    assert payload["mode"] == "local-rag"
    assert "jet lag" in payload["answer"].lower() or "travel" in payload["answer"].lower()


def test_chat_endpoint_applies_guardrails(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    if hasattr(app.state, "rag_pipeline"):
        delattr(app.state, "rag_pipeline")
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"question": "What is the weather in Bangkok today?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["in_scope"] is False
    assert payload["sources"] == []
