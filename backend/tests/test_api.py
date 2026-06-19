from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.rag_pipeline import RAGPipeline
from tests.helpers import ConstantTestEmbedder


FAQ_PATH = Path(__file__).resolve().parent.parent / "data" / "faq.md"


def install_test_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    app.state.rag_pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
        embedder=ConstantTestEmbedder(),
    )


def test_chat_endpoint_returns_grounded_answer(tmp_path, monkeypatch):
    install_test_pipeline(tmp_path, monkeypatch)
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


def test_chat_endpoint_applies_guardrails(tmp_path, monkeypatch):
    install_test_pipeline(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"question": "What is the weather in Bangkok today?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["in_scope"] is False
    assert payload["sources"] == []
