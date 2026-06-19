from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.rag_pipeline import RAGPipeline
from tests.helpers import CosineTestEmbedder


FAQ_PATH = Path(__file__).resolve().parent.parent / "data" / "faq.md"


def install_test_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.delenv("LLM_DEBUG_ENABLED", raising=False)
    monkeypatch.delenv("LLM_DEBUG_OUTPUT", raising=False)
    app.state.rag_pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
        embedder=CosineTestEmbedder(),
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
    assert payload["debug"] is None
    assert "jet lag" in payload["answer"].lower() or "travel" in payload["answer"].lower()


def test_chat_endpoint_returns_web_debug_when_requested(tmp_path, monkeypatch):
    install_test_pipeline(tmp_path, monkeypatch)
    monkeypatch.setenv("LLM_DEBUG_ENABLED", "true")
    monkeypatch.setenv("LLM_DEBUG_OUTPUT", "web")
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"question": "Can SleepPilot help with jet lag?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["retrieval_query"] == "Can SleepPilot help with jet lag?"


def test_chat_endpoint_omits_debug_for_server_only_output(tmp_path, monkeypatch, capsys):
    install_test_pipeline(tmp_path, monkeypatch)
    monkeypatch.setenv("LLM_DEBUG_ENABLED", "true")
    monkeypatch.setenv("LLM_DEBUG_OUTPUT", "server")
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"question": "Can SleepPilot help with jet lag?"},
    )

    assert response.status_code == 200
    assert response.json()["debug"] is None
    assert "SleepPilot LLM debug" in capsys.readouterr().out


def test_chat_endpoint_accepts_recent_history(tmp_path, monkeypatch):
    install_test_pipeline(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "question": "What about Garmin?",
            "history": [
                {
                    "role": "user",
                    "content": "Does SleepPilot support wearable devices?",
                },
                {
                    "role": "assistant",
                    "content": "SleepPilot supports common wearable platforms.",
                },
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"]


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
