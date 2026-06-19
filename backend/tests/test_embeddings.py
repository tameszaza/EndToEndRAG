import pytest

from app.embeddings import (
    LocalSentenceTransformerEmbedder,
    create_default_embedder,
)


def test_create_default_embedder_uses_local_model_by_default(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    monkeypatch.setenv("LOCAL_EMBEDDING_DIMENSIONS", "384")

    embedder = create_default_embedder()

    assert isinstance(embedder, LocalSentenceTransformerEmbedder)
    assert embedder.model_name == "intfloat/multilingual-e5-small"
    assert embedder.dimensions == 384
    assert "local-sentence-transformers:intfloat/multilingual-e5-small" in embedder.namespace


def test_create_default_embedder_rejects_hosted_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hosted")

    with pytest.raises(RuntimeError, match="EMBEDDING_PROVIDER must be one of"):
        create_default_embedder()


def test_local_embedder_uses_retrieval_prompts(monkeypatch):
    captured = {}

    class FakeModel:
        prompts = {"query": "Query: ", "document": "Document: "}

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, **kwargs):
            captured.update(kwargs)
            return [3.0, 4.0]

    embedder = LocalSentenceTransformerEmbedder(
        model_name="test-model",
        query_prompt_name="query",
        document_prompt_name="document",
    )
    embedder._model = FakeModel()

    vector = embedder.embed("hello", task_type="RETRIEVAL_QUERY")

    assert captured["sentences"] == "hello"
    assert captured["prompt_name"] == "query"
    assert captured["normalize_embeddings"] is True
    assert vector == [0.6, 0.8]


def test_local_embedder_prefixes_prompt_when_model_has_no_registered_prompts(monkeypatch):
    captured = {}

    class FakeModel:
        def encode(self, **kwargs):
            captured.update(kwargs)
            return [3.0, 4.0]

    embedder = LocalSentenceTransformerEmbedder(
        model_name="test-model",
        document_prompt_name="passage",
    )
    embedder._model = FakeModel()

    embedder.embed("FAQ text", task_type="RETRIEVAL_DOCUMENT")

    assert captured["sentences"] == "passage: FAQ text"
    assert "prompt_name" not in captured
