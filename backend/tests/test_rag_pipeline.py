from pathlib import Path

import pytest

from app.embeddings import HashingEmbedder
from app.faq_loader import load_and_chunk_faq
from app.rag_pipeline import RAGPipeline
from app.vector_store import SQLiteVectorStore


FAQ_PATH = Path(__file__).resolve().parent.parent / "data" / "faq.md"


def make_test_store(tmp_path) -> SQLiteVectorStore:
    return SQLiteVectorStore(
        tmp_path / "vectors.sqlite3",
        embedder=HashingEmbedder(),
    )


@pytest.mark.parametrize(
    ("question", "expected_chunk_id"),
    [
        ("What is this thing?", "faq-001"),
        ("How does SleepPilot improve my sleep?", "faq-002"),
        ("Can it diagnose insomnia or sleep disorders?", "faq-003"),
        ("What data does the product collect?", "faq-004"),
        ("Can I use it without a wearable?", "faq-005"),
        ("Does it support Garmin or Fitbit?", "faq-006"),
        ("How does SleepPilot protect my privacy?", "faq-007"),
        ("Can it help with jet lag when I travel?", "faq-008"),
        ("What is the sleep score?", "faq-009"),
        ("Does it have a smart alarm?", "faq-010"),
        ("Can it help me plan when to go to bed?", "faq-011"),
        ("Is it useful for students during exams?", "faq-012"),
        ("Can it help if I sleep too late every night?", "faq-013"),
        ("How much does premium cost?", "faq-014"),
        ("What if SleepPilot cannot answer my question?", "faq-015"),
    ],
)
def test_vector_store_routes_diverse_questions_to_expected_chunks(
    tmp_path,
    question,
    expected_chunk_id,
):
    chunks = load_and_chunk_faq(FAQ_PATH)
    store = make_test_store(tmp_path)
    store.ensure_built(chunks)

    results = store.similarity_search(question, top_k=3)

    assert results[0].chunk.id == expected_chunk_id


def test_vector_store_retrieves_pricing_chunk(tmp_path):
    chunks = load_and_chunk_faq(FAQ_PATH)
    store = make_test_store(tmp_path)
    store.ensure_built(chunks)

    results = store.similarity_search("How much does the premium plan cost?", top_k=3)

    assert results[0].chunk.id == "faq-014"
    assert results[0].score > 0
    assert "cost" in results[0].chunk.question.lower()


def test_vector_store_retrieves_product_overview_for_generic_product_question(tmp_path):
    chunks = load_and_chunk_faq(FAQ_PATH)
    store = make_test_store(tmp_path)
    store.ensure_built(chunks)

    results = store.similarity_search("What is the product?", top_k=3)

    assert results[0].chunk.id == "faq-001"
    assert results[0].score > results[1].score


def test_vector_store_retrieves_product_overview_for_this_thing_question(tmp_path):
    chunks = load_and_chunk_faq(FAQ_PATH)
    store = make_test_store(tmp_path)
    store.ensure_built(chunks)

    results = store.similarity_search("What is this thing?", top_k=3)

    assert results[0].chunk.id == "faq-001"


def test_vector_store_retrieves_sleep_improvement_chunk(tmp_path):
    chunks = load_and_chunk_faq(FAQ_PATH)
    store = make_test_store(tmp_path)
    store.ensure_built(chunks)

    results = store.similarity_search("How does SleepPilot improve my sleep?", top_k=3)

    assert results[0].chunk.id == "faq-002"


def test_vector_store_retrieves_bedtime_routine_for_go_to_bed_question(tmp_path):
    chunks = load_and_chunk_faq(FAQ_PATH)
    store = make_test_store(tmp_path)
    store.ensure_built(chunks)

    results = store.similarity_search("Can it help me plan when to go to bed?", top_k=3)

    assert results[0].chunk.id == "faq-011"
    assert "faq-009" not in [result.chunk.id for result in results]


def test_vector_store_retrieves_data_collection_before_privacy(tmp_path):
    chunks = load_and_chunk_faq(FAQ_PATH)
    store = make_test_store(tmp_path)
    store.ensure_built(chunks)

    results = store.similarity_search("What data does SleepPilot collect?", top_k=3)

    assert results[0].chunk.id == "faq-004"


def test_vector_store_distinguishes_wearable_support_from_no_wearable(tmp_path):
    chunks = load_and_chunk_faq(FAQ_PATH)
    store = make_test_store(tmp_path)
    store.ensure_built(chunks)

    support_results = store.similarity_search("Does SleepPilot work with Garmin or Fitbit?", top_k=3)
    no_wearable_results = store.similarity_search("Can I use SleepPilot without a wearable?", top_k=3)

    assert support_results[0].chunk.id == "faq-006"
    assert no_wearable_results[0].chunk.id == "faq-005"


def test_rag_pipeline_answers_in_scope_question(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
    )

    result = pipeline.answer("Does SleepPilot work with Garmin or Fitbit?")

    assert result.in_scope is True
    assert result.sources
    assert result.mode == "local-rag"
    assert "Garmin" in result.answer or "Fitbit" in result.answer


def test_rag_pipeline_sends_only_clean_context_for_clear_match(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
    )

    result = pipeline.answer("What is this thing?")

    assert [source.id for source in result.sources] == ["faq-001"]


def test_rag_pipeline_can_send_three_chunks_for_ambiguous_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
    )
    results = pipeline.retrieve("sleep habits and bedtime schedule", top_k=4)

    selected = pipeline._select_context_results(results)

    assert 1 <= len(selected) <= 3


def test_rag_pipeline_removes_model_citation_markers(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
    )

    assert pipeline._clean_answer("Yes, SleepPilot can help. [2]") == "Yes, SleepPilot can help."


def test_rag_pipeline_declines_out_of_scope_question(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
    )

    result = pipeline.answer("Write Python code for a todo app")

    assert result.in_scope is False
    assert result.sources == []
    assert "SleepPilot" in result.answer
