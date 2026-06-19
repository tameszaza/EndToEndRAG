from pathlib import Path

import pytest

from app.conversation import ConversationTurn, HISTORY_INPUT_CHAR_LIMIT, clean_history
from app.context_selector import select_context_results
from app.faq_loader import load_and_chunk_faq
from app.llm_client import LLMResponse
from app.rag_pipeline import RAGPipeline
from app.vector_store import SQLiteVectorStore
from tests.helpers import CosineTestEmbedder


FAQ_PATH = Path(__file__).resolve().parent.parent / "data" / "faq.md"


def make_test_store(tmp_path) -> SQLiteVectorStore:
    return SQLiteVectorStore(
        tmp_path / "vectors.sqlite3",
        embedder=CosineTestEmbedder(),
    )


class RecordingLLMClient:
    is_configured = True

    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 420,
    ) -> LLMResponse:
        self.calls.append(("rewrite", user_prompt))
        if "Rewrite the user's current question" in system_prompt:
            return LLMResponse(
                text="Does SleepPilot support Garmin or Fitbit?",
                provider="rewrite-test",
            )
        if "You rewrite follow-up messages" in system_prompt:
            return LLMResponse(
                text="Does SleepPilot support Garmin or Fitbit?",
                provider="rewrite-test",
            )
        return LLMResponse(text="Grounded answer from test LLM.", provider="answer-test")

    def complete_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 420,
    ) -> LLMResponse:
        if "You rewrite follow-up messages" in messages[0]["content"]:
            self.calls.append(("rewrite", messages))
            return LLMResponse(
                text="Does SleepPilot support Garmin or Fitbit?",
                provider="rewrite-test",
            )
        self.calls.append(("answer", messages))
        return LLMResponse(text="Grounded answer from test LLM.", provider="answer-test")


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
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
        embedder=CosineTestEmbedder(),
    )

    result = pipeline.answer("Does SleepPilot work with Garmin or Fitbit?")

    assert result.in_scope is True
    assert result.sources
    assert result.mode == "local-rag"
    assert "Garmin" in result.answer or "Fitbit" in result.answer


def test_rag_pipeline_skips_query_rewrite_when_history_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    llm_client = RecordingLLMClient()
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
        embedder=CosineTestEmbedder(),
        llm_client=llm_client,
    )

    result = pipeline.answer("Does SleepPilot work with Garmin or Fitbit?", history=[])

    assert result.sources[0].id == "faq-006"
    assert result.mode == "answer-test"
    assert len(llm_client.calls) == 1
    assert llm_client.calls[0][0] == "answer"
    assert result.debug is not None
    assert result.debug.retrieval_query == "Does SleepPilot work with Garmin or Fitbit?"
    assert len(result.debug.calls) == 1
    assert result.debug.calls[0].stage == "answer_generation"
    answer_messages = llm_client.calls[0][1]
    assert isinstance(answer_messages, list)
    assert "Current user question" in answer_messages[-1]["content"]


def test_rag_pipeline_rewrites_follow_up_question_before_retrieval(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    llm_client = RecordingLLMClient()
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
        embedder=CosineTestEmbedder(),
        llm_client=llm_client,
    )
    history = [
        ConversationTurn(role="user", content="Does SleepPilot support wearable devices?"),
        ConversationTurn(role="assistant", content="It supports common wearable platforms."),
    ]

    result = pipeline.answer("What about Garmin or Fitbit?", history=history)

    assert result.sources[0].id == "faq-006"
    assert result.answer == "Grounded answer from test LLM."
    assert len(llm_client.calls) == 2
    rewrite_messages = llm_client.calls[0][1]
    answer_messages = llm_client.calls[1][1]
    assert isinstance(rewrite_messages, list)
    assert isinstance(answer_messages, list)
    rewrite_prompt = rewrite_messages[-1]["content"]
    assert "Current user message:\nWhat about Garmin or Fitbit?" in rewrite_prompt
    assert "Current user question:\nWhat about Garmin or Fitbit?" in answer_messages[-1]["content"]
    assert answer_messages[1] == {
        "role": "user",
        "content": "Does SleepPilot support wearable devices?",
    }
    assert result.debug is not None
    assert result.debug.retrieval_query == "Does SleepPilot support Garmin or Fitbit?"
    assert [call.stage for call in result.debug.calls] == [
        "query_rewrite",
        "answer_generation",
    ]


def test_rag_pipeline_limits_history_to_four_messages_and_char_budget(tmp_path):
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
        embedder=CosineTestEmbedder(),
    )
    history = [
        ConversationTurn(role="user", content="old 1"),
        ConversationTurn(role="assistant", content="old 2"),
        ConversationTurn(role="user", content="recent 1"),
        ConversationTurn(role="assistant", content="recent 2"),
        ConversationTurn(role="user", content="recent 3"),
        ConversationTurn(role="assistant", content="recent 4"),
    ]

    cleaned_history = clean_history(history)

    assert [turn.content for turn in cleaned_history] == [
        "recent 1",
        "recent 2",
        "recent 3",
        "recent 4",
    ]

    long_history = [
        ConversationTurn(role="user", content="x" * 700),
        ConversationTurn(role="assistant", content="y" * 700),
        ConversationTurn(role="user", content="z" * 700),
        ConversationTurn(role="assistant", content="w" * 700),
    ]
    capped_history = clean_history(long_history)

    assert sum(len(turn.content) for turn in capped_history) <= HISTORY_INPUT_CHAR_LIMIT


def test_rag_pipeline_sends_only_clean_context_for_clear_match(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
        embedder=CosineTestEmbedder(),
    )

    result = pipeline.answer("What is this thing?")

    assert [source.id for source in result.sources] == ["faq-001"]


def test_rag_pipeline_can_send_three_chunks_for_ambiguous_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
        embedder=CosineTestEmbedder(),
    )
    results = pipeline.retrieve("sleep habits and bedtime schedule", top_k=4)

    selected = select_context_results(results)

    assert 1 <= len(selected) <= 3


def test_rag_pipeline_removes_model_citation_markers(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
        embedder=CosineTestEmbedder(),
    )

    assert pipeline._clean_answer("Yes, SleepPilot can help. [2]") == "Yes, SleepPilot can help."


def test_rag_pipeline_declines_out_of_scope_question(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    pipeline = RAGPipeline(
        faq_path=FAQ_PATH,
        vector_db_path=tmp_path / "vectors.sqlite3",
        embedder=CosineTestEmbedder(),
    )

    result = pipeline.answer("Write Python code for a todo app")

    assert result.in_scope is False
    assert result.sources == []
    assert "SleepPilot" in result.answer
