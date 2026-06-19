from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.answer_generator import AnswerGenerator
from app.conversation import ConversationTurn, clean_history
from app.context_selector import select_context_results
from app.embeddings import Embedder
from app.faq_loader import load_and_chunk_faq
from app.guardrails import DECLINE_MESSAGE, UNKNOWN_MESSAGE, is_out_of_scope
from app.llm_client import OpenAICompatibleClient
from app.llm_debug import LLMDebug
from app.prompt_config import PromptConfig, load_prompt_config
from app.query_rewriter import QueryRewriter
from app.vector_store import SQLiteVectorStore, SearchResult


@dataclass(frozen=True)
class ChatSource:
    id: str
    question: str
    answer: str
    score: float


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[ChatSource]
    in_scope: bool
    mode: str
    confidence: float
    debug: LLMDebug | None = None


class RAGPipeline:
    def __init__(
        self,
        faq_path: str | Path,
        vector_db_path: str | Path,
        llm_client: OpenAICompatibleClient | None = None,
        embedder: Embedder | None = None,
        prompt_config: PromptConfig | None = None,
    ):
        load_dotenv()
        self.faq_path = Path(faq_path)
        self.vector_store = SQLiteVectorStore(vector_db_path, embedder=embedder)
        self.llm_client = llm_client or OpenAICompatibleClient()
        self.prompt_config = prompt_config or load_prompt_config()
        self.query_rewriter = QueryRewriter(
            llm_client=self.llm_client,
            prompt_config=self.prompt_config,
        )
        self.answer_generator = AnswerGenerator(
            llm_client=self.llm_client,
            prompt_config=self.prompt_config,
        )
        self.refresh()

    def refresh(self) -> None:
        chunks = load_and_chunk_faq(self.faq_path)
        self.vector_store.ensure_built(chunks)

    def answer(
        self,
        question: str,
        history: list[ConversationTurn] | None = None,
    ) -> ChatResult:
        normalized_question = question.strip()
        clean_conversation = clean_history(history or [])
        debug = LLMDebug(retrieval_query=normalized_question)
        if not normalized_question:
            return ChatResult(
                answer="Ask me a question about SleepPilot and I’ll ground the answer in the FAQ.",
                sources=[],
                in_scope=True,
                mode="empty",
                confidence=0.0,
                debug=debug,
            )

        if len(normalized_question) > 700:
            return ChatResult(
                answer="That question is a bit long for this FAQ assistant. Please ask one SleepPilot question at a time.",
                sources=[],
                in_scope=True,
                mode="too_long",
                confidence=0.0,
                debug=debug,
            )

        standalone_question = self.query_rewriter.rewrite(
            question=normalized_question,
            history=clean_conversation,
            debug=debug,
        )
        debug.retrieval_query = standalone_question
        results = self.vector_store.similarity_search(standalone_question, top_k=4)
        best_score = results[0].score if results else 0.0

        if is_out_of_scope(standalone_question, best_score):
            return ChatResult(
                answer=DECLINE_MESSAGE,
                sources=[],
                in_scope=False,
                mode="guardrail",
                confidence=best_score,
                debug=debug,
            )

        if best_score < 0.08:
            return ChatResult(
                answer=UNKNOWN_MESSAGE,
                sources=[],
                in_scope=True,
                mode="no_context",
                confidence=best_score,
                debug=debug,
            )

        useful_results = select_context_results(results)
        sources = [
            ChatSource(
                id=result.chunk.id,
                question=result.chunk.question,
                answer=result.chunk.answer,
                score=result.score,
            )
            for result in useful_results[:3]
        ]

        response = self.answer_generator.generate(
            question=normalized_question,
            results=useful_results,
            history=clean_conversation,
            debug=debug,
        )
        return ChatResult(
            answer=self._clean_answer(response.text),
            sources=sources,
            in_scope=True,
            mode=response.provider,
            confidence=best_score,
            debug=debug,
        )

    def retrieve(self, question: str, top_k: int = 4) -> list[SearchResult]:
        return self.vector_store.similarity_search(question, top_k=top_k)

    def _clean_answer(self, answer: str) -> str:
        cleaned = re.sub(r"\s*\[\d+\]", "", answer)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()
