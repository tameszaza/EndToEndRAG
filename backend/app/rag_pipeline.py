from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from app.faq_loader import load_and_chunk_faq
from app.llm_client import GeminiClient, LLMResponse, OpenAICompatibleClient
from app.vector_store import SQLiteVectorStore, SearchResult


SYSTEM_PROMPT = """You are SleepPilot Coach, the embedded FAQ assistant for SleepPilot.
SleepPilot is a sleep optimization app with the tagline "Smarter nights. Sharper days."
Answer only questions about SleepPilot, its sleep tracking features, pricing, privacy,
wearable integrations, smart alarm, travel support, and sleep routine guidance.
Ground every answer in the provided FAQ context. If the context does not answer the
question, say you do not have enough information. Politely decline unrelated requests,
including coding, homework, politics, legal advice, financial advice, or general medical
diagnosis. SleepPilot is wellness guidance, not a medical device."""


DECLINE_MESSAGE = (
    "I can only help with SleepPilot, sleep tracking features, privacy, pricing, "
    "wearable integrations, and sleep routine guidance. I can’t help with that request here."
)


UNKNOWN_MESSAGE = (
    "I don’t have enough information in the SleepPilot FAQ to answer that confidently. "
    "I can help with SleepPilot features, privacy, pricing, supported wearables, smart alarms, "
    "jet lag, and bedtime routine guidance."
)


IN_SCOPE_TERMS = {
    "alarm",
    "apple",
    "bed",
    "bedtime",
    "caffeine",
    "cost",
    "data",
    "device",
    "diagnose",
    "fitbit",
    "garmin",
    "google",
    "habit",
    "health",
    "insomnia",
    "integration",
    "jet",
    "lag",
    "medical",
    "premium",
    "price",
    "privacy",
    "routine",
    "schedule",
    "score",
    "sleep",
    "sleeppilot",
    "smart",
    "student",
    "travel",
    "wearable",
    "wind",
}


OUT_OF_SCOPE_TERMS = {
    "basketball",
    "code",
    "crypto",
    "debug",
    "essay",
    "football",
    "homework",
    "javascript",
    "movie",
    "politics",
    "python",
    "recipe",
    "sql",
    "stock",
    "weather",
}


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


class RAGPipeline:
    def __init__(
        self,
        faq_path: str | Path,
        vector_db_path: str | Path,
        gemini_client: GeminiClient | None = None,
        llm_client: OpenAICompatibleClient | None = None,
    ):
        load_dotenv()
        self.faq_path = Path(faq_path)
        self.vector_store = SQLiteVectorStore(vector_db_path)
        self.gemini_client = gemini_client or GeminiClient()
        self.llm_client = llm_client or OpenAICompatibleClient()
        self.refresh()

    def refresh(self) -> None:
        chunks = load_and_chunk_faq(self.faq_path)
        self.vector_store.ensure_built(chunks)

    def answer(self, question: str) -> ChatResult:
        normalized_question = question.strip()
        if not normalized_question:
            return ChatResult(
                answer="Ask me a question about SleepPilot and I’ll ground the answer in the FAQ.",
                sources=[],
                in_scope=True,
                mode="empty",
                confidence=0.0,
            )

        if len(normalized_question) > 700:
            return ChatResult(
                answer="That question is a bit long for this FAQ assistant. Please ask one SleepPilot question at a time.",
                sources=[],
                in_scope=True,
                mode="too_long",
                confidence=0.0,
            )

        results = self.vector_store.similarity_search(normalized_question, top_k=4)
        best_score = results[0].score if results else 0.0

        if self._is_out_of_scope(normalized_question, best_score):
            return ChatResult(
                answer=DECLINE_MESSAGE,
                sources=[],
                in_scope=False,
                mode="guardrail",
                confidence=best_score,
            )

        if best_score < 0.08:
            return ChatResult(
                answer=UNKNOWN_MESSAGE,
                sources=[],
                in_scope=True,
                mode="no_context",
                confidence=best_score,
            )

        useful_results = self._select_context_results(results)
        sources = [
            ChatSource(
                id=result.chunk.id,
                question=result.chunk.question,
                answer=result.chunk.answer,
                score=result.score,
            )
            for result in useful_results[:3]
        ]

        response = self._generate_answer(normalized_question, useful_results)
        return ChatResult(
            answer=self._clean_answer(response.text),
            sources=sources,
            in_scope=True,
            mode=response.provider,
            confidence=best_score,
        )

    def retrieve(self, question: str, top_k: int = 4) -> list[SearchResult]:
        return self.vector_store.similarity_search(question, top_k=top_k)

    def _select_context_results(self, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []

        best_score = results[0].score
        selected = [results[0]]
        for result in results[1:]:
            if best_score >= 0.55:
                is_confident_support = result.score >= 0.18 and result.score >= best_score * 0.55
            else:
                is_confident_support = result.score >= 0.1 and result.score >= best_score * 0.5
            if is_confident_support:
                selected.append(result)
            if len(selected) == 3:
                break
        return selected

    def _generate_answer(self, question: str, results: list[SearchResult]) -> LLMResponse:
        context = self._format_context(results)
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        user_prompt = (
            f"FAQ context:\n{context}\n\n"
            f"User question: {question}\n\n"
            "Answer in a warm, concise SleepPilot voice. Do not use information outside "
            "the FAQ context. Only mention that SleepPilot is not a medical device when "
            "the user asks about diagnosis, treatment, sleep disorders, symptoms, or "
            "medical boundaries. Do not include bracket citations like [1], [2], source "
            "IDs, or context numbers in the answer text."
        )

        if provider == "gemini" and self.gemini_client.is_configured:
            try:
                return self.gemini_client.complete(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
            except (RuntimeError, requests.RequestException):
                pass

        if provider in {"api", "openai-compatible"} and self.llm_client.is_configured:
            try:
                return self.llm_client.complete(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
            except (RuntimeError, requests.RequestException):
                pass

        return LLMResponse(text=self._local_answer(results), provider="local-rag")

    def _local_answer(self, results: list[SearchResult]) -> str:
        primary = results[0].chunk
        answer = primary.answer

        related = [
            result.chunk.question
            for result in results[1:]
            if result.score >= 0.12 and result.chunk.id != primary.id
        ]
        if related:
            answer = (
                f"{answer}\n\n"
                f"Related FAQ topics: {'; '.join(related[:2])}."
            )

        return answer

    def _format_context(self, results: list[SearchResult]) -> str:
        lines = []
        for index, result in enumerate(results, start=1):
            lines.append(
                f"[{index}] {result.chunk.id} (score {result.score:.3f})\n"
                f"Q: {result.chunk.question}\n"
                f"A: {result.chunk.answer}"
            )
        return "\n\n".join(lines)

    def _clean_answer(self, answer: str) -> str:
        cleaned = re.sub(r"\s*\[\d+\]", "", answer)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def _is_out_of_scope(self, question: str, best_score: float) -> bool:
        tokens = {token.strip(".,?!:;()[]{}'\"").lower() for token in question.split()}
        has_scope_term = bool(tokens & IN_SCOPE_TERMS)
        has_out_of_scope_term = bool(tokens & OUT_OF_SCOPE_TERMS)

        if has_out_of_scope_term and not has_scope_term:
            return True
        return best_score < 0.08 and not has_scope_term
