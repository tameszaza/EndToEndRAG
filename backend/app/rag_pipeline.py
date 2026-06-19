from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from app.embeddings import Embedder
from app.faq_loader import load_and_chunk_faq
from app.llm_client import LLMResponse, OpenAICompatibleClient
from app.vector_store import SQLiteVectorStore, SearchResult


SYSTEM_PROMPT = """You are SleepPilot Coach, the embedded FAQ assistant for SleepPilot.
SleepPilot is a sleep optimization app with the tagline "Smarter nights. Sharper days."
Answer only questions about SleepPilot, its sleep tracking features, pricing, privacy,
wearable integrations, smart alarm, travel support, and sleep routine guidance.
Ground every answer in the provided FAQ context. If the context does not answer the
question, say you do not have enough information. Politely decline unrelated requests,
including coding, homework, politics, legal advice, financial advice, or general medical
diagnosis. SleepPilot is wellness guidance, not a medical device."""


QUERY_REWRITE_PROMPT = """You rewrite follow-up messages into standalone search queries.

Rules:
1. Use the conversation history only to resolve references and omitted context.
2. Do not answer the question.
3. Do not add facts that are not present in the conversation.
4. Preserve the user's original language.
5. Return only the standalone query.
6. If the current question is already standalone, return it unchanged.
7. Keep the rewritten query concise."""


HISTORY_MESSAGE_LIMIT = 4
HISTORY_INPUT_CHAR_LIMIT = 1800
MAX_QUESTION_LENGTH = 700


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
class ConversationTurn:
    role: str
    content: str


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
        llm_client: OpenAICompatibleClient | None = None,
        embedder: Embedder | None = None,
    ):
        load_dotenv()
        self.faq_path = Path(faq_path)
        self.vector_store = SQLiteVectorStore(vector_db_path, embedder=embedder)
        self.llm_client = llm_client or OpenAICompatibleClient()
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
        clean_history = self._clean_history(history or [])
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

        standalone_question = self._rewrite_question(normalized_question, clean_history)
        results = self.vector_store.similarity_search(standalone_question, top_k=4)
        best_score = results[0].score if results else 0.0

        if self._is_out_of_scope(standalone_question, best_score):
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

        response = self._generate_answer(normalized_question, useful_results, clean_history)
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

    def _generate_answer(
        self,
        question: str,
        results: list[SearchResult],
        history: list[ConversationTurn] | None = None,
    ) -> LLMResponse:
        context = self._format_context(results)
        messages = self._build_answer_messages(
            question=question,
            context=context,
            history=history or [],
        )
        provider = os.getenv("LLM_PROVIDER", "local").lower()

        if provider in {"api", "openai-compatible"} and self.llm_client.is_configured:
            try:
                return self.llm_client.complete_messages(
                    messages=messages,
                    temperature=0.2,
                    max_tokens=420,
                )
            except (RuntimeError, requests.RequestException):
                pass

        return LLMResponse(text=self._local_answer(results), provider="local-rag")

    def _rewrite_question(
        self,
        question: str,
        history: list[ConversationTurn],
    ) -> str:
        if not history:
            return question

        provider = os.getenv("LLM_PROVIDER", "local").lower()
        if provider not in {"api", "openai-compatible"} or not self.llm_client.is_configured:
            return question

        conversation = self._format_history(history)
        user_prompt = (
            f"Conversation history:\n{conversation}\n\n"
            f"Current user message:\n{question}\n\n"
            "Standalone search query:"
        )
        try:
            response = self.llm_client.complete(
                system_prompt=QUERY_REWRITE_PROMPT,
                user_prompt=user_prompt,
                temperature=0,
                max_tokens=120,
            )
        except (RuntimeError, requests.RequestException):
            return question

        rewritten = self._clean_rewritten_question(response.text)
        return rewritten or question

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

    def _build_answer_messages(
        self,
        question: str,
        context: str,
        history: list[ConversationTurn],
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(
            {"role": turn.role, "content": turn.content}
            for turn in history
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"FAQ context:\n{context}\n\n"
                    f"Current user question:\n{question}\n\n"
                    "Answer in a warm, concise SleepPilot voice. Use conversation history "
                    "only to understand references in the current question. Use the FAQ "
                    "context as the only factual source. Only mention that SleepPilot is "
                    "not a medical device when the user asks about diagnosis, treatment, "
                    "sleep disorders, symptoms, or medical boundaries. Do not include "
                    "bracket citations like [1], [2], source IDs, or context numbers in "
                    "the answer text."
                ),
            }
        )
        return messages

    def _clean_history(self, history: list[ConversationTurn]) -> list[ConversationTurn]:
        clean_turns: list[ConversationTurn] = []
        remaining_chars = HISTORY_INPUT_CHAR_LIMIT
        recent_turns: list[ConversationTurn] = []

        for turn in reversed(history[-HISTORY_MESSAGE_LIMIT:]):
            role = turn.role.strip().lower()
            content = turn.content.strip()
            if role not in {"user", "assistant"} or not content:
                continue
            if remaining_chars <= 0:
                break
            content = content[: min(MAX_QUESTION_LENGTH, remaining_chars)]
            remaining_chars -= len(content)
            recent_turns.append(
                ConversationTurn(
                    role=role,
                    content=content,
                )
            )

        for turn in reversed(recent_turns):
            clean_turns.append(turn)
        return clean_turns

    def _format_history(self, history: list[ConversationTurn]) -> str:
        lines = []
        for turn in history:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.content}")
        return "\n".join(lines)

    def _clean_rewritten_question(self, question: str) -> str:
        cleaned = question.strip().strip("\"'")
        cleaned = re.sub(
            r"^(standalone\s+(search\s+)?query|standalone question|query)\s*:\s*",
            "",
            cleaned,
            flags=re.I,
        )
        if len(cleaned) > MAX_QUESTION_LENGTH:
            return ""
        return cleaned.strip()

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
