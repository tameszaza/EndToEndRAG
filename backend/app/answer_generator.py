from __future__ import annotations

import os

import requests

from app.conversation import ConversationTurn
from app.llm_client import LLMResponse, OpenAICompatibleClient
from app.llm_debug import LLMDebug, LLMDebugCall
from app.prompt_config import PromptConfig
from app.vector_store import SearchResult


class AnswerGenerator:
    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        prompt_config: PromptConfig,
    ):
        self.llm_client = llm_client
        self.prompt_config = prompt_config

    def generate(
        self,
        question: str,
        results: list[SearchResult],
        history: list[ConversationTurn],
        debug: LLMDebug,
    ) -> LLMResponse:
        context = self.format_context(results)
        messages = self.build_messages(
            question=question,
            context=context,
            history=history,
        )
        provider = os.getenv("LLM_PROVIDER", "local").lower()

        if provider in {"api", "openai-compatible"} and self.llm_client.is_configured:
            try:
                response = self.llm_client.complete_messages(
                    messages=messages,
                    temperature=0.2,
                    max_tokens=420,
                )
                debug.calls.append(
                    LLMDebugCall(
                        stage="answer_generation",
                        messages=messages,
                        temperature=0.2,
                        max_tokens=420,
                        response_text=response.text,
                    )
                )
                return response
            except (RuntimeError, requests.RequestException) as error:
                debug.calls.append(
                    LLMDebugCall(
                        stage="answer_generation",
                        messages=messages,
                        temperature=0.2,
                        max_tokens=420,
                        error=str(error),
                    )
                )

        return LLMResponse(text=self.local_answer(results), provider="local-rag")

    def build_messages(
        self,
        question: str,
        context: str,
        history: list[ConversationTurn],
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.prompt_config.answer_system}]
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
                    f"{self.prompt_config.answer_user_instructions}"
                ),
            }
        )
        return messages

    def format_context(self, results: list[SearchResult]) -> str:
        lines = []
        for index, result in enumerate(results, start=1):
            lines.append(
                f"[{index}] {result.chunk.id} (score {result.score:.3f})\n"
                f"Q: {result.chunk.question}\n"
                f"A: {result.chunk.answer}"
            )
        return "\n\n".join(lines)

    def local_answer(self, results: list[SearchResult]) -> str:
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
