from __future__ import annotations

import os
import re

import requests

from app.conversation import ConversationTurn, MAX_QUESTION_LENGTH, format_history
from app.llm_client import OpenAICompatibleClient
from app.llm_debug import LLMDebug, LLMDebugCall
from app.prompt_config import PromptConfig


class QueryRewriter:
    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        prompt_config: PromptConfig,
    ):
        self.llm_client = llm_client
        self.prompt_config = prompt_config

    def rewrite(
        self,
        question: str,
        history: list[ConversationTurn],
        debug: LLMDebug,
    ) -> str:
        if not history:
            return question

        provider = os.getenv("LLM_PROVIDER", "local").lower()
        if provider not in {"api", "openai-compatible"} or not self.llm_client.is_configured:
            return question

        user_prompt = self.prompt_config.query_rewrite_user_template.format(
            history=format_history(history),
            question=question,
        )
        messages = [
            {"role": "system", "content": self.prompt_config.query_rewrite_system},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self.llm_client.complete_messages(
                messages=messages,
                temperature=0,
                max_tokens=120,
            )
        except (RuntimeError, requests.RequestException) as error:
            debug.calls.append(
                LLMDebugCall(
                    stage="query_rewrite",
                    messages=messages,
                    temperature=0,
                    max_tokens=120,
                    error=str(error),
                )
            )
            return question

        debug.calls.append(
            LLMDebugCall(
                stage="query_rewrite",
                messages=messages,
                temperature=0,
                max_tokens=120,
                response_text=response.text,
            )
        )
        rewritten = self._clean_rewritten_question(response.text)
        return rewritten or question

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
