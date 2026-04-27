from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str


class GeminiClient:
    """Google Gemini generateContent client.

    Defaults to Gemini 2.5 Flash-Lite because it is the current lightweight
    free-tier-friendly text model for this use case.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite"
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.model)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not self.is_configured:
            raise RuntimeError("Gemini client is not configured.")

        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent"
        )
        response = requests.post(
            endpoint,
            params={"key": self.api_key},
            json={
                "systemInstruction": {
                    "parts": [{"text": system_prompt}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.9,
                    "maxOutputTokens": 420,
                },
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return LLMResponse(
            text=self._extract_text(payload),
            provider=f"gemini:{self.model}",
        )

    def _extract_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise RuntimeError("Gemini returned an empty answer.")
        return text


class OpenAICompatibleClient:
    """Optional chat-completions client for Typhoon, OpenAI, or a local router."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.getenv("OPENAI_COMPATIBLE_MODEL") or os.getenv("LLM_MODEL")
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not self.is_configured:
            raise RuntimeError("LLM client is not configured.")

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 420,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"].strip()
        return LLMResponse(text=text, provider=self.model)
