from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMDebugCall:
    stage: str
    messages: list[dict[str, str]]
    temperature: float
    max_tokens: int
    response_text: str | None = None
    skipped_reason: str | None = None
    error: str | None = None


@dataclass
class LLMDebug:
    retrieval_query: str
    calls: list[LLMDebugCall] = field(default_factory=list)
