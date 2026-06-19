from __future__ import annotations

from dataclasses import dataclass


HISTORY_MESSAGE_LIMIT = 4
HISTORY_INPUT_CHAR_LIMIT = 1800
MAX_QUESTION_LENGTH = 700


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str


def clean_history(history: list[ConversationTurn]) -> list[ConversationTurn]:
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


def format_history(history: list[ConversationTurn]) -> str:
    lines = []
    for turn in history:
        label = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{label}: {turn.content}")
    return "\n".join(lines)
