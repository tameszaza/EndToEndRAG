from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Literal

from app.llm_debug import LLMDebug


DebugOutput = Literal["web", "server", "both"]


@dataclass(frozen=True)
class DebugSettings:
    enabled: bool
    output: DebugOutput

    @property
    def send_to_web(self) -> bool:
        return self.enabled and self.output in {"web", "both"}

    @property
    def send_to_server(self) -> bool:
        return self.enabled and self.output in {"server", "both"}


def load_debug_settings() -> DebugSettings:
    return DebugSettings(
        enabled=_bool_from_env("LLM_DEBUG_ENABLED", False),
        output=_debug_output_from_env("LLM_DEBUG_OUTPUT", "web"),
    )


def serialize_debug(debug: LLMDebug | None) -> dict[str, object] | None:
    if debug is None:
        return None
    return asdict(debug)


def log_server_debug(debug: LLMDebug | None) -> None:
    payload = serialize_debug(debug)
    if payload is None:
        return
    print(
        "[SleepPilot LLM debug] "
        + json.dumps(payload, ensure_ascii=False, indent=2),
        flush=True,
    )


def _debug_output_from_env(name: str, default: DebugOutput) -> DebugOutput:
    value = os.getenv(name, default).strip().lower()
    if value in {"web", "server", "both"}:
        return value
    return default


def _bool_from_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
