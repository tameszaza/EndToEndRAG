from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib


PROMPT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "prompts.toml"


@dataclass(frozen=True)
class PromptConfig:
    answer_system: str
    answer_user_instructions: str
    query_rewrite_system: str
    query_rewrite_user_template: str


@lru_cache(maxsize=1)
def load_prompt_config(path: str | Path = PROMPT_CONFIG_PATH) -> PromptConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    answer = data["answer"]
    query_rewrite = data["query_rewrite"]
    return PromptConfig(
        answer_system=answer["system"].strip(),
        answer_user_instructions=answer["user_instructions"].strip(),
        query_rewrite_system=query_rewrite["system"].strip(),
        query_rewrite_user_template=query_rewrite["user_template"].strip(),
    )
