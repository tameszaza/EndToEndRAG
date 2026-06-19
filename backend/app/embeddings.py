from __future__ import annotations

import math
import os
import re
from typing import Any, Iterable, Protocol


STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "help",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "question",
    "sleep",
    "sleeppilot",
    "that",
    "the",
    "this",
    "to",
    "use",
    "user",
    "users",
    "what",
    "when",
    "which",
    "with",
    "you",
    "your",
}


DOMAIN_SYNONYMS = {
    "alarm": ["wake", "wakeup", "morning", "window"],
    "app": ["product", "overview", "description", "optimization"],
    "bed": ["bedtime", "routine", "schedule"],
    "bedtime": ["routine", "schedule", "winddown", "night"],
    "caffeine": ["coffee", "stimulant", "late"],
    "cost": ["price", "pricing", "plan", "premium", "free"],
    "data": ["privacy", "delete", "collect", "permission"],
    "device": ["wearable", "fitbit", "garmin", "apple", "google"],
    "diagnose": ["medical", "doctor", "disorder", "treat"],
    "fitbit": ["wearable", "device", "integration"],
    "garmin": ["wearable", "device", "integration"],
    "habit": ["routine", "schedule", "consistency"],
    "insomnia": ["medical", "doctor", "disorder"],
    "jet": ["travel", "timezone", "lag"],
    "plan": ["routine", "schedule", "bedtime"],
    "price": ["cost", "pricing", "plan", "premium", "free"],
    "product": ["app", "overview", "description", "optimization"],
    "privacy": ["data", "delete", "permission", "advertisers"],
    "routine": ["bedtime", "schedule", "winddown", "relaxation"],
    "schedule": ["bedtime", "routine", "consistency", "wake"],
    "score": ["rating", "trend", "quality", "consistency"],
    "student": ["school", "exam", "study", "schedule"],
    "travel": ["jet", "timezone", "lag", "trip"],
    "wearable": ["device", "fitbit", "garmin", "apple", "google", "integration"],
    "wind": ["winddown", "routine", "relaxation"],
}


PHRASE_TOKENS = {
    "apple health": "apple_health",
    "google fit": "google_fit",
    "smart alarm": "smart_alarm",
    "sleep score": "sleep_score",
    "sleep disorder": "sleep_disorder",
    "jet lag": "jet_lag",
    "time zone": "time_zone",
    "screen time": "screen_time",
    "wake up": "wake_up",
    "go to bed": "go_to_bed",
    "wind down": "wind_down",
}


def normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []

    for phrase, replacement in PHRASE_TOKENS.items():
        if phrase in lowered:
            tokens.append(replacement)

    for raw in re.findall(r"[a-z0-9]+", lowered):
        raw_token = raw.lower()

        if raw_token in STOPWORDS:
            continue

        token = normalize_token(raw_token)

        if token and token not in STOPWORDS:
            tokens.append(token)

    return tokens


class Embedder(Protocol):
    @property
    def dimensions(self) -> int:
        ...

    @property
    def namespace(self) -> str:
        ...

    def embed(
        self,
        text: str,
        task_type: str | None = None,
        title: str | None = None,
    ) -> list[float]:
        ...


def expand_tokens(tokens: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(DOMAIN_SYNONYMS.get(token, []))
    return expanded


class LocalSentenceTransformerEmbedder:
    """Self-hosted embedding model loaded through SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v5-text-small-retrieval",
        device: str | None = None,
        trust_remote_code: bool = False,
        normalize_embeddings: bool = True,
        batch_size: int = 8,
        query_prompt_name: str = "query",
        document_prompt_name: str = "document",
        truncate_dim: int | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.query_prompt_name = query_prompt_name
        self.document_prompt_name = document_prompt_name
        self.truncate_dim = truncate_dim
        self._model: Any | None = None
        self._dimensions: int | None = truncate_dim

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            dimension = self._load_model().get_sentence_embedding_dimension()
            if dimension is None:
                raise RuntimeError(
                    f"Could not determine embedding dimensions for {self.model_name}."
                )
            self._dimensions = int(dimension)
        return self._dimensions

    @property
    def namespace(self) -> str:
        return (
            "local-sentence-transformers:"
            f"{self.model_name}:{self.dimensions}:"
            f"q={self.query_prompt_name}:d={self.document_prompt_name}:"
            f"normalize={self.normalize_embeddings}"
        )

    def embed(
        self,
        text: str,
        task_type: str | None = None,
        title: str | None = None,
    ) -> list[float]:
        if title and task_type == "RETRIEVAL_DOCUMENT":
            text = f"{title}\n{text}"

        model = self._load_model()
        prompt_name = self._prompt_name(task_type)
        text, prompt_name = self._prepare_prompt(model, text, prompt_name)
        encode_kwargs: dict[str, object] = {
            "sentences": text,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
        }
        if prompt_name:
            encode_kwargs["prompt_name"] = prompt_name
        if self.truncate_dim:
            encode_kwargs["truncate_dim"] = self.truncate_dim

        embedding = model.encode(**encode_kwargs)
        values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        return normalize_vector([float(value) for value in values])

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is required for local embeddings. "
                    "Install backend/requirements.txt, then run the app again."
                ) from error

            kwargs: dict[str, object] = {
                "trust_remote_code": self.trust_remote_code,
            }
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def _prompt_name(self, task_type: str | None) -> str | None:
        if task_type == "RETRIEVAL_QUERY":
            return self.query_prompt_name
        if task_type == "RETRIEVAL_DOCUMENT":
            return self.document_prompt_name
        return None

    def _prepare_prompt(
        self,
        model: Any,
        text: str,
        prompt_name: str | None,
    ) -> tuple[str, str | None]:
        if not prompt_name:
            return text, None

        prompts = getattr(model, "prompts", None)
        if isinstance(prompts, dict) and prompt_name in prompts:
            return text, prompt_name

        return f"{prompt_name}: {text}", None


def create_default_embedder() -> Embedder:
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    if provider in {"local", "sentence-transformers", "huggingface"}:
        return create_local_embedder()

    raise RuntimeError(
        "EMBEDDING_PROVIDER must be one of: local, sentence-transformers, huggingface."
    )


def create_local_embedder() -> LocalSentenceTransformerEmbedder:
    truncate_dim = _optional_positive_int_from_env("LOCAL_EMBEDDING_TRUNCATE_DIM")
    configured_dimensions = _optional_positive_int_from_env("LOCAL_EMBEDDING_DIMENSIONS")
    if truncate_dim is None:
        truncate_dim = configured_dimensions

    return LocalSentenceTransformerEmbedder(
        model_name=os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "jinaai/jina-embeddings-v5-text-small-retrieval",
        ),
        device=os.getenv("LOCAL_EMBEDDING_DEVICE") or None,
        trust_remote_code=_bool_from_env("LOCAL_EMBEDDING_TRUST_REMOTE_CODE", False),
        normalize_embeddings=_bool_from_env("LOCAL_EMBEDDING_NORMALIZE", True),
        batch_size=_positive_int_from_env("LOCAL_EMBEDDING_BATCH_SIZE", 8),
        query_prompt_name=os.getenv("LOCAL_EMBEDDING_QUERY_PROMPT", "query"),
        document_prompt_name=os.getenv("LOCAL_EMBEDDING_DOCUMENT_PROMPT", "document"),
        truncate_dim=truncate_dim,
    )


def _positive_int_from_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer.") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero.")
    return value


def _optional_positive_int_from_env(name: str) -> int | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer.") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero.")
    return value


def _bool_from_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value.")


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 8) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
