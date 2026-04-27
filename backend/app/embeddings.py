from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Iterable


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
        token = normalize_token(raw)
        if token and token not in STOPWORDS:
            tokens.append(token)

    return tokens


def expand_tokens(tokens: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(DOMAIN_SYNONYMS.get(token, []))
    return expanded


@dataclass(frozen=True)
class HashingEmbedder:
    """Small deterministic embedder for local RAG demos.

    It creates normalized hashed bag-of-word and bigram vectors. This keeps the
    assignment runnable without an external embedding API while still exercising
    the same retrieve-from-vector flow a hosted embedder would use.
    """

    dimensions: int = 384
    namespace: str = "sleeppilot-v3"
    bigram_weight: float = 1.35
    synonym_weight: float = 0.65
    _hash_cache: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        if not tokens:
            return [0.0] * self.dimensions

        vector = [0.0] * self.dimensions
        self._add_features(vector, tokens, weight=1.0)

        bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        self._add_features(vector, bigrams, weight=self.bigram_weight)

        expanded = [token for token in expand_tokens(tokens) if token not in tokens]
        self._add_features(vector, expanded, weight=self.synonym_weight)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [round(value / norm, 8) for value in vector]

    def _add_features(self, vector: list[float], features: Iterable[str], weight: float) -> None:
        for feature in features:
            index = self._stable_index(feature)
            vector[index] += weight

    def _stable_index(self, value: str) -> int:
        cached = self._hash_cache.get(value)
        if cached is not None:
            return cached
        digest = hashlib.blake2b(
            f"{self.namespace}:{value}".encode("utf-8"), digest_size=8
        ).digest()
        index = int.from_bytes(digest, byteorder="big") % self.dimensions
        self._hash_cache[value] = index
        return index


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions.")
    return sum(a * b for a, b in zip(left, right))
