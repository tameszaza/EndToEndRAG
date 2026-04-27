from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.embeddings import HashingEmbedder, cosine_similarity, expand_tokens, tokenize


ENTITY_TOKENS = {
    "apple",
    "apple_health",
    "caffeine",
    "fitbit",
    "garmin",
    "google",
    "google_fit",
    "insomnia",
    "jet_lag",
    "premium",
    "privacy",
    "smart_alarm",
    "student",
    "wearable",
}


FAQ_SEARCH_HINTS = {
    "faq-001": "product overview app description what is this thing what does it do sleep optimization",
    "faq-002": "improve sleep sleep better habits schedule duration screen time caffeine relaxation consistency",
    "faq-003": "diagnose disorder medical device insomnia snoring breathing pauses daytime sleepiness doctor",
    "faq-004": "data collection collect sleep logs bedtime wake-up time quality ratings lifestyle wearable private messages browsing files",
    "faq-005": "without wearable manual entry no device required bedtime wake-up sleep quality daily habits",
    "faq-006": "wearable support integrations Apple Health Google Fit Fitbit Garmin device permissions",
    "faq-007": "privacy protect data sell advertisers delete disconnect integrations settings personal insights",
    "faq-008": "jet lag travel timezone trip light exposure schedule shift adapt new time zone",
    "faq-009": "sleep score rating trend consistency duration quality routine stability low score",
    "faq-010": "smart alarm wake-up window wake up morning schedule recent sleep pattern",
    "faq-011": "bedtime routine plan when to go to bed target wake-up time wind down relaxation room environment consistent schedule",
    "faq-012": "students school exams studying irregular schedule stress caffeine healthy habits",
    "faq-013": "sleep too late late every night move bedtime earlier gradual adjustment screen exposure consistent wake-up",
    "faq-014": "cost price pricing free plan premium plan advanced trend smart alarm wearable travel support",
    "faq-015": "cannot answer unknown not enough information FAQ scope unsupported question",
}


INTENT_RULES = [
    (
        "faq-001",
        [
            "what is the product",
            "what is this",
            "what is this app",
            "what is this thing",
            "what is this product",
            "what is it",
            "what is sleeppilot",
            "tell me about sleeppilot",
            "tell me about the product",
            "what does sleeppilot do",
            "product overview",
            "product name",
        ],
    ),
    (
        "faq-002",
        [
            "improve my sleep",
            "improve sleep",
            "sleep better",
            "better sleep",
            "how does sleeppilot improve",
            "how can sleeppilot improve",
        ],
    ),
    (
        "faq-011",
        [
            "bedtime routine",
            "create a bedtime routine",
            "plan bedtime",
            "plan my bedtime",
            "plan when to go to bed",
            "when to go to bed",
            "when should i go to bed",
            "when should i sleep",
            "go to bed",
            "target wake-up time",
            "target wake up time",
            "routine for me",
        ],
    ),
    (
        "faq-013",
        [
            "sleep too late",
            "sleep late",
            "go to bed earlier",
            "move bedtime earlier",
            "bedtime earlier",
            "too late every night",
        ],
    ),
    (
        "faq-004",
        [
            "what data",
            "which data",
            "collect data",
            "data collect",
            "data does sleeppilot collect",
            "does the product collect data",
            "is the product collect data",
        ],
    ),
    (
        "faq-007",
        [
            "protect my privacy",
            "protect privacy",
            "privacy policy",
            "sell data",
            "delete my data",
            "delete data",
            "disconnect integrations",
        ],
    ),
]


@dataclass(frozen=True)
class StoredChunk:
    id: str
    question: str
    answer: str
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    chunk: StoredChunk
    score: float


class SQLiteVectorStore:
    """Tiny persistent vector store backed by SQLite.

    SQLite stores the FAQ chunks and their embedding vectors as JSON. Similarity
    search is an in-process cosine scan, which is more than enough for a 10-15
    item FAQ and keeps setup light for reviewers.
    """

    def __init__(self, db_path: str | Path, embedder: HashingEmbedder | None = None):
        self.db_path = Path(db_path)
        self.embedder = embedder or HashingEmbedder()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def ensure_built(self, chunks: Iterable[dict[str, str]]) -> None:
        chunk_list = list(chunks)
        signature = self._signature(chunk_list)
        current = self.get_metadata("content_signature")
        dimensions = self.get_metadata("embedding_dimensions")
        namespace = self.get_metadata("embedding_namespace")

        if (
            current == signature
            and dimensions == str(self.embedder.dimensions)
            and namespace == self.embedder.namespace
        ):
            return

        self.rebuild(chunk_list, signature)

    def rebuild(self, chunks: Iterable[dict[str, str]], signature: str | None = None) -> None:
        chunk_list = list(chunks)
        signature = signature or self._signature(chunk_list)
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM metadata")
            for chunk in chunk_list:
                embedding = self.embedder.embed(self._retrieval_text(chunk))
                connection.execute(
                    """
                    INSERT INTO chunks (id, question, answer, text, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["id"],
                        chunk["question"],
                        chunk["answer"],
                        chunk["text"],
                        json.dumps(embedding),
                    ),
                )
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                [
                    ("content_signature", signature),
                    ("embedding_dimensions", str(self.embedder.dimensions)),
                    ("embedding_namespace", self.embedder.namespace),
                    ("chunk_count", str(len(chunk_list))),
                ],
            )

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0])

    def all_chunks(self) -> list[StoredChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, question, answer, text, embedding
                FROM chunks
                ORDER BY id
                """
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def similarity_search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.embedder.embed(query)
        query_tokens = set(expand_tokens(tokenize(query)))
        chunks = self.all_chunks()
        chunk_term_sets = {
            chunk.id: set(tokenize(self._retrieval_text(chunk)))
            for chunk in chunks
        }
        document_frequency = Counter(
            token
            for terms in chunk_term_sets.values()
            for token in terms
        )
        scored = []
        for chunk in chunks:
            chunk_tokens = chunk_term_sets[chunk.id]
            overlap = query_tokens & chunk_tokens
            entity_overlap = overlap & ENTITY_TOKENS
            lexical_bonus = self._lexical_bonus(
                overlap=overlap,
                entity_overlap=entity_overlap,
                document_frequency=document_frequency,
                document_count=len(chunks),
                query_term_count=len(query_tokens),
            )
            intent_bonus = self._intent_bonus(query, chunk)
            score = max(
                0.0,
                min(
                    1.0,
                    cosine_similarity(query_embedding, chunk.embedding)
                    + lexical_bonus
                    + intent_bonus,
                ),
            )
            scored.append(SearchResult(chunk=chunk, score=round(score, 6)))
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def get_metadata(self, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else str(row[0])

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _row_to_chunk(self, row: sqlite3.Row | tuple[str, str, str, str, str]) -> StoredChunk:
        return StoredChunk(
            id=row[0],
            question=row[1],
            answer=row[2],
            text=row[3],
            embedding=json.loads(row[4]),
        )

    def _retrieval_text(self, chunk: StoredChunk | dict[str, str]) -> str:
        if isinstance(chunk, StoredChunk):
            chunk_id = chunk.id
            text = chunk.text
        else:
            chunk_id = chunk["id"]
            text = chunk["text"]
        hints = FAQ_SEARCH_HINTS.get(chunk_id, "")
        return f"{text}\nRetrieval hints: {hints}"

    def _lexical_bonus(
        self,
        overlap: set[str],
        entity_overlap: set[str],
        document_frequency: Counter[str],
        document_count: int,
        query_term_count: int,
    ) -> float:
        if not overlap or query_term_count == 0:
            return 0.0

        idf_sum = sum(
            1.0 + math.log((document_count + 1) / (document_frequency[token] + 1))
            for token in overlap
        )
        normalized_idf = idf_sum / query_term_count
        entity_bonus = 0.055 * len(entity_overlap)
        return min(0.28, (0.09 * normalized_idf) + entity_bonus)

    def _intent_bonus(self, query: str, chunk: StoredChunk) -> float:
        normalized_query = " ".join(query.lower().split())
        asks_without_wearable = "without" in normalized_query and any(
            term in normalized_query
            for term in ["wearable", "device", "fitbit", "garmin", "apple watch"]
        )
        asks_supported_wearable = any(
            term in normalized_query
            for term in [
                "apple health",
                "fitbit",
                "garmin",
                "google fit",
                "support wearable",
                "supported wearable",
                "wearable support",
                "which wearable",
                "work with",
            ]
        )

        if chunk.id == "faq-005" and asks_without_wearable:
            return 0.7
        if chunk.id == "faq-006" and asks_supported_wearable and not asks_without_wearable:
            return 0.7

        asks_bedtime_planning = any(
            term in normalized_query
            for term in [
                "bedtime",
                "go to bed",
                "when should i sleep",
                "when to sleep",
                "sleep schedule",
                "target wake",
                "wind down",
            ]
        )
        asks_sleep_score = "score" in normalized_query
        asks_pricing = any(
            term in normalized_query
            for term in ["cost", "price", "pricing", "premium", "free plan"]
        )

        if asks_bedtime_planning and chunk.id == "faq-009" and not asks_sleep_score:
            return -0.25
        if asks_bedtime_planning and chunk.id == "faq-014" and not asks_pricing:
            return -0.25

        for chunk_id, phrases in INTENT_RULES:
            if chunk.id == chunk_id and any(phrase in normalized_query for phrase in phrases):
                return 0.65
        return 0.0

    def _signature(self, chunks: Iterable[dict[str, str]]) -> str:
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk["id"].encode("utf-8"))
            digest.update(chunk["text"].encode("utf-8"))
            digest.update(FAQ_SEARCH_HINTS.get(chunk["id"], "").encode("utf-8"))
        return digest.hexdigest()
