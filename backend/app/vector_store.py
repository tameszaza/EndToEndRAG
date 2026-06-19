from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.embeddings import Embedder, cosine_similarity, create_default_embedder


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

    def __init__(self, db_path: str | Path, embedder: Embedder | None = None):
        self.db_path = Path(db_path)
        self.embedder = embedder or create_default_embedder()
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
                embedding = self.embedder.embed(
                    self._retrieval_text(chunk),
                    task_type="RETRIEVAL_DOCUMENT",
                    title=chunk["question"],
                )
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

        query_embedding = self.embedder.embed(query, task_type="RETRIEVAL_QUERY")
        chunks = self.all_chunks()
        scored: list[tuple[float, SearchResult]] = []

        for chunk in chunks:
            raw_score = cosine_similarity(query_embedding, chunk.embedding)
            display_score = max(0.0, min(1.0, raw_score))

            scored.append(
                (
                    raw_score,
                    SearchResult(
                        chunk=chunk,
                        score=round(display_score, 6),
                    ),
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)

        return [result for _, result in scored[:top_k]]

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
            return chunk.text
        return chunk["text"]

    def _signature(self, chunks: Iterable[dict[str, str]]) -> str:
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk["id"].encode("utf-8"))
            digest.update(chunk["text"].encode("utf-8"))
        return digest.hexdigest()
