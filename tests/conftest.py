"""Shared test fixtures for DeepRecall."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from deeprecall.core.types import SearchResult
from deeprecall.vectorstores.base import BaseVectorStore


class MockVectorStore(BaseVectorStore):
    """In-memory mock vector store for testing."""

    def __init__(self):
        super().__init__()
        self._documents: dict[str, dict[str, Any]] = {}

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        import uuid

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        for i, (doc_id, doc) in enumerate(zip(ids, documents, strict=False)):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            self._documents[doc_id] = {"content": doc, "metadata": meta}

        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        results = []
        for doc_id, doc in self._documents.items():
            # Simple keyword match scoring
            query_lower = query.lower()
            content_lower = doc["content"].lower()
            score = 0.0
            for word in query_lower.split():
                if word in content_lower:
                    score += 1.0 / len(query_lower.split())

            if filters:
                skip = False
                for key, value in filters.items():
                    if doc["metadata"].get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            results.append(
                SearchResult(
                    content=doc["content"],
                    metadata=doc["metadata"],
                    score=score,
                    id=doc_id,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._documents.pop(doc_id, None)

    def count(self) -> int:
        return len(self._documents)


@pytest.fixture
def mock_vectorstore() -> MockVectorStore:
    """Provide a mock vector store pre-loaded with sample documents."""
    store = MockVectorStore()
    store.add_documents(
        documents=[
            "Python is a high-level programming language created by Guido van Rossum.",
            "Rust is a systems programming language focused on safety and performance.",
            "JavaScript is the language of the web, running in browsers worldwide.",
            "Machine learning uses statistical methods to learn from data.",
            "Vector databases store and search high-dimensional embeddings efficiently.",
        ],
        metadatas=[
            {"topic": "python", "type": "language"},
            {"topic": "rust", "type": "language"},
            {"topic": "javascript", "type": "language"},
            {"topic": "ml", "type": "concept"},
            {"topic": "vectordb", "type": "concept"},
        ],
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
    )
    return store


@pytest.fixture
def mock_rlm_result() -> MagicMock:
    """Provide a mock RLM completion result."""
    result = MagicMock()
    result.response = "This is a test answer from the recursive reasoning engine."
    result.usage_summary = MagicMock()
    result.usage_summary.model_usage_summaries = {
        "gpt-4o-mini": MagicMock(
            total_input_tokens=100,
            total_output_tokens=50,
            total_calls=3,
        )
    }
    result.execution_time = 1.5
    return result
