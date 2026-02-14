"""Base vector store interface for DeepRecall."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from deeprecall.core.types import SearchResult


class BaseVectorStore(ABC):
    """Abstract base class for all vector store adapters.

    Every vector store adapter must implement these four methods to provide
    a consistent interface for the DeepRecall engine.

    Args:
        embedding_fn: Optional custom embedding function. If provided, it will be used
            instead of the vector store's built-in embedding. Signature:
            (texts: list[str]) -> list[list[float]]
    """

    def __init__(self, embedding_fn: Callable[[list[str]], list[list[float]]] | None = None):
        self.embedding_fn = embedding_fn

    @abstractmethod
    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of document texts to add.
            metadatas: Optional metadata dicts for each document.
            ids: Optional unique IDs. Auto-generated if not provided.
            embeddings: Optional pre-computed embeddings. If not provided,
                the store will generate them using its embedding function.

        Returns:
            List of document IDs that were added.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the vector store for documents similar to the query.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the total number of documents in the store."""
        raise NotImplementedError

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings using the custom embedding function if provided."""
        if self.embedding_fn is not None:
            return self.embedding_fn(texts)
        return None
