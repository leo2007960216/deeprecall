"""ChromaDB vector store adapter for DeepRecall."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from deeprecall.core.exceptions import VectorStoreConnectionError, VectorStoreError
from deeprecall.core.types import SearchResult
from deeprecall.vectorstores.base import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """Vector store adapter for ChromaDB.

    Supports both in-memory and persistent ChromaDB clients. If no embeddings
    or embedding_fn are provided, ChromaDB's default embedding function is used.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Path for persistent storage. None uses in-memory.
        host: ChromaDB server host for client/server mode.
        port: ChromaDB server port for client/server mode.
        embedding_fn: Optional custom embedding function.

    Example:
        ```python
        store = ChromaStore(collection_name="my_docs")
        store.add_documents(["Hello world", "Foo bar"])
        results = store.search("greeting", top_k=1)
        ```
    """

    def __init__(
        self,
        collection_name: str = "deeprecall",
        persist_directory: str | None = None,
        host: str | None = None,
        port: int = 8000,
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ):
        super().__init__(embedding_fn=embedding_fn)

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaStore. "
                "Install it with: pip install deeprecall[chroma]"
            ) from None

        try:
            if host is not None:
                self._client = chromadb.HttpClient(host=host, port=port)
            elif persist_directory is not None:
                self._client = chromadb.PersistentClient(path=persist_directory)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(name=collection_name)
        except ImportError:
            raise
        except Exception as e:
            raise VectorStoreConnectionError(f"Failed to connect to ChromaDB: {e}") from e

        self._collection_name = collection_name

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        self._validate_inputs(documents, metadatas, ids, embeddings)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        if embeddings is None:
            embeddings = self._generate_embeddings(documents)

        kwargs: dict[str, Any] = {
            "documents": documents,
            "ids": ids,
        }
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        try:
            self._collection.add(**kwargs)
        except Exception as e:
            raise VectorStoreError(f"ChromaDB add_documents failed: {e}") from e
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        try:
            doc_count = self._collection.count()
            if doc_count == 0:
                return []

            kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": min(top_k, doc_count),
            }

            if filters is not None:
                kwargs["where"] = filters

            query_embeddings = self._generate_embeddings([query])
            if query_embeddings is not None:
                kwargs["query_embeddings"] = query_embeddings
                kwargs.pop("query_texts", None)

            results = self._collection.query(**kwargs)
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"ChromaDB search failed: {e}") from e

        search_results: list[SearchResult] = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{} for _ in docs]
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            result_ids = results["ids"][0] if results["ids"] else [""] * len(docs)

            for doc, meta, dist, rid in zip(docs, metas, distances, result_ids, strict=False):
                # ChromaDB returns distances (lower = more similar).
                # For L2: score = 1 / (1 + dist) maps [0, inf) -> (0, 1]
                # For cosine/IP: dist is typically in [0, 2], so same formula works.
                score = 1.0 / (1.0 + dist) if dist >= 0 else 0.0
                search_results.append(
                    SearchResult(
                        content=doc,
                        metadata=dict(meta) if meta else {},
                        score=score,
                        id=rid,
                    )
                )

        return search_results

    def delete(self, ids: list[str]) -> None:
        try:
            self._collection.delete(ids=ids)
        except Exception as e:
            raise VectorStoreError(f"ChromaDB delete failed: {e}") from e

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as e:
            raise VectorStoreError(f"ChromaDB count failed: {e}") from e
