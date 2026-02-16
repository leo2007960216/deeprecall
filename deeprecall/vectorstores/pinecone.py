"""Pinecone vector store adapter for DeepRecall."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from deeprecall.core.exceptions import VectorStoreConnectionError, VectorStoreError
from deeprecall.core.types import SearchResult
from deeprecall.vectorstores.base import BaseVectorStore

_DEFAULT_DIMENSION = 1536


class PineconeStore(BaseVectorStore):
    """Vector store adapter for Pinecone.

    Connects to a Pinecone index for managed vector similarity search.
    Requires an embedding function and a Pinecone API key.

    Args:
        index_name: Name of the Pinecone index.
        api_key: Pinecone API key.
        dimension: Vector embedding dimension.
        metric: Distance metric ("cosine", "euclidean", "dotproduct").
        namespace: Optional namespace within the index.
        embedding_fn: Function to generate embeddings from text.
        cloud: Cloud provider for serverless ("aws", "gcp", "azure").
        region: Cloud region for serverless index.

    Example:
        ```python
        from deeprecall.vectorstores import PineconeStore

        store = PineconeStore(
            index_name="my-index",
            api_key="pc-...",
            embedding_fn=my_embed_fn,
        )
        store.add_documents(["Hello world"])
        results = store.search("greeting")
        ```
    """

    def __init__(
        self,
        index_name: str = "deeprecall",
        api_key: str | None = None,
        dimension: int = _DEFAULT_DIMENSION,
        metric: str = "cosine",
        namespace: str = "",
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        super().__init__(embedding_fn=embedding_fn)

        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError:
            raise ImportError(
                "pinecone is required for PineconeStore. "
                "Install it with: pip install deeprecall[pinecone]"
            ) from None

        if api_key is None:
            import os

            api_key = os.getenv("PINECONE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Pinecone API key is required. Pass api_key or set PINECONE_API_KEY."
                )

        try:
            self._pc = Pinecone(api_key=api_key)
            self._namespace = namespace
            self._dimension = dimension

            # Create index if it doesn't exist
            existing = [idx.name for idx in self._pc.list_indexes()]
            if index_name not in existing:
                self._pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )

            self._index = self._pc.Index(index_name)
        except ImportError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise VectorStoreConnectionError(f"Failed to connect to Pinecone: {e}") from e

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
            if embeddings is None:
                raise ValueError(
                    "PineconeStore requires embeddings. Either provide an embedding_fn "
                    "in the constructor or pass pre-computed embeddings."
                )

        vectors = []
        for i, (doc_id, doc, emb) in enumerate(zip(ids, documents, embeddings, strict=False)):
            meta: dict[str, Any] = {}
            if metadatas and i < len(metadatas):
                meta.update(metadatas[i])
            # Store document text under a reserved key so user metadata
            # with a "content" key cannot overwrite it.
            meta["_deeprecall_content"] = doc

            vectors.append({"id": doc_id, "values": emb, "metadata": meta})

        # Upsert in batches of 100
        batch_size = 100
        for batch_start in range(0, len(vectors), batch_size):
            batch = vectors[batch_start : batch_start + batch_size]
            try:
                self._index.upsert(vectors=batch, namespace=self._namespace)
            except Exception as e:
                raise VectorStoreError(
                    f"Pinecone upsert failed at batch starting index {batch_start}: {e}"
                ) from e

        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query_embeddings = self._generate_embeddings([query])
        if query_embeddings is None:
            raise ValueError(
                "PineconeStore requires an embedding_fn for search. Provide one in the constructor."
            )

        query_kwargs: dict[str, Any] = {
            "vector": query_embeddings[0],
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self._namespace,
        }

        if filters is not None:
            query_kwargs["filter"] = filters

        try:
            response = self._index.query(**query_kwargs)
        except Exception as e:
            raise VectorStoreError(f"Pinecone search failed: {e}") from e

        # Pinecone SDK v5+ returns objects with attribute access (.matches, .id, .score)
        # Older versions may return dicts. Handle both gracefully.
        # Use `is None` checks instead of `or` to preserve falsy values (0.0, "", {}).
        _matches = getattr(response, "matches", None)
        matches = _matches if _matches is not None else response.get("matches", [])

        search_results: list[SearchResult] = []
        for match in matches:
            _id = getattr(match, "id", None)
            match_id = _id if _id is not None else match.get("id", "")
            _score = getattr(match, "score", None)
            match_score = _score if _score is not None else match.get("score", 0.0)
            _meta = getattr(match, "metadata", None)
            raw_metadata = _meta if _meta is not None else match.get("metadata", {})
            if raw_metadata is None:
                raw_metadata = {}

            # Read from reserved key; fall back to legacy "content" for old data.
            # Use `is not None` to preserve empty-string content.
            content = raw_metadata.pop("_deeprecall_content", None)
            if content is None:
                content = raw_metadata.pop("content", "")
            metadata = dict(raw_metadata)
            search_results.append(
                SearchResult(
                    content=content,
                    metadata=metadata,
                    score=float(match_score),
                    id=str(match_id),
                )
            )

        return search_results

    def delete(self, ids: list[str]) -> None:
        try:
            self._index.delete(ids=ids, namespace=self._namespace)
        except Exception as e:
            raise VectorStoreError(f"Pinecone delete failed: {e}") from e

    def count(self) -> int:
        try:
            stats = self._index.describe_index_stats()
            # SDK v5+ returns object with attributes; older returns dict.
            # Use `is None` to preserve 0 counts.
            if self._namespace:
                _ns = getattr(stats, "namespaces", None)
                namespaces = _ns if _ns is not None else stats.get("namespaces", {})
                ns_stats = namespaces.get(self._namespace, {})
                _vc = getattr(ns_stats, "vector_count", None)
                count = _vc if _vc is not None else ns_stats.get("vector_count", 0)
            else:
                _tc = getattr(stats, "total_vector_count", None)
                count = _tc if _tc is not None else stats.get("total_vector_count", 0)
            return int(count)
        except Exception as e:
            raise VectorStoreError(f"Pinecone count failed: {e}") from e
