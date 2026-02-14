"""Pinecone vector store adapter for DeepRecall."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

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

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
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
            meta: dict[str, Any] = {"content": doc}
            if metadatas and i < len(metadatas):
                meta.update(metadatas[i])

            vectors.append({"id": doc_id, "values": emb, "metadata": meta})

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=self._namespace)

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

        results = self._index.query(**query_kwargs)

        search_results: list[SearchResult] = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            content = metadata.pop("content", "")
            search_results.append(
                SearchResult(
                    content=content,
                    metadata=metadata,
                    score=match.get("score", 0.0),
                    id=match.get("id", ""),
                )
            )

        return search_results

    def delete(self, ids: list[str]) -> None:
        self._index.delete(ids=ids, namespace=self._namespace)

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        if self._namespace:
            ns_stats = stats.get("namespaces", {}).get(self._namespace, {})
            return ns_stats.get("vector_count", 0)
        return stats.get("total_vector_count", 0)
