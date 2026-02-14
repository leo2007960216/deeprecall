"""Vector store adapters for DeepRecall.

Supported stores:
    - ChromaStore: ChromaDB (local-first, built-in embeddings)
    - MilvusStore: Milvus (scalable, production-ready)
    - QdrantStore: Qdrant (fast, Rust-based)
    - PineconeStore: Pinecone (managed cloud service)
"""

from deeprecall.core.types import SearchResult
from deeprecall.vectorstores.base import BaseVectorStore

__all__ = ["BaseVectorStore", "SearchResult"]


def _lazy_import(name: str):
    """Lazy import vector store adapters to avoid requiring all dependencies."""
    if name == "ChromaStore":
        from deeprecall.vectorstores.chroma import ChromaStore

        return ChromaStore
    if name == "MilvusStore":
        from deeprecall.vectorstores.milvus import MilvusStore

        return MilvusStore
    if name == "QdrantStore":
        from deeprecall.vectorstores.qdrant import QdrantStore

        return QdrantStore
    if name == "PineconeStore":
        from deeprecall.vectorstores.pinecone import PineconeStore

        return PineconeStore
    raise AttributeError(f"module 'deeprecall.vectorstores' has no attribute {name!r}")


def __getattr__(name: str):
    return _lazy_import(name)
