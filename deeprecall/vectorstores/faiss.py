"""FAISS vector store adapter for DeepRecall."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import uuid
from collections.abc import Callable
from typing import Any

from deeprecall.core.exceptions import ConfigurationError, VectorStoreError
from deeprecall.core.types import SearchResult
from deeprecall.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)

_DEFAULT_DIMENSION = 1536


class FAISSStore(BaseVectorStore):
    """Vector store adapter for FAISS.

    Uses ``faiss.IndexIDMap2`` wrapping a flat index to support ID-based
    deletion.  Documents and metadata are stored in a parallel dict since
    FAISS only handles raw vectors.

    Args:
        dimension: Vector embedding dimension.
        index_type: ``"flat_l2"`` (default) or ``"flat_ip"`` (inner product).
        metric: Alias for *index_type* (``"l2"`` or ``"ip"``).
        persist_path: Optional directory for ``save()`` / ``load()`` persistence.
        embedding_fn: **Required** -- function that turns text into vectors.

    Example:
        ```python
        from deeprecall.vectorstores import FAISSStore

        store = FAISSStore(dimension=384, embedding_fn=my_embed_fn)
        store.add_documents(["Hello world"])
        results = store.search("greeting")
        ```
    """

    def __init__(
        self,
        dimension: int = _DEFAULT_DIMENSION,
        index_type: str = "flat_l2",
        metric: str | None = None,
        persist_path: str | None = None,
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ):
        super().__init__(embedding_fn=embedding_fn)

        if embedding_fn is None:
            raise ConfigurationError(
                "FAISSStore requires an embedding_fn. "
                "FAISS does not generate embeddings -- you must provide one."
            )

        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISSStore. "
                "Install it with: pip install deeprecall[faiss]"
            ) from None

        self._faiss = faiss
        self._np = np
        self._dimension = dimension
        self._persist_path = persist_path

        # Resolve index type
        idx_type = (metric or index_type).lower().replace("_", "").replace("-", "")
        if idx_type in ("flatl2", "l2"):
            base_index = faiss.IndexFlatL2(dimension)
        elif idx_type in ("flatip", "ip"):
            base_index = faiss.IndexFlatIP(dimension)
        else:
            raise ConfigurationError(
                f"Unknown FAISS index type: {index_type!r}. Use 'flat_l2' or 'flat_ip'."
            )

        # Wrap with IDMap2 so we can map int64 IDs and support deletion
        self._index = faiss.IndexIDMap2(base_index)

        # Parallel storage for docs and metadata (FAISS only stores vectors)
        self._docs: dict[int, str] = {}
        self._metas: dict[int, dict[str, Any]] = {}
        self._str_to_int: dict[str, int] = {}
        self._int_to_str: dict[int, str] = {}
        self._next_int_id: int = 0

    # ------------------------------------------------------------------
    # ID helpers -- FAISS requires int64 IDs; we map string IDs to ints
    # ------------------------------------------------------------------

    def _allocate_int_id(self, str_id: str) -> int:
        if str_id in self._str_to_int:
            return self._str_to_int[str_id]
        int_id = self._next_int_id
        self._next_int_id += 1
        self._str_to_int[str_id] = int_id
        self._int_to_str[int_id] = str_id
        return int_id

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

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
                raise ConfigurationError("FAISSStore requires embeddings. Provide an embedding_fn.")

        np = self._np
        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.shape[1] != self._dimension:
            raise VectorStoreError(
                f"Embedding dimension {vectors.shape[1]} != index dimension {self._dimension}"
            )

        int_ids = np.array([self._allocate_int_id(sid) for sid in ids], dtype=np.int64)

        try:
            self._index.add_with_ids(vectors, int_ids)
        except Exception as e:
            raise VectorStoreError(f"FAISS add_with_ids failed: {e}") from e

        for i, sid in enumerate(ids):
            iid = self._str_to_int[sid]
            self._docs[iid] = documents[i]
            self._metas[iid] = metadatas[i] if metadatas and i < len(metadatas) else {}

        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query_embeddings = self._generate_embeddings([query])
        if query_embeddings is None:
            raise ConfigurationError("FAISSStore requires an embedding_fn for search.")

        np = self._np
        query_vec = np.array(query_embeddings, dtype=np.float32)

        # Fetch more candidates when filtering (post-filter may discard many)
        fetch_k = top_k * 4 if filters else top_k
        k = min(fetch_k, self._index.ntotal) if self._index.ntotal > 0 else fetch_k
        if k == 0:
            return []

        try:
            distances, int_ids = self._index.search(query_vec, k)
        except Exception as e:
            raise VectorStoreError(f"FAISS search failed: {e}") from e

        results: list[SearchResult] = []
        for dist, iid in zip(distances[0], int_ids[0], strict=False):
            if iid == -1:
                continue  # FAISS sentinel for "no result"
            sid = self._int_to_str.get(int(iid), str(iid))
            content = self._docs.get(int(iid), "")
            meta = self._metas.get(int(iid), {})

            if filters:
                if not all(meta.get(fk) == fv for fk, fv in filters.items()):
                    continue

            # Convert L2 distance to a similarity-like score
            score = max(0.0, 1.0 / (1.0 + float(dist)))
            results.append(SearchResult(content=content, metadata=meta, score=score, id=sid))

            if len(results) >= top_k:
                break

        return results

    def delete(self, ids: list[str]) -> None:
        np = self._np
        int_ids = []
        for sid in ids:
            iid = self._str_to_int.get(sid)
            if iid is not None:
                int_ids.append(iid)
                self._docs.pop(iid, None)
                self._metas.pop(iid, None)
                del self._str_to_int[sid]
                del self._int_to_str[iid]

        if int_ids:
            try:
                self._index.remove_ids(np.array(int_ids, dtype=np.int64))
            except Exception as e:
                raise VectorStoreError(f"FAISS delete failed: {e}") from e

    def count(self) -> int:
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_path(path: str) -> str:
        """Normalise and validate a persist path to prevent traversal."""
        normalised = os.path.normpath(os.path.abspath(path))
        if "\0" in normalised:
            raise ConfigurationError("Persist path contains null bytes.")
        return normalised

    def save(self, path: str | None = None) -> None:
        """Save the FAISS index and metadata to disk.

        Args:
            path: Directory to save into. Falls back to ``persist_path``.
        """
        save_dir = self._validate_path(path or self._persist_path or "")
        if not save_dir or save_dir == os.path.normpath(os.path.abspath("")):
            raise ConfigurationError("No persist_path configured and no path provided to save().")

        os.makedirs(save_dir, exist_ok=True)
        self._faiss.write_index(self._index, os.path.join(save_dir, "index.faiss"))

        meta = {
            "docs": {str(k): v for k, v in self._docs.items()},
            "metas": {str(k): v for k, v in self._metas.items()},
            "str_to_int": self._str_to_int,
            "int_to_str": {str(k): v for k, v in self._int_to_str.items()},
            "next_int_id": self._next_int_id,
            "dimension": self._dimension,
        }

        # Atomic write: write to temp file then rename
        import tempfile

        meta_path = os.path.join(save_dir, "metadata.json")
        fd, tmp_path = tempfile.mkstemp(dir=save_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            os.replace(tmp_path, meta_path)
        except BaseException:
            # Clean up temp file on failure
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    @classmethod
    def load(
        cls,
        path: str,
        embedding_fn: Callable[[list[str]], list[list[float]]],
    ) -> FAISSStore:
        """Load a FAISSStore from disk.

        Args:
            path: Directory previously saved via ``save()``.
            embedding_fn: The embedding function (must match the one used at save time).

        Returns:
            A restored ``FAISSStore`` instance.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISSStore. "
                "Install it with: pip install deeprecall[faiss]"
            ) from None

        path = FAISSStore._validate_path(path)
        with open(os.path.join(path, "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)

        store = cls(
            dimension=meta["dimension"],
            embedding_fn=embedding_fn,
            persist_path=path,
        )
        store._index = faiss.read_index(os.path.join(path, "index.faiss"))
        store._docs = {int(k): v for k, v in meta["docs"].items()}
        store._metas = {int(k): v for k, v in meta["metas"].items()}
        store._str_to_int = meta["str_to_int"]
        store._int_to_str = {int(k): v for k, v in meta["int_to_str"].items()}
        store._next_int_id = meta["next_int_id"]
        return store
