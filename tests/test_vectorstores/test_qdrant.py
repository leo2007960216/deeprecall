"""Tests for QdrantStore (mocked -- no qdrant-client required)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from deeprecall.core.types import SearchResult

# ---------------------------------------------------------------------------
# Mock qdrant-client SDK
# ---------------------------------------------------------------------------


def _build_qdrant_mocks():
    """Create fake qdrant_client module with models."""
    mod = types.ModuleType("qdrant_client")
    models_mod = types.ModuleType("qdrant_client.models")

    class Distance:
        COSINE = "Cosine"
        EUCLID = "Euclid"
        DOT = "Dot"

    class VectorParams:
        def __init__(self, size=0, distance=None):
            self.size = size
            self.distance = distance

    class PointStruct:
        def __init__(self, id=None, vector=None, payload=None):
            self.id = id
            self.vector = vector
            self.payload = payload

    class PointIdsList:
        def __init__(self, points=None):
            self.points = points or []

    class FieldCondition:
        def __init__(self, key="", match=None):
            self.key = key
            self.match = match

    class MatchValue:
        def __init__(self, value=None):
            self.value = value

    class Filter:
        def __init__(self, must=None):
            self.must = must or []

    class FakeClient:
        def __init__(self, url=None, api_key=None, **kwargs):
            self._data = {}
            self._collections = []

        def get_collections(self):
            resp = MagicMock()
            resp.collections = [MagicMock(name=n) for n in self._collections]
            return resp

        def create_collection(self, collection_name=None, vectors_config=None):
            self._collections.append(collection_name)

        def upsert(self, collection_name=None, points=None):
            for p in points or []:
                self._data[p.id] = p

        def search(
            self,
            collection_name=None,
            query_vector=None,
            limit=5,
            with_payload=True,
            query_filter=None,
        ):
            results = []
            for pid, point in list(self._data.items())[:limit]:
                m = MagicMock()
                m.id = pid
                m.score = 0.85
                m.payload = point.payload
                results.append(m)
            return results

        def delete(self, collection_name=None, points_selector=None):
            for pid in points_selector.points:
                # Handle both int and string keys (Qdrant adapter converts numeric strings)
                self._data.pop(pid, None)
                self._data.pop(str(pid), None)

        def get_collection(self, collection_name):
            info = MagicMock()
            info.points_count = len(self._data)
            return info

        def close(self):
            pass

    # Wire up
    models_mod.Distance = Distance
    models_mod.VectorParams = VectorParams
    models_mod.PointStruct = PointStruct
    models_mod.PointIdsList = PointIdsList
    models_mod.FieldCondition = FieldCondition
    models_mod.MatchValue = MatchValue
    models_mod.Filter = Filter
    mod.QdrantClient = FakeClient
    mod.models = models_mod

    return {"qdrant_client": mod, "qdrant_client.models": models_mod}


def _embed_fn(texts: list[str]) -> list[list[float]]:
    return [[0.1] * 4 for _ in texts]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQdrantStore:
    """Tests for QdrantStore with mocked qdrant-client."""

    def _make_store(self, **kwargs):
        mods = _build_qdrant_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.vectorstores.qdrant" in sys.modules:
                del sys.modules["deeprecall.vectorstores.qdrant"]
            from deeprecall.vectorstores.qdrant import QdrantStore

            return QdrantStore(
                collection_name="test_col",
                dimension=4,
                embedding_fn=_embed_fn,
                **kwargs,
            )

    def test_add_and_count(self):
        store = self._make_store()
        ids = store.add_documents(["doc1", "doc2"])
        assert len(ids) == 2
        assert store.count() == 2

    def test_add_with_custom_ids(self):
        store = self._make_store()
        ids = store.add_documents(["a"], ids=["myid"])
        assert ids == ["myid"]

    def test_add_with_metadata(self):
        store = self._make_store()
        store.add_documents(["hello"], metadatas=[{"topic": "test"}], ids=["m1"])
        assert store.count() == 1

    def test_search_returns_results(self):
        store = self._make_store()
        store.add_documents(["hello world"], ids=["hw"])
        results = store.search("hello")
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)

    def test_delete(self):
        store = self._make_store()
        store.add_documents(["a", "b"], ids=["d1", "d2"])
        store.delete(["d1"])
        assert store.count() == 1

    def test_close(self):
        store = self._make_store()
        store.close()  # should not raise

    def test_invalid_distance_raises(self):
        mods = _build_qdrant_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.vectorstores.qdrant" in sys.modules:
                del sys.modules["deeprecall.vectorstores.qdrant"]
            # ValueError inside __init__ is wrapped by generic except -> VectorStoreConnectionError
            from deeprecall.core.exceptions import VectorStoreConnectionError
            from deeprecall.vectorstores.qdrant import QdrantStore

            with pytest.raises(VectorStoreConnectionError, match="Invalid distance"):
                QdrantStore(
                    collection_name="x",
                    dimension=4,
                    distance="L2",
                    embedding_fn=_embed_fn,
                )

    def test_missing_embedding_fn_raises_on_add(self):
        mods = _build_qdrant_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.vectorstores.qdrant" in sys.modules:
                del sys.modules["deeprecall.vectorstores.qdrant"]
            from deeprecall.vectorstores.qdrant import QdrantStore

            store = QdrantStore(
                collection_name="x",
                dimension=4,
                embedding_fn=None,
            )
            with pytest.raises(ValueError, match="requires embeddings"):
                store.add_documents(["hello"])

    def test_context_manager(self):
        mods = _build_qdrant_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.vectorstores.qdrant" in sys.modules:
                del sys.modules["deeprecall.vectorstores.qdrant"]
            from deeprecall.vectorstores.qdrant import QdrantStore

            with QdrantStore(
                collection_name="ctx_test",
                dimension=4,
                embedding_fn=_embed_fn,
            ) as store:
                ids = store.add_documents(["context manager test"])
                assert len(ids) == 1

    def test_delete_with_numeric_ids(self):
        store = self._make_store()
        store.add_documents(["a"], ids=["123"])
        store.delete(["123"])
        assert store.count() == 0
