"""Tests for MilvusStore (mocked -- no pymilvus required)."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from deeprecall.core.exceptions import VectorStoreError
from deeprecall.core.types import SearchResult

# ---------------------------------------------------------------------------
# Mock pymilvus SDK
# ---------------------------------------------------------------------------


def _build_milvus_mocks():
    """Create fake pymilvus module."""
    mod = types.ModuleType("pymilvus")

    class FakeClient:
        def __init__(self, uri=None, **kwargs):
            self._data: dict[str, dict] = {}
            self._collections: list[str] = []

        def has_collection(self, name):
            return name in self._collections

        def create_collection(
            self, collection_name=None, dimension=None, metric_type=None, auto_id=False
        ):
            self._collections.append(collection_name)

        def insert(self, collection_name=None, data=None):
            for record in data or []:
                self._data[record["id"]] = record

        def search(self, collection_name=None, data=None, limit=5, output_fields=None, filter=None):
            hits = []
            for rid, record in list(self._data.items())[:limit]:
                hits.append(
                    {
                        "id": rid,
                        "distance": 0.9,
                        "entity": {
                            "content": record.get("content", ""),
                            "metadata": record.get("metadata", {}),
                        },
                    }
                )
            return [hits]

        def delete(self, collection_name=None, ids=None):
            for _id in ids or []:
                self._data.pop(_id, None)

        def get_collection_stats(self, collection_name):
            return {"row_count": len(self._data)}

        def close(self):
            pass

    mod.MilvusClient = FakeClient
    return {"pymilvus": mod}


def _embed_fn(texts: list[str]) -> list[list[float]]:
    return [[0.1] * 4 for _ in texts]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMilvusStore:
    """Tests for MilvusStore with mocked pymilvus."""

    def _make_store(self, **kwargs):
        mods = _build_milvus_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.vectorstores.milvus" in sys.modules:
                del sys.modules["deeprecall.vectorstores.milvus"]
            from deeprecall.vectorstores.milvus import MilvusStore

            return MilvusStore(
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
        ids = store.add_documents(["a"], ids=["custom1"])
        assert ids == ["custom1"]

    def test_add_with_metadata(self):
        store = self._make_store()
        store.add_documents(["hi"], metadatas=[{"topic": "test"}], ids=["m1"])
        assert store.count() == 1

    def test_search_returns_results(self):
        store = self._make_store()
        store.add_documents(["hello world"], ids=["hw"])
        results = store.search("hello")
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert results[0].content == "hello world"

    def test_search_with_filter(self):
        store = self._make_store()
        store.add_documents(["test"], ids=["f1"])
        results = store.search("test", filters={"topic": "demo"})
        assert isinstance(results, list)

    def test_delete(self):
        store = self._make_store()
        store.add_documents(["a", "b"], ids=["d1", "d2"])
        store.delete(["d1"])
        assert store.count() == 1

    def test_close(self):
        store = self._make_store()
        store.close()  # should not raise

    def test_missing_embedding_fn_raises_on_add(self):
        mods = _build_milvus_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.vectorstores.milvus" in sys.modules:
                del sys.modules["deeprecall.vectorstores.milvus"]
            from deeprecall.vectorstores.milvus import MilvusStore

            store = MilvusStore(
                collection_name="x",
                dimension=4,
                embedding_fn=None,
            )
            with pytest.raises(ValueError, match="requires embeddings"):
                store.add_documents(["hello"])

    def test_invalid_filter_key_raises(self):
        store = self._make_store()
        store.add_documents(["x"], ids=["ik1"])
        with pytest.raises(VectorStoreError, match="Invalid filter key"):
            store.search("q", filters={"bad key!": "val"})

    def test_nan_filter_value_raises(self):
        store = self._make_store()
        store.add_documents(["x"], ids=["nf1"])
        with pytest.raises(VectorStoreError, match="NaN"):
            store.search("q", filters={"score": float("nan")})

    def test_inf_filter_value_raises(self):
        store = self._make_store()
        store.add_documents(["x"], ids=["if1"])
        with pytest.raises(VectorStoreError, match="NaN/Inf"):
            store.search("q", filters={"score": float("inf")})

    def test_string_filter_escaping(self):
        store = self._make_store()
        store.add_documents(["x"], ids=["sf1"])
        # Should not raise -- value is escaped
        results = store.search("q", filters={"name": 'value"with"quotes'})
        assert isinstance(results, list)

    def test_bool_filter(self):
        store = self._make_store()
        store.add_documents(["x"], ids=["bf1"])
        results = store.search("q", filters={"active": True})
        assert isinstance(results, list)
