"""Tests for PineconeStore (mocked -- no pinecone SDK required)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from deeprecall.core.types import SearchResult

# ---------------------------------------------------------------------------
# Mock Pinecone SDK
# ---------------------------------------------------------------------------


def _build_pinecone_mocks():
    """Create fake pinecone module."""
    mod = types.ModuleType("pinecone")

    class ServerlessSpec:
        def __init__(self, cloud="aws", region="us-east-1"):
            self.cloud = cloud
            self.region = region

    class FakeIndex:
        def __init__(self):
            self.data = {}

        def upsert(self, vectors=None, namespace=""):
            for v in vectors or []:
                self.data[v["id"]] = v

        def query(self, vector=None, top_k=5, include_metadata=True, namespace="", filter=None):
            matches = []
            for vid, v in list(self.data.items())[:top_k]:
                m = MagicMock()
                m.id = vid
                m.score = 0.9
                m.metadata = dict(v.get("metadata", {}))
                matches.append(m)
            result = MagicMock()
            result.matches = matches
            return result

        def delete(self, ids=None, namespace=""):
            for _id in ids or []:
                self.data.pop(_id, None)

        def describe_index_stats(self):
            stats = MagicMock()
            stats.total_vector_count = len(self.data)
            stats.namespaces = {}
            return stats

    class FakePinecone:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self._indexes = {}

        def list_indexes(self):
            return [MagicMock(name=n) for n in self._indexes]

        def create_index(self, name=None, dimension=None, metric=None, spec=None):
            self._indexes[name] = FakeIndex()

        def Index(self, name):
            if name not in self._indexes:
                self._indexes[name] = FakeIndex()
            return self._indexes[name]

    mod.Pinecone = FakePinecone
    mod.ServerlessSpec = ServerlessSpec
    return {"pinecone": mod}


def _embed_fn(texts: list[str]) -> list[list[float]]:
    return [[0.1] * 4 for _ in texts]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPineconeStore:
    """Tests for PineconeStore with mocked Pinecone SDK."""

    def _make_store(self, **kwargs):
        mods = _build_pinecone_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.vectorstores.pinecone" in sys.modules:
                del sys.modules["deeprecall.vectorstores.pinecone"]
            from deeprecall.vectorstores.pinecone import PineconeStore

            return PineconeStore(
                index_name="test-idx",
                api_key="test-key",
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
        ids = store.add_documents(["a", "b"], ids=["id1", "id2"])
        assert ids == ["id1", "id2"]

    def test_add_with_metadata(self):
        store = self._make_store()
        store.add_documents(["hello"], metadatas=[{"key": "val"}], ids=["m1"])
        assert store.count() == 1

    def test_search_returns_results(self):
        store = self._make_store()
        store.add_documents(["hello world"], ids=["hw1"])
        results = store.search("hello")
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)

    def test_delete(self):
        store = self._make_store()
        store.add_documents(["a", "b"], ids=["d1", "d2"])
        store.delete(["d1"])
        assert store.count() == 1

    def test_missing_api_key_raises(self):
        mods = _build_pinecone_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.vectorstores.pinecone" in sys.modules:
                del sys.modules["deeprecall.vectorstores.pinecone"]
            from deeprecall.vectorstores.pinecone import PineconeStore

            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="API key is required"):
                    PineconeStore(
                        index_name="x",
                        api_key=None,
                        embedding_fn=_embed_fn,
                        dimension=4,
                    )

    def test_missing_embedding_fn_raises_on_add(self):
        mods = _build_pinecone_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.vectorstores.pinecone" in sys.modules:
                del sys.modules["deeprecall.vectorstores.pinecone"]
            from deeprecall.vectorstores.pinecone import PineconeStore

            store = PineconeStore(
                index_name="x",
                api_key="k",
                dimension=4,
                embedding_fn=None,
            )
            with pytest.raises(ValueError, match="requires embeddings"):
                store.add_documents(["hello"])

    def test_deeprecall_content_key_in_metadata(self):
        store = self._make_store()
        store.add_documents(["my text"], ids=["c1"])
        idx = store._index
        vec = idx.data["c1"]
        assert vec["metadata"]["_deeprecall_content"] == "my text"

    def test_close_does_not_raise(self):
        store = self._make_store()
        store.add_documents(["test"])
        store.close()  # Should not raise

    def test_context_manager(self):
        mods = _build_pinecone_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.vectorstores.pinecone" in sys.modules:
                del sys.modules["deeprecall.vectorstores.pinecone"]
            from deeprecall.vectorstores.pinecone import PineconeStore

            with PineconeStore(
                index_name="ctx-test",
                api_key="test-key",
                dimension=4,
                embedding_fn=_embed_fn,
            ) as store:
                ids = store.add_documents(["ctx mgr test"])
                assert len(ids) == 1

    def test_search_without_embedding_fn_raises(self):
        mods = _build_pinecone_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.vectorstores.pinecone" in sys.modules:
                del sys.modules["deeprecall.vectorstores.pinecone"]
            from deeprecall.vectorstores.pinecone import PineconeStore

            store = PineconeStore(
                index_name="x",
                api_key="k",
                dimension=4,
                embedding_fn=None,
            )
            with pytest.raises(ValueError, match="requires an embedding_fn"):
                store.search("q")
