"""Tests for the FAISS vector store adapter."""

import tempfile

import pytest

from deeprecall.core.exceptions import ConfigurationError


def _dummy_embed(texts):
    """Deterministic dummy embedding function (4-dim vectors)."""
    return [[float(ord(t[0]) if t else 0), 0.1, 0.2, 0.3] for t in texts]


@pytest.fixture
def faiss_store():
    """Create a FAISSStore for testing (requires faiss-cpu)."""
    pytest.importorskip("faiss")
    from deeprecall.vectorstores.faiss import FAISSStore

    return FAISSStore(dimension=4, embedding_fn=_dummy_embed)


class TestFAISSStore:
    def test_add_and_count(self, faiss_store):
        ids = faiss_store.add_documents(["hello", "world"])
        assert len(ids) == 2
        assert faiss_store.count() == 2

    def test_search_returns_results(self, faiss_store):
        faiss_store.add_documents(["hello", "world", "foo"])
        results = faiss_store.search("hello", top_k=2)
        assert len(results) > 0
        assert results[0].content in ("hello", "world", "foo")

    def test_search_with_metadata(self, faiss_store):
        faiss_store.add_documents(
            ["doc1", "doc2"],
            metadatas=[{"tag": "a"}, {"tag": "b"}],
        )
        results = faiss_store.search("doc1", top_k=2)
        assert len(results) > 0

    def test_search_with_filters(self, faiss_store):
        faiss_store.add_documents(
            ["doc1", "doc2"],
            metadatas=[{"tag": "a"}, {"tag": "b"}],
        )
        results = faiss_store.search("doc1", top_k=10, filters={"tag": "a"})
        for r in results:
            assert r.metadata.get("tag") == "a"

    def test_delete(self, faiss_store):
        ids = faiss_store.add_documents(["hello", "world"])
        assert faiss_store.count() == 2
        faiss_store.delete([ids[0]])
        assert faiss_store.count() == 1

    def test_custom_ids(self, faiss_store):
        ids = faiss_store.add_documents(["hello"], ids=["my-id-1"])
        assert ids == ["my-id-1"]
        assert faiss_store.count() == 1

    def test_empty_search(self, faiss_store):
        results = faiss_store.search("anything")
        assert results == []

    def test_missing_embedding_fn_raises(self):
        pytest.importorskip("faiss")
        from deeprecall.vectorstores.faiss import FAISSStore

        with pytest.raises(ConfigurationError, match="embedding_fn"):
            FAISSStore(dimension=4, embedding_fn=None)


class TestFAISSPersistence:
    def test_save_and_load(self):
        pytest.importorskip("faiss")
        from deeprecall.vectorstores.faiss import FAISSStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSStore(dimension=4, embedding_fn=_dummy_embed, persist_path=tmpdir)
            store.add_documents(["hello", "world"], metadatas=[{"k": "v1"}, {"k": "v2"}])
            store.save()

            loaded = FAISSStore.load(tmpdir, embedding_fn=_dummy_embed)
            assert loaded.count() == 2
            results = loaded.search("hello", top_k=2)
            assert len(results) > 0

    def test_save_without_path_raises(self):
        pytest.importorskip("faiss")
        from deeprecall.vectorstores.faiss import FAISSStore

        store = FAISSStore(dimension=4, embedding_fn=_dummy_embed)
        with pytest.raises(ConfigurationError, match="persist_path"):
            store.save()


class TestFAISSCloseAndContextManager:
    def test_close_does_not_raise(self):
        pytest.importorskip("faiss")
        from deeprecall.vectorstores.faiss import FAISSStore

        store = FAISSStore(dimension=4, embedding_fn=_dummy_embed)
        store.add_documents(["test"])
        store.close()  # Should not raise

    def test_context_manager(self):
        pytest.importorskip("faiss")
        from deeprecall.vectorstores.faiss import FAISSStore

        with FAISSStore(dimension=4, embedding_fn=_dummy_embed) as store:
            ids = store.add_documents(["ctx manager test"])
            assert len(ids) == 1
            assert store.count() == 1


class TestFAISSLazyImport:
    def test_lazy_import_from_vectorstores(self):
        pytest.importorskip("faiss")
        from deeprecall.vectorstores import FAISSStore

        assert FAISSStore is not None
