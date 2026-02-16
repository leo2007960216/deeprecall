"""Tests for ChromaStore adapter."""

from __future__ import annotations

import pytest

try:
    import chromadb  # noqa: F401

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

from deeprecall.core.types import SearchResult


@pytest.mark.skipif(not HAS_CHROMA, reason="chromadb not installed")
class TestChromaStore:
    def test_init_in_memory(self):
        from deeprecall.vectorstores.chroma import ChromaStore

        store = ChromaStore(collection_name="test_collection")
        assert store.count() == 0

    def test_add_and_search(self):
        from deeprecall.vectorstores.chroma import ChromaStore

        store = ChromaStore(collection_name="test_add_search")
        ids = store.add_documents(
            documents=["Hello world", "Goodbye world"],
            metadatas=[{"lang": "en"}, {"lang": "en"}],
        )
        assert len(ids) == 2
        assert store.count() == 2

        results = store.search("Hello", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert "Hello" in results[0].content

    def test_delete(self):
        from deeprecall.vectorstores.chroma import ChromaStore

        store = ChromaStore(collection_name="test_delete")
        ids = store.add_documents(documents=["Doc 1", "Doc 2"])
        assert store.count() == 2

        store.delete([ids[0]])
        assert store.count() == 1

    def test_custom_ids(self):
        from deeprecall.vectorstores.chroma import ChromaStore

        store = ChromaStore(collection_name="test_custom_ids")
        ids = store.add_documents(
            documents=["Test doc"],
            ids=["my-custom-id"],
        )
        assert ids == ["my-custom-id"]

    def test_search_with_filter(self):
        from deeprecall.vectorstores.chroma import ChromaStore

        store = ChromaStore(collection_name="test_filter")
        store.add_documents(
            documents=["Python doc", "Rust doc"],
            metadatas=[{"lang": "python"}, {"lang": "rust"}],
        )

        results = store.search("programming", top_k=5, filters={"lang": "python"})
        assert all(r.metadata.get("lang") == "python" for r in results)

    def test_close_does_not_raise(self):
        from deeprecall.vectorstores.chroma import ChromaStore

        store = ChromaStore(collection_name="test_close")
        store.add_documents(["test"])
        store.close()  # Should not raise

    def test_context_manager(self):
        from deeprecall.vectorstores.chroma import ChromaStore

        with ChromaStore(collection_name="test_ctx_mgr") as store:
            ids = store.add_documents(["Hello context manager"])
            assert len(ids) == 1
            assert store.count() == 1
        # After exit, store.close() was called -- no assertion needed,
        # just verify the block completed without error
