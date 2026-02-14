"""Tests for BaseVectorStore and SearchResult."""

from __future__ import annotations

from deeprecall.core.types import SearchResult, Source


class TestSearchResult:
    def test_to_dict(self):
        result = SearchResult(
            content="Hello world",
            metadata={"source": "test"},
            score=0.95,
            id="doc1",
        )
        d = result.to_dict()
        assert d["content"] == "Hello world"
        assert d["score"] == 0.95
        assert d["metadata"]["source"] == "test"

    def test_default_values(self):
        result = SearchResult(content="text")
        assert result.metadata == {}
        assert result.score == 0.0
        assert result.id == ""


class TestSource:
    def test_from_search_result(self):
        sr = SearchResult(
            content="Test content",
            metadata={"key": "value"},
            score=0.8,
            id="abc",
        )
        source = Source.from_search_result(sr)
        assert source.content == "Test content"
        assert source.score == 0.8
        assert source.id == "abc"


class TestMockVectorStore:
    """Test the mock vector store from conftest to ensure it works correctly."""

    def test_add_and_count(self, mock_vectorstore):
        assert mock_vectorstore.count() == 5

    def test_search_returns_results(self, mock_vectorstore):
        results = mock_vectorstore.search("python language", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_with_filter(self, mock_vectorstore):
        results = mock_vectorstore.search("language", top_k=10, filters={"type": "language"})
        assert all(r.metadata.get("type") == "language" for r in results)

    def test_delete(self, mock_vectorstore):
        mock_vectorstore.delete(["doc1"])
        assert mock_vectorstore.count() == 4

    def test_search_scoring(self, mock_vectorstore):
        results = mock_vectorstore.search("python", top_k=5)
        # "python" should match the Python document highest
        assert results[0].content.lower().startswith("python")
