"""Tests for the reranking system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from deeprecall.core.reranker import BaseReranker
from deeprecall.core.types import SearchResult


class SimpleReranker(BaseReranker):
    """Test reranker that reverses the score order."""

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5):
        reversed_results = list(reversed(results))
        reranked = []
        for i, r in enumerate(reversed_results[:top_k]):
            reranked.append(
                SearchResult(
                    content=r.content,
                    metadata=r.metadata,
                    score=1.0 - (i * 0.1),
                    id=r.id,
                )
            )
        return reranked


class TestBaseReranker:
    def test_rerank_basic(self):
        reranker = SimpleReranker()
        results = [
            SearchResult(content="Doc A", score=0.9, id="1"),
            SearchResult(content="Doc B", score=0.8, id="2"),
            SearchResult(content="Doc C", score=0.7, id="3"),
        ]

        reranked = reranker.rerank("test query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].content == "Doc C"  # Reversed order
        assert reranked[0].score == 1.0

    def test_rerank_empty_results(self):
        reranker = SimpleReranker()
        reranked = reranker.rerank("test", [], top_k=5)
        assert reranked == []

    def test_top_k_limits_results(self):
        reranker = SimpleReranker()
        results = [SearchResult(content=f"Doc {i}", score=0.5, id=str(i)) for i in range(10)]

        reranked = reranker.rerank("test", results, top_k=3)
        assert len(reranked) == 3


# ---------------------------------------------------------------------------
# CohereReranker (with mocked cohere client)
# ---------------------------------------------------------------------------


class TestCohereReranker:
    """Tests for CohereReranker with fully mocked cohere module."""

    def _make_reranker(self, mock_cohere_module):
        """Build a CohereReranker using a mocked cohere module."""
        import sys

        with patch.dict(sys.modules, {"cohere": mock_cohere_module}):
            # Reload to pick up the mocked module
            import importlib

            import deeprecall.core.reranker as _mod

            importlib.reload(_mod)
            return _mod.CohereReranker(api_key="test-key")

    def test_rerank_returns_reordered_results(self):
        mock_cohere = MagicMock()
        mock_client = MagicMock()
        mock_cohere.Client.return_value = mock_client

        mock_result_1 = MagicMock(index=1, relevance_score=0.95)
        mock_result_2 = MagicMock(index=0, relevance_score=0.70)
        mock_client.rerank.return_value = MagicMock(results=[mock_result_1, mock_result_2])

        reranker = self._make_reranker(mock_cohere)
        results = [
            SearchResult(content="Doc A about cats", score=0.8, id="a"),
            SearchResult(content="Doc B about dogs", score=0.9, id="b"),
        ]
        reranked = reranker.rerank("pets", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].content == "Doc B about dogs"
        assert reranked[0].score == 0.95
        assert reranked[1].content == "Doc A about cats"
        assert reranked[1].score == 0.70

    def test_empty_results(self):
        mock_cohere = MagicMock()
        reranker = self._make_reranker(mock_cohere)
        assert reranker.rerank("query", [], top_k=5) == []

    def test_api_error_falls_back_to_original(self):
        mock_cohere = MagicMock()
        mock_client = MagicMock()
        mock_cohere.Client.return_value = mock_client
        mock_client.rerank.side_effect = RuntimeError("API timeout")

        reranker = self._make_reranker(mock_cohere)
        results = [
            SearchResult(content="A", score=0.9, id="1"),
            SearchResult(content="B", score=0.8, id="2"),
        ]
        reranked = reranker.rerank("query", results, top_k=1)
        assert len(reranked) == 1
        assert reranked[0].content == "A"


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker with fully mocked sentence_transformers."""

    def _make_reranker(self, mock_st_module, model_name="test-model"):
        import sys

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            import importlib

            import deeprecall.core.reranker as _mod

            importlib.reload(_mod)
            return _mod.CrossEncoderReranker(model_name=model_name)

    def test_rerank_by_cross_encoder_scores(self):
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_st.CrossEncoder.return_value = mock_model
        mock_model.predict.return_value = [0.3, 0.9, 0.6]

        reranker = self._make_reranker(mock_st)
        results = [
            SearchResult(content="Low relevance", score=0.5, id="1"),
            SearchResult(content="High relevance", score=0.5, id="2"),
            SearchResult(content="Mid relevance", score=0.5, id="3"),
        ]
        reranked = reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].content == "High relevance"
        assert reranked[0].score == 0.9
        assert reranked[1].content == "Mid relevance"
        assert reranked[1].score == 0.6

    def test_empty_results(self):
        mock_st = MagicMock()
        reranker = self._make_reranker(mock_st)
        assert reranker.rerank("q", [], top_k=5) == []

    def test_model_error_falls_back(self):
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_st.CrossEncoder.return_value = mock_model
        mock_model.predict.side_effect = RuntimeError("CUDA OOM")

        reranker = self._make_reranker(mock_st)
        results = [
            SearchResult(content="A", score=0.9, id="1"),
            SearchResult(content="B", score=0.8, id="2"),
        ]
        reranked = reranker.rerank("q", results, top_k=1)
        assert len(reranked) == 1
        assert reranked[0].content == "A"
