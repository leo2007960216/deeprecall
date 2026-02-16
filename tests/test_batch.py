"""Tests for the batch query API."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from deeprecall.core.engine import DeepRecallEngine
from deeprecall.core.types import DeepRecallResult, UsageInfo


def _make_mock_result(query: str) -> DeepRecallResult:
    return DeepRecallResult(
        answer=f"Answer to: {query}",
        sources=[],
        reasoning_trace=[],
        usage=UsageInfo(),
        execution_time=0.1,
        query=query,
        budget_status={},
        error=None,
        confidence=0.9,
    )


def _make_mock_engine() -> DeepRecallEngine:
    """Create a mock DeepRecallEngine with required internal attributes."""
    engine = DeepRecallEngine.__new__(DeepRecallEngine)
    engine.config = MagicMock()
    engine.config.retry = None
    engine._batch_lock = threading.Lock()
    return engine


class TestQueryBatch:
    @patch.object(DeepRecallEngine, "query")
    @patch.object(DeepRecallEngine, "__init__", return_value=None)
    def test_returns_correct_count(self, mock_init, mock_query):
        mock_query.side_effect = lambda q, **kw: _make_mock_result(q)
        engine = _make_mock_engine()

        queries = ["q1", "q2", "q3"]
        results = engine.query_batch(queries)
        assert len(results) == 3

    @patch.object(DeepRecallEngine, "query")
    @patch.object(DeepRecallEngine, "__init__", return_value=None)
    def test_preserves_order(self, mock_init, mock_query):
        import time

        def slow_query(q, **kw):
            # Make queries complete in reverse order
            delays = {"q1": 0.05, "q2": 0.02, "q3": 0.01}
            time.sleep(delays.get(q, 0))
            return _make_mock_result(q)

        mock_query.side_effect = slow_query
        engine = _make_mock_engine()

        queries = ["q1", "q2", "q3"]
        results = engine.query_batch(queries, max_concurrency=3)
        assert [r.query for r in results] == ["q1", "q2", "q3"]

    @patch.object(DeepRecallEngine, "query")
    @patch.object(DeepRecallEngine, "__init__", return_value=None)
    def test_error_isolation(self, mock_init, mock_query):
        def failing_query(q, **kw):
            if q == "q2":
                raise RuntimeError("boom")
            return _make_mock_result(q)

        mock_query.side_effect = failing_query
        engine = _make_mock_engine()

        results = engine.query_batch(["q1", "q2", "q3"])
        assert len(results) == 3
        assert results[0].answer == "Answer to: q1"
        assert results[1].error == "boom"
        assert results[2].answer == "Answer to: q3"

    @patch.object(DeepRecallEngine, "query")
    @patch.object(DeepRecallEngine, "__init__", return_value=None)
    def test_max_concurrency_respected(self, mock_init, mock_query):
        max_concurrent = 0
        current = 0
        lock = threading.Lock()

        def counting_query(q, **kw):
            nonlocal max_concurrent, current
            with lock:
                current += 1
                max_concurrent = max(max_concurrent, current)
            import time

            time.sleep(0.05)
            with lock:
                current -= 1
            return _make_mock_result(q)

        mock_query.side_effect = counting_query
        engine = _make_mock_engine()

        engine.query_batch(["q1", "q2", "q3", "q4", "q5"], max_concurrency=2)
        assert max_concurrent <= 2


class TestAsyncQueryBatch:
    @pytest.mark.asyncio
    async def test_async_batch_returns_correct_count(self):
        import asyncio

        from deeprecall.core.async_engine import AsyncDeepRecallEngine

        mock_engine = MagicMock(spec=DeepRecallEngine)
        mock_engine.query.side_effect = lambda q, **kw: _make_mock_result(q)
        mock_engine.config = MagicMock()
        mock_engine.config.reuse_search_server = False

        async_engine = AsyncDeepRecallEngine.__new__(AsyncDeepRecallEngine)
        async_engine._engine = mock_engine
        async_engine._batch_lock = asyncio.Lock()

        async def mock_async_query(q, **kw):
            return _make_mock_result(q)

        async_engine.query = mock_async_query
        results = await async_engine.query_batch(["q1", "q2", "q3"])
        assert len(results) == 3
