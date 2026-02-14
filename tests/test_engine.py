"""Tests for the DeepRecall engine."""

from __future__ import annotations

import pytest

from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.engine import DeepRecallEngine
from deeprecall.core.search_server import SearchServer
from deeprecall.core.types import DeepRecallResult


class TestDeepRecallConfig:
    def test_default_config(self):
        config = DeepRecallConfig()
        assert config.backend == "openai"
        assert config.max_iterations == 15
        assert config.max_depth == 1
        assert config.top_k == 5
        assert config.verbose is False

    def test_custom_config(self):
        config = DeepRecallConfig(
            backend="anthropic",
            backend_kwargs={"model_name": "claude-3-sonnet"},
            max_iterations=20,
            verbose=True,
        )
        assert config.backend == "anthropic"
        assert config.max_iterations == 20
        assert config.verbose is True

    def test_config_to_dict_excludes_api_key(self):
        config = DeepRecallConfig(backend_kwargs={"model_name": "gpt-4o", "api_key": "sk-secret"})
        d = config.to_dict()
        assert "api_key" not in d["backend_kwargs"]
        assert d["backend_kwargs"]["model_name"] == "gpt-4o"


class TestDeepRecallEngine:
    def test_init_with_config(self, mock_vectorstore):
        config = DeepRecallConfig(backend="openai")
        engine = DeepRecallEngine(vectorstore=mock_vectorstore, config=config)
        assert engine.config.backend == "openai"

    def test_init_with_kwargs(self, mock_vectorstore):
        engine = DeepRecallEngine(
            vectorstore=mock_vectorstore,
            backend="anthropic",
            backend_kwargs={"model_name": "claude-3"},
            verbose=True,
        )
        assert engine.config.backend == "anthropic"
        assert engine.config.verbose is True

    def test_init_requires_vectorstore(self):
        with pytest.raises(ValueError, match="vectorstore is required"):
            DeepRecallEngine(vectorstore=None)

    def test_add_documents(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        ids = engine.add_documents(
            documents=["New doc 1", "New doc 2"],
            metadatas=[{"source": "test"}, {"source": "test"}],
        )
        assert len(ids) == 2
        assert mock_vectorstore.count() == 7  # 5 existing + 2 new

    def test_repr(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        r = repr(engine)
        assert "DeepRecall" in r
        assert "MockVectorStore" in r

    def test_context_manager(self, mock_vectorstore):
        with DeepRecallEngine(vectorstore=mock_vectorstore) as engine:
            assert engine is not None


class TestSearchServer:
    def test_start_stop(self, mock_vectorstore):
        server = SearchServer(mock_vectorstore)
        server.start()
        assert server.port > 0
        server.stop()

    def test_search_endpoint(self, mock_vectorstore):
        import json
        import urllib.request

        server = SearchServer(mock_vectorstore)
        server.start()

        try:
            data = json.dumps({"query": "python programming", "top_k": 3}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.port}/search",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                results = json.loads(resp.read())

            assert isinstance(results, list)
            assert len(results) <= 3
            assert all("content" in r for r in results)
        finally:
            server.stop()

    def test_get_accessed_sources(self, mock_vectorstore):
        import json
        import urllib.request

        server = SearchServer(mock_vectorstore)
        server.start()

        try:
            data = json.dumps({"query": "python", "top_k": 2}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.port}/search",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)

            sources = server.get_accessed_sources()
            assert len(sources) > 0
        finally:
            server.stop()

    def test_reset_sources(self, mock_vectorstore):
        server = SearchServer(mock_vectorstore)
        server.reset_sources()
        assert server.get_accessed_sources() == []


class TestDeepRecallResult:
    def test_result_to_dict(self):
        result = DeepRecallResult(
            answer="Test answer",
            query="Test query",
            execution_time=1.5,
        )
        d = result.to_dict()
        assert d["answer"] == "Test answer"
        assert d["query"] == "Test query"
        assert d["execution_time"] == 1.5
