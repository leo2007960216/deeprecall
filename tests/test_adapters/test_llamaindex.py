"""Tests for the LlamaIndex adapter (mocked -- no llama-index-core required)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from deeprecall.core.types import DeepRecallResult, Source, UsageInfo

# ---------------------------------------------------------------------------
# Helpers -- build mock llama-index modules
# ---------------------------------------------------------------------------


def _build_li_mocks() -> dict[str, types.ModuleType]:
    """Create fake llama_index.core modules."""
    li = types.ModuleType("llama_index")
    li_core = types.ModuleType("llama_index.core")
    li_qe = types.ModuleType("llama_index.core.query_engine")
    li_resp = types.ModuleType("llama_index.core.response")
    li_resp_schema = types.ModuleType("llama_index.core.response.schema")
    li_retrievers = types.ModuleType("llama_index.core.retrievers")
    li_schema = types.ModuleType("llama_index.core.schema")

    class TextNode:
        def __init__(self, text="", metadata=None, id_=""):
            self.text = text
            self.metadata = metadata or {}
            self.id_ = id_

    class NodeWithScore:
        def __init__(self, node=None, score=0.0):
            self.node = node
            self.score = score

    class QueryBundle:
        def __init__(self, query_str=""):
            self.query_str = query_str

    class Response:
        def __init__(self, response="", source_nodes=None, metadata=None):
            self.response = response
            self.source_nodes = source_nodes or []
            self.metadata = metadata or {}

    class CustomQueryEngine:
        class Config:
            arbitrary_types_allowed = True

        def __init_subclass__(cls, **kw):
            super().__init_subclass__(**kw)

        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class BaseRetriever:
        def __init__(self, **kw):
            pass

    # Wire up
    li_schema.TextNode = TextNode
    li_schema.NodeWithScore = NodeWithScore
    li_schema.QueryBundle = QueryBundle
    li_resp_schema.Response = Response
    li_qe.CustomQueryEngine = CustomQueryEngine
    li_retrievers.BaseRetriever = BaseRetriever

    return {
        "llama_index": li,
        "llama_index.core": li_core,
        "llama_index.core.query_engine": li_qe,
        "llama_index.core.response": li_resp,
        "llama_index.core.response.schema": li_resp_schema,
        "llama_index.core.retrievers": li_retrievers,
        "llama_index.core.schema": li_schema,
    }


def _fake_result(answer: str = "test answer") -> DeepRecallResult:
    return DeepRecallResult(
        answer=answer,
        sources=[
            Source(content="src1", metadata={"topic": "a"}, score=0.9, id="id1"),
            Source(content="src2", metadata={}, score=0.5, id="id2"),
        ],
        reasoning_trace=[],
        query="test query",
        execution_time=1.5,
        usage=UsageInfo(total_input_tokens=10, total_output_tokens=5, total_calls=2),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeepRecallQueryEngine:
    """Tests for the LlamaIndex query engine adapter."""

    def _make_engine(self, mock_engine: MagicMock):
        mods = _build_li_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.adapters.llamaindex" in sys.modules:
                del sys.modules["deeprecall.adapters.llamaindex"]
            from deeprecall.adapters.llamaindex import DeepRecallQueryEngine

            return DeepRecallQueryEngine(engine=mock_engine)

    def test_custom_query_returns_response(self):
        engine = MagicMock()
        engine.query.return_value = _fake_result("Final answer")
        qe = self._make_engine(engine)

        resp = qe.custom_query("What is X?")
        assert resp.response == "Final answer"
        engine.query.assert_called_once_with("What is X?")

    def test_source_nodes_have_score_and_id(self):
        engine = MagicMock()
        engine.query.return_value = _fake_result()
        qe = self._make_engine(engine)

        resp = qe.custom_query("q")
        assert len(resp.source_nodes) == 2
        assert resp.source_nodes[0].score == 0.9
        assert resp.source_nodes[0].node.id_ == "id1"
        assert resp.source_nodes[0].node.text == "src1"

    def test_metadata_includes_execution_time_and_usage(self):
        engine = MagicMock()
        engine.query.return_value = _fake_result()
        qe = self._make_engine(engine)

        resp = qe.custom_query("q")
        assert resp.metadata["execution_time"] == 1.5
        assert "usage" in resp.metadata
        assert resp.metadata["query"] == "test query"


class TestDeepRecallRetriever:
    """Tests for the LlamaIndex retriever adapter."""

    def _make_retriever(self, mock_engine: MagicMock, top_k: int = 5):
        mods = _build_li_mocks()
        with patch.dict(sys.modules, mods):
            if "deeprecall.adapters.llamaindex" in sys.modules:
                del sys.modules["deeprecall.adapters.llamaindex"]
            from deeprecall.adapters.llamaindex import DeepRecallRetriever

            return DeepRecallRetriever(engine=mock_engine, top_k=top_k)

    def test_retrieve_calls_vectorstore_search(self):
        engine = MagicMock()
        from deeprecall.core.types import SearchResult

        engine.vectorstore.search.return_value = [
            SearchResult(content="hello", metadata={"k": "v"}, score=0.8, id="r1"),
        ]
        retriever = self._make_retriever(engine, top_k=3)

        mods = _build_li_mocks()
        QueryBundle = mods["llama_index.core.schema"].QueryBundle
        nodes = retriever._retrieve(QueryBundle(query_str="greeting"))

        engine.vectorstore.search.assert_called_once_with(query="greeting", top_k=3)
        assert len(nodes) == 1
        assert nodes[0].node.text == "hello"
        assert nodes[0].score == 0.8
        assert nodes[0].node.id_ == "r1"

    def test_retrieve_empty_results(self):
        engine = MagicMock()
        engine.vectorstore.search.return_value = []
        retriever = self._make_retriever(engine)

        mods = _build_li_mocks()
        QueryBundle = mods["llama_index.core.schema"].QueryBundle
        nodes = retriever._retrieve(QueryBundle(query_str="nothing"))
        assert nodes == []
