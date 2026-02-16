"""Tests for core type definitions."""

from __future__ import annotations

from deeprecall.core.types import ReasoningStep, SearchResult, Source, UsageInfo


class TestSearchResult:
    def test_to_dict(self):
        r = SearchResult(content="test doc", metadata={"k": "v"}, score=0.85, id="r1")
        d = r.to_dict()
        assert d == {"content": "test doc", "metadata": {"k": "v"}, "score": 0.85, "id": "r1"}

    def test_defaults(self):
        r = SearchResult(content="minimal")
        assert r.metadata == {}
        assert r.score == 0.0
        assert r.id == ""


class TestSource:
    def test_from_search_result(self):
        sr = SearchResult(content="doc", metadata={"page": 1}, score=0.9, id="s1")
        source = Source.from_search_result(sr)
        assert source.content == "doc"
        assert source.metadata == {"page": 1}
        assert source.score == 0.9
        assert source.id == "s1"

    def test_to_dict(self):
        s = Source(content="src", score=0.7, id="x")
        d = s.to_dict()
        assert d["content"] == "src"
        assert d["score"] == 0.7


class TestReasoningStep:
    def test_to_dict_full(self):
        step = ReasoningStep(
            iteration=3,
            action="search_and_reasoning",
            code="search_db('test')",
            output="Found 5 results",
            searches=[{"query": "test", "has_results": True}],
            sub_llm_calls=2,
            iteration_time=1.5,
        )
        d = step.to_dict()
        assert d["iteration"] == 3
        assert d["action"] == "search_and_reasoning"
        assert d["code"] == "search_db('test')"
        assert d["output"] == "Found 5 results"
        assert len(d["searches"]) == 1
        assert d["sub_llm_calls"] == 2
        assert d["iteration_time"] == 1.5

    def test_to_dict_minimal(self):
        step = ReasoningStep(iteration=1, action="reasoning")
        d = step.to_dict()
        assert d["code"] is None
        assert d["output"] is None
        assert d["searches"] == []
        assert d["sub_llm_calls"] == 0


class TestUsageInfo:
    def test_to_dict(self):
        usage = UsageInfo(
            total_input_tokens=500,
            total_output_tokens=200,
            total_calls=5,
            model_breakdown={
                "gpt-4o-mini": {"input_tokens": 500, "output_tokens": 200, "calls": 5}
            },
        )
        d = usage.to_dict()
        assert d["total_input_tokens"] == 500
        assert d["total_output_tokens"] == 200
        assert d["total_calls"] == 5
        assert "gpt-4o-mini" in d["model_breakdown"]

    def test_defaults(self):
        usage = UsageInfo()
        assert usage.total_input_tokens == 0
        assert usage.total_output_tokens == 0
        assert usage.total_calls == 0
        assert usage.model_breakdown == {}
