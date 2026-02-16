"""Tests for the OpenAI-compatible API server."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from deeprecall.core.callbacks import UsageTrackingCallback
from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.types import DeepRecallResult, Source, UsageInfo


def _make_client(app: object) -> httpx.AsyncClient:
    """Build an async httpx client around a FastAPI/Starlette ASGI app."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.config = DeepRecallConfig()
    engine._callback_manager = None
    return engine


@pytest.fixture
def app(mock_engine):
    from deeprecall.adapters.openai_server import create_app

    return create_app(mock_engine)


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_returns_ok(self, app):
        async with _make_client(app) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "engine" in data


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_lists_models(self, app):
        async with _make_client(app) as client:
            resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "deeprecall"
        assert data["data"][0]["owned_by"] == "deeprecall"


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_basic_completion(self, app, mock_engine):
        mock_engine.query.return_value = DeepRecallResult(
            answer="Paris is the capital of France.",
            sources=[Source(content="doc1", score=0.9, id="s1")],
            usage=UsageInfo(total_input_tokens=100, total_output_tokens=50, total_calls=2),
            execution_time=1.5,
            query="What is the capital of France?",
        )

        async with _make_client(app) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is the capital of France?"}],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "Paris" in data["choices"][0]["message"]["content"]
        assert data["usage"]["prompt_tokens"] == 100
        assert data["usage"]["completion_tokens"] == 50
        assert data["usage"]["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_empty_messages_returns_400(self, app):
        async with _make_client(app) as client:
            resp = await client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_no_user_message_returns_400(self, app):
        async with _make_client(app) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "system", "content": "You are helpful."}]},
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, app, mock_engine):
        mock_engine.query.return_value = DeepRecallResult(
            answer="It has about 2 million people.",
            usage=UsageInfo(total_input_tokens=200, total_output_tokens=30),
            execution_time=1.0,
        )

        async with _make_client(app) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "You are a geography expert."},
                        {"role": "user", "content": "What is the capital of France?"},
                        {"role": "assistant", "content": "Paris."},
                        {"role": "user", "content": "What is the population?"},
                    ],
                },
            )

        assert resp.status_code == 200
        call_args = mock_engine.query.call_args
        query_text = call_args[0][0]
        assert "geography expert" in query_text
        assert "User: What is the capital of France?" in query_text
        assert "Assistant: Paris." in query_text
        assert "What is the population?" in query_text

    @pytest.mark.asyncio
    async def test_engine_error_returns_500(self, app, mock_engine):
        mock_engine.query.side_effect = RuntimeError("LLM crashed")

        async with _make_client(app) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_streaming_response(self, app, mock_engine):
        mock_engine.query.return_value = DeepRecallResult(
            answer="A short test answer for streaming.",
            usage=UsageInfo(total_input_tokens=50, total_output_tokens=20),
            execution_time=0.5,
        )

        async with _make_client(app) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Test streaming"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        chunks = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))

        assert len(chunks) >= 2
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


class TestDocumentsEndpoint:
    @pytest.mark.asyncio
    async def test_add_documents(self, app, mock_engine):
        mock_engine.add_documents.return_value = ["id-1", "id-2"]

        async with _make_client(app) as client:
            resp = await client.post(
                "/v1/documents",
                json={"documents": ["Doc one", "Doc two"]},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["ids"] == ["id-1", "id-2"]


class TestUsageEndpoint:
    @pytest.mark.asyncio
    async def test_no_tracker_configured(self, app):
        async with _make_client(app) as client:
            resp = await client.get("/v1/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["usage"] is None

    @pytest.mark.asyncio
    async def test_with_tracker(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app
        from deeprecall.core.callbacks import CallbackManager

        tracker = UsageTrackingCallback()
        tracker.total_queries = 5
        tracker.total_tokens = 1000
        mock_engine._callback_manager = CallbackManager([tracker])

        app = create_app(mock_engine)
        async with _make_client(app) as client:
            resp = await client.get("/v1/usage")

        assert resp.status_code == 200
        data = resp.json()
        assert data["usage"]["total_queries"] == 5
        assert data["usage"]["total_tokens"] == 1000


class TestCacheClearEndpoint:
    @pytest.mark.asyncio
    async def test_no_cache_configured(self, app, mock_engine):
        mock_engine.config.cache = None
        async with _make_client(app) as client:
            resp = await client.post("/v1/cache/clear")
        assert resp.status_code == 200
        assert "No cache" in resp.json()["message"]

    @pytest.mark.asyncio
    async def test_cache_cleared(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app

        mock_cache = MagicMock()
        mock_engine.config.cache = mock_cache
        app = create_app(mock_engine)

        async with _make_client(app) as client:
            resp = await client.post("/v1/cache/clear")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_cache.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_clear_error(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app

        mock_cache = MagicMock()
        mock_cache.clear.side_effect = RuntimeError("Redis down")
        mock_engine.config.cache = mock_cache
        app = create_app(mock_engine)

        async with _make_client(app) as client:
            resp = await client.post("/v1/cache/clear")

        assert resp.status_code == 500
