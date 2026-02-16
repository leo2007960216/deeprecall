"""Tests for authentication and rate limiting middleware."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.types import DeepRecallResult, UsageInfo


def _make_client(app: object) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.config = DeepRecallConfig()
    engine._callback_manager = None
    engine.query.return_value = DeepRecallResult(answer="ok", usage=UsageInfo(), execution_time=0.1)
    return engine


# ---------- Auth middleware ----------


class TestAPIKeyAuth:
    @pytest.mark.asyncio
    async def test_no_keys_disables_auth(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app

        app = create_app(mock_engine, api_keys=None)
        async with _make_client(app) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_valid_key_passes(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app

        app = create_app(mock_engine, api_keys=["test-key-123"])
        async with _make_client(app) as client:
            resp = await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer test-key-123"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_key_returns_401(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app

        app = create_app(mock_engine, api_keys=["real-key"])
        async with _make_client(app) as client:
            resp = await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer wrong-key"},
            )
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_header_returns_401(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app

        app = create_app(mock_engine, api_keys=["real-key"])
        async with _make_client(app) as client:
            resp = await client.get("/v1/models")
        assert resp.status_code == 401
        assert "Missing API key" in resp.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_exempt_paths_bypass_auth(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app

        app = create_app(mock_engine, api_keys=["key"])
        async with _make_client(app) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_malformed_bearer_returns_401(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app

        app = create_app(mock_engine, api_keys=["key"])
        async with _make_client(app) as client:
            resp = await client.get(
                "/v1/models",
                headers={"Authorization": "Basic dXNlcjpwYXNz"},
            )
        assert resp.status_code == 401


# ---------- Rate limiter middleware ----------


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_under_limit(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app

        app = create_app(mock_engine, requests_per_minute=60)
        async with _make_client(app) as client:
            resp = await client.get("/v1/models")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_429_when_exhausted(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app

        app = create_app(mock_engine, requests_per_minute=2)
        async with _make_client(app) as client:
            assert (await client.get("/v1/models")).status_code == 200
            assert (await client.get("/v1/models")).status_code == 200
            resp = await client.get("/v1/models")
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.json()["error"]["message"]
        assert "retry-after" in resp.headers

    @pytest.mark.asyncio
    async def test_exempt_paths_bypass_rate_limit(self, mock_engine):
        from deeprecall.adapters.openai_server import create_app

        app = create_app(mock_engine, requests_per_minute=1)
        async with _make_client(app) as client:
            await client.get("/v1/models")
            await client.get("/v1/models")
            resp = await client.get("/health")
        assert resp.status_code == 200
