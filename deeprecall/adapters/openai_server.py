"""OpenAI-compatible REST API server for DeepRecall.

Exposes DeepRecall as an OpenAI-compatible API so any tool that speaks the
OpenAI protocol can use DeepRecall as a backend.

Install: pip install deeprecall[server]
Usage: deeprecall serve --vectorstore chroma --collection my_docs --port 8000
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "fastapi is required for the OpenAI-compatible server. "
        "Install it with: pip install deeprecall[server]"
    ) from None

import logging

from deeprecall import __version__
from deeprecall.core.engine import DeepRecallEngine

_logger = logging.getLogger(__name__)

# --- Request/Response Models ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "deeprecall"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "deeprecall"
    choices: list[ChatCompletionChoice]
    usage: TokenUsage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "deeprecall"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class DocumentAddRequest(BaseModel):
    documents: list[str]
    metadatas: list[dict[str, Any]] | None = None
    ids: list[str] | None = None


class DocumentAddResponse(BaseModel):
    ids: list[str]
    count: int


# --- Server Factory ---


def create_app(
    engine: DeepRecallEngine,
    api_keys: list[str] | None = None,
    requests_per_minute: int | None = None,
) -> FastAPI:
    """Create a FastAPI app wrapping a DeepRecall engine.

    Args:
        engine: A configured DeepRecallEngine instance.
        api_keys: Optional list of valid API keys for auth.
        requests_per_minute: Optional rate limit per key.

    Returns:
        FastAPI app with OpenAI-compatible endpoints.
    """
    app = FastAPI(
        title="DeepRecall API",
        description="OpenAI-compatible API powered by DeepRecall recursive reasoning.",
        version=__version__,
    )

    # Add middleware (order matters: auth first, then rate limit)
    if api_keys:
        from deeprecall.middleware.auth import APIKeyAuth

        app.add_middleware(APIKeyAuth, api_keys=api_keys)

    if requests_per_minute:
        from deeprecall.middleware.rate_limit import RateLimiter

        app.add_middleware(RateLimiter, requests_per_minute=requests_per_minute)

    @app.get("/v1/models")
    async def list_models() -> ModelListResponse:
        backend = engine.config.backend_kwargs.get("model_name", "unknown")
        return ModelListResponse(
            data=[
                ModelInfo(id="deeprecall"),
                ModelInfo(id=f"deeprecall-{backend}"),
            ]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty.")

        # Build context from the full conversation (multi-turn support).
        # System messages become instructions, prior user/assistant turns
        # become conversation history, and the last user message is the query.
        system_parts: list[str] = []
        history_parts: list[str] = []
        query = ""

        for msg in request.messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            elif msg.role == "user":
                if query:
                    history_parts.append(f"User: {query}")
                query = msg.content
            elif msg.role == "assistant":
                history_parts.append(f"Assistant: {msg.content}")

        if not query:
            raise HTTPException(status_code=400, detail="No user message found.")

        # Prepend system context and conversation history to query
        prefix_parts: list[str] = []
        if system_parts:
            prefix_parts.append(f"[System instructions: {' '.join(system_parts)}]")
        if history_parts:
            prefix_parts.append("[Conversation history:\n" + "\n".join(history_parts) + "]")
        if prefix_parts:
            query = "\n\n".join(prefix_parts) + "\n\n" + query

        if request.stream:
            return StreamingResponse(
                _stream_response(engine, query, request.model),
                media_type="text/event-stream",
            )

        try:
            result = await asyncio.to_thread(engine.query, query)
        except Exception as e:
            _logger.error("Chat completion failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error") from None

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=result.answer),
                )
            ],
            usage=TokenUsage(
                prompt_tokens=result.usage.total_input_tokens,
                completion_tokens=result.usage.total_output_tokens,
                total_tokens=(result.usage.total_input_tokens + result.usage.total_output_tokens),
            ),
        )

    @app.post("/v1/documents")
    async def add_documents(request: DocumentAddRequest) -> DocumentAddResponse:
        """Add documents to the vector store (DeepRecall extension endpoint)."""
        ids = await asyncio.to_thread(
            engine.add_documents,
            documents=request.documents,
            metadatas=request.metadatas,
            ids=request.ids,
        )
        return DocumentAddResponse(ids=ids, count=len(ids))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "engine": repr(engine)}

    @app.get("/v1/usage")
    async def usage_stats() -> dict[str, Any]:
        """Return usage statistics if a UsageTrackingCallback is registered."""
        from deeprecall.core.callbacks import UsageTrackingCallback

        cb_manager = getattr(engine, "_callback_manager", None)
        if cb_manager:
            for cb in cb_manager.callbacks:
                if isinstance(cb, UsageTrackingCallback):
                    return {"status": "ok", "usage": cb.summary()}
        return {"status": "ok", "usage": None, "message": "No usage tracker configured"}

    @app.post("/v1/cache/clear")
    async def clear_cache() -> dict[str, str]:
        """Clear the query and search cache."""
        if engine.config.cache:
            try:
                await asyncio.to_thread(engine.config.cache.clear)
            except Exception as e:
                _logger.error("Cache clear failed: %s", e, exc_info=True)
                raise HTTPException(status_code=500, detail="Cache clear failed") from None
            return {"status": "ok", "message": "Cache cleared"}
        return {"status": "ok", "message": "No cache configured"}

    return app


async def _stream_response(
    engine: DeepRecallEngine,
    query: str,
    model: str,
) -> Any:
    """Stream a DeepRecall response in OpenAI SSE format.

    .. warning::
        This is *simulated* streaming. The full query runs to completion first,
        then the answer is sliced into chunks for SSE delivery. True token-level
        streaming requires upstream RLM support and is planned for v0.5.0.
    """
    import asyncio
    import json

    # Run query in thread to not block
    try:
        result = await asyncio.to_thread(engine.query, query)
    except Exception as e:
        _logger.error("Streaming query failed: %s", e, exc_info=True)
        error_data = {
            "error": {"message": "Internal server error", "type": "server_error"},
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Stream the answer in chunks
    chunk_size = 50
    answer = result.answer
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    for i in range(0, len(answer), chunk_size):
        chunk = answer[i : i + chunk_size]
        delta: dict[str, str] = {"content": chunk}
        # OpenAI spec requires role: assistant on the first chunk
        if i == 0:
            delta["role"] = "assistant"
        data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0)  # yield control

    # Final chunk with finish_reason
    final_data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_data)}\n\n"
    yield "data: [DONE]\n\n"
