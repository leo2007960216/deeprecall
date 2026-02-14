"""OpenAI-compatible REST API server for DeepRecall.

Exposes DeepRecall as an OpenAI-compatible API so any tool that speaks the
OpenAI protocol can use DeepRecall as a backend.

Install: pip install deeprecall[server]
Usage: deeprecall serve --vectorstore chroma --collection my_docs --port 8000
"""

from __future__ import annotations

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

from deeprecall.core.engine import DeepRecallEngine

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


def create_app(engine: DeepRecallEngine) -> FastAPI:
    """Create a FastAPI app wrapping a DeepRecall engine.

    Args:
        engine: A configured DeepRecallEngine instance.

    Returns:
        FastAPI app with OpenAI-compatible endpoints.
    """
    app = FastAPI(
        title="DeepRecall API",
        description="OpenAI-compatible API powered by DeepRecall recursive reasoning.",
        version="0.1.0",
    )

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

        # Extract the last user message as the query
        query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                query = msg.content
                break

        if not query:
            raise HTTPException(status_code=400, detail="No user message found.")

        if request.stream:
            return StreamingResponse(
                _stream_response(engine, query, request.model),
                media_type="text/event-stream",
            )

        result = engine.query(query)

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
        ids = engine.add_documents(
            documents=request.documents,
            metadatas=request.metadatas,
            ids=request.ids,
        )
        return DocumentAddResponse(ids=ids, count=len(ids))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "engine": repr(engine)}

    return app


async def _stream_response(
    engine: DeepRecallEngine,
    query: str,
    model: str,
) -> Any:
    """Stream a DeepRecall response in OpenAI SSE format."""
    import json

    result = engine.query(query)

    # Stream the answer in chunks
    chunk_size = 50
    answer = result.answer
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    for i in range(0, len(answer), chunk_size):
        chunk = answer[i : i + chunk_size]
        data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"

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
