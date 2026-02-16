"""Lightweight HTTP server that exposes vector store search to the RLM REPL.

This server runs in a background thread and provides a /search endpoint
that the REPL's search_db() function calls via urllib.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any

from deeprecall.core.exceptions import VectorStoreError
from deeprecall.core.types import Source
from deeprecall.vectorstores.base import BaseVectorStore

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from deeprecall.core.cache import BaseCache
    from deeprecall.core.callbacks import CallbackManager
    from deeprecall.core.reranker import BaseReranker


class _SearchHandler(BaseHTTPRequestHandler):
    """HTTP handler for vector store search requests."""

    vectorstore: BaseVectorStore
    reranker: BaseReranker | None
    cache: BaseCache | None
    callback_manager: CallbackManager | None
    accessed_sources: list[Source]
    _lock: threading.Lock

    def do_POST(self) -> None:
        if self.path == "/search":
            self._handle_search()
        else:
            self.send_error(404, "Not found")

    def _handle_search(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            query = data.get("query", "")
            top_k = data.get("top_k", 5)
            filters = data.get("filters")

            # Check search cache first
            cache_key = None
            if self.cache is not None:
                import hashlib

                # Sort filters for deterministic cache keys regardless of dict order
                filters_str = json.dumps(filters, sort_keys=True) if filters else ""
                cache_key = (
                    "search:"
                    + hashlib.sha256(f"{query}|{top_k}|{filters_str}".encode()).hexdigest()
                )
                cached = self.cache.get(cache_key)
                if cached is not None:
                    self._send_json(200, cached)
                    return

            # Fetch more results if reranking (reranker will trim to top_k)
            fetch_k = top_k * 3 if self.reranker else top_k
            search_start = time.perf_counter()
            try:
                results = self.vectorstore.search(query=query, top_k=fetch_k, filters=filters)
            except VectorStoreError:
                raise
            except Exception as e:
                raise VectorStoreError(f"Vector store search failed: {e}") from e

            # Apply reranking
            if self.reranker and results:
                results = self.reranker.rerank(query=query, results=results, top_k=top_k)

            elapsed_ms = (time.perf_counter() - search_start) * 1000

            # Fire on_search callback
            if self.callback_manager is not None:
                self.callback_manager.on_search(query, len(results), elapsed_ms)

            # Track accessed sources for the final result
            with self._lock:
                for r in results:
                    self.accessed_sources.append(Source.from_search_result(r))

            response_data = [r.to_dict() for r in results]

            # Store in search cache
            if self.cache is not None and cache_key:
                self.cache.set(cache_key, response_data, ttl=300)

            self._send_json(200, response_data)

        except VectorStoreError as e:
            _logger.error("Search failed: %s", e, exc_info=True)
            self._send_json(500, {"error": "Search failed"})
        except Exception as e:
            _logger.error("Unexpected search error: %s", e, exc_info=True)
            self._send_json(500, {"error": "Internal server error"})

    def _send_json(self, status: int, data: Any) -> None:
        response = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence HTTP server logs."""


class SearchServer:
    """Background HTTP server wrapping a vector store for REPL access.

    Usage:
        server = SearchServer(vectorstore)
        server.start()
        # ... REPL can now call http://127.0.0.1:{server.port}/search
        sources = server.get_accessed_sources()
        server.stop()
    """

    def __init__(
        self,
        vectorstore: BaseVectorStore,
        reranker: BaseReranker | None = None,
        cache: BaseCache | None = None,
        callback_manager: CallbackManager | None = None,
    ):
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.cache = cache
        self.callback_manager = callback_manager
        self.port = self._find_free_port()
        self._accessed_sources: list[Source] = []
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the search server in a background thread."""
        if self._server is not None:
            return  # Already running; prevent double-start leak

        # Re-acquire a free port in case the old one was reclaimed after stop().
        self.port = self._find_free_port()

        handler_class = type(
            "_BoundSearchHandler",
            (_SearchHandler,),
            {
                "vectorstore": self.vectorstore,
                "reranker": self.reranker,
                "cache": self.cache,
                "callback_manager": self.callback_manager,
                "accessed_sources": self._accessed_sources,
                "_lock": self._lock,
            },
        )
        self._server = HTTPServer(("127.0.0.1", self.port), handler_class)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the search server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def get_accessed_sources(self) -> list[Source]:
        """Return all sources that were accessed during search calls."""
        import hashlib

        with self._lock:
            seen: dict[str, Source] = {}
            for source in self._accessed_sources:
                # Use ID if available; otherwise hash full content to avoid
                # collisions from documents sharing the same prefix.
                if source.id:
                    key = source.id
                else:
                    key = hashlib.sha256(source.content.encode()).hexdigest()
                if key not in seen or source.score > seen[key].score:
                    seen[key] = source
            return list(seen.values())

    def reset_sources(self) -> None:
        """Clear tracked sources for a new query."""
        with self._lock:
            self._accessed_sources.clear()

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
