"""Lightweight HTTP server that exposes vector store search to the RLM REPL.

This server runs in a background thread and provides a /search endpoint
that the REPL's search_db() function calls via urllib.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from deeprecall.core.types import Source
from deeprecall.vectorstores.base import BaseVectorStore


class _SearchHandler(BaseHTTPRequestHandler):
    """HTTP handler for vector store search requests."""

    vectorstore: BaseVectorStore
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

            results = self.vectorstore.search(query=query, top_k=top_k, filters=filters)

            # Track accessed sources for the final result
            with self._lock:
                for r in results:
                    self.accessed_sources.append(Source.from_search_result(r))

            response_data = [r.to_dict() for r in results]
            self._send_json(200, response_data)

        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _send_json(self, status: int, data: Any) -> None:
        response = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence HTTP server logs."""
        pass


class SearchServer:
    """Background HTTP server wrapping a vector store for REPL access.

    Usage:
        server = SearchServer(vectorstore)
        server.start()
        # ... REPL can now call http://127.0.0.1:{server.port}/search
        sources = server.get_accessed_sources()
        server.stop()
    """

    def __init__(self, vectorstore: BaseVectorStore):
        self.vectorstore = vectorstore
        self.port = self._find_free_port()
        self._accessed_sources: list[Source] = []
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the search server in a background thread."""
        handler_class = type(
            "_BoundSearchHandler",
            (_SearchHandler,),
            {
                "vectorstore": self.vectorstore,
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
        with self._lock:
            # Deduplicate by ID, keeping highest score
            seen: dict[str, Source] = {}
            for source in self._accessed_sources:
                key = source.id or source.content[:100]
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
