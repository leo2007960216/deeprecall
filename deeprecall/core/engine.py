"""DeepRecall Engine -- the core recursive reasoning engine.

Bridges RLM (Recursive Language Models) with vector databases to enable
recursive, multi-hop retrieval and reasoning.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.panel import Panel

from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.search_server import SearchServer
from deeprecall.core.types import DeepRecallResult, UsageInfo
from deeprecall.prompts.templates import DEEPRECALL_SYSTEM_PROMPT, build_search_setup_code
from deeprecall.vectorstores.base import BaseVectorStore

console = Console()


class DeepRecallEngine:
    """Recursive reasoning engine powered by RLM and vector databases.

    Combines RLM's recursive decomposition with vector DB retrieval to
    enable multi-hop reasoning over large document collections.

    Args:
        vectorstore: A vector store adapter instance (ChromaStore, MilvusStore, etc.).
        config: Engine configuration. Uses defaults if not provided.

    Example:
        ```python
        from deeprecall import DeepRecall
        from deeprecall.vectorstores import ChromaStore

        store = ChromaStore(collection_name="my_docs")
        store.add_documents(["Document 1 text...", "Document 2 text..."])

        engine = DeepRecall(vectorstore=store, backend="openai",
                            backend_kwargs={"model_name": "gpt-4o-mini"})
        result = engine.query("What are the main themes?")
        print(result.answer)
        ```
    """

    def __init__(
        self,
        vectorstore: BaseVectorStore,
        config: DeepRecallConfig | None = None,
        *,
        backend: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
        max_iterations: int | None = None,
        **kwargs: Any,
    ):
        self.vectorstore = vectorstore

        # Allow either config object or individual kwargs
        if config is not None:
            self.config = config
        else:
            config_kwargs: dict[str, Any] = {}
            if backend is not None:
                config_kwargs["backend"] = backend
            if backend_kwargs is not None:
                config_kwargs["backend_kwargs"] = backend_kwargs
            if max_iterations is not None:
                config_kwargs["max_iterations"] = max_iterations
            config_kwargs["verbose"] = verbose
            config_kwargs.update(kwargs)
            self.config = DeepRecallConfig(**config_kwargs)

        self._search_server: SearchServer | None = None
        self._validate_setup()

    def _validate_setup(self) -> None:
        """Validate that the engine is properly configured."""
        if self.vectorstore is None:
            raise ValueError("A vectorstore is required. Pass a BaseVectorStore instance.")

    def query(
        self,
        query: str,
        root_prompt: str | None = None,
        top_k: int | None = None,
    ) -> DeepRecallResult:
        """Execute a recursive reasoning query over the vector database.

        Args:
            query: The question or task to answer.
            root_prompt: Optional short prompt visible to the root LM. If None,
                the query itself is used as the root prompt.
            top_k: Override the default top_k for this query.

        Returns:
            DeepRecallResult with answer, sources, reasoning trace, and usage info.
        """
        try:
            from rlm import RLM
            from rlm.logger import RLMLogger
        except ImportError:
            raise ImportError(
                "rlms is required for DeepRecall. Install it with: pip install rlms"
            ) from None

        time_start = time.perf_counter()
        effective_top_k = top_k or self.config.top_k

        if self.config.verbose:
            console.print(
                Panel(
                    f"[bold]Query:[/bold] {query}\n"
                    f"[bold]Vector Store:[/bold] {type(self.vectorstore).__name__} "
                    f"({self.vectorstore.count()} docs)\n"
                    f"[bold]Backend:[/bold] {self.config.backend} / "
                    f"{self.config.backend_kwargs.get('model_name', 'unknown')}",
                    title="[bold blue]DeepRecall[/bold blue]",
                    border_style="blue",
                )
            )

        # Start the search server
        search_server = SearchServer(self.vectorstore)
        search_server.start()

        try:
            # Build setup code that injects search_db() into the REPL
            setup_code = build_search_setup_code(search_server.port)

            # Build environment kwargs
            env_kwargs = {**self.config.environment_kwargs, "setup_code": setup_code}

            # Build context -- include query and DB stats as context for the RLM
            context = self._build_context(query, effective_top_k)

            # Setup logger if configured
            logger = None
            if self.config.log_dir:
                logger = RLMLogger(log_dir=self.config.log_dir)

            # Create and run the RLM
            rlm = RLM(
                backend=self.config.backend,
                backend_kwargs=self.config.backend_kwargs,
                environment=self.config.environment,
                environment_kwargs=env_kwargs,
                max_iterations=self.config.max_iterations,
                max_depth=self.config.max_depth,
                custom_system_prompt=DEEPRECALL_SYSTEM_PROMPT,
                other_backends=self.config.other_backends,
                other_backend_kwargs=self.config.other_backend_kwargs,
                logger=logger,
                verbose=self.config.verbose,
            )

            # Run recursive completion
            root = root_prompt or query
            result = rlm.completion(prompt=context, root_prompt=root)

            # Collect results
            execution_time = time.perf_counter() - time_start
            sources = search_server.get_accessed_sources()
            usage = self._extract_usage(result)

            return DeepRecallResult(
                answer=result.response,
                sources=sources,
                reasoning_trace=[],  # TODO: extract from RLM iterations
                usage=usage,
                execution_time=execution_time,
                query=query,
            )

        finally:
            search_server.stop()

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Convenience method to add documents to the vector store.

        Args:
            documents: List of document texts.
            metadatas: Optional metadata for each document.
            ids: Optional IDs for each document.

        Returns:
            List of document IDs.
        """
        return self.vectorstore.add_documents(documents=documents, metadatas=metadatas, ids=ids)

    def _build_context(self, query: str, top_k: int) -> str:
        """Build context string for the RLM with query info and DB stats."""
        doc_count = self.vectorstore.count()
        context_parts = [
            f"USER QUERY: {query}",
            "",
            f"VECTOR DATABASE: {type(self.vectorstore).__name__} with {doc_count} documents.",
            f"You have access to search_db(query, top_k={top_k}) to search this database.",
            "",
            "INSTRUCTIONS:",
            "1. Use search_db() to find relevant documents for the query.",
            "2. Analyze retrieved documents and search again if needed.",
            "3. Use llm_query() to reason over large amounts of retrieved text.",
            "4. Provide a comprehensive final answer with FINAL().",
        ]
        return "\n".join(context_parts)

    def _extract_usage(self, rlm_result: Any) -> UsageInfo:
        """Extract token usage from the RLM result."""
        usage = UsageInfo()
        try:
            summary = rlm_result.usage_summary
            if hasattr(summary, "model_usage_summaries"):
                for model_name, model_usage in summary.model_usage_summaries.items():
                    usage.total_input_tokens += model_usage.total_input_tokens or 0
                    usage.total_output_tokens += model_usage.total_output_tokens or 0
                    usage.total_calls += model_usage.total_calls or 0
                    usage.model_breakdown[model_name] = {
                        "input_tokens": model_usage.total_input_tokens or 0,
                        "output_tokens": model_usage.total_output_tokens or 0,
                        "calls": model_usage.total_calls or 0,
                    }
        except Exception:
            pass
        return usage

    def close(self) -> None:
        """Clean up resources."""
        if self._search_server is not None:
            self._search_server.stop()
            self._search_server = None

    def __enter__(self) -> DeepRecallEngine:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"DeepRecall(vectorstore={type(self.vectorstore).__name__}, "
            f"backend={self.config.backend!r}, "
            f"docs={self.vectorstore.count()})"
        )
