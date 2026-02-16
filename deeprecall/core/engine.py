"""DeepRecall Engine -- core recursive reasoning engine."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any

from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.exceptions import (
    BudgetExceededError,
    LLMProviderError,
    SearchServerError,
)
from deeprecall.core.guardrails import QueryBudget
from deeprecall.core.search_server import SearchServer
from deeprecall.core.tracer import DeepRecallTracer
from deeprecall.core.types import DeepRecallResult, Source, UsageInfo
from deeprecall.prompts.templates import DEEPRECALL_SYSTEM_PROMPT, build_search_setup_code
from deeprecall.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class DeepRecallEngine:
    """Recursive reasoning engine powered by RLM and vector databases.

    Combines RLM's recursive decomposition with vector DB retrieval to
    enable multi-hop reasoning over large document collections.

    Args:
        vectorstore: A vector store adapter instance (ChromaStore, MilvusStore, etc.).
        config: Engine configuration. Uses defaults if not provided.
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

        self._callback_manager: Any = None
        self._search_server: SearchServer | None = None
        self._batch_lock = threading.Lock()
        self._setup_callbacks()
        self._validate_setup()

    def _validate_setup(self) -> None:
        """Validate that the engine is properly configured."""
        if self.vectorstore is None:
            raise ValueError("A vectorstore is required. Pass a BaseVectorStore instance.")

    def _setup_callbacks(self) -> None:
        """Initialize callback manager; auto-creates JSONLCallback for log_dir."""
        callbacks = list(self.config.callbacks) if self.config.callbacks else []
        if self.config.log_dir:
            from deeprecall.core.callbacks import JSONLCallback

            if not any(isinstance(c, JSONLCallback) for c in callbacks):
                callbacks.append(JSONLCallback(log_dir=self.config.log_dir))
        if callbacks:
            from deeprecall.core.callbacks import CallbackManager

            self._callback_manager = CallbackManager(callbacks)

    def query(
        self,
        query: str,
        root_prompt: str | None = None,
        top_k: int | None = None,
        budget: QueryBudget | None = None,
    ) -> DeepRecallResult:
        """Execute a recursive reasoning query over the vector database.

        Args:
            query: The question or task to answer.
            root_prompt: Optional short prompt visible to the root LM.
            top_k: Override the default top_k for this query.
            budget: Per-query budget override. Falls back to config.budget.

        Returns:
            DeepRecallResult with answer, sources, reasoning trace, and usage info.
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        try:
            from rlm import RLM
        except ImportError:
            raise ImportError(
                "rlms is required for DeepRecall. Install it with: pip install rlms"
            ) from None

        time_start = time.perf_counter()
        effective_top_k = top_k if top_k is not None else self.config.top_k
        effective_budget = budget if budget is not None else self.config.budget

        # Check query cache
        cache_key: str | None = None
        if self.config.cache:
            cache_key = self._build_cache_key(query, effective_top_k)
            cached = self.config.cache.get(cache_key)
            if cached is not None:
                if isinstance(cached, DeepRecallResult):
                    return cached
                # Persistent caches (Disk/Redis) deserialise to dict
                if isinstance(cached, dict):
                    return DeepRecallResult.from_dict(cached)

        # Fire on_query_start callback
        if self._callback_manager:
            self._callback_manager.on_query_start(query, self.config)

        if self.config.verbose:
            self._print_query_panel(query, effective_budget)

        # Start or reuse the search server
        owns_server = False
        if self.config.reuse_search_server and self._search_server is not None:
            search_server = self._search_server
            search_server.reset_sources()
        else:
            search_server = SearchServer(
                self.vectorstore,
                reranker=self.config.reranker,
                cache=self.config.cache,
                callback_manager=self._callback_manager,
            )
            try:
                search_server.start()
            except SearchServerError:
                raise
            except Exception as e:
                raise SearchServerError(f"Failed to start search server: {e}") from e

            if self.config.reuse_search_server:
                self._search_server = search_server
            else:
                owns_server = True

        # Create tracer for this query
        tracer = DeepRecallTracer(
            budget=effective_budget,
            callback_manager=self._callback_manager,
            start_time=time_start,
        )

        try:
            # Build setup code that injects search_db() into the REPL
            setup_code = build_search_setup_code(
                server_port=search_server.port,
                max_search_calls=(effective_budget.max_search_calls if effective_budget else None),
            )

            # Build environment kwargs
            env_kwargs = {**self.config.environment_kwargs, "setup_code": setup_code}

            # Build context
            context = self._build_context(query, effective_top_k)

            # Create and run the RLM with our tracer as the logger
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
                logger=tracer,
                verbose=self.config.verbose,
            )

            # Run recursive completion (with retry if configured)
            root = root_prompt or query

            def _run_completion() -> Any:
                try:
                    return rlm.completion(prompt=context, root_prompt=root)
                except BudgetExceededError:
                    raise
                except LLMProviderError:
                    raise
                except Exception as e:
                    raise LLMProviderError(f"LLM completion failed: {e}") from e

            if self.config.retry is not None:
                from deeprecall.core.retry import retry_with_backoff

                rlm_result = retry_with_backoff(_run_completion, self.config.retry)
            else:
                rlm_result = _run_completion()

            # Build final result
            result = self._build_result(
                rlm_result=rlm_result,
                search_server=search_server,
                tracer=tracer,
                query=query,
                time_start=time_start,
            )

            # Store in cache
            if self.config.cache and cache_key and not result.error:
                self.config.cache.set(cache_key, result, ttl=self.config.cache_ttl)

            # Fire on_query_end callback
            if self._callback_manager:
                self._callback_manager.on_query_end(result)

            return result

        except BudgetExceededError as e:
            # Return partial result when budget exceeded
            result = self._build_result(
                rlm_result=None,
                search_server=search_server,
                tracer=tracer,
                query=query,
                time_start=time_start,
                error=str(e),
            )
            if self._callback_manager:
                self._callback_manager.on_budget_warning(e.status)
                self._callback_manager.on_query_end(result)
            return result

        except Exception as e:
            if self._callback_manager:
                self._callback_manager.on_error(e)
            raise

        finally:
            if owns_server:
                search_server.stop()

    def _print_query_panel(self, query: str, budget: QueryBudget | None) -> None:
        """Print verbose query panel using rich (best-effort, never raises)."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            limits: list[str] = []
            if budget:
                for attr, label in [("max_search_calls", "searches"), ("max_tokens", "tokens")]:
                    val = getattr(budget, attr, None)
                    if val is not None:
                        limits.append(f"{label}\u2264{val}")
                if budget.max_time_seconds is not None:
                    limits.append(f"time\u2264{budget.max_time_seconds}s")
            budget_info = f"\n[bold]Budget:[/bold] {', '.join(limits)}" if limits else ""
            try:
                doc_count: int | str = self.vectorstore.count()
            except Exception:
                doc_count = "?"
            vs = type(self.vectorstore).__name__
            model = self.config.backend_kwargs.get("model_name", "unknown")
            Console().print(
                Panel(
                    f"[bold]Query:[/bold] {query}\n"
                    f"[bold]Store:[/bold] {vs} ({doc_count} docs)\n"
                    f"[bold]Backend:[/bold] {self.config.backend} / {model}{budget_info}",
                    title="[bold blue]DeepRecall[/bold blue]",
                    border_style="blue",
                )
            )
        except Exception:
            pass

    def _build_result(
        self,
        rlm_result: Any | None,
        search_server: SearchServer,
        tracer: DeepRecallTracer,
        query: str,
        time_start: float,
        error: str | None = None,
    ) -> DeepRecallResult:
        """Build a DeepRecallResult from RLM output and tracer data."""
        execution_time = time.perf_counter() - time_start
        sources = search_server.get_accessed_sources()
        usage = self._extract_usage(rlm_result) if rlm_result else UsageInfo()

        # Update tracer budget with token info
        tracer.budget_status.tokens_used = usage.total_input_tokens + usage.total_output_tokens

        # Calculate confidence from source scores
        confidence = self._compute_confidence(sources)

        # Get answer (partial from trace if budget exceeded)
        if rlm_result:
            answer = rlm_result.response
        elif tracer.steps:
            last_output = tracer.steps[-1].output or ""
            answer = f"[Partial - budget exceeded] {last_output[:500]}"
        else:
            answer = "[No answer - budget exceeded before first iteration]"

        return DeepRecallResult(
            answer=answer,
            sources=sources,
            reasoning_trace=tracer.get_trace(),
            usage=usage,
            execution_time=execution_time,
            query=query,
            budget_status=tracer.budget_status.to_dict(),
            error=error,
            confidence=confidence,
        )

    def _compute_confidence(self, sources: list[Source]) -> float | None:
        """Compute a confidence score based on source relevance scores."""
        if not sources:
            return None
        scores = [s.score for s in sources if s.score > 0]
        if not scores:
            return None
        # Average of top-3 source scores, clamped to [0.0, 1.0]
        top_scores = sorted(scores, reverse=True)[:3]
        raw = sum(top_scores) / len(top_scores)
        return round(min(max(raw, 0.0), 1.0), 4)

    def query_batch(
        self,
        queries: list[str],
        *,
        max_concurrency: int = 4,
        root_prompt: str | None = None,
        top_k: int | None = None,
        budget: QueryBudget | None = None,
    ) -> list[DeepRecallResult]:
        """Execute multiple queries concurrently using a thread pool.

        Returns a list of DeepRecallResult in the same order as *queries*.

        Raises:
            ValueError: If the batch exceeds 10 000 queries or max_concurrency < 1.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if len(queries) > 10_000:
            raise ValueError(
                f"Batch size {len(queries)} exceeds maximum of 10,000. Split into smaller batches."
            )
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")

        results: list[DeepRecallResult | None] = [None] * len(queries)

        def _run(index: int, q: str) -> tuple[int, DeepRecallResult]:
            result = self.query(q, root_prompt=root_prompt, top_k=top_k, budget=budget)
            return index, result

        # Hold the lock for the full batch lifetime so overlapping batch calls
        # don't corrupt the config.
        with self._batch_lock:
            saved_reuse = self.config.reuse_search_server
            self.config.reuse_search_server = False
            try:
                with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
                    futures = {pool.submit(_run, i, q): i for i, q in enumerate(queries)}
                    for future in as_completed(futures):
                        try:
                            idx, res = future.result()
                            results[idx] = res
                        except Exception as e:
                            idx = futures[future]
                            results[idx] = DeepRecallResult(
                                answer="",
                                sources=[],
                                reasoning_trace=[],
                                usage=UsageInfo(),
                                execution_time=0.0,
                                query=queries[idx],
                                budget_status={},
                                error=str(e),
                                confidence=None,
                            )
            finally:
                self.config.reuse_search_server = saved_reuse

        # Ensure no None slots remain (shouldn't happen, but be safe)
        for i, r in enumerate(results):
            if r is None:
                results[i] = DeepRecallResult(
                    answer="",
                    sources=[],
                    reasoning_trace=[],
                    usage=UsageInfo(),
                    execution_time=0.0,
                    query=queries[i],
                    budget_status={},
                    error="Query failed: no result produced",
                    confidence=None,
                )

        return results  # type: ignore[return-value]

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Convenience method to add documents to the vector store."""
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
            logger.debug("Could not extract usage info from RLM result", exc_info=True)
        return usage

    def _build_cache_key(self, query: str, top_k: int) -> str:
        """Build a deterministic cache key (backend + model + top_k + query)."""
        key_data = (
            f"{query}|{self.config.backend}|"
            f"{self.config.backend_kwargs.get('model_name', '')}|"
            f"{top_k}"
        )
        return hashlib.sha256(key_data.encode()).hexdigest()

    def close(self) -> None:
        """Clean up resources (search server, cache, callbacks)."""
        if self._search_server is not None:
            self._search_server.stop()
            self._search_server = None
        if self.config.cache:
            logger.debug("Engine closed; cache retained with %s", self.config.cache.stats())

    def __enter__(self) -> DeepRecallEngine:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        # Avoid calling vectorstore.count() -- it may trigger a network call,
        # which violates the Python __repr__ contract (must be side-effect free).
        return (
            f"DeepRecall(vectorstore={type(self.vectorstore).__name__}, "
            f"backend={self.config.backend!r})"
        )
