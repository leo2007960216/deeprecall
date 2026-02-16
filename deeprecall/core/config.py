"""Configuration for DeepRecall engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from deeprecall.core.cache import BaseCache
    from deeprecall.core.callbacks import BaseCallback
    from deeprecall.core.guardrails import QueryBudget
    from deeprecall.core.reranker import BaseReranker
    from deeprecall.core.retry import RetryConfig

# Supported LLM backends (mirrors RLM's ClientBackend)
BackendType = Literal[
    "openai",
    "portkey",
    "openrouter",
    "vercel",
    "vllm",
    "litellm",
    "anthropic",
    "azure_openai",
    "gemini",
]

# Supported environment types (mirrors RLM's EnvironmentType)
EnvironmentType = Literal["local", "docker", "modal", "prime", "daytona", "e2b"]


@dataclass
class DeepRecallConfig:
    """Configuration for the DeepRecall engine.

    Args:
        backend: LLM provider to use (e.g., "openai", "anthropic").
        backend_kwargs: Provider-specific kwargs (model_name, api_key, etc.).
        environment: REPL sandbox type for code execution.
        environment_kwargs: Environment-specific kwargs.
        max_iterations: Max recursive reasoning iterations before forcing an answer.
        max_depth: Max recursion depth for sub-LM calls.
        top_k: Default number of results to retrieve from vector store.
        verbose: Enable rich console output for debugging.
        log_dir: Directory for JSONL trajectory logs. None disables logging.
        other_backends: Additional LLM backends for sub-calls.
        other_backend_kwargs: Kwargs for additional backends.
        budget: Resource limits for queries (token, time, cost budgets).
        callbacks: List of callback handlers for observability.
        cache: Cache backend for query and search result caching.
        cache_ttl: Time-to-live for cached results in seconds.
        reranker: Post-retrieval reranker for improving search quality.
    """

    backend: BackendType = "openai"
    backend_kwargs: dict[str, Any] = field(default_factory=lambda: {"model_name": "gpt-4o-mini"})
    environment: EnvironmentType = "local"
    environment_kwargs: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 15
    max_depth: int = 1
    top_k: int = 5
    verbose: bool = False
    log_dir: str | None = None
    other_backends: list[BackendType] | None = None
    other_backend_kwargs: list[dict[str, Any]] | None = None
    budget: QueryBudget | None = None
    callbacks: list[BaseCallback] | None = None
    cache: BaseCache | None = None
    cache_ttl: int = 3600
    reranker: BaseReranker | None = None
    retry: RetryConfig | None = None
    reuse_search_server: bool = True

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.cache_ttl < 0:
            raise ValueError(f"cache_ttl must be >= 0, got {self.cache_ttl}")

    @staticmethod
    def _strip_secrets(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of kwargs with api_key and similar secrets removed."""
        _secret_keys = {"api_key", "api_secret", "secret_key", "token"}
        return {k: v for k, v in kwargs.items() if k not in _secret_keys}

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_kwargs": self._strip_secrets(self.backend_kwargs),
            "environment": self.environment,
            "environment_kwargs": self.environment_kwargs,
            "max_iterations": self.max_iterations,
            "max_depth": self.max_depth,
            "top_k": self.top_k,
            "verbose": self.verbose,
            "log_dir": self.log_dir,
            "other_backends": self.other_backends,
            "other_backend_kwargs": (
                [self._strip_secrets(d) for d in self.other_backend_kwargs]
                if self.other_backend_kwargs
                else None
            ),
            "budget": self.budget.to_dict() if self.budget else None,
            "callbacks": [type(c).__name__ for c in self.callbacks] if self.callbacks else None,
            "cache": type(self.cache).__name__ if self.cache else None,
            "cache_ttl": self.cache_ttl,
            "reranker": type(self.reranker).__name__ if self.reranker else None,
            "retry": (
                {
                    "max_retries": self.retry.max_retries,
                    "base_delay": self.retry.base_delay,
                    "max_delay": self.retry.max_delay,
                }
                if self.retry
                else None
            ),
            "reuse_search_server": self.reuse_search_server,
        }
