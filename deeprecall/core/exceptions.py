"""Custom exception hierarchy for DeepRecall.

All DeepRecall-specific exceptions inherit from ``DeepRecallError``,
allowing production teams to ``except DeepRecallError`` at the boundary.
"""

from __future__ import annotations

from typing import Any


class DeepRecallError(Exception):
    """Base exception for all DeepRecall errors."""


class ConfigurationError(DeepRecallError):
    """Raised when the engine or a component is misconfigured."""


class BudgetExceededError(DeepRecallError):
    """Raised when a query exceeds its allocated budget."""

    def __init__(self, reason: str, status: Any):
        self.reason = reason
        self.status = status
        super().__init__(f"Budget exceeded: {reason}")


class VectorStoreError(DeepRecallError):
    """Raised when a vector store operation fails."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when a connection to the vector store cannot be established."""


class LLMProviderError(DeepRecallError):
    """Raised when an LLM provider call fails."""


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM call times out."""


class LLMRateLimitError(LLMProviderError):
    """Raised when the LLM provider returns a rate-limit response.

    Attributes:
        retry_after: Seconds to wait before retrying, if provided by the provider.
    """

    def __init__(self, message: str = "Rate limited", retry_after: float | None = None):
        if retry_after is not None and retry_after < 0:
            retry_after = None
        self.retry_after = retry_after
        super().__init__(message)


class CacheError(DeepRecallError):
    """Raised when a cache operation fails."""


class SearchServerError(DeepRecallError):
    """Raised when the internal HTTP search server encounters an error."""
