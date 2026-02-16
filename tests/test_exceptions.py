"""Tests for the custom exception hierarchy."""

import pytest

from deeprecall.core.exceptions import (
    BudgetExceededError,
    CacheError,
    ConfigurationError,
    DeepRecallError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    SearchServerError,
    VectorStoreConnectionError,
    VectorStoreError,
)


class TestExceptionHierarchy:
    """All custom exceptions should be instances of DeepRecallError."""

    def test_deep_recall_error_is_base(self):
        assert issubclass(DeepRecallError, Exception)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            BudgetExceededError,
            VectorStoreError,
            VectorStoreConnectionError,
            LLMProviderError,
            LLMTimeoutError,
            LLMRateLimitError,
            CacheError,
            SearchServerError,
        ],
    )
    def test_subclass_of_deep_recall_error(self, exc_cls):
        assert issubclass(exc_cls, DeepRecallError)

    def test_vector_store_connection_inherits_vector_store(self):
        assert issubclass(VectorStoreConnectionError, VectorStoreError)

    def test_llm_timeout_inherits_llm_provider(self):
        assert issubclass(LLMTimeoutError, LLMProviderError)

    def test_llm_rate_limit_inherits_llm_provider(self):
        assert issubclass(LLMRateLimitError, LLMProviderError)

    def test_budget_exceeded_has_status(self):
        status = {"iterations_used": 5}
        exc = BudgetExceededError("Too many iterations", status)
        assert exc.reason == "Too many iterations"
        assert exc.status == status
        assert "Budget exceeded" in str(exc)

    def test_llm_rate_limit_has_retry_after(self):
        exc = LLMRateLimitError("slow down", retry_after=5.0)
        assert exc.retry_after == 5.0

    def test_llm_rate_limit_default_retry_after(self):
        exc = LLMRateLimitError()
        assert exc.retry_after is None

    def test_catch_all_deep_recall_error(self):
        """Production code can catch DeepRecallError to handle all errors."""
        errors = [
            ConfigurationError("bad config"),
            VectorStoreError("db down"),
            LLMProviderError("api error"),
            CacheError("cache miss"),
            SearchServerError("port taken"),
        ]
        for err in errors:
            with pytest.raises(DeepRecallError):
                raise err


class TestBackwardCompat:
    """BudgetExceededError should still be importable from guardrails."""

    def test_import_from_guardrails(self):
        from deeprecall.core.guardrails import BudgetExceededError as FromGuardrails

        assert FromGuardrails is BudgetExceededError

    def test_import_from_core(self):
        from deeprecall.core import BudgetExceededError as FromCore

        assert FromCore is BudgetExceededError
