"""Core module for DeepRecall."""

from deeprecall.core.async_engine import AsyncDeepRecallEngine
from deeprecall.core.cache import BaseCache, DiskCache, InMemoryCache
from deeprecall.core.callbacks import (
    BaseCallback,
    CallbackManager,
    ConsoleCallback,
    JSONLCallback,
    ProgressCallback,
    UsageTrackingCallback,
)
from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.engine import DeepRecallEngine
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
from deeprecall.core.guardrails import BudgetStatus, QueryBudget
from deeprecall.core.reranker import BaseReranker
from deeprecall.core.retry import RetryConfig
from deeprecall.core.tracer import DeepRecallTracer
from deeprecall.core.types import DeepRecallResult, ReasoningStep, SearchResult, Source, UsageInfo

__all__ = [
    "AsyncDeepRecallEngine",
    "BaseCache",
    "BaseCallback",
    "BaseReranker",
    "BudgetExceededError",
    "BudgetStatus",
    "CacheError",
    "CallbackManager",
    "ConfigurationError",
    "ConsoleCallback",
    "DeepRecallConfig",
    "DeepRecallEngine",
    "DeepRecallError",
    "DeepRecallResult",
    "DeepRecallTracer",
    "DiskCache",
    "InMemoryCache",
    "JSONLCallback",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "ProgressCallback",
    "QueryBudget",
    "RetryConfig",
    "ReasoningStep",
    "SearchResult",
    "SearchServerError",
    "Source",
    "UsageInfo",
    "UsageTrackingCallback",
    "VectorStoreConnectionError",
    "VectorStoreError",
]

# --- Lazy imports for optional-dependency classes ---
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "RedisCache": ("deeprecall.core.cache_redis", "RedisCache"),
    "OpenTelemetryCallback": ("deeprecall.core.callback_otel", "OpenTelemetryCallback"),
    "CohereReranker": ("deeprecall.core.reranker", "CohereReranker"),
    "CrossEncoderReranker": ("deeprecall.core.reranker", "CrossEncoderReranker"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'deeprecall.core' has no attribute {name!r}")
