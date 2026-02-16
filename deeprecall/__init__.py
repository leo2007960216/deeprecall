"""DeepRecall -- Recursive reasoning engine for vector databases.

Usage:
    from deeprecall import DeepRecall
    from deeprecall.vectorstores import ChromaStore

    store = ChromaStore(collection_name="my_docs")
    store.add_documents(["Document text..."])

    engine = DeepRecall(vectorstore=store, backend="openai",
                        backend_kwargs={"model_name": "gpt-4o-mini"})
    result = engine.query("Your question here")
    print(result.answer)
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from deeprecall.core.async_engine import AsyncDeepRecallEngine
from deeprecall.core.cache import DiskCache, InMemoryCache
from deeprecall.core.callbacks import (
    BaseCallback,
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
from deeprecall.core.guardrails import QueryBudget
from deeprecall.core.logging_config import configure_logging
from deeprecall.core.retry import RetryConfig
from deeprecall.core.types import DeepRecallResult, SearchResult, Source

# Convenience aliases
DeepRecall = DeepRecallEngine
AsyncDeepRecall = AsyncDeepRecallEngine

try:
    __version__ = _pkg_version("deeprecall")
except PackageNotFoundError:
    __version__ = "0.3.0"

# --- Lazy imports for optional-dependency classes ---
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "RedisCache": ("deeprecall.core.cache_redis", "RedisCache"),
    "OpenTelemetryCallback": ("deeprecall.core.callback_otel", "OpenTelemetryCallback"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'deeprecall' has no attribute {name!r}")


__all__ = [
    "AsyncDeepRecall",
    "AsyncDeepRecallEngine",
    "BaseCallback",
    "BudgetExceededError",
    "CacheError",
    "ConfigurationError",
    "ConsoleCallback",
    "DeepRecall",
    "DeepRecallConfig",
    "DeepRecallEngine",
    "DeepRecallError",
    "DeepRecallResult",
    "DiskCache",
    "InMemoryCache",
    "JSONLCallback",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "OpenTelemetryCallback",
    "ProgressCallback",
    "QueryBudget",
    "RedisCache",
    "RetryConfig",
    "SearchResult",
    "SearchServerError",
    "Source",
    "UsageTrackingCallback",
    "VectorStoreConnectionError",
    "VectorStoreError",
    "configure_logging",
]
