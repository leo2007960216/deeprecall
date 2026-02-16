"""Deprecation utilities for DeepRecall.

Provides decorators and helpers for gracefully deprecating API elements
with clear warnings and migration guidance.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any


def deprecated(
    reason: str,
    removal_version: str,
    alternative: str | None = None,
) -> Any:
    """Decorator to mark a function or class as deprecated.

    Args:
        reason: Why this is deprecated.
        removal_version: The version where this will be removed (e.g., "0.5.0").
        alternative: Suggested replacement, if any.

    Example:
        @deprecated("Use query_batch() instead", "0.5.0", alternative="query_batch")
        def old_batch_query(...):
            ...
    """

    def decorator(func: Any) -> Any:
        message = (
            f"{func.__qualname__} is deprecated: {reason}. Will be removed in v{removal_version}."
        )
        if alternative:
            message += f" Use {alternative} instead."

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__deprecated__ = message  # type: ignore[attr-defined]
        return wrapper

    return decorator


def warn_deprecated_param(
    param_name: str,
    alternative: str | None = None,
    removal_version: str = "0.5.0",
) -> None:
    """Emit a deprecation warning for a function parameter.

    Call this inside a function when a deprecated parameter is used.

    Args:
        param_name: Name of the deprecated parameter.
        alternative: Suggested replacement parameter, if any.
        removal_version: The version where this parameter will be removed.
    """
    message = f"Parameter '{param_name}' is deprecated and will be removed in v{removal_version}."
    if alternative:
        message += f" Use '{alternative}' instead."
    warnings.warn(message, DeprecationWarning, stacklevel=2)
