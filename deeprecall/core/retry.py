"""Retry with exponential backoff for transient failures.

Provides a ``RetryConfig`` dataclass and a ``retry_with_backoff`` function
for wrapping calls to LLM providers and vector stores.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from deeprecall.core.exceptions import (
    LLMRateLimitError,
    VectorStoreError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default set of exceptions considered transient and retryable.
_DEFAULT_RETRYABLE: tuple[type[BaseException], ...] = (
    LLMRateLimitError,
    VectorStoreError,
    ConnectionError,
    TimeoutError,
    OSError,
)


@dataclass
class RetryConfig:
    """Configuration for retry behaviour.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Upper-bound cap on the computed delay.
        exponential_base: Multiplier for exponential growth (delay * base^attempt).
        jitter: If ``True``, add uniform random jitter in ``[0, delay]``.
        retryable_exceptions: Tuple of exception types considered transient.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[BaseException], ...] = field(
        default_factory=lambda: _DEFAULT_RETRYABLE
    )


def retry_with_backoff(
    fn: Any,
    config: RetryConfig,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call *fn* with retry logic.

    On transient failures (those matching ``config.retryable_exceptions``),
    wait with exponential backoff + optional jitter, then retry up to
    ``config.max_retries`` times.

    If the caught exception is :class:`LLMRateLimitError` and carries a
    ``retry_after`` value, the delay is set to at least that value.

    Args:
        fn: The callable to invoke.
        config: Retry configuration.
        *args: Positional arguments forwarded to *fn*.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn* on success.

    Raises:
        The last exception raised by *fn* when retries are exhausted,
        or immediately for non-retryable exceptions.
    """
    last_exc: BaseException | None = None

    for attempt in range(1 + config.max_retries):
        try:
            return fn(*args, **kwargs)
        except config.retryable_exceptions as exc:
            last_exc = exc

            if attempt >= config.max_retries:
                logger.warning(
                    "Retry exhausted after %d attempts for %s: %s",
                    config.max_retries,
                    fn,
                    exc,
                )
                raise

            # Compute delay with overflow protection
            try:
                delay = config.base_delay * (config.exponential_base**attempt)
            except OverflowError:
                delay = config.max_delay
            delay = min(delay, config.max_delay)

            # Honor Retry-After from rate limit responses
            if isinstance(exc, LLMRateLimitError) and exc.retry_after is not None:
                delay = max(delay, min(exc.retry_after, config.max_delay))

            # Apply jitter: randomise within [0, delay] to decorrelate retries
            if config.jitter:
                delay = random.uniform(0, delay)  # noqa: S311

            logger.info(
                "Retrying %s (attempt %d/%d) after %.2fs: %s",
                fn,
                attempt + 1,
                config.max_retries,
                delay,
                exc,
            )
            time.sleep(delay)

    # Should not reach here, but satisfy type checker
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_with_backoff: unreachable")
