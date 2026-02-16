"""Structured logging configuration for DeepRecall.

Provides a helper to configure logging for the ``deeprecall.*`` logger
namespace without touching the root logger or any other library's loggers.
"""

from __future__ import annotations

import logging


def configure_logging(
    level: int | str = logging.INFO,
    fmt: str | None = None,
    handler: logging.Handler | None = None,
) -> logging.Logger:
    """Configure logging for all ``deeprecall.*`` loggers.

    This only affects the ``deeprecall`` logger namespace and never modifies
    the root logger.

    Args:
        level: Logging level (e.g., ``logging.DEBUG``, ``"WARNING"``).
        fmt: Optional format string. Defaults to a concise timestamped format.
        handler: Optional custom handler. Defaults to a ``StreamHandler``.

    Returns:
        The configured ``deeprecall`` logger instance.

    Example:
        >>> from deeprecall import configure_logging
        >>> configure_logging(level="DEBUG")
    """
    logger = logging.getLogger("deeprecall")

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    if handler is None:
        handler = logging.StreamHandler()

    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handler.setFormatter(logging.Formatter(fmt))

    # Avoid duplicate handlers on repeated calls
    logger.handlers = [h for h in logger.handlers if not isinstance(h, type(handler))]
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger
