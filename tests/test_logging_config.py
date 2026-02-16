"""Tests for the structured logging configuration helper."""

import logging

from deeprecall.core.logging_config import configure_logging


class TestConfigureLogging:
    def test_sets_level_on_deeprecall_logger(self):
        logger = configure_logging(level=logging.DEBUG)
        assert logger.name == "deeprecall"
        assert logger.level == logging.DEBUG

    def test_accepts_string_level(self):
        logger = configure_logging(level="WARNING")
        assert logger.level == logging.WARNING

    def test_root_logger_unaffected(self):
        root_level = logging.getLogger().level
        configure_logging(level=logging.DEBUG)
        assert logging.getLogger().level == root_level

    def test_does_not_propagate(self):
        logger = configure_logging(level=logging.INFO)
        assert logger.propagate is False

    def test_custom_handler(self):
        handler = logging.StreamHandler()
        logger = configure_logging(handler=handler)
        assert handler in logger.handlers

    def test_custom_format(self):
        logger = configure_logging(fmt="%(message)s")
        # Find the handler we added
        for h in logger.handlers:
            if h.formatter:
                assert h.formatter._fmt == "%(message)s"
                break

    def test_repeated_calls_dont_duplicate_handlers(self):
        configure_logging(level=logging.INFO)
        configure_logging(level=logging.DEBUG)
        logger = logging.getLogger("deeprecall")
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) == 1
