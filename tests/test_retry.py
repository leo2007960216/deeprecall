"""Tests for the retry / resilience layer."""

from unittest.mock import MagicMock, patch

import pytest

from deeprecall.core.exceptions import LLMProviderError, LLMRateLimitError, VectorStoreError
from deeprecall.core.retry import RetryConfig, retry_with_backoff


class TestRetryConfig:
    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 60.0
        assert cfg.exponential_base == 2.0
        assert cfg.jitter is True

    def test_custom(self):
        cfg = RetryConfig(max_retries=1, base_delay=0.5, jitter=False)
        assert cfg.max_retries == 1
        assert cfg.base_delay == 0.5
        assert cfg.jitter is False


class TestRetryWithBackoff:
    def test_success_on_first_call(self):
        fn = MagicMock(return_value="ok")
        result = retry_with_backoff(fn, RetryConfig(max_retries=3))
        assert result == "ok"
        assert fn.call_count == 1

    @patch("deeprecall.core.retry.time.sleep")
    def test_retries_on_transient_error(self, mock_sleep):
        fn = MagicMock(side_effect=[VectorStoreError("fail"), "ok"])
        cfg = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        result = retry_with_backoff(fn, cfg)
        assert result == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("deeprecall.core.retry.time.sleep")
    def test_max_retries_respected(self, mock_sleep):
        fn = MagicMock(side_effect=VectorStoreError("always fails"))
        cfg = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)
        with pytest.raises(VectorStoreError, match="always fails"):
            retry_with_backoff(fn, cfg)
        assert fn.call_count == 3  # initial + 2 retries

    @patch("deeprecall.core.retry.time.sleep")
    def test_non_retryable_propagates_immediately(self, mock_sleep):
        fn = MagicMock(side_effect=ValueError("bad input"))
        cfg = RetryConfig(max_retries=3, base_delay=0.01)
        with pytest.raises(ValueError, match="bad input"):
            retry_with_backoff(fn, cfg)
        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("deeprecall.core.retry.time.sleep")
    def test_retry_after_honored(self, mock_sleep):
        exc = LLMRateLimitError("rate limited", retry_after=10.0)
        fn = MagicMock(side_effect=[exc, "ok"])
        cfg = RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        result = retry_with_backoff(fn, cfg)
        assert result == "ok"
        # Should sleep at least 10.0 (retry_after)
        actual_delay = mock_sleep.call_args[0][0]
        assert actual_delay >= 10.0

    @patch("deeprecall.core.retry.time.sleep")
    def test_backoff_increases(self, mock_sleep):
        fn = MagicMock(
            side_effect=[
                VectorStoreError("1"),
                VectorStoreError("2"),
                "ok",
            ]
        )
        cfg = RetryConfig(max_retries=3, base_delay=1.0, exponential_base=2.0, jitter=False)
        retry_with_backoff(fn, cfg)
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        # First delay: 1.0 * 2^0 = 1.0, second: 1.0 * 2^1 = 2.0
        assert delays[0] == pytest.approx(1.0)
        assert delays[1] == pytest.approx(2.0)

    @patch("deeprecall.core.retry.time.sleep")
    def test_jitter_adds_randomness(self, mock_sleep):
        fn = MagicMock(side_effect=[VectorStoreError("fail"), "ok"])
        cfg = RetryConfig(max_retries=3, base_delay=1.0, jitter=True)
        retry_with_backoff(fn, cfg)
        delay = mock_sleep.call_args[0][0]
        # With jitter, delay is uniform in [0, base_delay] for attempt 0
        assert 0.0 <= delay <= 1.0

    @patch("deeprecall.core.retry.time.sleep")
    def test_retries_connection_error(self, mock_sleep):
        fn = MagicMock(side_effect=[ConnectionError("refused"), "ok"])
        cfg = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)
        result = retry_with_backoff(fn, cfg)
        assert result == "ok"

    @patch("deeprecall.core.retry.time.sleep")
    def test_retries_timeout_error(self, mock_sleep):
        fn = MagicMock(side_effect=[TimeoutError("timed out"), "ok"])
        cfg = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)
        result = retry_with_backoff(fn, cfg)
        assert result == "ok"

    @patch("deeprecall.core.retry.time.sleep")
    def test_retries_llm_provider_error(self, mock_sleep):
        """LLMProviderError is not in default retryable list (only LLMRateLimitError is)."""
        fn = MagicMock(side_effect=LLMProviderError("provider down"))
        cfg = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)
        with pytest.raises(LLMProviderError):
            retry_with_backoff(fn, cfg)
        assert fn.call_count == 1  # not retried

    @patch("deeprecall.core.retry.time.sleep")
    def test_max_delay_cap(self, mock_sleep):
        fn = MagicMock(
            side_effect=[
                VectorStoreError("1"),
                VectorStoreError("2"),
                VectorStoreError("3"),
                "ok",
            ]
        )
        cfg = RetryConfig(
            max_retries=3, base_delay=10.0, max_delay=5.0, exponential_base=10.0, jitter=False
        )
        retry_with_backoff(fn, cfg)
        for call in mock_sleep.call_args_list:
            assert call[0][0] <= 5.0
