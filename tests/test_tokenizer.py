"""Tests for tokenizer module."""

from prompts_lab.tokenizer import (
    count_tokens_approx,
    estimate_tokens,
    format_token_count,
)


class TestTokenCounting:
    def test_empty_text(self):
        assert count_tokens_approx("") == 0
        assert estimate_tokens("") == 0

    def test_approx_count(self):
        text = "Hello world this is a test"
        count = count_tokens_approx(text)
        assert count > 0

    def test_format_token_count(self):
        assert format_token_count(1234) == "1,234"
        assert format_token_count(1234567) == "1,234,567"

    def test_estimate_with_model(self):
        count = estimate_tokens("Hello world", "gpt-4o")
        assert count > 0

    def test_estimate_fallback(self):
        count = estimate_tokens("Hello world", None)
        assert count > 0
