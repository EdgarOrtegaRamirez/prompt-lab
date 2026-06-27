"""Token counting and estimation utilities."""

from __future__ import annotations


def count_tokens_tiktoken(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken library.

    Args:
        text: The text to count tokens for.
        model: The model name to determine the encoding.

    Returns:
        Number of tokens in the text.
    """
    try:
        import tiktoken
    except ImportError:
        return _fallback_token_count(text)

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def count_tokens_approx(text: str) -> int:
    """Approximate token count using heuristics.

    Rule of thumb: ~4 characters per token for English text.
    """
    if not text:
        return 0
    # Count words as a rough proxy
    words = len(text.split())
    # English average is ~1.3 tokens per word
    return max(1, int(words * 1.3))


def _fallback_token_count(text: str) -> int:
    """Fallback token counting when tiktoken is not available."""
    return count_tokens_approx(text)


def estimate_tokens(text: str, model: str | None = None) -> int:
    """Estimate token count, falling back gracefully.

    Args:
        text: Text to estimate.
        model: Model name for tiktoken (optional).

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    if model:
        return count_tokens_tiktoken(text, model)
    return count_tokens_approx(text)


def format_token_count(tokens: int) -> str:
    """Format token count with comma separators."""
    return f"{tokens:,}"
