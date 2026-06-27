"""Mock API client for testing prompts without real API calls."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from prompts_lab.models import ModelInfo, ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class MockResponse:
    """Simulated API response."""

    model: str
    provider: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None
    response_time_ms: float = 0.0


class MockAPIClient:
    """Client that simulates AI model responses for testing.

    Generates realistic-looking responses based on the prompt content
    to allow testing the full pipeline without real API calls.
    """

    # Response templates for different prompt types
    RESPONSE_TEMPLATES: dict[str, list[str]] = {
        "coding": [
            "Here's how you can implement this:\n\n```python\ndef example():\n    # Implementation details\n    return result\n```\n\nThe key considerations are:\n1. Error handling for edge cases\n2. Performance optimization\n3. Code readability and maintainability",
            "This is a common pattern. Here's a robust implementation:\n\n```python\nclass Handler:\n    def __init__(self, config):\n        self.config = config\n        self._state = {}\n\n    def process(self, data):\n        validated = self._validate(data)\n        return self._transform(validated)\n```\n\nNote: Make sure to handle the timeout scenarios.",
        ],
        "analysis": [
            "Based on the analysis, here are the key findings:\n\n1. **Performance**: The current approach has O(n²) complexity\n2. **Scalability**: Limited to ~10K records with current design\n3. **Recommendation**: Consider using a streaming approach for large datasets\n\nOverall, the code is functional but could benefit from refactoring.",
            "Here's my analysis:\n\n- Strengths: Clear separation of concerns, good test coverage\n- Weaknesses: No error handling for network failures, hardcoded paths\n- Suggestions: Add retry logic, use environment variables for config",
        ],
        "creative": [
            "Here's a creative take on that concept:\n\nImagine a world where code writes itself. Every function call triggers an AI that generates the implementation based on the function's documentation and test suite. Developers become architects, specifying what rather than how.\n\nThis vision is closer than most think. The key breakthroughs:\n- Better context windows\n- Multi-step reasoning\n- Self-correction loops",
            "A novel approach would be to draw parallels between neural architecture search and prompt engineering. Both are about finding the optimal configuration in a vast search space...\n\nThe implications are fascinating: just as NAS discovered architectures humans missed, the right prompt might reveal solutions we'd never consider.",
        ],
        "default": [
            "Thank you for your question. Here's my response:\n\nThis is an interesting topic with several dimensions to consider. Let me break it down:\n\n1. First, understanding the core concept\n2. Then exploring practical applications\n3. Finally, considering potential improvements\n\nWould you like me to elaborate on any specific aspect?",
            "I'd be happy to help with that. Here's what I can share:\n\nThe key insight here is that the answer depends on your specific use case. For most scenarios, the recommended approach is to:\n\n- Start with the simplest solution\n- Measure the impact\n- Iterate based on results\n\nLet me know if you need more specific guidance.",
        ],
    }

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._call_count: int = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset(self) -> None:
        """Reset call counter."""
        self._call_count = 0

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: ModelInfo,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> MockResponse:
        """Simulate a chat completion API call.

        Args:
            messages: List of {role, content} message dicts.
            model: Model info for the response.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            MockResponse with simulated content.
        """
        self._call_count += 1
        start = time.monotonic()

        # Combine all messages into one string for classification
        all_text = " ".join(msg.get("content", "") for msg in messages)

        # Determine response category
        category = self._classify_prompt(all_text)

        # Pick a response
        templates = self.RESPONSE_TEMPLATES.get(category, self.RESPONSE_TEMPLATES["default"])
        content = self._rng.choice(templates)

        # Truncate to max_tokens if specified
        if max_tokens:
            approx_chars = max_tokens * 4
            content = content[:approx_chars]

        # Calculate token counts
        prompt_text = all_text
        prompt_tokens = len(prompt_text.split()) * 1.3
        completion_tokens = len(content.split()) * 1.3

        # Simulate response time
        response_time_ms = 100 + self._rng.uniform(50, 500)
        await asyncio.sleep(response_time_ms / 1000)

        elapsed = (time.monotonic() - start) * 1000

        return MockResponse(
            model=model.id,
            provider=model.provider.value,
            content=content,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            response_time_ms=max(elapsed, response_time_ms),
        )

    def _classify_prompt(self, text: str) -> str:
        """Classify a prompt to pick appropriate response template."""
        lower = text.lower()
        if any(w in lower for w in ["code", "function", "implement", "class", "def ", "import"]):
            return "coding"
        if any(w in lower for w in ["analyze", "review", "compare", "evaluate", "metric"]):
            return "analysis"
        if any(w in lower for w in ["creative", "imagine", "brainstorm", "novel", "idea"]):
            return "creative"
        return "default"
