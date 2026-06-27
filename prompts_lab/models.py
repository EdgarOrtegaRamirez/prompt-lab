"""Pydantic models for PromptLab."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_serializer


class ModelProvider(str, Enum):
    """Supported AI model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    LOCAL = "local"
    CUSTOM = "custom"


class ModelInfo(BaseModel):
    """Information about an AI model."""

    id: str = Field(description="Model identifier, e.g. 'gpt-4o'")
    provider: ModelProvider = Field(description="Provider name")
    max_context: int = Field(default=128000, description="Max context window in tokens")
    pricing: dict[str, float] = Field(
        default_factory=dict,
        description="Pricing per 1M tokens: {input: cost, output: cost}",
    )

    def cost_estimate(self, input_tokens: int, output_tokens: int) -> float | None:
        """Estimate cost in USD for given token counts."""
        pricing = self.pricing
        if not pricing:
            return None
        input_cost = pricing.get("input", 0)
        output_cost = pricing.get("output", 0)
        return (input_tokens / 1_000_000) * input_cost + (output_tokens / 1_000_000) * output_cost


# Default model catalog
DEFAULT_MODELS: list[dict[str, Any]] = [
    {
        "id": "gpt-4o",
        "provider": "openai",
        "max_context": 128000,
        "pricing": {"input": 2.50, "output": 10.00},
    },
    {
        "id": "gpt-4o-mini",
        "provider": "openai",
        "max_context": 128000,
        "pricing": {"input": 0.15, "output": 0.60},
    },
    {
        "id": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "max_context": 200000,
        "pricing": {"input": 3.00, "output": 15.00},
    },
    {
        "id": "claude-haiku-3-5",
        "provider": "anthropic",
        "max_context": 200000,
        "pricing": {"input": 0.80, "output": 4.00},
    },
    {
        "id": "gemini-2.0-flash",
        "provider": "google",
        "max_context": 1048576,
        "pricing": {"input": 0.15, "output": 0.60},
    },
    {
        "id": "llama-3.1-405b",
        "provider": "groq",
        "max_context": 128000,
        "pricing": {"input": 0.00, "output": 0.00},
    },
]


class ScoringCriteria(BaseModel):
    """Configurable scoring criteria for prompt evaluation."""

    name: str = Field(description="Criterion name, e.g. 'helpfulness'")
    weight: float = Field(default=1.0, ge=0.0, description="Weight relative to other criteria")
    description: str = Field(default="", description="What this criterion measures")

    @property
    def normalized_weight(self) -> float:
        return self.weight / sum(c.weight for c in [self]) if self.weight > 0 else 0.0


class PromptVariant(BaseModel):
    """A single prompt variant to test."""

    id: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    name: str = Field(description="Human-readable name for this variant")
    template: str = Field(description="Jinja2 prompt template")
    context: dict[str, Any] = Field(default_factory=dict, description="Context variables to inject")
    system_prompt: str | None = Field(default=None, description="Optional system prompt")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Max output tokens")


class TestResult(BaseModel):
    """Result of sending a prompt to a model."""

    variant_id: str
    variant_name: str
    model_id: str
    model_provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    output: str = ""
    error: str | None = None
    cost_usd: float | None = None
    response_time_ms: float = 0.0
    scored: bool = False
    scores: dict[str, float] = Field(default_factory=dict)
    overall_score: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ScoringResult(BaseModel):
    """Scored evaluation of a prompt variant."""

    variant: PromptVariant
    model: ModelInfo
    test_result: TestResult
    scores: dict[str, float] = Field(default_factory=dict)
    overall_score: float = 0.0
    grade: str = "N/A"

    def grade_from_score(self, score: float) -> str:
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"


class Config(BaseModel):
    """PromptLab configuration."""

    models: list[ModelInfo] = Field(default_factory=lambda: [ModelInfo(**m) for m in DEFAULT_MODELS])
    default_scoring: list[ScoringCriteria] = Field(
        default_factory=lambda: [
            ScoringCriteria(name="helpfulness", weight=3.0, description="How helpful is the response?"),
            ScoringCriteria(name="accuracy", weight=2.5, description="How accurate and factual is the response?"),
            ScoringCriteria(name="creativity", weight=1.0, description="How creative and original is the response?"),
            ScoringCriteria(name="relevance", weight=2.0, description="How relevant is the response to the prompt?"),
        ]
    )
    default_provider: ModelProvider = Field(default=ModelProvider.OPENAI)
    max_concurrent: int = Field(default=3, ge=1, le=20, description="Max concurrent API calls")
    timeout_seconds: int = Field(default=60, ge=5, le=300, description="Request timeout in seconds")
    retry_attempts: int = Field(default=2, ge=0, le=5, description="Number of retry attempts on failure")


# Storage models
class TestRecord(BaseModel):
    """Persisted test run record."""

    id: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
    test_id: str
    variant: PromptVariant
    model: ModelInfo
    result: TestResult

    @field_serializer("variant", "model", "result")
    def _serialize(self, value: BaseModel) -> dict:
        return value.model_dump()


class HistoryEntry(BaseModel):
    """A single history entry for a test run."""

    id: str
    test_id: str
    timestamp: str
    variant_name: str
    model_id: str
    provider: str
    overall_score: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None
    summary: str = ""

    @field_serializer("timestamp")
    def _serialize_ts(self, v: str) -> str:
        return v
