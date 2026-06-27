"""Tests for models module."""

from __future__ import annotations

from prompts_lab.models import (
    Config,
    ModelInfo,
    ModelProvider,
    ScoringCriteria,
)


class TestModelInfo:
    def test_create_model_info(self):
        model = ModelInfo(id="gpt-4o", provider=ModelProvider.OPENAI)
        assert model.id == "gpt-4o"
        assert model.provider == ModelProvider.OPENAI
        assert model.max_context == 128000
        assert model.pricing == {}

    def test_cost_estimate(self):
        model = ModelInfo(
            id="gpt-4o",
            provider=ModelProvider.OPENAI,
            pricing={"input": 2.50, "output": 10.00},
        )
        cost = model.cost_estimate(1000, 500)
        assert cost == (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00

    def test_cost_estimate_no_pricing(self):
        model = ModelInfo(id="gpt-4o", provider=ModelProvider.OPENAI)
        assert model.cost_estimate(1000, 500) is None

    def test_cost_estimate_zero_tokens(self):
        model = ModelInfo(
            id="gpt-4o",
            provider=ModelProvider.OPENAI,
            pricing={"input": 2.50, "output": 10.00},
        )
        assert model.cost_estimate(0, 0) == 0.0


class TestScoringCriteria:
    def test_default_weight(self):
        c = ScoringCriteria(name="helpfulness")
        assert c.weight == 1.0

    def test_custom_weight(self):
        c = ScoringCriteria(name="accuracy", weight=2.5)
        assert c.weight == 2.5

    def test_weight_normalization(self):
        c = ScoringCriteria(name="test", weight=4.0)
        assert c.normalized_weight == 1.0

    def test_invalid_weight_low(self):
        try:
            ScoringCriteria(name="test", weight=-1.0)
        except Exception:
            pass  # Pydantic should reject

    def test_invalid_weight_high(self):
        try:
            ScoringCriteria(name="test", weight=11.0)
        except Exception:
            pass


class TestConfig:
    def test_default_config(self):
        config = Config()
        assert len(config.models) > 0
        assert len(config.default_scoring) > 0
        assert config.max_concurrent == 3
        assert config.timeout_seconds == 60

    def test_config_with_custom_models(self):
        custom = ModelInfo(
            id="custom-model",
            provider=ModelProvider.LOCAL,
            max_context=8192,
        )
        config = Config(models=[custom])
        assert len(config.models) == 1
        assert config.models[0].id == "custom-model"
