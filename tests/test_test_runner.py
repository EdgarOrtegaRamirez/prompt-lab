"""Tests for test runner."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from prompts_lab.models import ModelInfo, ModelProvider, PromptVariant
from prompts_lab.test_runner import PromptTestRunner


@pytest.fixture
def runner():
    return PromptTestRunner(use_mock=True, mock_seed=42)


@pytest.fixture
def sample_model():
    return ModelInfo(
        id="gpt-4o",
        provider=ModelProvider.OPENAI,
        max_context=128000,
        pricing={"input": 2.50, "output": 10.00},
    )


@pytest.fixture
def sample_variant():
    return PromptVariant(
        id="v1",
        name="Test Variant",
        template="Write a {{ task }} for me.",
        context={"task": "sorting function"},
    )


class TestPromptTestRunner:
    @pytest.mark.asyncio
    async def test_run_single_test(self, runner, sample_variant, sample_model):
        result = await runner.run_test(
            variant=sample_variant,
            model=sample_model,
            evaluate=True,
        )
        assert result is not None
        assert result.variant_name == "Test Variant"
        assert result.model_id == "gpt-4o"
        assert len(result.output) > 0
        assert not result.error
        assert result.scored
        assert result.overall_score >= 0

    @pytest.mark.asyncio
    async def test_run_test_without_evaluation(self, runner, sample_variant, sample_model):
        result = await runner.run_test(
            variant=sample_variant,
            model=sample_model,
            evaluate=False,
        )
        assert not result.scored
        assert result.overall_score == 0.0

    @pytest.mark.asyncio
    async def test_run_batch(self, runner, sample_model):
        variants = [
            PromptVariant(id="v1", name="Variant A", template="Write a sorting function", context={"task": "sorting function"}),
            PromptVariant(id="v2", name="Variant B", template="Create a sorting function", context={"task": "sorting function"}),
        ]
        results = await runner.run_batch(
            variants=variants,
            model=sample_model,
            evaluate=True,
        )
        assert len(results) == 2
        assert all(r is not None for r in results)
        assert all(r.scored for r in results)

    @pytest.mark.asyncio
    async def test_run_cross_model(self, runner):
        variant = PromptVariant(id="v1", name="Test", template="Hello world")
        models = [
            ModelInfo(id="gpt-4o", provider=ModelProvider.OPENAI),
            ModelInfo(id="claude-sonnet-4-20250514", provider=ModelProvider.ANTHROPIC),
        ]
        results = await runner.run_cross_model(
            variant=variant,
            models=models,
            evaluate=False,
        )
        assert len(results) == 2

    def test_model_names(self, runner):
        names = runner.model_names
        assert isinstance(names, list)
        assert "gpt-4o" in names

    def test_model_details(self, runner):
        details = runner.model_details
        assert isinstance(details, list)
        assert len(details) > 0
        assert "id" in details[0]
        assert "provider" in details[0]

    def test_get_history_empty(self, runner):
        history = runner.get_history()
        assert history == []

    def test_results_summary_empty(self, runner):
        summary = runner.get_results_summary()
        assert summary["total_tests"] == 0

    def test_results_summary_after_test(self, runner, sample_variant, sample_model):
        asyncio.run(runner.run_test(sample_variant, sample_model, evaluate=True))
        summary = runner.get_results_summary()
        assert summary["total_tests"] == 1
        assert summary["variants"] == 1
        assert summary["scored"] == 1
        assert summary["errors"] == 0

    def test_export_results(self, runner, sample_variant, sample_model):
        asyncio.run(runner.run_test(sample_variant, sample_model))
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            runner.export_results(f.name)
            data = json.loads(Path(f.name).read_text())
            assert data["total_tests"] == 1
