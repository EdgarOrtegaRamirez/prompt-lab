"""Prompt test runner — sends prompts to models and collects results."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prompts_lab.mock_client import MockAPIClient
from prompts_lab.models import (
    Config,
    HistoryEntry,
    ModelInfo,
    PromptVariant,
    ScoringCriteria,
    TestRecord,
    TestResult,
)
from prompts_lab.scorer import ScoringEngine
from prompts_lab.template_engine import PromptTemplateEngine

logger = logging.getLogger(__name__)


class PromptTestRunner:
    """Runs prompt tests against models and collects results."""

    def __init__(
        self,
        config: Config | None = None,
        api_key: str | None = None,
        use_mock: bool = False,
        mock_seed: int | None = None,
    ):
        self._config = config or Config()
        self._api_key = api_key
        self._engine = PromptTemplateEngine()
        self._scorer = ScoringEngine()
        self._use_mock = use_mock

        if use_mock:
            self._mock_client = MockAPIClient(seed=mock_seed)
        else:
            self._mock_client = None

        self._test_history: list[TestRecord] = []
        self._evaluation_history: list[HistoryEntry] = []

    @property
    def model_names(self) -> list[str]:
        """List of available model names."""
        return [m.id for m in self._config.models]

    @property
    def model_details(self) -> list[dict[str, Any]]:
        """List of model info as dicts."""
        return [m.model_dump() for m in self._config.models]

    async def run_test(
        self,
        variant: PromptVariant,
        model: ModelInfo,
        context: dict[str, Any] | None = None,
        evaluate: bool = True,
        criteria: list[ScoringCriteria] | None = None,
    ) -> TestResult:
        """Run a single prompt test against a model.

        Args:
            variant: Prompt variant to test.
            model: Model to send the prompt to.
            context: Additional context variables.
            evaluate: Whether to auto-score the response.
            criteria: Scoring criteria to use.

        Returns:
            TestResult with output and scores.
        """
        # Merge context variables
        merged_context = dict(variant.context)
        if context:
            merged_context.update(context)

        # Render template
        rendered = self._engine.render(
            template=variant.template,
            context=merged_context,
            system_prompt=variant.system_prompt,
            model=model.id,
        )

        # Build messages list
        messages: list[dict[str, str]] = []
        if "system" in rendered:
            messages.append({"role": "system", "content": rendered["system"]})
        messages.append({"role": "user", "content": rendered["user"]})

        user_prompt = rendered["user"]

        # Call the API (mock or real)
        test_result = await self._call_model(messages, model, variant)

        # Calculate cost
        test_result.cost_usd = model.cost_estimate(test_result.prompt_tokens, test_result.completion_tokens)

        # Auto-evaluate
        if evaluate and not test_result.error:
            criteria_list = criteria or self._config.default_scoring
            criteria_weights = {c.name: c.weight for c in criteria_list}

            evaluation = self._scorer.score_weighted(
                user_prompt=user_prompt,
                response=test_result.output,
                criteria_weights=criteria_weights,
            )

            test_result.scored = True
            test_result.scores = {
                c.name: round(c.score, 1) for c in evaluation.criteria
            }
            test_result.overall_score = evaluation.overall_score

            logger.info(
                "Evaluated variant '%s' vs %s: %.1f/%s",
                variant.name,
                model.id,
                test_result.overall_score,
                evaluation.grade,
            )

        # Store record
        record = TestRecord(
            test_id=datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            variant=variant,
            model=model,
            result=test_result,
        )
        self._test_history.append(record)

        return test_result

    async def run_batch(
        self,
        variants: list[PromptVariant],
        model: ModelInfo,
        evaluate: bool = True,
        criteria: list[ScoringCriteria] | None = None,
    ) -> list[TestResult]:
        """Run tests for multiple variants against a single model.

        Args:
            variants: List of prompt variants to test.
            model: Model to send prompts to.
            evaluate: Whether to auto-score.
            criteria: Scoring criteria.

        Returns:
            List of TestResults.
        """
        tasks = [
            self.run_test(v, model, evaluate=evaluate, criteria=criteria)
            for v in variants
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def run_cross_model(
        self,
        variant: PromptVariant,
        models: list[ModelInfo],
        evaluate: bool = True,
        criteria: list[ScoringCriteria] | None = None,
    ) -> list[TestResult]:
        """Run the same prompt against multiple models (concurrent).

        Args:
            variant: Prompt variant to test.
            models: Models to test against.
            evaluate: Whether to auto-score.
            criteria: Scoring criteria.

        Returns:
            List of TestResults, one per model.
        """
        semaphore = asyncio.Semaphore(self._config.max_concurrent)

        async def _run_with_semaphore(m: ModelInfo) -> TestResult:
            async with semaphore:
                return await self.run_test(variant, m, evaluate=evaluate, criteria=criteria)

        tasks = [_run_with_semaphore(m) for m in models]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _call_model(
        self,
        messages: list[dict[str, str]],
        model: ModelInfo,
        variant: PromptVariant,
    ) -> TestResult:
        """Call the AI model (mock or real)."""
        if self._use_mock or not self._api_key:
            return await self._mock_call(messages, model, variant)
        return await self._real_call(messages, model, variant)

    async def _mock_call(
        self,
        messages: list[dict[str, str]],
        model: ModelInfo,
        variant: PromptVariant,
    ) -> TestResult:
        """Call the mock API client."""
        try:
            response = await self._mock_client.chat(
                messages=messages,
                model=model,
                temperature=variant.temperature,
                max_tokens=variant.max_tokens,
            )
            return TestResult(
                variant_id=variant.id,
                variant_name=variant.name,
                model_id=response.model,
                model_provider=response.provider,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.prompt_tokens + response.completion_tokens,
                output=response.content,
                cost_usd=response.prompt_tokens * 0.000001,  # rough estimate
                response_time_ms=response.response_time_ms,
            )
        except Exception as exc:
            return TestResult(
                variant_id=variant.id,
                variant_name=variant.name,
                model_id=model.id,
                model_provider=model.provider.value,
                error=str(exc),
            )

    async def _real_call(
        self,
        messages: list[dict[str, str]],
        model: ModelInfo,
        variant: PromptVariant,
    ) -> TestResult:
        """Call a real API (placeholder — user must provide API key)."""
        # This is a placeholder for future real API integration.
        # For now, fall back to mock for all testing.
        logger.warning("Real API calls not yet implemented; using mock response")
        return await self._mock_call(messages, model, variant)

    def get_history(self) -> list[HistoryEntry]:
        """Get test history as HistoryEntry list."""
        entries: list[HistoryEntry] = []
        for record in self._test_history:
            entries.append(
                HistoryEntry(
                    id=record.id,
                    test_id=record.test_id,
                    timestamp=record.result.created_at,
                    variant_name=record.variant.name,
                    model_id=record.result.model_id,
                    provider=record.result.model_provider,
                    overall_score=record.result.overall_score,
                    prompt_tokens=record.result.prompt_tokens,
                    completion_tokens=record.result.completion_tokens,
                    cost_usd=record.result.cost_usd,
                    error=record.result.error,
                    summary=record.result.output[:200] if record.result.output else "",
                )
            )
        return entries

    def get_results_summary(self) -> dict[str, Any]:
        """Get a summary of all test results."""
        if not self._test_history:
            return {"total_tests": 0, "variants": 0, "models": 0, "results": []}

        variant_names = set(r.variant.name for r in self._test_history)
        model_ids = set(r.result.model_id for r in self._test_history)
        scored = [r for r in self._test_history if r.result.scored]
        errors = [r for r in self._test_history if r.result.error]

        return {
            "total_tests": len(self._test_history),
            "variants": len(variant_names),
            "models": len(model_ids),
            "scored": len(scored),
            "errors": len(errors),
            "total_cost_usd": round(
                sum(r.result.cost_usd or 0 for r in self._test_history),
                4,
            ),
            "avg_response_time_ms": round(
                sum(r.result.response_time_ms for r in self._test_history) / len(self._test_history),
                1,
            ),
            "results": [
                {
                    "variant": r.variant.name,
                    "model": r.result.model_id,
                    "overall_score": r.result.overall_score,
                    "grade": self._scorer._score_to_grade(r.result.overall_score) if r.result.scored else "N/A",
                    "tokens": r.result.total_tokens,
                    "cost": r.result.cost_usd,
                    "error": r.result.error,
                }
                for r in self._test_history
            ],
        }

    def export_results(self, filepath: str | Path) -> None:
        """Export all test results to a JSON file."""
        filepath = Path(filepath)
        data = self.get_results_summary()
        filepath.write_text(json.dumps(data, indent=2))
        logger.info("Exported results to %s", filepath)
