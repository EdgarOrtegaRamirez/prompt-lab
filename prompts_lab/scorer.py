"""Prompt evaluation scoring engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CriterionResult:
    """Score for a single evaluation criterion."""

    name: str
    score: float  # 0-100
    weight: float
    weighted_score: float


@dataclass
class EvaluationResult:
    """Complete evaluation result with scored criteria and overall grade."""

    overall_score: float  # 0-100
    grade: str  # A-F
    criteria: list[CriterionResult]
    feedback: list[str]

    @property
    def best_criteria(self) -> CriterionResult | None:
        if not self.criteria:
            return None
        return max(self.criteria, key=lambda c: c.score)

    @property
    def weakest_criteria(self) -> CriterionResult | None:
        if not self.criteria:
            return None
        return min(self.criteria, key=lambda c: c.score)


class ScoringEngine:
    """Evaluates AI model responses against configurable scoring criteria.

    Supports two modes:
    - Automated: Uses keyword/metric-based heuristics to score responses
    - Manual: Returns the structure so the user can fill in scores
    """

    # Scoring keywords for automated evaluation
    POSITIVE_PATTERNS: dict[str, list[str]] = {
        "helpfulness": ["helpful", "useful", "solution", "works", "recommend", "clear", "good", "best", "effective", "solves"],
        "accuracy": ["correct", "accurate", "precise", "fact", "verify", "source", "data", "evidence", "proven", "confirmed"],
        "creativity": ["creative", "novel", "unique", "innovative", "imagine", "imagine", "imagine", "idea", "original", "insightful"],
        "relevance": ["relevant", "on-topic", "directly", "specifically", "addressed", "answering", "exactly", "precisely", "focused"],
        "clarity": ["clear", "concise", "easy", "understand", "explained", "simple", "readable", "well-structured"],
    }

    NEGATIVE_PATTERNS: dict[str, list[str]] = {
        "helpfulness": ["sorry", "unable", "cannot", "don't know", "not sure", "may", "probably", "vague"],
        "accuracy": ["incorrect", "wrong", "outdated", "deprecated", "mistake", "error", "bug", "flawed"],
        "creativity": ["cliché", "generic", "boring", "standard", "obvious", "common", "routine"],
        "relevance": ["unrelated", "off-topic", "irrelevant", "not related", "doesn't address", "missed"],
        "clarity": ["confusing", "unclear", "vague", "ambiguous", "complex", "convoluted", "hard to follow"],
    }

    def score(
        self,
        user_prompt: str,
        response: str,
        criteria_names: list[str],
    ) -> EvaluationResult:
        """Score a response against the given criteria.

        Args:
            user_prompt: The original user prompt.
            response: The model's response to evaluate.
            criteria_names: List of criterion names to score.

        Returns:
            EvaluationResult with scores and feedback.
        """
        response_lower = response.lower()
        prompt_lower = user_prompt.lower()

        criteria_results: list[CriterionResult] = []
        feedback: list[str] = []

        for name in criteria_names:
            score = self._score_single(name, prompt_lower, response_lower)
            # Default weight of 1.0 for all criteria
            criteria_results.append(CriterionResult(name=name, score=score, weight=1.0, weighted_score=score))
            feedback.extend(self._generate_feedback(name, score))

        overall = self._compute_overall(criteria_results)
        grade = self._score_to_grade(overall)

        logger.info("Evaluation complete: overall=%.1f, grade=%s", overall, grade)

        return EvaluationResult(
            overall_score=round(overall, 1),
            grade=grade,
            criteria=criteria_results,
            feedback=feedback,
        )

    def score_weighted(
        self,
        user_prompt: str,
        response: str,
        criteria_weights: dict[str, float],
    ) -> EvaluationResult:
        """Score a response with weighted criteria.

        Args:
            user_prompt: The original user prompt.
            response: The model's response to evaluate.
            criteria_weights: Dict of {criterion_name: weight}.

        Returns:
            EvaluationResult with weighted scores.
        """
        response_lower = response.lower()
        prompt_lower = user_prompt.lower()

        criteria_results: list[CriterionResult] = []

        for name, weight in criteria_weights.items():
            score = self._score_single(name, prompt_lower, response_lower)
            weighted = score * weight
            criteria_results.append(CriterionResult(name=name, score=score, weight=weight, weighted_score=weighted))

        total_weight = sum(c.weight for c in criteria_results)
        if total_weight == 0:
            overall = sum(c.score for c in criteria_results) / len(criteria_results) if criteria_results else 0
        else:
            overall = sum(c.weighted_score for c in criteria_results) / total_weight

        grade = self._score_to_grade(overall)

        return EvaluationResult(
            overall_score=round(overall, 1),
            grade=grade,
            criteria=criteria_results,
            feedback=[],
        )

    def _score_single(self, criterion: str, prompt_lower: str, response_lower: str) -> float:
        """Score a single criterion on a 0-100 scale."""
        positive = self.POSITIVE_PATTERNS.get(criterion, [])
        negative = self.NEGATIVE_PATTERNS.get(criterion, [])

        pos_count = sum(1 for word in positive if word in response_lower)
        neg_count = sum(1 for word in negative if word in response_lower)

        # Base score: start at 50, adjust based on patterns
        base_score = 50.0
        adjustment = (pos_count * 5.0) - (neg_count * 8.0)
        score = base_score + adjustment

        # Check for length-based signals: longer, detailed responses tend to score higher
        response_words = len(response_lower.split())
        if response_words < 20:
            score -= 15  # Too short
        elif response_words > 200:
            score += 5  # Comprehensive

        # Clamp to 0-100
        return max(0.0, min(100.0, score))

    def _compute_overall(self, criteria: list[CriterionResult]) -> float:
        """Compute overall score from weighted criteria results."""
        if not criteria:
            return 0.0

        total_weight = sum(c.weight for c in criteria)
        if total_weight == 0:
            return sum(c.score for c in criteria) / len(criteria)

        return sum(c.weighted_score for c in criteria) / total_weight

    def _score_to_grade(self, score: float) -> str:
        """Convert a 0-100 score to a letter grade."""
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"

    def _generate_feedback(self, criterion: str, score: float) -> list[str]:
        """Generate actionable feedback for a criterion score."""
        feedback: list[str] = []

        if score < 40:
            feedback.append(f"{criterion.capitalize()}: Low score. Try making the prompt more specific and include examples.")
        elif score < 60:
            feedback.append(f"{criterion.capitalize()}: Could improve. Consider adding more context and clearer instructions.")
        elif score >= 90:
            feedback.append(f"{criterion.capitalize()}: Excellent! The prompt is well-crafted for this criterion.")
        elif score >= 80:
            feedback.append(f"{criterion.capitalize()}: Good score. Minor tweaks could make it even better.")

        return feedback
