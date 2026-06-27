"""Tests for scorer module."""

from __future__ import annotations

from prompts_lab.scorer import ScoringEngine


class TestScoringEngine:
    def setup_method(self):
        self.engine = ScoringEngine()

    def test_score_basic(self):
        result = self.engine.score(
            user_prompt="Write a sorting function",
            response="Here's a helpful sorting function that works correctly and efficiently.",
            criteria_names=["helpfulness", "accuracy"],
        )
        assert result.overall_score >= 0
        assert result.overall_score <= 100
        assert result.grade in ("A", "B", "C", "D", "F")
        assert len(result.criteria) == 2

    def test_score_with_feedback(self):
        result = self.engine.score(
            user_prompt="Help me",
            response="I don't know sorry cannot.",
            criteria_names=["helpfulness"],
        )
        assert len(result.feedback) > 0

    def test_score_best_criteria(self):
        result = self.engine.score(
            user_prompt="Write code",
            response="Great helpful correct accurate relevant creative clear concise solution works well.",
            criteria_names=["helpfulness", "accuracy", "creativity"],
        )
        assert result.best_criteria is not None
        assert result.weakest_criteria is not None

    def test_score_empty_criteria(self):
        result = self.engine.score(
            user_prompt="test",
            response="response",
            criteria_names=[],
        )
        assert result.overall_score == 0.0
        assert result.grade == "F"
        assert len(result.criteria) == 0

    def test_score_short_response(self):
        result = self.engine.score(
            user_prompt="Help me",
            response="No.",
            criteria_names=["helpfulness"],
        )
        # Short responses should score lower
        assert result.criteria[0].score < 50

    def test_grade_a(self):
        assert self.engine._score_to_grade(95) == "A"
        assert self.engine._score_to_grade(90) == "A"

    def test_grade_b(self):
        assert self.engine._score_to_grade(85) == "B"
        assert self.engine._score_to_grade(80) == "B"

    def test_grade_c(self):
        assert self.engine._score_to_grade(75) == "C"
        assert self.engine._score_to_grade(70) == "C"

    def test_grade_d(self):
        assert self.engine._score_to_grade(65) == "D"
        assert self.engine._score_to_grade(60) == "D"

    def test_grade_f(self):
        assert self.engine._score_to_grade(50) == "F"
        assert self.engine._score_to_grade(0) == "F"

    def test_weighted_score(self):
        result = self.engine.score_weighted(
            user_prompt="Write code",
            response="Great helpful accurate relevant code.",
            criteria_weights={"helpfulness": 3.0, "accuracy": 1.0},
        )
        assert result.overall_score >= 0
        assert result.overall_score <= 100
        # The helpfulness criterion should have higher weighted score
        help_score = next(c for c in result.criteria if c.name == "helpfulness")
        acc_score = next(c for c in result.criteria if c.name == "accuracy")
        assert help_score.weighted_score > acc_score.weighted_score
