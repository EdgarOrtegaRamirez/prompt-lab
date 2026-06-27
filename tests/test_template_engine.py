"""Tests for template engine."""

from __future__ import annotations

from prompts_lab.template_engine import PromptTemplateEngine


class TestTemplateEngine:
    def setup_method(self):
        self.engine = PromptTemplateEngine()

    def test_render_simple(self):
        result = self.engine.render(
            template="Write code for: {{ task }}",
            context={"task": "sorting function"},
        )
        assert result["user"] == "Write code for: sorting function"

    def test_render_with_system_prompt(self):
        result = self.engine.render(
            template="{{ message }}",
            context={"message": "Hello"},
            system_prompt="You are a helpful assistant.",
        )
        assert "system" in result
        assert result["system"] == "You are a helpful assistant."
        assert result["user"] == "Hello"

    def test_render_missing_variable_raises(self):
        try:
            self.engine.render("{{ missing_var }}")
        except RuntimeError as exc:
            assert "Template rendering failed" in str(exc)

    def test_validate_valid_template(self):
        is_valid, errors = self.engine.validate_template("Hello {{ name }}!")
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_template(self):
        is_valid, errors = self.engine.validate_template("{% invalid")
        assert is_valid is False
        assert len(errors) > 0

    def test_extract_variables(self):
        vars_list = self.engine.extract_variables(
            "Hello {{ name }}, your role is {{ role }}"
        )
        assert "name" in vars_list
        assert "role" in vars_list

    def test_extract_variables_no_vars(self):
        vars_list = self.engine.extract_variables("Plain text with no variables")
        assert len(vars_list) == 0

    def test_user_only(self):
        result = self.engine.render_user_only(
            "Task: {{ task }}\nContext: {{ context }}",
            context={"task": "coding", "context": "web app"},
        )
        assert "Task:" in result

    def test_render_with_multiple_context(self):
        result = self.engine.render(
            template="For user: {{ name }}, role: {{ role }}, task: {{ task }}",
            context={"name": "Alice", "role": "dev", "task": "fix bug"},
        )
        assert "Alice" in result["user"]
        assert "dev" in result["user"]
        assert "fix bug" in result["user"]

    def test_empty_context(self):
        result = self.engine.render(template="Static text")
        assert result["user"] == "Static text"

    def test_empty_template(self):
        result = self.engine.render(template="")
        assert result["user"] == ""
