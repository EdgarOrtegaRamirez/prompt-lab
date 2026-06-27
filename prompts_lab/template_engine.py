"""Prompt template engine with variable interpolation and context injection."""

from __future__ import annotations

import logging
from typing import Any

from jinja2 import BaseLoader, Environment, StrictUndefined

from prompts_lab.tokenizer import count_tokens_tiktoken

logger = logging.getLogger(__name__)


class PromptTemplateEngine:
    """Engine for rendering Jinja2 prompt templates with context variables."""

    def __init__(self, autoescape: bool = True):
        self._env = Environment(
            loader=BaseLoader(),
            undefined=StrictUndefined,
            autoescape=autoescape,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(
        self,
        template: str,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> dict[str, str]:
        """Render a prompt template with given context.

        Args:
            template: Jinja2 prompt template string.
            context: Variables to inject into the template.
            system_prompt: Optional system prompt.
            model: Model name for token counting.

        Returns:
            Dict with 'system', 'user', and 'token_counts' keys.

        Raises:
            RuntimeError: If template rendering fails due to missing variables.
        """
        context = context or {}

        try:
            jinja_template = self._env.from_string(template)
            user_message = jinja_template.render(**context)
        except Exception as exc:
            raise RuntimeError(f"Template rendering failed: {exc}") from exc

        result: dict[str, str] = {}
        if system_prompt:
            result["system"] = system_prompt
        result["user"] = user_message

        token_counts: dict[str, int] = {}
        if model:
            for role, content in result.items():
                token_counts[role] = count_tokens_tiktoken(content, model)
            result["token_counts"] = token_counts

        logger.debug(
            "Rendered prompt: system=%d tokens, user=%d tokens",
            result.get("token_counts", {}).get("system", 0),
            result.get("token_counts", {}).get("user", 0),
        )

        return result

    def render_user_only(self, template: str, context: dict[str, Any] | None = None) -> str:
        """Render only the user message from a template.

        Args:
            template: Jinja2 template string.
            context: Variables to inject.

        Returns:
            Rendered user message string.
        """
        result = self.render(template, context)
        return result["user"]

    def validate_template(self, template: str) -> tuple[bool, list[str]]:
        """Validate that a template is syntactically correct.

        Args:
            template: Template string to validate.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors: list[str] = []
        try:
            self._env.from_string(template)
        except Exception as exc:
            errors.append(str(exc))
        return (len(errors) == 0, errors)

    def extract_variables(self, template: str) -> list[str]:
        """Extract all template variable names from a Jinja2 template.

        Args:
            template: Jinja2 template string.

        Returns:
            List of variable names.
        """
        try:
            ast = self._env.parse(template)
            from jinja2 import meta

            variables = meta.find_undeclared_variables(ast)
            return sorted(variables)
        except Exception:
            return []
