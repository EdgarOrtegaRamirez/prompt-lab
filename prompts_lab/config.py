"""Configuration management for PromptLab."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from prompts_lab.models import Config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "prompt-lab" / "config.yaml"


class ConfigLoader:
    """Loads and saves PromptLab configuration."""

    def __init__(self, config_path: Path | None = None):
        self._config_path = config_path or DEFAULT_CONFIG_PATH

    @property
    def config_path(self) -> Path:
        return self._config_path

    def load(self) -> Config:
        """Load configuration from file. Returns defaults if file doesn't exist.

        Raises:
            ValueError: If the config file has invalid YAML.
        """
        if not self._config_path.exists():
            logger.info("No config file found at %s, using defaults", self._config_path)
            return Config()

        try:
            raw = yaml.safe_load(self._config_path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in config file: {exc}") from exc

        if not raw or not isinstance(raw, dict):
            logger.warning("Empty config file, using defaults")
            return Config()

        return self._parse_config(raw)

    def save(self, config: Config) -> None:
        """Save configuration to file.

        Args:
            config: The Config object to save.
        """
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        data = config.model_dump(mode="json")
        self._config_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        logger.info("Saved config to %s", self._config_path)

    def exists(self) -> bool:
        """Check if config file exists."""
        return self._config_path.exists()

    def _parse_config(self, raw: dict[str, Any]) -> Config:
        """Parse raw config dict into Config model."""
        config_data: dict[str, Any] = {}

        # Parse models
        if "models" in raw:
            from prompts_lab.models import ModelInfo

            config_data["models"] = [
                ModelInfo(**m) if isinstance(m, dict) else m
                for m in raw["models"]
            ]

        # Parse default scoring
        if "default_scoring" in raw:
            from prompts_lab.models import ScoringCriteria

            config_data["default_scoring"] = [
                ScoringCriteria(**s) if isinstance(s, dict) else s
                for s in raw["default_scoring"]
            ]

        # Parse scalar fields
        for field in ["default_provider", "max_concurrent", "timeout_seconds", "retry_attempts"]:
            if field in raw:
                config_data[field] = raw[field]

        return Config(**config_data)


def get_sample_config() -> str:
    """Return a sample configuration as a YAML string."""
    return """# PromptLab Configuration
# Location: ~/.config/prompt-lab/config.yaml

# Models to test against
models:
  - id: "gpt-4o"
    provider: "openai"
    max_context: 128000
    pricing:
      input: 2.50
      output: 10.00
  - id: "claude-sonnet-4-20250514"
    provider: "anthropic"
    max_context: 200000
    pricing:
      input: 3.00
      output: 15.00
  - id: "gemini-2.0-flash"
    provider: "google"
    max_context: 1048576
    pricing:
      input: 0.15
      output: 0.60

# Default scoring criteria
default_scoring:
  - name: "helpfulness"
    weight: 3.0
    description: "How helpful is the response?"
  - name: "accuracy"
    weight: 2.5
    description: "How accurate and factual is the response?"
  - name: "creativity"
    weight: 1.0
    description: "How creative and original is the response?"
  - name: "relevance"
    weight: 2.0
    description: "How relevant is the response to the prompt?"

# API settings
default_provider: "openai"
max_concurrent: 3
timeout_seconds: 60
retry_attempts: 2
"""
