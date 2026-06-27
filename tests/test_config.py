"""Tests for config module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from prompts_lab.config import ConfigLoader, get_sample_config
from prompts_lab.models import Config, ModelProvider, ScoringCriteria


class TestConfigLoader:
    def test_load_defaults(self):
        loader = ConfigLoader(config_path=Path("/nonexistent/config.yaml"))
        config = loader.load()
        assert isinstance(config, Config)
        assert len(config.models) > 0

    def test_load_custom_config(self, tmp_path):
        config_data = """
models:
  - id: "custom-model"
    provider: "local"
    max_context: 8192
default_scoring:
  - name: "helpfulness"
    weight: 2.0
max_concurrent: 5
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_data)

        loader = ConfigLoader(config_path=config_path)
        config = loader.load()
        assert config.max_concurrent == 5
        assert len(config.models) >= 1

    def test_save_and_load(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        loader = ConfigLoader(config_path=config_path)

        original = Config(max_concurrent=5, timeout_seconds=120)
        loader.save(original)

        reloaded = loader.load()
        assert reloaded.max_concurrent == 5
        assert reloaded.timeout_seconds == 120

    def test_exists_returns_false(self):
        loader = ConfigLoader(config_path=Path("/nonexistent/path/config.yaml"))
        assert loader.exists() is False

    def test_sample_config_not_empty(self):
        config = get_sample_config()
        assert len(config) > 100
        assert "models:" in config
        assert "default_scoring:" in config
