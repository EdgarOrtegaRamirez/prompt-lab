# AGENTS.md

## PromptLab - AI Agent Notes

This repo is a CLI tool and Python library for testing, evaluating, and comparing AI prompts.

### Quick Overview

- **Language**: Python 3.10+
- **Build**: `uv pip install -e ".[dev]"` or `pip install -e ".[dev]"`
- **Tests**: `pytest tests/ -v` (59 tests)
- **CLI**: `python -m prompts_lab.cli` or `prompt-lab`

### Project Structure

```
prompts_lab/          # Main package
├── cli.py            # CLI entry point (argparse)
├── config.py         # YAML config loader
├── models.py         # Pydantic models (Config, ModelInfo, PromptVariant, etc.)
├── mock_client.py    # Mock API client for testing without real API calls
├── scorer.py         # Automated scoring engine (keyword-based heuristics)
├── template_engine.py# Jinja2 prompt template rendering
├── test_runner.py    # Main test orchestration (PromptTestRunner)
└── tokenizer.py      # Token counting utilities

tests/                # Comprehensive test suite
├── test_models.py    # Model tests
├── test_config.py    # Config loader tests
├── test_template_engine.py
├── test_scorer.py
├── test_mock_client.py
├── test_test_runner.py
└── test_tokenizer.py
```

### Key Classes

- `PromptTestRunner` - Main test runner class. Use `use_mock=True` for testing without API keys.
- `PromptTemplateEngine` - Jinja2 template rendering with variable injection
- `ScoringEngine` - Automated response scoring with configurable criteria
- `MockAPIClient` - Simulates AI model responses for testing

### Running Tests

```bash
uv venv && uv pip install -e ".[dev]"
pytest tests/ -v
```

### Adding Features

1. Add new models in `models.py` DEFAULT_MODELS
2. Add new scoring criteria in `scorer.py` POSITIVE_PATTERNS/NEGATIVE_PATTERNS
3. Add CLI commands in `cli.py` with subparser
4. Add tests in `tests/` following existing patterns
5. Update README.md

### Important Notes

- The mock client provides deterministic responses via seed-based random
- Real API integration is planned but not yet implemented
- Config uses YAML with Pydantic v2 models
- Token counting uses tiktoken when available, falls back to heuristic counting
