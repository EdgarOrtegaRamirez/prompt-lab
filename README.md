# PromptLab

AI Prompt Playground & Evaluator — test, evaluate, and compare AI prompts across multiple models with configurable scoring criteria.

## What It Does

PromptLab helps prompt engineers and developers systematically evaluate AI prompts by:

- **Template Engine** — Render Jinja2 prompt templates with variable interpolation and context injection
- **Model Testing** — Send prompts to multiple AI models (via mock client or real API adapters)
- **A/B Comparison** — Compare multiple prompt variants against the same model
- **Automated Scoring** — Score responses on configurable criteria (helpfulness, accuracy, creativity, relevance, clarity)
- **Cross-Model Evaluation** — Test the same prompt across multiple models concurrently
- **Cost Tracking** — Estimate token usage and cost for each test

Complements [Prompt Diff](https://github.com/EdgarOrtegaRamirez/prompt-diff) (which handles versioning/diffing) by providing the testing and evaluation layer.

## Quick Start

```bash
# Install
pip install prompts-lab

# Test a prompt (mock mode)
prompt-lab test -p "Write a {{ task }}" -c '{"task": "sorting function"}' --use-mock --evaluate

# Compare two prompt variants
prompt-lab compare --variants variant_a.yaml variant_b.yaml --model gpt-4o --evaluate

# List available models
prompt-lab models

# Evaluate a response manually
prompt-lab evaluate --prompt "Explain Python" --response "Python is a programming language..."

# Generate sample config
prompt-lab sample-config > ~/.config/prompt-lab/config.yaml
```

## Installation

```bash
pip install prompts-lab
```

Or from source:

```bash
git clone https://github.com/EdgarOrtegaRamirez/prompt-lab.git
cd prompt-lab
uv venv && uv pip install -e ".[dev]"
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `test` | Run a single prompt test against a model |
| `compare` | Compare multiple prompt variants |
| `models` | List available models with pricing |
| `evaluate` | Score a response against criteria |
| `history` | View test history |
| `sample-config` | Print sample configuration |
| `--version` | Show version |

### Test Command Options

```
-p, --prompt      Prompt template (Jinja2)
-c, --context     Context variables as JSON string
-m, --model       Model ID to test against
-e, --evaluate    Auto-score the response
-o, --output      Output file path (JSON)
--system          System prompt
--use-mock        Use mock responses
```

## Python API

```python
from prompts_lab.test_runner import PromptTestRunner
from prompts_lab.models import ModelInfo, ModelProvider, PromptVariant

runner = PromptTestRunner(use_mock=True)

variant = PromptVariant(
    id="v1",
    name="My Prompt",
    template="Write a {{ task }}",
    context={"task": "sorting function"},
)

model = ModelInfo(id="gpt-4o", provider=ModelProvider.OPENAI)

result = await runner.run_test(variant, model, evaluate=True)
print(f"Score: {result.overall_score}/100")
```

## Architecture

```
prompt-lab/
├── prompts_lab/
│   ├── __init__.py          # Package init
│   ├── cli.py               # CLI interface (argparse)
│   ├── config.py            # YAML config loader
│   ├── models.py            # Pydantic models
│   ├── mock_client.py       # Mock API client for testing
│   ├── scorer.py            # Automated scoring engine
│   ├── template_engine.py   # Jinja2 prompt templates
│   ├── test_runner.py       # Main test orchestration
│   └── tokenizer.py         # Token counting utilities
├── tests/                   # Comprehensive test suite
├── pyproject.toml           # Build config (hatchling)
├── README.md
├── LICENSE
└── .github/workflows/ci.yml
```

## Scoring Criteria

Built-in criteria with keyword-based scoring:

| Criterion | Description | Default Weight |
|-----------|-------------|----------------|
| `helpfulness` | How helpful is the response? | 3.0 |
| `accuracy` | How accurate and factual? | 2.5 |
| `creativity` | How creative and original? | 1.0 |
| `relevance` | How relevant to the prompt? | 2.0 |
| `clarity` | How clear and concise? | N/A |

Scores are 0-100, converted to letter grades (A-F). Custom weights can be specified.

## Configuration

Default config location: `~/.config/prompt-lab/config.yaml`

```yaml
models:
  - id: "gpt-4o"
    provider: "openai"
    pricing: { input: 2.50, output: 10.00 }

default_scoring:
  - name: "helpfulness"
    weight: 3.0

max_concurrent: 3
timeout_seconds: 60
```

## License

MIT — see [LICENSE](LICENSE) file.
