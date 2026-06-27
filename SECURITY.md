# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please report security vulnerabilities via GitHub Issues with the label `security`.

## Security Considerations

### What This Project Handles

- **Template Rendering**: Uses Jinja2 with `StrictUndefined` to prevent template injection attacks
- **Context Variables**: JSON-parsed context variables are validated before injection
- **File Paths**: Config file paths are validated to exist before reading/writing
- **Input Validation**: Pydantic models enforce type and range constraints on all inputs

### What Users Should Be Aware Of

- **Mock Mode**: When using `--use-mock`, no real API calls are made — responses are simulated
- **Real API Keys**: Future real API integration will require API keys stored in environment variables
- **Config Files**: Config is stored in `~/.config/prompt-lab/config.yaml` — keep this file secure
- **Token Output**: No PII is transmitted — the mock client generates fake responses

### Recommendations

- Never commit real API keys or tokens
- Use environment variables for sensitive configuration
- Keep dependencies updated via `uv pip install -U`
- Run `pytest` before pushing changes
