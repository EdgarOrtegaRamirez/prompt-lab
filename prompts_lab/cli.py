"""CLI interface for PromptLab."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from prompts_lab import __version__
from prompts_lab.config import ConfigLoader, get_sample_config
from prompts_lab.models import Config, Config as ConfigModel, ModelInfo, ModelProvider, ScoringCriteria
from prompts_lab.models import PromptVariant
from prompts_lab.test_runner import PromptTestRunner


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="prompt-lab",
        description="AI Prompt Playground & Evaluator — test, evaluate, and compare AI prompts",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # test command
    test_parser = subparsers.add_parser("test", help="Run a prompt test")
    test_parser.add_argument("--prompt", "-p", required=True, help="Prompt template (Jinja2)")
    test_parser.add_argument("--context", "-c", help="Context variables as JSON string")
    test_parser.add_argument("--model", "-m", help="Model ID to test against")
    test_parser.add_argument("--evaluate", "-e", action="store_true", help="Auto-score the response")
    test_parser.add_argument("--output", "-o", help="Output file path (JSON)")
    test_parser.add_argument("--system", help="System prompt")
    test_parser.add_argument("--use-mock", action="store_true", help="Use mock responses")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple prompt variants")
    compare_parser.add_argument(
        "--variants",
        "-v",
        required=True,
        nargs="+",
        help="Prompt variant files (YAML or JSON, one per variant)",
    )
    compare_parser.add_argument(
        "--model",
        "-m",
        help="Model to test against",
    )
    compare_parser.add_argument("--evaluate", action="store_true", help="Auto-score variants")
    compare_parser.add_argument("--output", "-o", help="Output file path")

    # models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a response")
    eval_parser.add_argument("--prompt", required=True, help="Original prompt")
    eval_parser.add_argument("--response", required=True, help="Response to evaluate")
    eval_parser.add_argument("--criteria", help="Comma-separated criterion names")

    # history command
    history_parser = subparsers.add_parser("history", help="View test history")

    # sample-config command
    subparsers.add_parser("sample-config", help="Print sample configuration")

    return parser


def cmd_test(args: argparse.Namespace, runner: PromptTestRunner) -> None:
    """Execute the test command."""
    import asyncio

    context = {}
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError as exc:
            print(f"Error: Invalid context JSON: {exc}", file=sys.stderr)
            sys.exit(1)

    # Select model
    model = None
    if args.model:
        model = next((m for m in runner.model_details if m["id"] == args.model), None)
        if not model:
            print(f"Error: Model '{args.model}' not found. Use 'prompt-lab models' to list available models.")
            sys.exit(1)
        model = ModelInfo(**model)
    else:
        model = ModelInfo(id="gpt-4o", provider=ModelProvider.OPENAI)

    variant = PromptVariant(
        id="cli-test",
        name="CLI Test",
        template=args.prompt,
        context=context,
        system_prompt=args.system,
    )

    result = asyncio.run(
        runner.run_test(
            variant=variant,
            model=model,
            context=context,
            evaluate=args.evaluate,
        )
    )

    # Output
    output_data = result.model_dump()
    if args.output:
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output_data, indent=2))


def cmd_compare(args: argparse.Namespace, runner: PromptTestRunner) -> None:
    """Execute the compare command."""
    import asyncio

    variants: list[PromptVariant] = []

    for vpath in args.variants:
        path = Path(vpath)
        if not path.exists():
            print(f"Error: Variant file not found: {vpath}", file=sys.stderr)
            sys.exit(1)

        raw = json.loads(path.read_text()) if path.suffix in (".json", ".yaml", ".yml") else {}

        if isinstance(raw, dict) and "template" in raw:
            variants.append(PromptVariant(**{k: v for k, v in raw.items() if k != "template"}))
            # Use template name as variant name if not provided
            if not raw.get("name"):
                variants[-1].name = path.stem
            variants[-1].template = raw["template"]

    if not variants:
        print("Error: No valid variants loaded.", file=sys.stderr)
        sys.exit(1)

    model = ModelInfo(id="gpt-4o", provider=ModelProvider.OPENAI)
    if args.model:
        model = ModelInfo(id=args.model, provider=ModelProvider.OPENAI)

    results = asyncio.run(runner.run_batch(variants, model, evaluate=args.evaluate))

    summary = runner.get_results_summary()
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(summary, indent=2))


def cmd_models(args: argparse.Namespace) -> None:
    """Execute the models command."""
    config = Config()
    if args.json:
        print(json.dumps([m.model_dump() for m in config.models], indent=2))
    else:
        print(f"{'ID':<40} {'Provider':<12} {'Context':<12} {'Input ($/1M)':<16} {'Output ($/1M)':<16}")
        print("-" * 96)
        for m in config.models:
            ip = f"${m.pricing.get('input', 0):.2f}" if m.pricing else "N/A"
            op = f"${m.pricing.get('output', 0):.2f}" if m.pricing else "N/A"
            print(f"{m.id:<40} {m.provider.value:<12} {m.max_context:<12} {ip:<16} {op:<16}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Execute the evaluate command."""
    from prompts_lab.scorer import ScoringEngine

    criteria_names = args.criteria.split(",") if args.criteria else ["helpfulness", "accuracy", "creativity", "relevance"]
    scorer = ScoringEngine()
    result = scorer.score(args.prompt, args.response, criteria_names)

    print(f"Overall Score: {result.overall_score}/100 (Grade: {result.grade})\n")
    for c in result.criteria:
        print(f"  {c.name:<15} {c.score:>5.1f}/100")
    print()
    for fb in result.feedback:
        print(f"  {fb}")


def cmd_history(args: argparse.Namespace, runner: PromptTestRunner) -> None:
    """Execute the history command."""
    entries = runner.get_history()
    if not entries:
        print("No test history found.")
        return

    print(f"{'ID':<20} {'Variant':<20} {'Model':<20} {'Score':<10} {'Status'}")
    print("-" * 90)
    for e in entries:
        grade = runner._scorer._score_to_grade(e.overall_score) if e.overall_score > 0 else "N/A"
        status = f"{grade} | {e.tokens}" if e.tokens > 0 else ("ERROR" if e.error else "N/A")
        print(f"{e.id:<20} {e.variant_name:<20} {e.model_id:<20} {e.overall_score:>6.1f}   {status}")


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(getattr(args, "verbose", False))

    # Handle subcommands that don't need a runner
    if args.command == "sample-config":
        print(get_sample_config())
        return

    if args.command == "models":
        cmd_models(args)
        return

    if args.command == "evaluate":
        cmd_evaluate(args)
        return

    if args.command == "history":
        runner = PromptTestRunner(use_mock=True)
        cmd_history(args, runner)
        return

    if args.command == "test":
        runner = PromptTestRunner(
            use_mock=getattr(args, "use_mock", False),
            mock_seed=42,
        )
        cmd_test(args, runner)
        return

    if args.command == "compare":
        runner = PromptTestRunner(use_mock=True, mock_seed=42)
        cmd_compare(args, runner)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
