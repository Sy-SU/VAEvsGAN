"""Model-agnostic argument dispatch for the scripts in ``tools/``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .registry import available_models, get_workflow, model_descriptions
from .workflow import Action


def _base_parser(action: Action, program: str) -> argparse.ArgumentParser:
    descriptions = model_descriptions()
    model_help = "; ".join(
        f"{name}: {description}" for name, description in descriptions.items()
    )
    parser = argparse.ArgumentParser(
        prog=program,
        description=f"{action.capitalize()} a registered generative model.",
        epilog=model_help,
    )
    parser.add_argument("model", choices=available_models(), help="model workflow")
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML config file; command-line arguments override its values",
    )
    return parser


def run_tool(
    action: Action,
    argv: Sequence[str] | None = None,
    *,
    program: str,
) -> int:
    """Resolve a model workflow, parse its arguments, and run an action."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    base_parser = _base_parser(action, program)
    if not arguments or arguments[0] in {"-h", "--help"}:
        base_parser.print_help()
        return 0 if arguments else 2

    model_name = arguments[0].lower()
    if model_name not in available_models():
        base_parser.error(
            f"unknown model {arguments[0]!r}; choose from "
            f"{', '.join(available_models())}"
        )

    try:
        workflow = get_workflow(model_name)
        parser = _base_parser(action, program)
        parser.description = f"{action.capitalize()} {workflow.description}."
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument("model")
        config_parser.add_argument("--config", type=Path)
        known_args, _ = config_parser.parse_known_args(arguments)
        workflow.configure_parser(action, parser, known_args.config)
        args = parser.parse_args(arguments)
        workflow.run(action, args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
