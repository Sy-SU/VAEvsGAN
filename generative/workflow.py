"""Public workflow contract implemented by each generative model family."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Protocol


Action = Literal["train", "test", "generate"]


class ModelWorkflow(Protocol):
    """Interface used by the model-agnostic scripts in ``tools/``."""

    name: str
    description: str

    def configure_parser(
        self,
        action: Action,
        parser: argparse.ArgumentParser,
        config_path: Path | None = None,
    ) -> None:
        """Add model-specific arguments for an action."""

    def run(self, action: Action, args: argparse.Namespace) -> None:
        """Execute one model-specific action."""
