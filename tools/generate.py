"""Generate samples with a registered generative model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generative.runner import run_tool


def main(argv: Sequence[str] | None = None) -> int:
    return run_tool("generate", argv, program="python tools/generate.py")


if __name__ == "__main__":
    raise SystemExit(main())

