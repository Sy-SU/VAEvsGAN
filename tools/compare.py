"""Run the reproducible, paired MNIST VAE / DCGAN comparison study."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/comparison.toml")
    parser.add_argument("--stage", choices=["train", "evaluate", "report", "all"], default="all")
    parser.add_argument("--smoke", action="store_true", help="Run an isolated small real-data study, never the formal experiment")
    args = parser.parse_args()
    from generative.comparison_study import run_study
    run_study(args.config, args.stage, smoke=args.smoke)


if __name__ == "__main__":
    main()
