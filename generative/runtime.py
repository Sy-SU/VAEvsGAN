"""Runtime helpers shared by current and future model workflows."""

from __future__ import annotations

import argparse
import math
import random

import torch


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1)")
    return parsed


def digit(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 9:
        raise argparse.ArgumentTypeError("digit must be between 0 and 9")
    return parsed


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs and select deterministic cuDNN behavior."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda into a usable torch device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available; use --device cpu or --device auto"
        )
    return torch.device(requested)


def add_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_workers: bool,
    seed: int = 42,
    device: str = "auto",
    num_workers: int = 0,
) -> None:
    parser.add_argument("--seed", type=nonnegative_int, default=seed)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default=device
    )
    if include_workers:
        parser.add_argument(
            "--num-workers", type=nonnegative_int, default=num_workers
        )
