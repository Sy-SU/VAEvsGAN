"""Conditional VAE checkpoint serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .model import ConvVAE


CHECKPOINT_VERSION = 2
MODEL_NAME = "conditional_vae"


def save_checkpoint(
    path: str | Path,
    model: ConvVAE,
    epoch: int,
    training_config: Mapping[str, str | int | float | bool],
) -> None:
    """Save the latest conditional VAE checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "model_name": MODEL_NAME,
        "epoch": epoch,
        "latent_dim": model.latent_dim,
        "num_classes": model.num_classes,
        "model_state_dict": model.state_dict(),
        "training_config": dict(training_config),
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[ConvVAE, dict[str, Any]]:
    """Load a conditional VAE checkpoint and reconstruct its model."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint["model_name"] != MODEL_NAME:
        raise ValueError(
            f"checkpoint contains model {checkpoint['model_name']!r}, "
            f"expected {MODEL_NAME!r}"
        )

    latent_dim = checkpoint["latent_dim"]
    num_classes = checkpoint["num_classes"]
    model = ConvVAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
