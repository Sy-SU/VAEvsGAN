"""Conditional DCGAN checkpoint serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .model import ConditionalDiscriminator, ConditionalGenerator


CHECKPOINT_VERSION = 2
MODEL_NAME = "conditional_dcgan"


def save_checkpoint(
    path: str | Path,
    generator: ConditionalGenerator,
    discriminator: ConditionalDiscriminator,
    epoch: int,
    training_config: Mapping[str, str | int | float | bool],
) -> None:
    """Save the latest generator and discriminator weights."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "model_name": MODEL_NAME,
        "epoch": epoch,
        "latent_dim": generator.latent_dim,
        "num_classes": generator.num_classes,
        "label_embedding_dim": generator.label_embedding_dim,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "training_config": dict(training_config),
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[ConditionalGenerator, ConditionalDiscriminator, dict[str, Any]]:
    """Load a conditional DCGAN checkpoint and reconstruct both networks."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint["model_name"] != MODEL_NAME:
        raise ValueError(
            f"checkpoint contains model {checkpoint['model_name']!r}, "
            f"expected {MODEL_NAME!r}"
        )

    generator = ConditionalGenerator(
        latent_dim=checkpoint["latent_dim"],
        num_classes=checkpoint["num_classes"],
        label_embedding_dim=checkpoint["label_embedding_dim"],
    ).to(device)
    discriminator = ConditionalDiscriminator(
        num_classes=checkpoint["num_classes"]
    ).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    return generator, discriminator, checkpoint
