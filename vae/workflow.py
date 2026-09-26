"""Conditional VAE workflow used by the model-agnostic tools."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm.auto import tqdm

from generative.config import (
    CONFIG_DIR,
    config_device,
    config_path,
    config_value,
    load_config_section,
)
from generative.data import MNISTPNGDataset, make_loader
from generative.runtime import (
    add_runtime_arguments,
    digit,
    nonnegative_int,
    positive_float,
    positive_int,
    resolve_device,
    set_seed,
)
from generative.visualization import (
    save_training_progress,
    training_snapshot_epochs,
)
from generative.workflow import Action

from .checkpoint import load_checkpoint, save_checkpoint
from .model import ConvVAE, VAELoss, vae_loss


DEFAULT_CONFIG = CONFIG_DIR / "vae.toml"


class MetricAccumulator:
    """Accumulate per-sample batch means into dataset-wide means."""

    def __init__(self) -> None:
        self.total = 0.0
        self.reconstruction = 0.0
        self.kl = 0.0
        self.samples = 0

    def update(self, losses: VAELoss, batch_size: int) -> None:
        self.total += losses.total.item() * batch_size
        self.reconstruction += losses.reconstruction.item() * batch_size
        self.kl += losses.kl.item() * batch_size
        self.samples += batch_size

    def averages(self) -> tuple[float, float, float]:
        return (
            self.total / self.samples,
            self.reconstruction / self.samples,
            self.kl / self.samples,
        )


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)
    dataset = MNISTPNGDataset(args.data_dir, "train")
    loader = make_loader(
        dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    model = ConvVAE(args.latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    training_config: dict[str, str | int | float | bool] = {
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "latent_dim": args.latent_dim,
        "num_classes": model.num_classes,
        "conditional": True,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "device": args.device,
        "progress_digit": args.progress_digit,
        "progress_snapshots": args.progress_snapshots,
        "training_progress_output": str(args.training_progress_output),
    }
    snapshot_epochs = training_snapshot_epochs(
        args.epochs, args.progress_snapshots
    )
    snapshot_images: list[Tensor] = []
    snapshot_rng = torch.Generator().manual_seed(args.seed)
    snapshot_latent = torch.randn(
        1, model.latent_dim, generator=snapshot_rng
    ).to(device)
    snapshot_label = torch.tensor(
        [args.progress_digit], dtype=torch.long, device=device
    )

    print(f"Training conditional VAE on {device} with {len(dataset)} samples")
    for epoch in range(1, args.epochs + 1):
        model.train()
        metrics = MetricAccumulator()
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch:02d}/{args.epochs:02d}",
            unit="batch",
            dynamic_ncols=True,
            disable=None,
        )
        for images, labels in progress:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(images, labels)
            losses = vae_loss(reconstruction, images, mu, logvar)
            losses.total.backward()
            optimizer.step()
            metrics.update(losses, images.shape[0])
            running_total, running_reconstruction, running_kl = metrics.averages()
            progress.set_postfix(
                loss=f"{running_total:.4f}",
                reconstruction=f"{running_reconstruction:.4f}",
                kl=f"{running_kl:.4f}",
            )

        average_total, average_reconstruction, average_kl = metrics.averages()
        if epoch in snapshot_epochs:
            model.eval()
            with torch.no_grad():
                snapshot_images.append(
                    model.decode(snapshot_latent, snapshot_label).cpu()[0]
                )
        save_checkpoint(args.checkpoint, model, epoch, training_config)
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} - "
            f"loss: {average_total:.4f}, "
            f"reconstruction: {average_reconstruction:.4f}, "
            f"kl: {average_kl:.4f}"
        )

    progress_output = save_training_progress(
        snapshot_images,
        snapshot_epochs,
        model_name="Conditional VAE",
        digit=args.progress_digit,
        output=args.training_progress_output,
    )
    print(f"Checkpoint saved to {args.checkpoint}")
    print(f"Training progress visualization saved to {progress_output}")


def test(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = MNISTPNGDataset(args.data_dir, "test")
    loader = make_loader(
        dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )

    model.eval()
    metrics = MetricAccumulator()
    original_images: list[Tensor] = []
    reconstructed_images: list[Tensor] = []
    collected = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            sampled_reconstruction, mu, logvar = model(images, labels)
            losses = vae_loss(sampled_reconstruction, images, mu, logvar)
            metrics.update(losses, images.shape[0])

            if collected < args.num_images:
                count = min(args.num_images - collected, images.shape[0])
                original_images.append(images[:count].cpu())
                reconstructed_images.append(
                    model.decode(mu[:count], labels[:count]).cpu()
                )
                collected += count

    average_total, average_reconstruction, average_kl = metrics.averages()
    originals = torch.cat(original_images)
    reconstructions = torch.cat(reconstructed_images)
    comparison = torch.cat((originals, reconstructions), dim=0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(comparison, output_path, nrow=originals.shape[0])

    print(f"Testing conditional VAE checkpoint from epoch {checkpoint['epoch']} on {device}")
    print(
        f"loss: {average_total:.4f}, "
        f"reconstruction: {average_reconstruction:.4f}, "
        f"kl: {average_kl:.4f}"
    )
    print(f"Reconstruction comparison saved to {output_path}")


def generate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    # Reset after model construction so the requested seed controls sampling alone.
    set_seed(args.seed)
    model.eval()
    with torch.no_grad():
        latent = torch.randn(args.num_samples, model.latent_dim, device=device)
        labels = torch.full(
            (args.num_samples,), args.digit, dtype=torch.long, device=device
        )
        generated = model.decode(latent, labels).cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = math.ceil(math.sqrt(args.num_samples))
    save_image(generated, output_path, nrow=columns)
    print(
        f"Generated {args.num_samples} images of digit {args.digit} from "
        f"checkpoint epoch {checkpoint['epoch']} on {device}"
    )
    print(f"Generated image grid saved to {output_path}")


class VAEWorkflow:
    """Workflow adapter that exposes the conditional VAE to shared tools."""

    name = "vae"
    description = "conditional convolutional VAE for MNIST"

    def configure_parser(
        self,
        action: Action,
        parser: argparse.ArgumentParser,
        config_path_override: Path | None = None,
    ) -> None:
        defaults = load_config_section(
            config_path_override or DEFAULT_CONFIG,
            action,
            expected_model=self.name,
        )
        data_dir = lambda: config_path(defaults, "data_dir")
        checkpoint = lambda: config_path(defaults, "checkpoint")
        runtime = {
            "seed": config_value(defaults, "seed", nonnegative_int),
            "device": config_device(defaults),
            "num_workers": config_value(
                defaults, "num_workers", nonnegative_int
            )
            if action != "generate"
            else 0,
        }
        if action == "train":
            parser.add_argument("--data-dir", type=Path, default=data_dir())
            parser.add_argument(
                "--checkpoint", type=Path, default=checkpoint()
            )
            parser.add_argument(
                "--epochs",
                type=positive_int,
                default=config_value(defaults, "epochs", positive_int),
            )
            parser.add_argument(
                "--batch-size",
                type=positive_int,
                default=config_value(defaults, "batch_size", positive_int),
            )
            parser.add_argument(
                "--learning-rate",
                type=positive_float,
                default=config_value(
                    defaults, "learning_rate", positive_float
                ),
            )
            parser.add_argument(
                "--latent-dim",
                type=positive_int,
                default=config_value(defaults, "latent_dim", positive_int),
            )
            parser.add_argument(
                "--progress-digit",
                type=digit,
                default=config_value(defaults, "progress_digit", digit),
                help="fixed digit shown in the training progress figure",
            )
            parser.add_argument(
                "--progress-snapshots",
                type=positive_int,
                default=config_value(
                    defaults, "progress_snapshots", positive_int
                ),
                help="maximum number of epochs shown in the progress figure",
            )
            parser.add_argument(
                "--training-progress-output",
                type=Path,
                default=config_path(defaults, "training_progress_output"),
            )
            add_runtime_arguments(parser, include_workers=True, **runtime)
            return

        if action == "test":
            parser.add_argument("--data-dir", type=Path, default=data_dir())
            parser.add_argument(
                "--checkpoint", type=Path, default=checkpoint()
            )
            parser.add_argument(
                "--output",
                type=Path,
                default=config_path(defaults, "output"),
            )
            parser.add_argument(
                "--num-images",
                type=positive_int,
                default=config_value(defaults, "num_images", positive_int),
            )
            parser.add_argument(
                "--batch-size",
                type=positive_int,
                default=config_value(defaults, "batch_size", positive_int),
            )
            add_runtime_arguments(parser, include_workers=True, **runtime)
            return

        if action == "generate":
            parser.add_argument(
                "--checkpoint", type=Path, default=checkpoint()
            )
            parser.add_argument(
                "--digit",
                type=digit,
                required=True,
                help="MNIST digit to generate (0-9)",
            )
            parser.add_argument(
                "--num-samples",
                type=positive_int,
                default=config_value(defaults, "num_samples", positive_int),
            )
            parser.add_argument(
                "--output",
                type=Path,
                default=config_path(defaults, "output"),
            )
            add_runtime_arguments(parser, include_workers=False, **runtime)
            return

        raise ValueError(f"unsupported VAE action: {action}")

    def run(self, action: Action, args: argparse.Namespace) -> None:
        handlers = {"train": train, "test": test, "generate": generate}
        handlers[action](args)


workflow = VAEWorkflow()
