"""Conditional DCGAN workflow used by the model-agnostic tools."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor, nn
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
    probability,
    resolve_device,
    set_seed,
)
from generative.visualization import (
    save_training_progress,
    training_snapshot_epochs,
)
from generative.workflow import Action

from .checkpoint import load_checkpoint, save_checkpoint
from .model import ConditionalDiscriminator, ConditionalGenerator


DEFAULT_CONFIG = CONFIG_DIR / "gan.toml"


class MetricAccumulator:
    """Accumulate per-sample GAN metrics into dataset-wide means."""

    def __init__(self) -> None:
        self.discriminator_loss = 0.0
        self.generator_loss = 0.0
        self.real_score = 0.0
        self.fake_score = 0.0
        self.wrong_label_score = 0.0
        self.samples = 0

    def update(
        self,
        discriminator_loss: Tensor,
        generator_loss: Tensor,
        real_logits: Tensor,
        fake_logits: Tensor,
        wrong_label_logits: Tensor,
        batch_size: int,
    ) -> None:
        self.discriminator_loss += discriminator_loss.item() * batch_size
        self.generator_loss += generator_loss.item() * batch_size
        self.real_score += real_logits.sigmoid().mean().item() * batch_size
        self.fake_score += fake_logits.sigmoid().mean().item() * batch_size
        self.wrong_label_score += (
            wrong_label_logits.sigmoid().mean().item() * batch_size
        )
        self.samples += batch_size

    def averages(self) -> tuple[float, float, float, float, float]:
        return (
            self.discriminator_loss / self.samples,
            self.generator_loss / self.samples,
            self.real_score / self.samples,
            self.fake_score / self.samples,
            self.wrong_label_score / self.samples,
        )


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _to_display_range(images: Tensor) -> Tensor:
    return images.add(1.0).div(2.0).clamp(0.0, 1.0)


def _make_mismatched_labels(labels: Tensor, num_classes: int) -> Tensor:
    """Return random class labels guaranteed to differ from each input label."""
    offsets = torch.randint(
        1, num_classes, labels.shape, device=labels.device, dtype=labels.dtype
    )
    return (labels + offsets) % num_classes


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
    generator = ConditionalGenerator(
        latent_dim=args.latent_dim,
        label_embedding_dim=args.label_embedding_dim,
    ).to(device)
    discriminator = ConditionalDiscriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    generator_optimizer = Adam(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    discriminator_optimizer = Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    training_config: dict[str, str | int | float | bool] = {
        "data_dir": str(args.data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "latent_dim": args.latent_dim,
        "num_classes": generator.num_classes,
        "label_embedding_dim": generator.label_embedding_dim,
        "conditional": True,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "conditional_negative_weight": args.conditional_negative_weight,
        "discriminator_batch_norm": False,
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
        1, generator.latent_dim, generator=snapshot_rng
    ).to(device)
    snapshot_label = torch.tensor(
        [args.progress_digit], dtype=torch.long, device=device
    )

    print(f"Training conditional DCGAN on {device} with {len(dataset)} samples")
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
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
            images = images.mul(2.0).sub(1.0)
            labels = labels.to(device, non_blocking=device.type == "cuda")
            batch_size = images.shape[0]
            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)
            wrong_labels = _make_mismatched_labels(
                labels, generator.num_classes
            )
            latent = torch.randn(batch_size, generator.latent_dim, device=device)
            generated = generator(latent, labels)

            discriminator_optimizer.zero_grad(set_to_none=True)
            real_logits = discriminator(images, labels)
            wrong_label_logits = discriminator(images, wrong_labels)
            fake_logits = discriminator(generated.detach(), labels)
            conditional_negative_loss = criterion(
                wrong_label_logits, fake_targets
            )
            discriminator_loss = criterion(real_logits, real_targets) + (
                args.conditional_negative_weight
                * (
                    criterion(fake_logits, fake_targets)
                    + conditional_negative_loss
                )
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()

            _set_requires_grad(discriminator, False)
            generator_optimizer.zero_grad(set_to_none=True)
            generator_logits = discriminator(generated, labels)
            generator_loss = criterion(generator_logits, real_targets)
            generator_loss.backward()
            generator_optimizer.step()
            _set_requires_grad(discriminator, True)

            metrics.update(
                discriminator_loss,
                generator_loss,
                real_logits.detach(),
                fake_logits.detach(),
                wrong_label_logits.detach(),
                batch_size,
            )
            running_d_loss, running_g_loss, _, _, _ = metrics.averages()
            progress.set_postfix(
                discriminator=f"{running_d_loss:.4f}",
                generator=f"{running_g_loss:.4f}",
            )

        d_loss, g_loss, real_score, fake_score, wrong_label_score = (
            metrics.averages()
        )
        if epoch in snapshot_epochs:
            generator.eval()
            with torch.no_grad():
                snapshot_images.append(
                    _to_display_range(
                        generator(snapshot_latent, snapshot_label)
                    ).cpu()[0]
                )
        save_checkpoint(
            args.checkpoint, generator, discriminator, epoch, training_config
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} - "
            f"discriminator_loss: {d_loss:.4f}, "
            f"generator_loss: {g_loss:.4f}, "
            f"real_score: {real_score:.4f}, "
            f"fake_score: {fake_score:.4f}, "
            f"wrong_label_score: {wrong_label_score:.4f}"
        )

    progress_output = save_training_progress(
        snapshot_images,
        snapshot_epochs,
        model_name="Conditional DCGAN",
        digit=args.progress_digit,
        output=args.training_progress_output,
    )
    print(f"Checkpoint saved to {args.checkpoint}")
    print(f"Training progress visualization saved to {progress_output}")


def test(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)
    generator, discriminator, checkpoint = load_checkpoint(args.checkpoint, device)
    set_seed(args.seed)
    dataset = MNISTPNGDataset(args.data_dir, "test")
    loader = make_loader(
        dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    criterion = nn.BCEWithLogitsLoss()
    negative_weight = float(
        checkpoint["training_config"].get("conditional_negative_weight", 0.5)
    )
    generator.eval()
    discriminator.eval()
    metrics = MetricAccumulator()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            images = images.mul(2.0).sub(1.0)
            labels = labels.to(device, non_blocking=device.type == "cuda")
            batch_size = images.shape[0]
            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)
            wrong_labels = _make_mismatched_labels(
                labels, generator.num_classes
            )
            latent = torch.randn(batch_size, generator.latent_dim, device=device)
            generated = generator(latent, labels)
            real_logits = discriminator(images, labels)
            wrong_label_logits = discriminator(images, wrong_labels)
            fake_logits = discriminator(generated, labels)
            conditional_negative_loss = criterion(
                wrong_label_logits, fake_targets
            )
            discriminator_loss = criterion(real_logits, real_targets) + (
                negative_weight
                * (
                    criterion(fake_logits, fake_targets)
                    + conditional_negative_loss
                )
            )
            generator_loss = criterion(fake_logits, real_targets)
            metrics.update(
                discriminator_loss,
                generator_loss,
                real_logits,
                fake_logits,
                wrong_label_logits,
                batch_size,
            )

        set_seed(args.seed)
        grid_labels = torch.arange(generator.num_classes, device=device)
        grid_labels = grid_labels.repeat_interleave(args.samples_per_class)
        grid_latent = torch.randn(
            grid_labels.shape[0], generator.latent_dim, device=device
        )
        grid = _to_display_range(generator(grid_latent, grid_labels)).cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path, nrow=args.samples_per_class)
    d_loss, g_loss, real_score, fake_score, wrong_label_score = metrics.averages()
    print(
        f"Testing conditional DCGAN checkpoint from epoch "
        f"{checkpoint['epoch']} on {device}"
    )
    print(
        f"discriminator_loss: {d_loss:.4f}, "
        f"generator_loss: {g_loss:.4f}, "
        f"real_score: {real_score:.4f}, "
        f"fake_score: {fake_score:.4f}, "
        f"wrong_label_score: {wrong_label_score:.4f}"
    )
    print(f"Balanced digit grid saved to {output_path}")


def generate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)
    generator, _, checkpoint = load_checkpoint(args.checkpoint, device)
    set_seed(args.seed)
    generator.eval()
    with torch.no_grad():
        latent = torch.randn(args.num_samples, generator.latent_dim, device=device)
        labels = torch.full(
            (args.num_samples,), args.digit, dtype=torch.long, device=device
        )
        generated = _to_display_range(generator(latent, labels)).cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = math.ceil(math.sqrt(args.num_samples))
    save_image(generated, output_path, nrow=columns)
    print(
        f"Generated {args.num_samples} images of digit {args.digit} from "
        f"checkpoint epoch {checkpoint['epoch']} on {device}"
    )
    print(f"Generated image grid saved to {output_path}")


class GANWorkflow:
    """Workflow adapter that exposes the conditional DCGAN to shared tools."""

    name = "gan"
    description = "conditional DCGAN for MNIST"

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
            parser.add_argument(
                "--data-dir",
                type=Path,
                default=config_path(defaults, "data_dir"),
            )
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
                "--label-embedding-dim",
                type=positive_int,
                default=config_value(
                    defaults, "label_embedding_dim", positive_int
                ),
            )
            parser.add_argument(
                "--adam-beta1",
                type=probability,
                default=config_value(defaults, "adam_beta1", probability),
            )
            parser.add_argument(
                "--adam-beta2",
                type=probability,
                default=config_value(defaults, "adam_beta2", probability),
            )
            parser.add_argument(
                "--conditional-negative-weight",
                type=positive_float,
                default=config_value(
                    defaults, "conditional_negative_weight", positive_float
                ),
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
            parser.add_argument(
                "--data-dir",
                type=Path,
                default=config_path(defaults, "data_dir"),
            )
            parser.add_argument(
                "--checkpoint", type=Path, default=checkpoint()
            )
            parser.add_argument(
                "--output",
                type=Path,
                default=config_path(defaults, "output"),
            )
            parser.add_argument(
                "--samples-per-class",
                type=positive_int,
                default=config_value(
                    defaults, "samples_per_class", positive_int
                ),
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

        raise ValueError(f"unsupported GAN action: {action}")

    def run(self, action: Action, args: argparse.Namespace) -> None:
        handlers = {"train": train, "test": test, "generate": generate}
        handlers[action](args)


workflow = GANWorkflow()
