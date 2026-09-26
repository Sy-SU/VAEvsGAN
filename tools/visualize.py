"""Create comparison visualizations for the conditional VAE and DCGAN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gan.checkpoint import load_checkpoint as load_gan_checkpoint
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
    positive_int,
    resolve_device,
    set_seed,
)
from generative.visualization import (
    save_latent_distribution,
    save_sample_comparison,
)
from vae.checkpoint import load_checkpoint as load_vae_checkpoint


DEFAULT_CONFIG = CONFIG_DIR / "visualization.toml"


def _latent_samples(count: int, dimensions: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, dimensions, generator=generator)


def compare_samples(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)
    vae, _ = load_vae_checkpoint(args.vae_checkpoint, device)
    gan, _, _ = load_gan_checkpoint(args.gan_checkpoint, device)
    if vae.latent_dim == gan.latent_dim:
        common_latent = _latent_samples(
            args.samples_per_class, vae.latent_dim, args.seed
        )
        vae_latent = common_latent.to(device)
        gan_latent = common_latent.to(device)
    else:
        vae_latent = _latent_samples(
            args.samples_per_class, vae.latent_dim, args.seed
        ).to(device)
        gan_latent = _latent_samples(
            args.samples_per_class, gan.latent_dim, args.seed
        ).to(device)
        print(
            "Warning: checkpoint latent dimensions differ; the same seed is used "
            "but exact latent vectors cannot be shared.",
            file=sys.stderr,
        )

    digits = range(10) if args.digit is None else (args.digit,)
    rows: list[tuple[str, torch.Tensor]] = []
    vae.eval()
    gan.eval()
    with torch.no_grad():
        for current_digit in digits:
            labels = torch.full(
                (args.samples_per_class,),
                current_digit,
                dtype=torch.long,
                device=device,
            )
            vae_images = vae.decode(vae_latent, labels).cpu()
            gan_images = gan(gan_latent, labels).add(1.0).div(2.0).clamp(0.0, 1.0)
            rows.append((f"VAE · {current_digit}", vae_images))
            rows.append((f"GAN · {current_digit}", gan_images.cpu()))

    output_path = save_sample_comparison(
        rows,
        output=args.output,
        title=(
            "Same-condition random generation · shared seed"
            + (
                ""
                if vae.latent_dim != gan.latent_dim
                else " and shared latent vectors"
            )
        ),
    )
    print(f"Sample comparison saved to {output_path}")


def _collect_vae_latents(
    args: argparse.Namespace, device: torch.device, vae
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = MNISTPNGDataset(args.data_dir, "test")
    loader = make_loader(
        dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )
    per_digit: list[list[torch.Tensor]] = [[] for _ in range(10)]
    counts = [0] * 10
    vae.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            means, _ = vae.encode(images, labels)
            for current_digit in range(10):
                needed = args.samples_per_class - counts[current_digit]
                if needed <= 0:
                    continue
                indices = torch.nonzero(
                    labels == current_digit, as_tuple=False
                ).flatten()[:needed]
                if indices.numel() > 0:
                    selected = means.index_select(0, indices).cpu()
                    per_digit[current_digit].append(selected)
                    counts[current_digit] += selected.shape[0]
            if all(count >= args.samples_per_class for count in counts):
                break

    latent_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    for current_digit, parts in enumerate(per_digit):
        if not parts:
            continue
        latent = torch.cat(parts, dim=0)
        latent_parts.append(latent)
        label_parts.append(
            torch.full((latent.shape[0],), current_digit, dtype=torch.long)
        )
    if any(count < args.samples_per_class for count in counts):
        print(
            "Warning: some digits contain fewer test samples than requested; "
            f"collected counts are {counts}.",
            file=sys.stderr,
        )
    return torch.cat(latent_parts), torch.cat(label_parts)


def latent_distribution(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)
    vae, _ = load_vae_checkpoint(args.vae_checkpoint, device)
    gan, _, _ = load_gan_checkpoint(args.gan_checkpoint, device)
    vae_latent, vae_labels = _collect_vae_latents(args, device, vae)
    gan_labels = torch.arange(10).repeat_interleave(args.samples_per_class)
    gan_noise = _latent_samples(
        gan_labels.shape[0], gan.latent_dim, args.seed
    )
    gan_embeddings = gan.label_embedding.weight.detach().cpu()
    output_path = save_latent_distribution(
        vae_latent,
        vae_labels,
        gan_noise,
        gan_labels,
        gan_embeddings,
        output=args.output,
    )
    print(f"Latent-space visualization saved to {output_path}")


def build_parser(config_path_override: Path | None = None) -> argparse.ArgumentParser:
    config_path_value = config_path_override or DEFAULT_CONFIG
    sample_defaults = load_config_section(config_path_value, "samples")
    latent_defaults = load_config_section(config_path_value, "latent")
    parser = argparse.ArgumentParser(
        prog="python tools/visualize.py",
        description="Visualize conditional VAE and DCGAN differences.",
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    samples_parser = subparsers.add_parser(
        "samples", help="compare random samples under the same digit condition"
    )
    samples_parser.add_argument(
        "--config",
        type=Path,
        help="TOML config file; command-line arguments override its values",
    )
    samples_parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=config_path(sample_defaults, "vae_checkpoint"),
    )
    samples_parser.add_argument(
        "--gan-checkpoint",
        type=Path,
        default=config_path(sample_defaults, "gan_checkpoint"),
    )
    samples_parser.add_argument(
        "--digit",
        type=digit,
        help="single digit to compare; omit to compare all digits",
    )
    samples_parser.add_argument(
        "--samples-per-class",
        type=positive_int,
        default=config_value(sample_defaults, "samples_per_class", positive_int),
    )
    samples_parser.add_argument(
        "--output",
        type=Path,
        default=config_path(sample_defaults, "output"),
    )
    add_runtime_arguments(
        samples_parser,
        include_workers=False,
        seed=config_value(sample_defaults, "seed", nonnegative_int),
        device=config_device(sample_defaults),
    )
    samples_parser.set_defaults(handler=compare_samples)

    latent_parser = subparsers.add_parser(
        "latent", help="plot digit-conditioned latent distributions with PCA"
    )
    latent_parser.add_argument(
        "--config",
        type=Path,
        help="TOML config file; command-line arguments override its values",
    )
    latent_parser.add_argument(
        "--data-dir",
        type=Path,
        default=config_path(latent_defaults, "data_dir"),
    )
    latent_parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=config_path(latent_defaults, "vae_checkpoint"),
    )
    latent_parser.add_argument(
        "--gan-checkpoint",
        type=Path,
        default=config_path(latent_defaults, "gan_checkpoint"),
    )
    latent_parser.add_argument(
        "--samples-per-class",
        type=positive_int,
        default=config_value(latent_defaults, "samples_per_class", positive_int),
    )
    latent_parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=config_value(latent_defaults, "batch_size", positive_int),
    )
    latent_parser.add_argument(
        "--output",
        type=Path,
        default=config_path(latent_defaults, "output"),
    )
    add_runtime_arguments(
        latent_parser,
        include_workers=True,
        seed=config_value(latent_defaults, "seed", nonnegative_int),
        device=config_device(latent_defaults),
        num_workers=config_value(
            latent_defaults, "num_workers", nonnegative_int
        ),
    )
    latent_parser.set_defaults(handler=latent_distribution)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument("experiment", nargs="?")
        config_parser.add_argument("--config", type=Path)
        known_args, _ = config_parser.parse_known_args(arguments)
        parser = build_parser(known_args.config)
        args = parser.parse_args(arguments)
        args.handler(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
