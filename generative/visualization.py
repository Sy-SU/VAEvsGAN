"""Static visualizations shared by the VAE and GAN experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import torch
from torch import Tensor

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


DIGIT_COLORS = plt.get_cmap("tab10")(np.arange(10))


def training_snapshot_epochs(total_epochs: int, count: int = 10) -> tuple[int, ...]:
    """Choose up to ``count`` distinct epochs spanning the full training run."""
    if total_epochs <= count:
        return tuple(range(1, total_epochs + 1))
    selected = {
        round(1 + index * (total_epochs - 1) / (count - 1))
        for index in range(count)
    }
    return tuple(sorted(selected))


def _prepare_output(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _image_array(image: Tensor) -> np.ndarray:
    return image.detach().cpu().squeeze().clamp(0.0, 1.0).numpy()


def save_training_progress(
    images: Sequence[Tensor],
    epochs: Sequence[int],
    *,
    model_name: str,
    digit: int,
    output: str | Path,
) -> Path:
    """Save fixed-latent samples from several epochs as one static figure."""
    columns = min(5, len(images))
    rows = math.ceil(len(images) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.1 * columns, 2.8 * rows + 0.35),
        squeeze=False,
        constrained_layout=True,
    )
    figure.get_layout_engine().set(
        w_pad=0.05,
        h_pad=0.12,
        wspace=0.05,
        hspace=0.18,
    )
    flat_axes = list(axes.flat)
    for axis, image, epoch in zip(flat_axes, images, epochs):
        axis.imshow(_image_array(image), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"Epoch {epoch}", pad=8)
        axis.axis("off")
    for axis in flat_axes[len(images) :]:
        axis.axis("off")

    figure.suptitle(
        f"{model_name} training progress · fixed latent · digit {digit}",
        fontsize=14,
    )
    output_path = _prepare_output(output)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_sample_comparison(
    rows: Sequence[tuple[str, Tensor]],
    *,
    output: str | Path,
    title: str,
) -> Path:
    """Save labeled rows of generated samples with a shared grayscale scale."""
    sample_count = rows[0][1].shape[0]

    figure, axes = plt.subplots(
        len(rows),
        sample_count,
        figsize=(1.3 * sample_count, 1.45 * len(rows)),
        squeeze=False,
    )
    for row_index, (label, images) in enumerate(rows):
        for column_index in range(sample_count):
            axis = axes[row_index, column_index]
            axis.imshow(
                _image_array(images[column_index]),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axis.axis("off")
        axes[row_index, 0].text(
            -0.08,
            0.5,
            label,
            transform=axes[row_index, 0].transAxes,
            rotation=0,
            ha="right",
            va="center",
            fontsize=11,
            clip_on=False,
        )

    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=(0.04, 0.0, 1.0, 0.97))
    output_path = _prepare_output(output)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def pca_2d(features: Tensor) -> Tensor:
    """Project a feature matrix to two dimensions with deterministic PCA."""
    centered = features.detach().cpu().float()
    centered = centered - centered.mean(dim=0, keepdim=True)
    if centered.shape[1] == 1:
        return torch.cat((centered, torch.zeros_like(centered)), dim=1)
    _, _, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    return centered @ right_vectors[:2].T


def _scatter_digits(axis, points: Tensor, labels: Tensor, title: str) -> None:
    points_array = points.detach().cpu().numpy()
    labels_array = labels.detach().cpu().numpy()
    for digit in range(10):
        selected = labels_array == digit
        if selected.any():
            axis.scatter(
                points_array[selected, 0],
                points_array[selected, 1],
                s=8,
                alpha=0.45,
                color=DIGIT_COLORS[digit],
            )
    axis.set_title(title)
    axis.set_xlabel("PC 1")
    axis.set_ylabel("PC 2")
    axis.grid(alpha=0.15)


def save_latent_distribution(
    vae_latent: Tensor,
    vae_labels: Tensor,
    gan_noise: Tensor,
    gan_labels: Tensor,
    gan_label_embeddings: Tensor,
    *,
    output: str | Path,
) -> Path:
    """Save PCA views of VAE posteriors and the two GAN condition inputs."""
    vae_points = pca_2d(vae_latent)
    gan_noise_points = pca_2d(gan_noise)
    embedding_points = pca_2d(gan_label_embeddings)

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    _scatter_digits(
        axes[0], vae_points, vae_labels, "VAE posterior means q(z | x, y)"
    )
    _scatter_digits(
        axes[1], gan_noise_points, gan_labels, "GAN noise z (label-independent)"
    )

    embedding_array = embedding_points.numpy()
    for digit in range(10):
        axes[2].scatter(
            embedding_array[digit, 0],
            embedding_array[digit, 1],
            s=90,
            color=DIGIT_COLORS[digit],
        )
        axes[2].annotate(
            str(digit),
            (embedding_array[digit, 0], embedding_array[digit, 1]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axes[2].set_title("GAN learned label embeddings")
    axes[2].set_xlabel("PC 1")
    axes[2].set_ylabel("PC 2")
    axes[2].grid(alpha=0.15)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=DIGIT_COLORS[digit],
            label=str(digit),
        )
        for digit in range(10)
    ]
    figure.legend(
        handles=legend_handles,
        title="Digit",
        loc="lower center",
        ncol=10,
        bbox_to_anchor=(0.5, 0.07),
    )
    figure.suptitle("Digit-conditioned latent spaces · PCA projection", fontsize=14)
    figure.text(
        0.5,
        0.02,
        "Both models receive y separately, so class overlap is not a failure. "
        "GAN samples z independently of y and injects class information through "
        "the learned embedding.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.18, 1.0, 0.94))
    output_path = _prepare_output(output)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path
