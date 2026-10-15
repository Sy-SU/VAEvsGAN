"""A shared MNIST evaluator and feature-space comparison metrics.

Metrics use the evaluator's unprojected features.  MMD is the unbiased
polynomial-kernel estimate and may legitimately be negative.  Precision and
recall use the union of closed k-nearest-neighbour balls in each sample set.
"""

from __future__ import annotations

from contextlib import contextmanager
import random
from typing import Iterator

import numpy as np
import torch
from torch import Tensor, nn


_BLOCK_SIZE = 512


class MNISTEvaluator(nn.Module):
    """MNIST classifier exposing a 128-dimensional penultimate feature vector."""

    def __init__(self) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, 10)

    def features(self, images: Tensor) -> Tensor:
        """Return features before the final ten-class linear classifier."""
        return self.embedding(self.convolutions(images))

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.features(images))


@contextmanager
def preserve_rng() -> Iterator[None]:
    """Restore process-global Python, NumPy and PyTorch RNGs, even on errors.

    All available CUDA default generators are included.  Independently created
    ``numpy.random.Generator`` and ``torch.Generator`` objects remain owned by
    their callers and are not modified by this context manager.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def extract_features(
    model: MNISTEvaluator,
    images: Tensor,
    device: torch.device | str,
    batch_size: int = 512,
) -> tuple[Tensor, Tensor]:
    """Extract CPU features and predicted labels from N x 1 x 28 x 28 images.

    Inputs must be floating point CPU tensors, already scaled to the evaluator's
    [0, 1] image convention.  The evaluator is moved to ``device``; its previous
    training/evaluation mode is restored after extraction.
    """
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError("images must have shape (N, 1, 28, 28)")
    if images.device.type != "cpu" or not images.is_floating_point():
        raise ValueError("images must be a floating point CPU tensor")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not torch.isfinite(images).all():
        raise ValueError("images must contain only finite values")
    if images.numel() and (images.min() < 0 or images.max() > 1):
        raise ValueError("images must be scaled to [0, 1]")
    if images.shape[0] == 0:
        return torch.empty((0, 128)), torch.empty((0,), dtype=torch.long)

    model.to(device)
    was_training = model.training
    feature_batches: list[Tensor] = []
    prediction_batches: list[Tensor] = []
    model.eval()
    try:
        with torch.inference_mode():
            dtype = next(model.parameters()).dtype
            for batch in images.split(batch_size):
                features = model.features(batch.to(device=device, dtype=dtype))
                predictions = model.classifier(features).argmax(dim=1)
                feature_batches.append(features.cpu())
                prediction_batches.append(predictions.cpu())
    finally:
        model.train(was_training)
    return torch.cat(feature_batches), torch.cat(prediction_batches)


def _feature_matrix(values: Tensor, name: str) -> Tensor:
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty-width two-dimensional tensor")
    values = values.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _kernel_sum(left: Tensor, right: Tensor, *, exclude_diagonal: bool) -> Tensor:
    total = torch.zeros((), dtype=torch.float64)
    dimensions = left.shape[1]
    for left_start in range(0, len(left), _BLOCK_SIZE):
        left_batch = left[left_start : left_start + _BLOCK_SIZE]
        for right_start in range(0, len(right), _BLOCK_SIZE):
            right_batch = right[right_start : right_start + _BLOCK_SIZE]
            kernel = (left_batch @ right_batch.T / dimensions + 1).pow(3)
            if exclude_diagonal and left_start == right_start:
                kernel.diagonal().zero_()
            total += kernel.sum()
    return total


def _squared_distances(left: Tensor, right: Tensor) -> Tensor:
    distances = (
        left.square().sum(dim=1, keepdim=True)
        + right.square().sum(dim=1).unsqueeze(0)
        - 2 * (left @ right.T)
    )
    return distances.clamp_min_(0)


def _manifold_radii_squared(features: Tensor, k: int) -> Tensor:
    radii: list[Tensor] = []
    for start in range(0, len(features), _BLOCK_SIZE):
        query = features[start : start + _BLOCK_SIZE]
        nearest = torch.full((len(query), k), torch.inf, dtype=torch.float64)
        for reference_start in range(0, len(features), _BLOCK_SIZE):
            reference = features[reference_start : reference_start + _BLOCK_SIZE]
            distances = _squared_distances(query, reference)
            if start == reference_start:
                # Exclude the point itself, not other points with equal features.
                distances.diagonal().fill_(torch.inf)
            nearest = torch.cat((nearest, distances), dim=1).topk(
                k, dim=1, largest=False, sorted=False
            ).values
        radii.append(nearest.max(dim=1).values)
    return torch.cat(radii)


def _coverage_fraction(samples: Tensor, centers: Tensor, radii: Tensor) -> float:
    covered_count = 0
    for start in range(0, len(samples), _BLOCK_SIZE):
        query = samples[start : start + _BLOCK_SIZE]
        covered = torch.zeros(len(query), dtype=torch.bool)
        for center_start in range(0, len(centers), _BLOCK_SIZE):
            center_batch = centers[center_start : center_start + _BLOCK_SIZE]
            distances = _squared_distances(query, center_batch)
            covered |= (
                distances <= radii[center_start : center_start + _BLOCK_SIZE].unsqueeze(0)
            ).any(dim=1)
            if covered.all():
                break
        covered_count += int(covered.sum())
    return covered_count / len(samples)


def distribution_metrics(
    real_features: Tensor, fake_features: Tensor, k: int = 3
) -> dict[str, float]:
    """Compute unbiased polynomial MMD squared and manifold precision/recall.

    The kernel is ``((x @ y.T) / feature_dimension + 1) ** 3``.  Each set must
    contain at least ``k + 1`` samples.  Distances and kernels use float64 and
    bounded blocks; no full sample-by-sample distance matrix is retained.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    real = _feature_matrix(real_features, "real_features")
    fake = _feature_matrix(fake_features, "fake_features")
    if real.shape[1] != fake.shape[1]:
        raise ValueError("real and fake features must have the same dimension")
    if min(len(real), len(fake)) < k + 1:
        raise ValueError("each feature set must contain at least k + 1 samples")
    real_count, fake_count = len(real), len(fake)
    mmd2 = (
        _kernel_sum(real, real, exclude_diagonal=True) / (real_count * (real_count - 1))
        + _kernel_sum(fake, fake, exclude_diagonal=True) / (fake_count * (fake_count - 1))
        - 2 * _kernel_sum(real, fake, exclude_diagonal=False) / (real_count * fake_count)
    )
    real_radii = _manifold_radii_squared(real, k)
    fake_radii = _manifold_radii_squared(fake, k)
    return {
        "mmd2": float(mmd2),
        "precision": _coverage_fraction(fake, real, real_radii),
        "recall": _coverage_fraction(real, fake, fake_radii),
    }


def _digit_labels(values: Tensor, count: int, name: str) -> Tensor:
    if values.ndim != 1 or len(values) != count:
        raise ValueError(f"{name} must be a one-dimensional tensor of length {count}")
    values = values.detach().cpu()
    if values.is_complex() or values.dtype == torch.bool:
        raise ValueError(f"{name} must contain integer digit labels")
    if values.is_floating_point() and (
        not torch.isfinite(values).all() or not torch.equal(values, values.round())
    ):
        raise ValueError(f"{name} must contain integer digit labels")
    if ((values < 0) | (values > 9)).any():
        raise ValueError(f"{name} must contain digits from 0 through 9")
    return values.to(dtype=torch.long)


def classwise_metrics(
    real_features: Tensor,
    real_labels: Tensor,
    fake_features: Tensor,
    requested_labels: Tensor,
    predictions: Tensor,
    k: int = 3,
) -> dict[str, object]:
    """Evaluate each requested digit and return their equally weighted means.

    ``confusion`` contains integer counts: rows are requested labels and columns
    are classifier predictions.  Every real and generated class must have at
    least ``k + 1`` examples.  Accuracy measures agreement with the requested
    condition; it is not an independent assessment of visual quality.
    """
    real = _feature_matrix(real_features, "real_features")
    fake = _feature_matrix(fake_features, "fake_features")
    labels = _digit_labels(real_labels, len(real), "real_labels")
    requested = _digit_labels(requested_labels, len(fake), "requested_labels")
    predicted = _digit_labels(predictions, len(fake), "predictions")
    per_class: list[dict[str, int | float]] = []
    for digit in range(10):
        real_mask = labels == digit
        fake_mask = requested == digit
        scores = distribution_metrics(real[real_mask], fake[fake_mask], k=k)
        accuracy = float((predicted[fake_mask] == digit).double().mean())
        per_class.append({"digit": digit, "accuracy": accuracy, **scores})
    names = ("accuracy", "mmd2", "precision", "recall")
    macro = {name: sum(float(row[name]) for row in per_class) / 10 for name in names}
    confusion = torch.bincount(requested * 10 + predicted, minlength=100).reshape(10, 10)
    return {"macro": macro, "per_class": per_class, "confusion": confusion.tolist()}
