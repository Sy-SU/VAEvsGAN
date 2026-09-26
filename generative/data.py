"""Shared datasets and data-loader helpers for generative workflows."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor


class MNISTPNGDataset(Dataset[tuple[Tensor, int]]):
    """Load one MNIST split from ``labels.csv`` and its PNG image directory."""

    def __init__(self, data_dir: str | Path, split: str) -> None:
        self.split_dir = Path(data_dir) / split
        labels_path = self.split_dir / "labels.csv"

        self.samples: list[tuple[Path, int]] = []
        with labels_path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                image_path = self.split_dir / row["image"].strip()
                self.samples.append((image_path, int(row["label"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            grayscale = image.convert("L")
            tensor = pil_to_tensor(grayscale).to(dtype=torch.float32).div_(255.0)
        return tensor, label


def make_loader(
    dataset: Dataset[tuple[Tensor, int]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create a deterministic loader with CUDA pinning when appropriate."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
