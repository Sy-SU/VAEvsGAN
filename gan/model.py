"""Label-conditional DCGAN models for 28x28 grayscale images."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


class ConditionalGenerator(nn.Module):
    """Generate MNIST images from noise and integer class labels."""

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        label_embedding_dim: int = 50,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_embedding_dim = label_embedding_dim
        self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)
        self.project = nn.Linear(latent_dim + label_embedding_dim, 128 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.apply(_initialize_weights)

    def forward(self, latent: Tensor, labels: Tensor) -> Tensor:
        labels = labels.to(device=latent.device, dtype=torch.long)
        conditioned = torch.cat((latent, self.label_embedding(labels)), dim=1)
        features = self.project(conditioned).view(-1, 128, 7, 7)
        return self.decoder(features)


class ConditionalDiscriminator(nn.Module):
    """Score MNIST images while conditioning on integer class labels."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, 28 * 28)
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(128 * 7 * 7, 1)
        self.apply(_initialize_weights)

    def forward(self, images: Tensor, labels: Tensor) -> Tensor:
        labels = labels.to(device=images.device, dtype=torch.long)
        condition = self.label_embedding(labels).view(-1, 1, 28, 28)
        conditioned_images = torch.cat((images, condition), dim=1)
        features = self.features(conditioned_images)
        return self.classifier(torch.flatten(features, start_dim=1)).view(-1)
