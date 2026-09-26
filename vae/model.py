"""Conditional convolutional variational autoencoder and its loss."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VAELoss(NamedTuple):
    """Per-sample mean loss values for a batch."""

    total: Tensor
    reconstruction: Tensor
    kl: Tensor


class ConvVAE(nn.Module):
    """A label-conditional convolutional VAE for 1x28x28 MNIST images."""

    def __init__(self, latent_dim: int = 20, num_classes: int = 10) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        conditioned_features = 64 * 7 * 7 + num_classes
        self.fc_mu = nn.Linear(conditioned_features, latent_dim)
        self.fc_logvar = nn.Linear(conditioned_features, latent_dim)

        self.decoder_input = nn.Linear(
            latent_dim + num_classes, 64 * 7 * 7
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def _condition(self, labels: Tensor, reference: Tensor) -> Tensor:
        labels = labels.to(device=reference.device, dtype=torch.long)
        one_hot = F.one_hot(labels, num_classes=self.num_classes)
        return one_hot.to(dtype=reference.dtype)

    def encode(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Return the mean and log variance of q(z|x,y)."""
        features = self.encoder(images)
        flattened = torch.flatten(features, start_dim=1)
        conditioned = torch.cat((flattened, self._condition(labels, flattened)), dim=1)
        return self.fc_mu(conditioned), self.fc_logvar(conditioned)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Draw a differentiable sample from q(z|x,y)."""
        standard_deviation = torch.exp(0.5 * logvar)
        noise = torch.randn_like(standard_deviation)
        return mu + noise * standard_deviation

    def decode(self, latent: Tensor, labels: Tensor) -> Tensor:
        """Decode latent vectors conditioned on class labels."""
        conditioned = torch.cat((latent, self._condition(labels, latent)), dim=1)
        features = self.decoder_input(conditioned)
        features = features.view(-1, 64, 7, 7)
        return self.decoder(features)

    def forward(
        self, images: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(images, labels)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent, labels), mu, logvar


def vae_loss(
    reconstruction: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
) -> VAELoss:
    """Compute BCE + KL divergence, averaged over samples in the batch."""
    batch_size = target.shape[0]
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction, target, reduction="sum"
    ) / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return VAELoss(
        total=reconstruction_loss + kl_loss,
        reconstruction=reconstruction_loss,
        kl=kl_loss,
    )
