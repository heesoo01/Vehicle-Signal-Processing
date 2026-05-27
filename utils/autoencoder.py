"""PyTorch autoencoder model for PSD vectors."""
from __future__ import annotations

import torch
import torch.nn as nn


class PSDAutoencoder(nn.Module):
    """Small fully-connected autoencoder for Welch PSD feature vectors."""

    def __init__(self, input_dim: int = 129, latent_dim: int = 4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
