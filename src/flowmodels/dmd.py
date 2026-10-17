import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, dim: int = 2, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def dmd_loss(
    x_fake: torch.Tensor,
    real_score: torch.Tensor,
    fake_score: torch.Tensor,
) -> torch.Tensor:
    direction = real_score - fake_score

    return -(x_fake * direction.detach()).sum(dim=-1).mean()