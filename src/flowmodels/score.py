import torch
from torch import nn


class ScoreNetwork(nn.Module):
    def __init__(self, dim: int = 2, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def score_matching_loss(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    x = x.requires_grad_(True)

    score = model(x)

    score_norm = 0.5 * score.pow(2).sum(dim=-1)

    divergence = torch.zeros_like(score_norm)

    for i in range(x.shape[-1]):
        grad = torch.autograd.grad(
            score[:, i].sum(),
            x,
            create_graph=True,
        )[0]

        divergence = divergence + grad[:, i]

    return (score_norm + divergence).mean()