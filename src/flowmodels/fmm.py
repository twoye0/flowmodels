import torch
from torch import nn


class FlowMap(nn.Module):
    def __init__(self, dim: int = 2, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        st = torch.cat([s, t], dim=-1)
        inp = torch.cat([x, st], dim=-1)

        return self.net(inp)


def flow_map_loss(
    model: nn.Module,
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    x_t: torch.Tensor,
) -> torch.Tensor:
    pred = model(x_s, s, t)

    return (pred - x_t).pow(2).mean()