import torch
from torch import nn


class ConsistencyFlowMap(nn.Module):
    def __init__(self, dim: int = 2, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def velocity(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        inp = torch.cat([x, s, t], dim=-1)
        return self.net(inp)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        velocity = self.velocity(x, s, t)

        return x + (t - s) * velocity


def consistency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (pred - target.detach()).pow(2).mean()


@torch.no_grad()
def update_teacher(
    student: nn.Module,
    teacher: nn.Module,
    decay: float = 0.99,
) -> None:
    for student_param, teacher_param in zip(
        student.parameters(),
        teacher.parameters(),
    ):
        teacher_param.mul_(decay)
        teacher_param.add_(
            student_param,
            alpha=1 - decay,
        )