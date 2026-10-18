import torch
import torch.nn as nn

from torch.func import jvp


# ============================================================
# 1. Naive flow-map baseline
# ============================================================

class FlowMap(nn.Module):
    """
    Simple flow-map baseline.

    Convention:
        t = 0 : noise
        t = 1 : data

    The model directly returns the mapped position:
        x_s = x_t + (s - t) * displacement
    """

    def __init__(self, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t, t, s):
        inp = torch.cat([x_t, t, s], dim=-1)

        displacement = self.net(inp)

        return x_t + (s - t) * displacement


def sample_path(batch_size, device="cpu"):
    """
    Original toy convention:

        t = 0 : noise
        t = 1 : data
    """

    # Noise
    x0 = torch.randn(batch_size, 2, device=device)

    # Data: shifted Gaussian
    x1 = torch.randn(batch_size, 2, device=device)
    x1[:, 0] += 4.0

    # Random time
    t = torch.rand(batch_size, 1, device=device)

    # Linear probability path
    x_t = (1.0 - t) * x0 + t * x1

    # Conditional velocity
    v_t = x1 - x0

    return x0, x1, t, x_t, v_t


def cfm_loss(model, batch_size=256, device="cpu"):
    """
    CFM loss for the naive FlowMap baseline.
    """

    _, _, t, x_t, v_t = sample_path(
        batch_size,
        device,
    )

    eps = 1e-3
    s = t + eps

    x_s = model(x_t, t, s)

    # Convert a small flow-map displacement into velocity
    v_pred = (x_s - x_t) / eps

    return ((v_pred - v_t) ** 2).mean()


def self_distillation_loss(
    model,
    batch_size=256,
    device="cpu",
):
    """
    Naive two-step self-distillation baseline.

    Direct:
        t -> s

    Composed:
        t -> m -> s
    """

    _, _, t, x_t, _ = sample_path(
        batch_size,
        device,
    )

    # Choose a later time
    delta = torch.rand_like(t) * (1.0 - t)
    s = t + delta

    # Direct prediction used as a stop-gradient teacher
    with torch.no_grad():
        x_s_teacher = model(x_t, t, s)

    # Midpoint
    m = 0.5 * (t + s)

    # t -> m
    x_m = model(x_t, t, m)

    # m -> s
    x_s_student = model(x_m, m, s)

    return ((x_s_student - x_s_teacher) ** 2).mean()


def naive_isd_loss(
    model,
    batch_size=256,
    lambda_sd=1.0,
    device="cpu",
):
    """
    Naive baseline:

        CFM + two-step self-distillation

    This is kept only as a comparison baseline.
    """

    loss_cfm = cfm_loss(
        model,
        batch_size=batch_size,
        device=device,
    )

    loss_sd = self_distillation_loss(
        model,
        batch_size=batch_size,
        device=device,
    )

    loss = loss_cfm + lambda_sd * loss_sd

    return loss, loss_cfm, loss_sd


# ============================================================
# 2. iSD flow-map representation
# ============================================================

class ISDFlowMap(nn.Module):
    """
    Flow-map representation used by the iSD toy model.

    Convention:
        t = 0 : data
        t = 1 : noise

    F_theta(x_t; t, s) represents the flow-map vector.

    On the diagonal:
        F_theta(x_t; t, t)

    acts as the velocity prediction.
    """

    def __init__(self, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t, t, s):
        inp = torch.cat([x_t, t, s], dim=-1)

        return self.net(inp)


# ============================================================
# 3. iSD probability path
# ============================================================

def sample_isd_path(batch_size, device="cpu"):
    """
    iSD convention:

        t = 0 : data
        t = 1 : noise

    Generation therefore runs backward:

        noise (t=1) -> data (t=0)
    """

    # Data distribution: shifted Gaussian
    x_data = torch.randn(batch_size, 2, device=device)
    x_data[:, 0] += 4.0

    # Noise distribution: standard Gaussian
    x_noise = torch.randn(batch_size, 2, device=device)

    # Random time
    t = torch.rand(batch_size, 1, device=device)

    # Linear interpolation
    x_t = (1.0 - t) * x_data + t * x_noise

    # Conditional velocity:
    #
    #     dx_t / dt = x_noise - x_data
    #
    v_t = x_noise - x_data

    return x_data, x_noise, t, x_t, v_t


# ============================================================
# 4. CFM grounding
# ============================================================

def isd_cfm_loss(
    model,
    batch_size=256,
    device="cpu",
):
    """
    Ground the diagonal flow-map prediction with CFM.

        F_theta(x_t; t, t) ~= v_t
    """

    _, _, t, x_t, v_t = sample_isd_path(
        batch_size,
        device,
    )

    # Diagonal flow-map prediction acts as velocity
    v_pred = model(x_t, t, t)

    return ((v_pred - v_t) ** 2).mean()


# ============================================================
# 5. Total derivative via JVP
# ============================================================

def flow_map_jvp(model, x_t, t, s):
    """
    Compute the total derivative

        d/dt F_theta(x_t; t, s)

    along the learned trajectory.

    The trajectory direction is

        dx_t / dt = v_theta(x_t, t)

    where

        v_theta(x_t, t)
            = F_theta(x_t; t, t).
    """

    # Learned velocity at the diagonal
    with torch.no_grad():
        v_t = model(x_t, t, t)

    def f(x, time):
        return model(x, time, s)

    # Direction:
    #
    #   x changes along v_t
    #   t changes at unit speed
    #
    _, dF_dt = jvp(
        f,
        (x_t, t),
        (
            v_t,
            torch.ones_like(t),
        ),
    )

    return dF_dt


# ============================================================
# 6. SD-R target
# ============================================================

def sd_r_target(model, x_t, t, s):
    """
    Construct the stop-gradient SD-R target.

    We use:

        0 <= s <= t

    and

        target
            = v_theta
              - (t - s) * dF_theta/dt
    """

    # Current learned velocity
    with torch.no_grad():
        v_t = model(x_t, t, t)

    # Total derivative along the trajectory
    dF_dt = flow_map_jvp(
        model,
        x_t,
        t,
        s,
    )

    # Stop gradient through the complete regression target
    target = (
        v_t
        - (t - s) * dF_dt
    ).detach()

    return target


# ============================================================
# 7. SD-R loss
# ============================================================

def sd_r_loss(
    model,
    batch_size=256,
    device="cpu",
):
    """
    Self-distillation regression loss.
    """

    _, _, t, x_t, _ = sample_isd_path(
        batch_size,
        device,
    )

    # Random earlier target time:
    #
    #     0 <= s <= t
    #
    s = torch.rand_like(t) * t

    # Flow-map vector prediction
    pred = model(x_t, t, s)

    # Stop-gradient SD-R target
    target = sd_r_target(
        model,
        x_t,
        t,
        s,
    )

    return ((pred - target) ** 2).mean()


# ============================================================
# 8. Final iSD objective
# ============================================================

def isd_loss(
    model,
    batch_size=256,
    lambda_sd=1.0,
    device="cpu",
):
    """
    Final toy iSD objective:

        L = L_CFM + lambda_sd * L_SD-R
    """

    loss_cfm = isd_cfm_loss(
        model,
        batch_size=batch_size,
        device=device,
    )

    loss_sd = sd_r_loss(
        model,
        batch_size=batch_size,
        device=device,
    )

    loss = loss_cfm + lambda_sd * loss_sd

    return loss, loss_cfm, loss_sd