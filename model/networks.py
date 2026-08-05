import torch
import torch.nn as nn


def _mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, output_dim)]
    return nn.Sequential(*layers)


class LatentAdapter(nn.Module):
    """G_theta: z_beta = z0 + alpha * MLP_theta([beta, z0])

    Bottleneck MLP over [beta, z0] (default hidden dims 256->512->512->256),
    output is a same-size delta added to z0 and scaled by alpha -- keeps
    z_beta close to z0 by construction, matching the model.md spec.
    """

    def __init__(self, beta_dim, z_dim, hidden_dims=(256, 512, 512, 256),
                 alpha=0.1, alpha_learnable=False):
        super().__init__()
        self.z_dim = z_dim
        self.mlp = _mlp(beta_dim + z_dim, list(hidden_dims), z_dim)
        if alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, beta: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        # beta: (B, beta_dim), z0: (B, z_dim)
        delta = self.mlp(torch.cat([beta, z0], dim=-1))
        return z0 + self.alpha * delta


class ActionHead(nn.Module):
    """Residual correction on top of the frozen actor's raw action mean,
    conditioned on beta -- accounts for the target body's different
    actuator/limb response even though the action space size is unchanged
    (robot_<label>.xml keeps the same actuator names/gear ratios)."""

    def __init__(self, action_dim, beta_dim, hidden_dims=(128, 128)):
        super().__init__()
        self.mlp = _mlp(action_dim + beta_dim, list(hidden_dims), action_dim)

    def forward(self, raw_action: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        delta = self.mlp(torch.cat([raw_action, beta], dim=-1))
        return raw_action + delta


class ActionResidual(nn.Module):
    """Residual correction on top of the frozen actor's raw action mean, with
    NO beta conditioning -- for the single-body, no-adapter, kinematics-only
    exploration experiment (model/train_explore.py) where z is fed to the
    frozen actor unmodified and there is no morphology descriptor to condition
    on."""

    def __init__(self, action_dim, hidden_dims=(128, 128)):
        super().__init__()
        self.mlp = _mlp(action_dim, list(hidden_dims), action_dim)

    def forward(self, raw_action: torch.Tensor) -> torch.Tensor:
        return raw_action + self.mlp(raw_action)
