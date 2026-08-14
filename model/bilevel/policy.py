"""Lower-level policy phi = LatentAdapter + ActionHead + RootWrenchHead,
wrapped around the frozen Meta Motivo FB-CPR actor.

Action space is 75-dim: 69 joint ctrl + a 6-DoF root wrench. Everything is
sampled from ONE diagonal Gaussian in raw (pre-squash) space, and PPO's
log-probs use the raw value -- so it stays a plain Normal with no tanh-Jacobian
correction. The squashing happens on the way into the simulator:

    ctrl   = clip(a[:69], -1, 1)          (MuJoCo clamps anyway, ctrllimited=true)
    force  = f_max * tanh(a[69:72])       in the ROOT frame
    torque = m_max * tanh(a[72:75])

Gradient into z_beta
--------------------
model.act()/model.actor() are @torch.no_grad()-wrapped (metamotivo/fb/model.py:110),
so we call the private model._normalize / model._actor instead. The frozen
actor's *weights* stay frozen (FBModel.__init__ already does requires_grad_(False));
its *activations* are differentiable, so gradients flow into z_beta. This is
the same trick as model/train.py:141-143 -- see that docstring for the full
argument.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from model.networks import ActionHead, LatentAdapter, RootWrenchHead, ValueNet

# Root features fed to the wrench head, all expressed in the root's own frame so
# the head is rotation-equivariant. Built by rollout.py from the qpos/qvel the
# sim workers ship back.
#   [0] root world z
#   [1] up_z          = 2*(qy*qz + qw*qx)   (see quat.py:quat_up_z -- NOT the textbook form)
#   [2:5] linear velocity in root frame
#   [5:8] angular velocity in root frame (qvel[3:6] is already body-local)
ROOT_FEAT_DIM = 8

CTRL_DIM = 69
WRENCH_DIM = 6
ACTION_DIM = CTRL_DIM + WRENCH_DIM  # 75


class LowerPolicy(nn.Module):
    def __init__(self, cfg, frozen_model, beta_dim: int):
        super().__init__()
        self.cfg = cfg
        self.frozen = frozen_model            # NOT registered as a submodule parameter set;
        self.z_dim = frozen_model.cfg.archi.z_dim
        self.obs_dim = frozen_model.cfg.obs_dim

        self.adapter = LatentAdapter(
            beta_dim=beta_dim,
            z_dim=self.z_dim,
            hidden_dims=cfg.adapter_hidden_dims,
            alpha=cfg.adapter_alpha,
            alpha_learnable=cfg.adapter_alpha_learnable,
            project=cfg.project_z,
        )
        self.action_head = ActionHead(
            action_dim=CTRL_DIM, beta_dim=beta_dim, hidden_dims=cfg.action_head_hidden_dims
        )
        self.wrench_head = RootWrenchHead(
            root_feat_dim=ROOT_FEAT_DIM, beta_dim=beta_dim, action_dim=CTRL_DIM,
            hidden_dims=cfg.wrench_head_hidden_dims,
        )
        self.register_buffer(
            "log_std",
            torch.log(torch.tensor(
                [cfg.exploration_std] * CTRL_DIM + [cfg.wrench_std] * WRENCH_DIM
            )),
        )
        self._std_cache = None

    # `frozen` must not appear in .parameters() -- it is 12x2048 of frozen actor
    # and would otherwise be handed to the optimizer and to clip_grad_norm_.
    def __setattr__(self, name, value):
        if name == "frozen":
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def trainable_parameters(self):
        return list(self.adapter.parameters()) + list(self.action_head.parameters()) + \
               list(self.wrench_head.parameters())

    # ------------------------------------------------------------------ forward

    def latent(self, beta: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        """z_beta = G(beta, z0), projected back onto the FB sphere if configured."""
        return self.adapter(beta, z0)

    def act_mean(
        self, obs: torch.Tensor, beta: torch.Tensor, z_beta: torch.Tensor,
        root_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (mean (B, 75), raw_prior (B, 69)).

        `raw_prior` is the frozen actor's own action mean before the residual
        head touches it. It is returned because the regularization reward
        penalizes ||a_ctrl - raw_prior|| (e_res), which is what keeps the head
        an actual residual instead of a replacement policy.
        """
        obs_norm = self.frozen._normalize(obs)
        raw = self.frozen._actor(obs_norm, z_beta, self.frozen.cfg.actor_std).mean  # (B, 69)
        mu_ctrl = self.action_head(raw, beta)
        mu_wr = self.wrench_head(root_feats, beta, raw)
        return torch.cat([mu_ctrl, mu_wr], dim=-1), raw

    def dist(self, mean: torch.Tensor) -> Normal:
        # validate_args=False is not cosmetic: torch.distributions runs its
        # `constraints` checks on every construction, and for this diagonal
        # Normal over (256, 75) that measured 17.8 ms per control step -- more
        # than the frozen 12x2048 actor forward it follows (6.2 ms), and 426 ms
        # of every rollout. The std is also cached rather than exp()'d each call.
        return Normal(mean, self.std.expand_as(mean), validate_args=False)

    @property
    def std(self) -> torch.Tensor:
        if getattr(self, "_std_cache", None) is None or self._std_cache.device != self.log_std.device:
            self._std_cache = self.log_std.exp()
        return self._std_cache

    def forward(self, obs, beta, z0, root_feats, z_beta=None):
        z_beta = self.latent(beta, z0) if z_beta is None else z_beta
        mean, raw = self.act_mean(obs, beta, z_beta, root_feats)
        return mean, raw, z_beta

    # ------------------------------------------------------------------ squashing

    @staticmethod
    def split_action(a: torch.Tensor, f_max: torch.Tensor, m_max: torch.Tensor):
        """Raw Gaussian sample -> what actually goes into MuJoCo.

        f_max / m_max are (B,) and carry the per-body magnitude AND the global
        anneal factor, so passing zeros disables the wrench entirely without
        changing the action distribution (log-probs stay well-defined).
        """
        ctrl = a[..., :CTRL_DIM].clamp(-1.0, 1.0)
        force = f_max.unsqueeze(-1) * torch.tanh(a[..., CTRL_DIM:CTRL_DIM + 3])
        torque = m_max.unsqueeze(-1) * torch.tanh(a[..., CTRL_DIM + 3:])
        return ctrl, force, torque


def load_frozen_model(cfg):
    """Frozen FB-CPR policy. eval() + requires_grad_(False) is already done in
    FBModel.__init__ (metamotivo/fb/model.py:80), repeated here as an assertion
    of intent rather than a fix."""
    from metamotivo.fb_cpr.huggingface import FBcprModel

    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(cfg.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_value_net(cfg, frozen_model, beta_dim: int) -> ValueNet:
    return ValueNet(
        obs_dim=frozen_model.cfg.obs_dim,
        z_dim=frozen_model.cfg.archi.z_dim,
        beta_dim=beta_dim,
        hidden_dims=cfg.value_hidden_dims,
    )
