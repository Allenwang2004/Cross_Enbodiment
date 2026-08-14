"""AMP: a STATIONARY motion-style reward, to replace the time-indexed one.

STATUS: TRIED AND ABANDONED (2026-08-10). `amp_enabled` is False and no stage
turns it on. The code is kept so the negative result can be re-measured, not
because it is on the critical path. Measured over 2000 stage-1 iterations with
w_amp ramped 0 -> 1 across iterations 200-1000:

    iter  w_amp   long/survive   amp_reward   d_real / d_fake
     200   0.00       0.089           --           --
     300   0.04       0.227         0.257      +0.73 / -0.77
     500   0.31       0.305         0.221      +0.83 / -0.79
     600   0.50       0.070         0.197      +0.84 / -0.80
    1000   1.00       0.066         0.161      +0.89 / -0.81
    1900   1.00       0.086         0.133      +0.87 / -0.87

The discriminator reached the LSGAN targets (+1 real, -1 fake) within ~100
iterations and never let go, so `r_amp = max(0, 1 - 0.25 (D-1)^2)` collapsed to
a near-constant ~0.14 -- a reward with no gradient in it. At w_amp = 1 that
constant occupied the whole 0.65 tracking budget, leaving the policy optimizing
only r_reg + r_surv, and long-horizon survival fell back to its untrained
value. So this run measured "what happens when you remove the time-indexed
reward", NOT "what happens when a stationary one replaces it".

The likely reason D wins so easily is that the "real" samples are the KINEMATIC
retargeted reference: it penetrates the floor, its joint velocities are exact
finite differences, and it has no contact impulses. Those are cues no physically
simulated rollout can ever reproduce, which is the known AMP failure mode when
the demonstrations are not dynamically feasible for the agent. Raising
amp_grad_penalty or lowering amp_lr would only delay saturation.

Two things worth keeping from the attempt: at w_amp ~ 0.3 (iteration 500)
long/survive hit 0.305, the best this project has measured -- though
long_eval_clips = 8 makes single points unreliable, and neighbouring samples
were 0.193 and 0.070. And the analysis below still stands on its own; it is the
IMPLEMENTATION that was not validated.

Why this exists
---------------
`r_track = exp(-k||q_t - q_hat_t||^2)` compares against frame t of a specific
trajectory, so it depends on an external clock. Three consequences, all
measured:

  1. The optimum is a moving point. Fall three frames behind and the reward for
     the pose you ARE holding has already expired; the only way to score is to
     lunge at a target you cannot reach.
  2. That lunge destabilizes, which costs more frames, which is a positive
     feedback loop -- error diverges rather than random-walks.
  3. Worst, the optimal policy under a time-indexed reward is pi*(s, t), but
     this policy is pi(s, z) with z constant over the window and no phase input.
     The same observation demands different actions depending on a lag that is
     not observable, so s -> best action is not even a function. We were asking
     the network to represent something outside its function class.

Training hid all three because RSI rewrites qpos from the reference every 24
steps, which silently re-synchronizes the clock. Measured on the same policy,
clip and body: 22.3/24 steps survived in training, 22/299 in one continuous
rollout.

A discriminator D(s, s') on TRANSITIONS has no t in it. The high-reward set is
the whole reference manifold rather than a point sliding along it, so lagging
costs nothing, the feedback loop is broken, and pi*(s) is a genuine function of
s -- realizable by the architecture we actually have. This is what makes the
mode-B deployment story (give z and beta, no reference) coherent.

What it does NOT fix, stated plainly:
  - physical instability still compounds; only the reward's active push toward
    it is removed;
  - stationary rewards have their own degenerate solution (freeze in one
    high-scoring pose). Scoring TRANSITIONS rather than states is the standard
    defence, but adversarial training can still drop modes;
  - frame-accurate reproduction is given up by construction. For tracking-mode
    numbers you would have to go Meta Motivo's route and recompute z per step
    through the backward map.

Design notes
------------
Features come from `torch_kin`, not from humenv's 358-dim observation. The obs
builder needs `data.sensordata`, which needs an `mj_forward` per reference
frame in the MAIN process -- 256 envs x 25 frames x ~0.1 ms is 0.64 s/iter,
more than the whole simulation budget. torch FK is batched, on GPU, and already
built.

Everything in the feature vector is expressed in the root's heading frame or is
already body-local, so a `child` transition and a `giant` transition are
compared on the same footing. The discriminator is additionally conditioned on
beta, so it can still learn morphology-specific style if that helps it.

LSGAN targets (+1 real, -1 fake) with a gradient penalty on real samples, both
from the AMP paper -- plain BCE-GAN discriminators saturate here and stop
producing a usable reward.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.bilevel.quat import quat_rot, quat_to_yaw, quat_up_z

# End effectors whose local positions carry most of the visible style.
AMP_EE = ["L_Hand", "R_Hand", "L_Toe", "R_Toe", "Head"]


def _heading_inv(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion that undoes the root's yaw. (..., 4) -> (..., 4).

    Yaw-only, not the full orientation: lean and roll ARE style and must stay in
    the features. Only which way the character happens to be facing is removed,
    for the same reason humenv's own observation removes it.
    """
    half = -0.5 * quat_to_yaw(quat)
    out = torch.zeros(*quat.shape[:-1], 4, dtype=quat.dtype, device=quat.device)
    out[..., 0] = torch.cos(half)
    out[..., 3] = torch.sin(half)
    return out


def amp_features(kin, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """A (s, s') transition -> style features. (..., 76) x2 -> (..., D).

    No absolute world position and no time index appear anywhere: that is the
    whole point. Velocities are finite differences between the two control
    steps, which keeps the reference and the robot on identical footing (the
    reference has no qvel of its own).
    """
    hinge0, hinge1 = q0[..., 7:], q1[..., 7:]
    root_q0 = q0[..., 3:7]
    hinv = _heading_inv(root_q0)

    xpos0, xquat0 = kin(q0)
    ee0 = kin.body_pos_of(xpos0, AMP_EE)                     # (..., 5, 3)
    root0 = xpos0[..., kin.free_body, :].unsqueeze(-2)
    # expand, NOT expand_as: the quaternion's last dim is 4 and the points' is
    # 3, so expand_as(ee0) tries to make a (...,1,4) into a (...,5,3). Same trap
    # as torch_kin.min_geom_z -- broadcast the LEADING dims and keep the 4.
    ee_local = quat_rot(hinv.unsqueeze(-2).expand(*ee0.shape[:-1], 4), ee0 - root0)

    d_root = quat_rot(hinv, q1[..., :3] - q0[..., :3])       # local displacement
    d_yaw = quat_to_yaw(q1[..., 3:7]) - quat_to_yaw(root_q0)
    d_yaw = torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw))  # wrap to (-pi, pi]

    return torch.cat([
        hinge0,                                  # 69  pose
        hinge1 - hinge0,                         # 69  joint velocity
        q0[..., 2:3],                            #  1  root height
        quat_up_z(root_q0).unsqueeze(-1),        #  1  lean
        (q1[..., 2:3] - q0[..., 2:3]),           #  1  vertical velocity
        ee_local.flatten(-2, -1),                # 15  end effectors, heading frame
        d_root,                                  #  3  root velocity, heading frame
        d_yaw.unsqueeze(-1),                     #  1  turn rate
    ], dim=-1)


AMP_FEAT_DIM = 69 + 69 + 1 + 1 + 1 + 15 + 3 + 1      # 160


class Discriminator(nn.Module):
    """[features, beta] -> a real-valued score. LSGAN, so no sigmoid."""

    def __init__(self, feat_dim: int, beta_dim: int, hidden=(512, 256)):
        super().__init__()
        dims = [feat_dim + beta_dim, *hidden]
        layers: List[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)
        nn.init.uniform_(self.head.weight, -1.0, 1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(torch.cat([feat, beta], dim=-1))).squeeze(-1)


class AmpTrainer:
    """Owns the discriminator, its optimizer, and the running feature normalizer.

    One iteration does two things:
      `reward()`  scores the policy's own transitions  -> the RL reward channel
      `update()`  trains D to tell reference from policy
    """

    def __init__(self, cfg, ds, device):
        self.cfg = cfg
        self.ds = ds
        self.device = device
        self.d = Discriminator(AMP_FEAT_DIM, ds.beta_dim,
                               tuple(cfg.amp_hidden_dims)).to(device)
        self.opt = torch.optim.Adam(self.d.parameters(), lr=cfg.amp_lr)
        # Running standardization of the features. Without it the 69 joint-angle
        # channels (order 1) and the root displacement channels (order 0.01)
        # arrive at the first layer three decades apart and D simply ignores the
        # small ones -- which are exactly the locomotion style.
        self.register_n = 0
        self.mean = torch.zeros(AMP_FEAT_DIM, device=device)
        self.var = torch.ones(AMP_FEAT_DIM, device=device)

    # ------------------------------------------------------------------ utils
    @torch.no_grad()
    def _update_norm(self, x: torch.Tensor) -> None:
        m = self.cfg.amp_norm_momentum
        bm, bv = x.mean(0), x.var(0, unbiased=False)
        if self.register_n == 0:
            self.mean, self.var = bm, bv.clamp(min=1e-6)
        else:
            self.mean = m * self.mean + (1 - m) * bm
            self.var = (m * self.var + (1 - m) * bv).clamp(min=1e-6)
        self.register_n += 1

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.var.sqrt()

    def _feats(self, kin_by_body, slices, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        """q0/q1: (T, N, 76) -> (T*N, AMP_FEAT_DIM), one kin call per body group."""
        out = torch.empty(*q0.shape[:-1], AMP_FEAT_DIM,
                          dtype=torch.float32, device=self.device)
        for b, sl in slices:
            f = amp_features(kin_by_body[b],
                             q0[:, sl].to(torch.float64),
                             q1[:, sl].to(torch.float64))
            out[:, sl] = f.to(torch.float32)
        return out.reshape(-1, AMP_FEAT_DIM)

    # --------------------------------------------------------------- interface
    @torch.no_grad()
    def reward(self, feat_fake: torch.Tensor, beta_rep: torch.Tensor,
               shape) -> torch.Tensor:
        """AMP style reward in [0, 1]. Paper's least-squares form:

            r = max(0, 1 - 0.25 (D - 1)^2)

        Bounded, and zero once D is confidently "not reference" -- which keeps it
        on the same scale as the r_track it replaces so the 0.65/0.15/0.20 split
        does not have to change.
        """
        s = self.d(self._norm(feat_fake), beta_rep)
        return (1.0 - 0.25 * (s - 1.0) ** 2).clamp(min=0.0).reshape(shape)

    def update(self, feat_real: torch.Tensor, feat_fake: torch.Tensor,
               beta_rep: torch.Tensor) -> Dict[str, float]:
        cfg = self.cfg
        self._update_norm(feat_real)
        real = self._norm(feat_real).requires_grad_(True)
        fake = self._norm(feat_fake)

        stats = {}
        n = real.shape[0]
        idx = torch.randperm(n, device=self.device)
        mb = max(1, n // cfg.amp_minibatches)
        for k in range(cfg.amp_minibatches):
            j = idx[k * mb:(k + 1) * mb]
            if j.numel() == 0:
                continue
            r_in = real[j].detach().requires_grad_(True)
            s_real = self.d(r_in, beta_rep[j])
            s_fake = self.d(fake[j], beta_rep[j])
            loss = 0.5 * ((s_real - 1.0) ** 2).mean() + 0.5 * ((s_fake + 1.0) ** 2).mean()

            # Gradient penalty on REAL samples only (AMP eq. 9). Without it D
            # wins outright within a few hundred iterations, the reward pins at
            # 0, and the policy gets no signal at all.
            g = torch.autograd.grad(s_real.sum(), r_in, create_graph=True)[0]
            gp = (g ** 2).sum(-1).mean()
            loss = loss + 0.5 * cfg.amp_grad_penalty * gp

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            stats = {
                "amp_loss": float(loss.detach()),
                "amp_d_real": float(s_real.detach().mean()),
                "amp_d_fake": float(s_fake.detach().mean()),
                "amp_grad_pen": float(gp.detach()),
            }
        return stats

    def step(self, ep, kin_by_body, slices) -> Dict[str, float]:
        """One iteration: build both feature sets, score, train. -> metrics + reward.

        `ep["qpos_all"]` is (H+1, N, 76) -- the RSI state followed by the state
        after each control step -- so the robot's transitions are the H
        consecutive pairs. The reference's are the matching pairs of
        `ep["ref"]`, which is already (N, H+1, 76).
        """
        qp = ep["qpos_all"]
        H = qp.shape[0] - 1
        fake = self._feats(kin_by_body, slices, qp[:-1], qp[1:])

        ref = ep["ref"].transpose(0, 1).to(torch.float32)      # (H+1, N, 76)
        real = self._feats(kin_by_body, slices, ref[:-1], ref[1:])

        beta_rep = ep["beta"].unsqueeze(0).expand(H, -1, -1).reshape(-1, ep["beta"].shape[-1])

        # Only score steps the window was still alive for; a frozen slot repeats
        # its last qpos, which would teach D that "not moving" is fake.
        valid = ep["valid"].reshape(-1) > 0
        m = self.update(real[valid], fake[valid], beta_rep[valid])
        r = self.reward(fake, beta_rep, (H, qp.shape[1]))
        m["amp_reward"] = float((r * ep["valid"]).sum() / ep["valid"].sum().clamp(min=1))
        return r, m
