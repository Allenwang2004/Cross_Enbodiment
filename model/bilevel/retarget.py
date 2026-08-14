"""Upper level: runtime, parameterized, differentiable retargeting.

Replaces scripts/qpos_retarget.py's offline pipeline. That script had exactly
one free knob -- a single scalar root scale (`retarget_qpos`, :91, whose entire
body is `out[:, 0:3] *= scale`) -- plus a non-differentiable per-frame ground
lift (`ground_correct_qpos`, :127). Here the correction is a 36-dim vector
emitted by a small net conditioned ONLY on the morphology descriptor beta, and
the ground lift becomes a learned global `dz_root` driven by the differentiable
penetration penalty in semantics.py.

Anti-degeneracy
---------------
The tractable part of the upper hypergradient is a pure "move ref onto tau"
pull (see upper.py). Three things stop that from destroying the motion; this
file implements the first and most important one:

    A HARD tanh BOX on every output.

At the box corners the reference is still within 16% root scale, 20% joint
amplitude and 0.15 rad of the naive retarget. Collapsing 540 clips onto a
standing pose is structurally impossible, not merely penalized -- which matters
because it is the only defence that cannot be broken by mis-setting a weight.
The other two (the fidelity term S, and the saturation diagnostic) live in
semantics.py and upper.py.

p = 0 reproduces scripts/qpos_retarget.py:91 retarget_qpos EXACTLY. The naive
retarget is the architectural origin, not a soft target -- asserted in
tests/test_retarget.py.

Joint groups
------------
The 69 hinges are tied into 14 groups with LEFT AND RIGHT SHARED, so no
chirality can be learned (a left/right-asymmetric global correction is never a
legitimate morphology adaptation -- all 13 bodies are laterally symmetric).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from model.bilevel.quat import exp_map_to_quat, quat_mul, quat_normalize
from model.bilevel.torch_kin import TorchKinematics

# 14 groups, in u-vector order. Derived from the body names: every hinge is
# named "<Body>_{x,y,z}" and every body is either unpaired (Torso, Spine, Chest,
# Neck, Head) or an L_/R_ pair. 9 paired groups x 6 + 5 unpaired x 3 = 69.
JOINT_GROUPS: List[str] = [
    "Hip", "Knee", "Ankle", "Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "Thorax", "Shoulder", "Elbow", "Wrist", "Hand",
]
N_GROUPS = len(JOINT_GROUPS)

# u layout (36 = 3 + 1 + 3 + 14 + 14 + 1)
U_ROOT_SCALE = slice(0, 3)
U_ROOT_DZ = 3
U_ROOT_ROT = slice(4, 7)
U_JOINT_GAIN = slice(7, 7 + N_GROUPS)
U_JOINT_BIAS = slice(7 + N_GROUPS, 7 + 2 * N_GROUPS)
U_LOG_TAU = 7 + 2 * N_GROUPS
U_DIM = 8 + 2 * N_GROUPS  # 36


def joint_group_index(hinge_names: List[str]) -> torch.Tensor:
    """Map each of the 69 hinges to its group id. -> (69,) long."""
    lookup: Dict[str, int] = {g: i for i, g in enumerate(JOINT_GROUPS)}
    out = []
    for name in hinge_names:
        base = name.rsplit("_", 1)[0]          # "L_Shoulder_x" -> "L_Shoulder"
        if base.startswith(("L_", "R_")):      # -> "Shoulder"
            base = base[2:]
        if base not in lookup:
            raise ValueError(f"hinge {name!r} -> unknown group {base!r}; JOINT_GROUPS is stale")
        out.append(lookup[base])
    return torch.as_tensor(out, dtype=torch.long)


class RetargetParams(nn.Module):
    """The hard box: pre-activation u (B, 36) -> named retarget parameters.

    Split out from RetargetNet so that the ES estimator in upper.py can perturb
    `u` directly (36 dims is tractable for ES; the net's ~10k weights are not)
    and still go through exactly the same box.
    """

    def __init__(self, cfg, source_rest_h: float, target_rest_h: float):
        super().__init__()
        self.cfg = cfg
        # This is the naive retarget's scale: the ratio of rest-pose pelvis
        # heights, i.e. what scripts/qpos_retarget.py:174 computes as
        # `target_h / source_h` from the two skeleton.json files. Read from
        # MjModel.qpos0[2] instead -- verified identical, and 11 of 13 bodies
        # have no skeleton.json on disk.
        self.register_buffer(
            "log_scale0", torch.log(torch.tensor(target_rest_h / source_rest_h, dtype=torch.float64))
        )
        # Per-dimension freeze. A zeroed dimension has u = 0 going into tanh, so
        # that parameter takes its p=0 value exactly and receives no gradient
        # -- which is what makes "unfreeze one axis at a time" possible without
        # a second code path.
        mask = torch.ones(U_DIM, dtype=torch.float64)
        if getattr(cfg, "upper_free_dims", None) is not None:
            mask.zero_()
            mask[list(cfg.upper_free_dims)] = 1.0
        self.register_buffer("free_mask", mask)

    def forward(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        c = self.cfg
        t = torch.tanh(u * self.free_mask)
        log_tau = (
            c.box_log_tau * t[..., U_LOG_TAU]
            if c.enable_time_warp
            else torch.zeros_like(t[..., U_LOG_TAU])
        )
        return {
            # root xyz scale; at u=0 this is the scalar h_tgt/h_src on all three
            # axes, exactly reproducing retarget_qpos's `out[:, 0:3] *= scale`.
            "root_scale": torch.exp(self.log_scale0 + c.box_root_scale * t[..., U_ROOT_SCALE]),
            "root_dz": c.box_root_dz * t[..., U_ROOT_DZ],
            "root_rot": c.box_root_rot * t[..., U_ROOT_ROT],
            "joint_gain": 1.0 + c.box_joint_gain * t[..., U_JOINT_GAIN],
            "joint_bias": c.box_joint_bias * t[..., U_JOINT_BIAS],
            "tau": torch.exp(log_tau),
        }

    def saturation(self, u: torch.Tensor, thresh: float = 0.9) -> torch.Tensor:
        """Fraction of FREE components pressed against the box wall.

        proposal.md 3.5: this is THE alarm metric. Above ~0.2 the objective is
        pushing toward collapse and lambda_fid is too small.

        Frozen dimensions are excluded from the denominator -- they sit at 0 by
        construction, so counting them would dilute the alarm exactly when p
        is most restricted (Stage 1 frees 1 of 36, which would cap the reported
        saturation at 0.028 no matter how hard that one dimension saturates).
        """
        free = self.free_mask > 0
        return (torch.tanh(u[..., free]).abs() > thresh).to(u.dtype).mean()


class RetargetNet(nn.Module):
    """beta (B, 8) -> u (B, 36).

    Deliberately tiny and beta-only. No clip identity, no timestep, no state: a
    per-clip or per-frame p would collapse immediately, and 36 global numbers
    shared across 540 clips x 10 bodies is what makes the capacity argument in
    proposal.md 3.1 hold.

    The output layer is zero-initialized so that u == 0 at step 0 -- training
    therefore STARTS at the exact naive retarget, which is what makes the
    `upper_warmup_iters` freeze meaningful and the S/G baselines comparable.
    """

    def __init__(self, beta_dim: int, hidden_dims=(64, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = beta_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        head = nn.Linear(prev, U_DIM)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.mlp = nn.Sequential(*layers)

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        return self.mlp(beta)

    @torch.no_grad()
    def warm_start(self, dim: int, target: float, half_width: float) -> float:
        """Set one output dimension's bias so u[dim] starts at a chosen value.

        Used for dz_root. Removing scripts/qpos_retarget.py:127
        ground_correct_qpos's per-frame lift left the p=0 reference sitting
        ~12-16 mm inside the floor on 88% of frames, and learning that offset
        from scratch is absurd: it is analytically computable (it is just the
        reference's own median ground penetration), while lr_upper=1e-5 -- sized
        for the delicate anti-degeneracy work of Stage 2/3 -- would need ~16,000
        upper steps to climb out. Measured: 0.00005 m of the required 0.014 m
        after 50 iterations.

        So the offset is computed and installed, and gradient descent only
        refines it. This does not weaken the p=0 anchor: that property is
        about apply_retarget's algebra (tests/test_retarget.py passes u=0
        explicitly), not about where training happens to initialize.

        `target` is in the parameter's own units (metres for dz_root),
        `half_width` the hard box half-width. Returns the pre-activation value.
        """
        frac = max(-0.999, min(0.999, target / half_width))
        u0 = float(np.arctanh(frac))
        self.mlp[-1].bias[dim] = u0
        return u0


def _resample_time(src: torch.Tensor, tau: torch.Tensor, n_out: int) -> torch.Tensor:
    """Differentiable linear resample along time. src: (B, T, D), tau: (B,).

    Output frame k is sampled at source time k * tau. tau == 1 hits integer
    indices exactly, so the p=0 path is bit-identical to plain slicing (the
    lerp weight is exactly 0). Only used when cfg.enable_time_warp is on;
    callers must supply enough source margin (see WindowDataset.src_window_len).
    """
    B, T, D = src.shape
    k = torch.arange(n_out, device=src.device, dtype=src.dtype)
    pos = k.unsqueeze(0) * tau.unsqueeze(1)                     # (B, n_out)
    pos = pos.clamp(max=float(T - 1) - 1e-9)
    i0 = pos.floor().long()
    i1 = (i0 + 1).clamp(max=T - 1)
    w = (pos - i0.to(src.dtype)).unsqueeze(-1)                  # (B, n_out, 1)
    g0 = torch.gather(src, 1, i0.unsqueeze(-1).expand(B, n_out, D))
    g1 = torch.gather(src, 1, i1.unsqueeze(-1).expand(B, n_out, D))
    return g0 * (1 - w) + g1 * w


def apply_retarget(
    src_qpos: torch.Tensor,
    params: Dict[str, torch.Tensor],
    kin: TorchKinematics,
    group_idx: torch.Tensor,
    n_out: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """src_qpos: (B, T, 76) on the SOURCE body -> (ref_raw, ref), each (B, n_out, 76).

    `ref_raw` is pre-clamp and keeps a live gradient outside jnt_range -- that
    is deliberate. torch.clamp has zero subgradient beyond the bound, which
    would silently kill exactly the gradient the feasibility penalty needs
    (95.8% of reference frames violate at least one limit, see proposal.md 1.2).
    `ref` is the clamped version and is what RSI and the tracking reward use.

    At params from u = 0 this computes, elementwise:
        ref[..., 0:3] = src[..., 0:3] * (h_tgt / h_src)
        ref[..., 3:]  = src[..., 3:]
    i.e. scripts/qpos_retarget.py:91 retarget_qpos, exactly.
    """
    n_out = n_out or src_qpos.shape[1]
    tau = params["tau"]
    if torch.any(tau != 1.0):
        src_qpos = _resample_time(src_qpos, tau, n_out)
    else:
        src_qpos = src_qpos[:, :n_out]

    root_pos = src_qpos[..., 0:3]
    root_quat = src_qpos[..., 3:7]
    hinge = src_qpos[..., 7:]

    scale = params["root_scale"].unsqueeze(1)                   # (B, 1, 3)
    dz = params["root_dz"].reshape(-1, 1, 1)                    # (B, 1, 1)
    out_pos = root_pos * scale + torch.cat(
        [torch.zeros_like(dz), torch.zeros_like(dz), dz], dim=-1
    )

    # Body-frame correction: right-multiply, so it is applied in the root's own
    # frame and is invariant to the clip's world heading.
    drot = exp_map_to_quat(params["root_rot"]).unsqueeze(1)     # (B, 1, 4)
    out_quat = quat_normalize(quat_mul(root_quat, drot.expand_as(root_quat)))

    gain = params["joint_gain"].index_select(-1, group_idx).unsqueeze(1)   # (B, 1, 69)
    bias = params["joint_bias"].index_select(-1, group_idx).unsqueeze(1)
    out_hinge = hinge * gain + bias

    ref_raw = torch.cat([out_pos, out_quat, out_hinge], dim=-1)
    return ref_raw, kin.clamp_hinges(ref_raw)


class Retargeter(nn.Module):
    """Convenience wrapper binding the net, the box and one target body together.

    One instance per training body (they differ only in `log_scale0` and the
    TorchKinematics they clamp against); the RetargetNet itself is SHARED across
    bodies -- that sharing is what forces p to be a genuine function of beta
    rather than 10 independent per-body corrections.
    """

    def __init__(self, cfg, net: RetargetNet, kin: TorchKinematics, source_rest_h: float):
        super().__init__()
        self.cfg = cfg
        self.net = net
        self.kin = kin
        self.params = RetargetParams(cfg, source_rest_h, kin.root_rest_height)
        self.register_buffer("group_idx", joint_group_index(kin.hinge_names))

    def u_from_beta(self, beta: torch.Tensor) -> torch.Tensor:
        return self.net(beta)

    def forward(
        self, src_qpos: torch.Tensor, beta: torch.Tensor,
        u_override: Optional[torch.Tensor] = None, n_out: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (ref_raw, ref, u). `u_override` is the ES hook (upper.py)."""
        u = self.u_from_beta(beta) if u_override is None else u_override
        p = self.params(u)
        ref_raw, ref = apply_retarget(src_qpos, p, self.kin, self.group_idx, n_out=n_out)
        return ref_raw, ref, u
