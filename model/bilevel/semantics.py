"""The two purely-p terms of the upper objective: S (fidelity) and C (feasibility).

Both depend on p ONLY through the reference -- never through the simulator --
so their gradients are exact and zero-variance. Together with the hard tanh box
in retarget.py they are what stops the upper level from collapsing.

S -- semantic fidelity
    The counterweight to G. G's tractable gradient is a pure "move ref onto the
    robot" pull; S is a pure "move ref back onto the source clip" pull. Their
    joint stationary point is a shrinkage estimator,

        ref* = (lambda_gap * tau + lambda_fid * src) / (lambda_gap + lambda_fid)

    so lambda_fid/lambda_gap sets, interpretably, how far the reference is
    allowed to drift toward the robot: 25% at the default 3:1.

    Every channel is dimensionless or scale-normalized, so a legitimate
    morphology change costs nothing and only a change of MOTION costs. That is
    the whole trick -- comparing raw world-space positions across a 0.58 m child
    and a 1.17 m giant would penalize exactly the adaptation we want.

C -- reference feasibility on the target body
    Encodes the finding that motivates the whole upper level: 95.8% of source
    frames ask for at least one joint angle outside the target body's own
    jnt_range, and MuJoCo silently clamps them. This is the term that has
    somewhere real to go before it starts eating into motion semantics.

    C is evaluated on `ref_raw` (pre-clamp), never on `ref`. torch.clamp has
    zero subgradient outside the bound, so evaluating it on the clamped
    reference would kill precisely the gradient that pulls violations back
    inside -- verified in tests/test_retarget.py.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from model.bilevel.quat import quat_log_map, quat_mul, quat_to_yaw
from model.bilevel.torch_kin import TorchKinematics
from model.kinematics import EE_BODIES, FOOT_BODIES

GRAVITY = 9.81


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + torch.pi) % (2 * torch.pi) - torch.pi


def contact_signal(foot_z: torch.Tensor, rest_z: torch.Tensor, k: float) -> torch.Tensor:
    """Soft foot-contact probability from foot height. (..., 2) -> (..., 2).

    Height is measured RELATIVE to that body's own rest-pose foot height, so the
    same threshold means the same thing on every morphology (see
    data.py:_foot_rest_heights).
    """
    return torch.sigmoid(-k * (foot_z - rest_z))


class UpperGeometry:
    """FK'd quantities for one (body, batch of windows), cached for reuse.

    S and C both need most of these, and G in upper.py needs the reference side
    again -- computing them once per body group per iteration keeps the upper
    level at the ~30 ms/iteration budgeted in proposal.md 6.7.
    """

    def __init__(self, kin: TorchKinematics, qpos: torch.Tensor, foot_rest_z: torch.Tensor):
        self.kin = kin
        self.qpos = qpos                                    # (B, T, 76)
        self.xpos, self.xquat = kin(qpos)                   # (B, T, nbody, 3/4)
        self.root = kin.body_pos_of(self.xpos, ["Pelvis"])[..., 0, :]      # (B, T, 3)
        self.root_quat = kin.body_quat_of(self.xquat, ["Pelvis"])[..., 0, :]
        self.ee = kin.body_pos_of(self.xpos, EE_BODIES)                    # (B, T, 5, 3)
        self.feet = kin.body_pos_of(self.xpos, FOOT_BODIES)                # (B, T, 2, 3)
        self.foot_rest_z = foot_rest_z                                     # (2,)
        self._com: Optional[torch.Tensor] = None
        self._minz: Optional[torch.Tensor] = None

    @property
    def com(self) -> torch.Tensor:
        if self._com is None:
            self._com = self.kin.com(self.xpos, self.xquat)
        return self._com

    @property
    def min_geom_z(self) -> torch.Tensor:
        if self._minz is None:
            self._minz = self.kin.min_geom_z(self.xpos, self.xquat)
        return self._minz

    @property
    def hinge(self) -> torch.Tensor:
        return self.qpos[..., 7:]

    def ee_relative(self) -> torch.Tensor:
        return self.ee - self.root.unsqueeze(-2)

    def foot_height(self) -> torch.Tensor:
        return self.feet[..., 2] - self.foot_rest_z

    def foot_vel_xy(self, dt: float) -> torch.Tensor:
        """(B, T-1, 2, 2) per-foot planar velocity."""
        return (self.feet[:, 1:, :, :2] - self.feet[:, :-1, :, :2]) / dt

    def heading(self) -> torch.Tensor:
        return quat_to_yaw(self.root_quat)

    def froude(self, leg_len: float, dt: float) -> torch.Tensor:
        """||v_root,xy|| / sqrt(g * L_leg): the dimensionless gait-speed number.

        Two bodies performing "the same walk" at proportional speeds have equal
        Froude numbers, which is exactly the invariance S needs.
        """
        v = (self.root[:, 1:, :2] - self.root[:, :-1, :2]) / dt
        return v.norm(dim=-1) / (GRAVITY * leg_len) ** 0.5


def weighted_sum(cfg, terms: Dict[str, torch.Tensor],
                 scales: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """Sum w_k * term_k / scale_k, where scale_k is the term's value at p = 0.

    Normalizing by the p=0 baseline is what makes the weights in
    BilevelConfig mean what they say. Without it the raw magnitudes span six
    orders of magnitude -- measured at p=0 on `child`:

        c_limit  9.7e-04     <- the term encoding the 95.8%-illegal finding
        c_slide  5.6e-01
        c_smooth 1.2e+03     (with a /dt^2, since removed)

    so a nominal 4:1 priority for c_limit over c_slide was in reality 1:140,
    and the upper level would have spent its entire budget smoothing.

    It also makes the Stage 2 gate directly readable: S is 1.0 at p=0 by
    construction, so "S(ref_p) < 2 * S(ref_0)" is literally "S < 2".
    """
    total = None
    for k, v in terms.items():
        w = getattr(cfg, k)
        if scales is not None:
            w = w / max(scales.get(k, 1.0), 1e-12)
        total = w * v if total is None else total + w * v
    return total


def semantic_fidelity(
    cfg, ref_geo: UpperGeometry, src_geo: UpperGeometry,
    ee_limb_len: torch.Tensor, src_ee_limb_len: torch.Tensor,
    leg_len: float, src_leg_len: float,
) -> Dict[str, torch.Tensor]:
    """S(ref_p, src) raw terms. Exact gradient w.r.t. p.

    Combine with weighted_sum(); the caller owns the p=0 normalization.
    """
    dt = cfg.dt

    # 1. joint angles -- already a morphology invariant, the topology is identical
    d_pose = (ref_geo.hinge - src_geo.hinge).pow(2).mean()

    # 2. end-effector reach, made dimensionless by each body's own limb lengths
    e_ref = ref_geo.ee_relative() / ee_limb_len.view(1, 1, -1, 1)
    e_src = src_geo.ee_relative() / src_ee_limb_len.view(1, 1, -1, 1)
    d_reach = (e_ref - e_src).pow(2).sum(-1).mean()

    # 3. contact pattern -- the workhorse. If p shrinks amplitudes or drops the
    #    root, foot-off/foot-strike timing shifts and this fires immediately.
    p_ref = contact_signal(ref_geo.foot_height(), ref_geo.foot_rest_z, cfg.s_contact_k)
    c_src = contact_signal(src_geo.foot_height(), src_geo.foot_rest_z, cfg.s_contact_k).detach()
    d_contact = F.binary_cross_entropy(p_ref.clamp(1e-6, 1 - 1e-6), c_src)

    # 4. per-step heading change (turning rate), wrapped so +-pi does not alias
    dpsi_ref = _wrap_pi(ref_geo.heading()[:, 1:] - ref_geo.heading()[:, :-1])
    dpsi_src = _wrap_pi(src_geo.heading()[:, 1:] - src_geo.heading()[:, :-1])
    d_heading = (dpsi_ref - dpsi_src).pow(2).mean()

    # 5. Froude number
    d_froude = (ref_geo.froude(leg_len, dt) - src_geo.froude(src_leg_len, dt)).pow(2).mean()

    return {"s_pose": d_pose, "s_reach": d_reach, "s_contact": d_contact,
            "s_heading": d_heading, "s_froude": d_froude}


def reference_feasibility(
    cfg, kin: TorchKinematics, ref_raw: torch.Tensor, ref_geo: UpperGeometry,
    src_geo: UpperGeometry, leg_len: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """C(ref_raw, beta) raw terms. Exact gradient w.r.t. p.

    `ref_raw` must be the PRE-CLAMP reference (see module docstring).
    """
    dt = cfg.dt

    # 1. actuator / joint-limit violation. |a_raw| > 1 is exactly the set of
    #    frames MuJoCo would silently clamp -- the 95.8% figure.
    a_raw = kin.normalized_ctrl(ref_raw[..., 7:])
    c_limit = F.relu(a_raw.abs() - 1.0).pow(2).mean()

    # 2. ground penetration, against the same tolerance
    #    scripts/qpos_retarget.py:161 uses for its (now removed) per-frame lift
    minz = ref_geo.min_geom_z
    c_pen = F.relu(-(minz - cfg.ground_tol)).pow(2).mean()

    # 3. floating: airborne in the reference while the SOURCE has a foot planted
    src_contact = (contact_signal(src_geo.foot_height(), src_geo.foot_rest_z,
                                  cfg.s_contact_k).max(dim=-1).values > 0.5).detach()
    c_float = (F.relu(minz - cfg.float_tol).pow(2) * src_contact).mean()

    # 4. foot slide during source-declared contact, in leg-lengths per second so
    #    a child and a giant are held to the same standard
    c_src = contact_signal(src_geo.foot_height(), src_geo.foot_rest_z,
                           cfg.s_contact_k).detach()[:, :-1]
    v = ref_geo.foot_vel_xy(dt) / leg_len
    c_slide = (c_src * v.pow(2).sum(-1)).mean()

    # 5. reference smoothness (second difference of the hinge angles)
    #    Kept in PER-FRAME units. Dividing by dt^2 and then squaring multiplies
    #    this by 810,000 at 30 Hz, which made it 99.9% of C and reduced the
    #    limit term -- the one encoding the 95.8%-illegal finding that motivates
    #    the whole upper level -- to 0.0007% of it. Per-term normalization
    #    (calibrate() below) makes the absolute scale irrelevant anyway; this
    #    just keeps the raw logged numbers readable.
    acc = ref_raw[:, 2:, 7:] - 2 * ref_raw[:, 1:-1, 7:] + ref_raw[:, :-2, 7:]
    c_smooth = acc.pow(2).mean()

    return {"c_limit": c_limit, "c_penetrate": c_pen, "c_float": c_float,
            "c_slide": c_slide, "c_smooth": c_smooth}


def gap(
    cfg, kin: TorchKinematics, rollout_qpos: torch.Tensor, ref_geo: UpperGeometry,
    valid: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """G(tau, ref_p): what new.md asks the upper level to measure.

    `rollout_qpos` is the realized trajectory and is treated as a CONSTANT --
    MuJoCo is not differentiable, so the only gradient this contributes is the
    reference-side one. That asymmetry is precisely why S, C and the hard box
    exist; on its own this term says "move the reference onto the robot".

    Scored against the CLAMPED reference: the robot cannot be blamed for failing
    to reach an angle its own actuators cannot command.

    `valid` (B, H) MUST be supplied for anything that feeds a gradient. A
    terminated slot is frozen by the worker, which leaves its last live qpos
    repeating in the buffer for the rest of the window -- so an unmasked mean
    scores "a corpse on the floor" against "the reference still mid-jump" for
    every step after the fall. That is not merely noise: it is a systematic pull
    toward references that match FALLEN poses, i.e. it feeds the exact
    degeneracy the hard box, S and the saturation diagnostic exist to prevent.

    Measured on the stage-3 run of 2026-08-10: mean_steps was ~30 of 60, so
    roughly HALF of every G in that run was post-mortem. `_gap_per_env` (the ES
    path) was already masking; this path was not, so the two estimates of the
    same quantity disagreed by construction.
    """
    tau = rollout_qpos.detach()
    tau_xpos, tau_xquat = kin(tau)
    tau_root = kin.body_pos_of(tau_xpos, ["Pelvis"])[..., 0, :]
    tau_root_q = kin.body_quat_of(tau_xquat, ["Pelvis"])[..., 0, :]
    tau_ee = kin.body_pos_of(tau_xpos, EE_BODIES) - tau_root.unsqueeze(-2)

    rel = quat_mul(
        torch.cat([ref_geo.root_quat[..., :1], -ref_geo.root_quat[..., 1:]], dim=-1), tau_root_q
    )

    if valid is None:
        avg = lambda x: x.mean()
    else:
        w = valid
        denom = w.sum().clamp(min=1.0)
        avg = lambda x: (x * w).sum() / denom

    d_pose = avg((tau[..., 7:] - ref_geo.hinge).pow(2).mean(-1))
    d_ee = avg((tau_ee - ref_geo.ee_relative()).pow(2).sum(-1).mean(-1))
    d_root_pos = avg((tau_root - ref_geo.root).pow(2).sum(-1))
    d_root_rot = avg(quat_log_map(rel).pow(2).sum(-1))

    return {"g_pose": d_pose, "g_ee": d_ee, "g_root_pos": d_root_pos, "g_root_rot": d_root_rot}


def frac_illegal_frames(kin: TorchKinematics, ref_raw: torch.Tensor) -> torch.Tensor:
    """Fraction of frames with at least one out-of-range hinge.

    The headline success metric for the upper level: measured at 0.958 for the
    naive retarget (p = 0) across all 540 clips, and Stage 2's gate is getting
    it below 0.2 (proposal.md 8.2).
    """
    a = kin.normalized_ctrl(ref_raw[..., 7:])
    return (a.abs() > 1.0).any(dim=-1).to(a.dtype).mean()
