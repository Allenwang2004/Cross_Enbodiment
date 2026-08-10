"""Per-step reward terms. PURE NUMPY -- this module must never import torch.

It is imported by model/bilevel/sim/worker.py, which runs in 32 separate
processes that have no business paying torch's import cost or its thread-pool
footprint.

Why not reuse model/losses.py
-----------------------------
losses.py is the wrong SHAPE, not the wrong content: every function there takes
a whole (T, nq) trajectory, runs its own per-frame mj_kinematics loop, and
returns one scalar for the episode. Here we need one value per control step,
computed inside the worker from an MjData that already exists. Bending
losses.py into that form would wreck it for its remaining job (eval_bilevel.py
still calls functional_equivalence / physics_penalty unchanged, so the new
numbers stay comparable with outputs/{baseline,eval}/report.json).

What IS carried over from losses.py:
  - the up-vector formula, verbatim, including its comment (see _up_z below)
  - com_support / penetration / smoothness, minus their outer np.mean
  - the joint-limit penalty, rewritten as one vectorized clip instead of a
    70-iteration Python loop (losses.py:167) -- that loop is pure overhead in a
    path that runs 256 x 24 x 10000 times
  - FOOT_CONTACT_HEIGHT's 0.05 m height proxy is REPLACED by real contact
    detection off data.contact. The worker owns the MjData, so it has the
    contact list that the post-hoc version never had. This is a genuine
    accuracy upgrade, not just a port.

Also note losses.py's callers all use its `dt=1.0` default, which leaves
foot_slide and smoothness in per-FRAME units (at 30 Hz the per-second values
are 900x and 810000x larger). Everything here uses a real dt.

The worker ships these 14 numbers UNWEIGHTED. Two reasons: the ES estimator in
upper.py has to be able to re-weight F_sim without re-simulating, and reward
weight sweeps become free.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import mujoco
import numpy as np

# Term layout. Order is the wire format between worker and main process -- do
# not reorder without bumping sim/protocol.py's PROTOCOL_VERSION.
TERM_NAMES: List[str] = [
    "r_pose", "r_vel", "r_ee", "r_root", "r_com",     # tracking kernels, each in (0, 1]
    "e_act", "e_smooth", "e_res", "e_tau", "e_ext", "e_slip",  # regularization penalties, >= 0
    "alive",                                           # 1.0 / 0.0
    "pose_err", "root_dist",                           # raw diagnostics (also drive termination)
]
N_TERMS = len(TERM_NAMES)
TERM_IDX: Dict[str, int] = {n: i for i, n in enumerate(TERM_NAMES)}

EE_BODIES = ["L_Hand", "R_Hand", "L_Toe", "R_Toe", "Head"]
FOOT_SIDES = {"L": ["L_Ankle", "L_Toe"], "R": ["R_Ankle", "R_Toe"]}
ROOT_BODY = "Pelvis"


def _up_z(quat_wxyz: np.ndarray) -> float:
    """World-z component of the body's up axis.

    up_z = 2 * (q_y*q_z + q_w*q_x) -- NOT the naive 1 - 2*(qx^2 + qy^2).
    Copied verbatim from model/losses.py:203 together with this comment: this
    asset's Pelvis carries euler="90 0 0", so its rest quaternion is
    (0.7071, 0.7071, 0, 0) and the body's local +Y, not +Z, is world up. The
    textbook formula is a real bug that has already been paid for once here.
    """
    w, x, y, z = quat_wxyz
    return float(2.0 * (y * z + w * x))


@dataclass
class RewardContext:
    """Per-body constants, resolved once per worker process."""
    model: mujoco.MjModel
    dt: float
    root_bid: int
    ee_bids: np.ndarray                # (5,)
    foot_bids: Dict[str, List[int]]    # "L"/"R" -> body ids
    foot_geoms: Dict[str, set]         # "L"/"R" -> geom ids
    floor_gid: int
    forcerange: np.ndarray             # (nu,) positive limit
    mass: float
    leg_len: float
    weight: float                      # M * g, for normalizing the external wrench

    @classmethod
    def build(cls, model: mujoco.MjModel, dt: float, leg_len: float) -> "RewardContext":
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        foot_bids, foot_geoms = {}, {}
        for side, bodies in FOOT_SIDES.items():
            bids = [model.body(b).id for b in bodies]
            foot_bids[side] = bids
            foot_geoms[side] = {
                g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bids
            }
        fr = np.abs(model.actuator_forcerange[:, 1]).copy()
        fr[fr == 0] = 1.0  # unlimited actuators would otherwise divide by zero
        mass = float(model.body_mass.sum())
        return cls(
            model=model, dt=dt,
            root_bid=model.body(ROOT_BODY).id,
            ee_bids=np.array([model.body(b).id for b in EE_BODIES]),
            foot_bids=foot_bids, foot_geoms=foot_geoms, floor_gid=floor,
            forcerange=fr, mass=mass, leg_len=leg_len,
            weight=mass * float(abs(model.opt.gravity[2])),
        )


def reference_cache(ctx: RewardContext, ref_qpos: np.ndarray) -> Dict[str, np.ndarray]:
    """FK the whole reference window once, at RSI time.

    ref_qpos: (H+1, nq) -> dict of arrays indexed by window step. Measured at
    0.0033 ms per mj_kinematics call, so 25 frames is ~0.08 ms -- free relative
    to the 24 x 1.15 ms of physics that follows.
    """
    m = ctx.model
    d = mujoco.MjData(m)
    T = ref_qpos.shape[0]
    xpos = np.zeros((T, m.nbody, 3))
    com = np.zeros((T, 3))
    for t in range(T):
        d.qpos[:] = ref_qpos[t]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        xpos[t] = d.xpos
        com[t] = d.subtree_com[0]
    hinge = ref_qpos[:, 7:]
    # Reference hinge velocity by finite difference. Safe for hinges (unlike the
    # root, whose qvel[3:6] is a body-local angular velocity -- see
    # data.py:ref_qvel_from_qpos); r_vel only scores the 69 hinges.
    hvel = np.zeros_like(hinge)
    hvel[:-1] = (hinge[1:] - hinge[:-1]) / ctx.dt
    hvel[-1] = hvel[-2] if T > 1 else 0.0
    return {
        "qpos": ref_qpos,
        "hinge": hinge,
        "hinge_vel": hvel,
        "xpos": xpos,
        "com": com,
        "ee": xpos[:, ctx.ee_bids, :],
        "root": xpos[:, ctx.root_bid, :],
    }


def foot_contacts(ctx: RewardContext, data: mujoco.MjData) -> Dict[str, bool]:
    """Real contact detection off data.contact, replacing losses.py's 0.05 m
    height proxy (losses.py:153 FOOT_CONTACT_HEIGHT)."""
    out = {"L": False, "R": False}
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        for side, geoms in ctx.foot_geoms.items():
            if (g1 in geoms and g2 == ctx.floor_gid) or (g2 in geoms and g1 == ctx.floor_gid):
                out[side] = True
    return out


def step_terms(
    ctx: RewardContext,
    cfg,
    data: mujoco.MjData,
    ref: Dict[str, np.ndarray],
    t: int,
    ctrl: np.ndarray,
    prev_ctrl: Optional[np.ndarray],
    raw_prior: np.ndarray,
    wrench: np.ndarray,
    prev_foot_xy: Optional[Dict[str, np.ndarray]],
    out: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Fill `out` (N_TERMS,) for the state in `data` against reference frame t.

    `t` indexes the reference window AFTER the step, i.e. the state produced by
    the t-th control step is scored against ref[t+1]; the caller passes t+1.
    Returns the current foot xy positions so the caller can carry them into the
    next step's slip term.
    """
    m = ctx.model
    q = data.qpos
    hinge = q[7:]
    hinge_vel = data.qvel[6:]

    # ---------------- tracking (DeepMimic-style exponential kernels) --------
    pose_err = float(np.mean((hinge - ref["hinge"][t]) ** 2))
    # Raw radians deliberately, NOT normalized by joint range: 12 of the 69
    # hinges have a range under 0.3 rad (an SMPL-fit artifact -- knee_y/z and
    # ankle_z are ~ +-4.6 deg), and normalizing would let them dominate r_pose
    # while contributing almost nothing visually. In raw radians they self-limit,
    # because the physics clamps them anyway.
    vel_err = float(np.mean((hinge_vel - ref["hinge_vel"][t]) ** 2))

    root_p = data.xpos[ctx.root_bid]
    ee_rel = data.xpos[ctx.ee_bids] - root_p
    ee_rel_ref = ref["ee"][t] - ref["root"][t]
    ee_err = float(np.mean(np.sum((ee_rel - ee_rel_ref) ** 2, axis=1)))

    root_dp = root_p - ref["root"][t]
    root_dist = float(np.linalg.norm(root_dp))
    # Orientation error as a squared geodesic angle, via the relative quaternion.
    dq = np.empty(4)
    rq = np.empty(4)
    mujoco.mju_negQuat(rq, np.ascontiguousarray(ref["qpos"][t, 3:7]))
    mujoco.mju_mulQuat(dq, rq, np.ascontiguousarray(q[3:7]))
    ang = 2.0 * np.arctan2(np.linalg.norm(dq[1:]), abs(dq[0]))
    root_err = float(root_dp @ root_dp) + 0.5 * ang * ang

    mujoco.mj_comPos(m, data)
    com_err = float(np.sum((data.subtree_com[0][:2] - ref["com"][t][:2]) ** 2))

    out[TERM_IDX["r_pose"]] = np.exp(-cfg.k_pose * pose_err)
    out[TERM_IDX["r_vel"]] = np.exp(-cfg.k_vel * vel_err)
    out[TERM_IDX["r_ee"]] = np.exp(-cfg.k_ee * ee_err)
    out[TERM_IDX["r_root"]] = np.exp(-cfg.k_root * root_err)
    out[TERM_IDX["r_com"]] = np.exp(-cfg.k_com * com_err)

    # ---------------- regularization ---------------------------------------
    n = ctrl.shape[0]
    out[TERM_IDX["e_act"]] = float(ctrl @ ctrl) / n
    if prev_ctrl is None:
        out[TERM_IDX["e_smooth"]] = 0.0
    else:
        d_ctrl = ctrl - prev_ctrl
        out[TERM_IDX["e_smooth"]] = float(d_ctrl @ d_ctrl) / n
    d_res = ctrl - raw_prior
    out[TERM_IDX["e_res"]] = float(d_res @ d_res) / n
    tau_n = data.actuator_force / ctx.forcerange
    out[TERM_IDX["e_tau"]] = float(tau_n @ tau_n) / n

    f, mm = wrench[:3], wrench[3:]
    denom_f = ctx.weight ** 2
    denom_m = (ctx.weight * ctx.leg_len) ** 2
    out[TERM_IDX["e_ext"]] = float(f @ f) / denom_f + float(mm @ mm) / denom_m

    contacts = foot_contacts(ctx, data)
    foot_xy = {s: data.xpos[ctx.foot_bids[s][0], :2].copy() for s in ("L", "R")}
    slip = 0.0
    if prev_foot_xy is not None:
        for s in ("L", "R"):
            if contacts[s]:
                v = (foot_xy[s] - prev_foot_xy[s]) / ctx.dt
                slip += float(v @ v)
    out[TERM_IDX["e_slip"]] = slip

    # ---------------- survival ----------------------------------------------
    # humenv's HumEnv.is_terminated() always returns False (humenv/env.py:138),
    # so this is evaluated here or not at all -- and without it r_surv would be
    # a constant carrying no signal whatsoever.
    up = _up_z(q[3:7])
    # Relative to the reference's OWN up-vector, exactly as the root-height test
    # above is relative to the reference's own root height. See
    # config.term_up_margin: an absolute threshold terminated the robot for
    # performing an inverted reference (headstand, crawl, lieonground)
    # correctly, and that -- not ground penetration -- was what pinned Stage 1's
    # term_rate at 0.95-0.99.
    ref_up = _up_z(ref["qpos"][t, 3:7])
    ref_root_z = float(ref["root"][t][2])
    alive = (
        q[2] > cfg.term_root_height_frac * ref_root_z
        and up > ref_up - cfg.term_up_margin
        and root_dist < cfg.term_root_dist * (ctx.leg_len / _ADULT_LEG_LEN)
        # Tracking-failure termination: the single highest-leverage trick in
        # motion-tracking RL, and what turns r_surv into a real signal.
        and pose_err < cfg.term_pose_err
    )
    out[TERM_IDX["alive"]] = 1.0 if alive else 0.0
    out[TERM_IDX["pose_err"]] = pose_err
    out[TERM_IDX["root_dist"]] = root_dist
    return foot_xy


# Reference leg length, used only to scale the root-drift termination threshold
# so a child and a giant are held to proportionally the same standard. Adult's
# thigh + shank from assets/robots/adult/robot.xml.
_ADULT_LEG_LEN = 0.7797


def set_adult_leg_len(v: float) -> None:
    global _ADULT_LEG_LEN
    _ADULT_LEG_LEN = float(v)


def combine(terms: np.ndarray, cfg) -> np.ndarray:
    """Unweighted terms (..., N_TERMS) -> scalar reward (...), bounded in [0, 1].

    Applied in the MAIN process, never in the worker -- see the module docstring.
    """
    i = TERM_IDX
    r_track = (
        cfg.w_pose * terms[..., i["r_pose"]]
        + cfg.w_vel * terms[..., i["r_vel"]]
        + cfg.w_ee * terms[..., i["r_ee"]]
        + cfg.w_root * terms[..., i["r_root"]]
        + cfg.w_com * terms[..., i["r_com"]]
    )
    pen = (
        cfg.e_act * terms[..., i["e_act"]]
        + cfg.e_smooth * terms[..., i["e_smooth"]]
        + cfg.e_res * terms[..., i["e_res"]]
        + cfg.e_tau * terms[..., i["e_tau"]]
        + cfg.e_ext * terms[..., i["e_ext"]]
        + cfg.e_slip * terms[..., i["e_slip"]]
    )
    r_reg = np.exp(-pen)
    r_surv = terms[..., i["alive"]]
    return (
        cfg.r_track_weight * r_track
        + cfg.r_reg_weight * r_reg
        + cfg.r_surv_weight * r_surv
    )
