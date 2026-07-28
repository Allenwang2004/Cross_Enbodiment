"""D = D_root + D_ee + D_contact + D_pose + D_velocity (functional-equivalence
loss between a generated rollout and a retargeted reference trajectory) plus
L_phys (physical-feasibility penalty on the generated rollout alone).

All computed on qpos arrays (numpy) via forward kinematics -- these are used
as scalar terms inside the REINFORCE return in train.py, not backpropagated
through directly (see train.py docstring for why: the MuJoCo rollout that
produced qpos_beta isn't autodiff-differentiable).

Known v1 simplifications (flagged, not silently swept under the rug):
- trajectories of different length are compared by truncating to the
  shorter one, not proper temporal alignment (e.g. DTW).
- D_velocity uses finite-difference qpos deltas as a stand-in for qvel,
  since reference trajectories only store qpos.
"""

import mujoco
import numpy as np

from . import kinematics as kin


def _align_length(a, b):
    T = min(len(a), len(b))
    return a[:T], b[:T]


def d_pose(qpos_a: np.ndarray, qpos_b: np.ndarray) -> float:
    a, b = _align_length(qpos_a[:, 7:], qpos_b[:, 7:])
    return float(np.mean((a - b) ** 2))


def d_velocity(qpos_a: np.ndarray, qpos_b: np.ndarray, dt: float = 1.0) -> float:
    va = np.diff(qpos_a, axis=0) / dt
    vb = np.diff(qpos_b, axis=0) / dt
    va, vb = _align_length(va, vb)
    return float(np.mean((va - vb) ** 2))


def d_root(model, qpos_a: np.ndarray, qpos_b: np.ndarray) -> float:
    pos_a, quat_a = kin.batch_forward_pose(model, qpos_a, [kin.ROOT_BODY])
    pos_b, quat_b = kin.batch_forward_pose(model, qpos_b, [kin.ROOT_BODY])
    root_a, root_b = _align_length(pos_a[kin.ROOT_BODY], pos_b[kin.ROOT_BODY])

    yaw_a = np.unwrap(kin.quat_to_yaw(quat_a[kin.ROOT_BODY]))
    yaw_b = np.unwrap(kin.quat_to_yaw(quat_b[kin.ROOT_BODY]))
    yaw_a, yaw_b = _align_length(yaw_a, yaw_b)
    heading_err = float(np.mean((yaw_a - yaw_b) ** 2))

    yaw_rate_a, yaw_rate_b = _align_length(np.diff(yaw_a), np.diff(yaw_b))
    yaw_rate_err = float(np.mean((yaw_rate_a - yaw_rate_b) ** 2)) if len(yaw_rate_a) else 0.0

    def curvature(traj_xy, min_speed=1e-3):
        # dheading/speed blows up when the root is nearly stationary (e.g.
        # crawling/crouching) -- curvature is meaningless there anyway, so
        # mask those samples out instead of dividing by ~0. Also wrap
        # dheading to [-pi, pi]: arctan2's branch cut otherwise turns a
        # small turn crossing +-pi into a fake ~2*pi jump.
        v = np.diff(traj_xy, axis=0)
        speed = np.linalg.norm(v, axis=-1)
        heading = np.arctan2(v[:, 1], v[:, 0])
        dheading = np.diff(heading)
        dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
        valid = speed[:-1] > min_speed
        curv = np.zeros_like(dheading)
        curv[valid] = dheading[valid] / speed[:-1][valid]
        return curv, valid

    curv_a, valid_a = curvature(root_a[:, :2])
    curv_b, valid_b = curvature(root_b[:, :2])
    curv_a, curv_b = _align_length(curv_a, curv_b)
    valid_a, valid_b = _align_length(valid_a, valid_b)
    both_valid = valid_a & valid_b
    curv_err = float(np.mean((curv_a[both_valid] - curv_b[both_valid]) ** 2)) if both_valid.any() else 0.0

    return heading_err + yaw_rate_err + curv_err


def d_ee(model, qpos_a: np.ndarray, qpos_b: np.ndarray) -> float:
    bodies = kin.EE_BODIES + [kin.ROOT_BODY]
    pos_a, _ = kin.batch_forward_pose(model, qpos_a, bodies)
    pos_b, _ = kin.batch_forward_pose(model, qpos_b, bodies)
    err = 0.0
    for name in kin.EE_BODIES:
        rel_a = pos_a[name] - pos_a[kin.ROOT_BODY]
        rel_b = pos_b[name] - pos_b[kin.ROOT_BODY]
        rel_a, rel_b = _align_length(rel_a, rel_b)
        err += float(np.mean((rel_a - rel_b) ** 2))
    return err / len(kin.EE_BODIES)


def d_contact(model, qpos_a: np.ndarray, qpos_b: np.ndarray) -> float:
    pos_a, _ = kin.batch_forward_pose(model, qpos_a, kin.FOOT_BODIES)
    pos_b, _ = kin.batch_forward_pose(model, qpos_b, kin.FOOT_BODIES)
    err = 0.0
    for name in kin.FOOT_BODIES:
        za, zb = _align_length(pos_a[name][:, 2], pos_b[name][:, 2])
        err += float(np.mean((za - zb) ** 2))
    return err / len(kin.FOOT_BODIES)


def functional_equivalence(model, qpos_beta: np.ndarray, qpos_ref, weights: dict):
    """weights: dict with keys root/ee/contact/pose/velocity.
    qpos_ref may be None (no retargeted reference attached yet for this
    sample) -> returns (0.0, {})."""
    if qpos_ref is None:
        return 0.0, {}
    terms = {
        "root": d_root(model, qpos_beta, qpos_ref),
        "ee": d_ee(model, qpos_beta, qpos_ref),
        "contact": d_contact(model, qpos_beta, qpos_ref),
        "pose": d_pose(qpos_beta, qpos_ref),
        "velocity": d_velocity(qpos_beta, qpos_ref),
    }
    total = sum(weights[k] * v for k, v in terms.items())
    return total, terms


def physics_penalty(model, qpos_seq: np.ndarray) -> float:
    """Joint-limit violation + a crude fall detector, from qpos_seq alone."""
    penalty = 0.0
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qadr = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]
        if lo == hi:
            continue
        vals = qpos_seq[:, qadr]
        viol = np.clip(lo - vals, 0, None) + np.clip(vals - hi, 0, None)
        penalty += float(np.mean(viol ** 2))

    pelvis_z = qpos_seq[:, 2]
    fell = pelvis_z < (0.5 * qpos_seq[0, 2])
    penalty += 5.0 * float(np.mean(fell))
    return penalty
