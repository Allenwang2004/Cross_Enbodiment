"""Forward-kinematics helpers for computing world-frame body positions and
orientations from raw qpos arrays, used by losses.py to score motion
similarity (D terms) without needing to re-run physics."""

import numpy as np
import mujoco

EE_BODIES = ["L_Hand", "R_Hand", "L_Toe", "R_Toe", "Head"]
FOOT_BODIES = ["L_Toe", "R_Toe"]
ROOT_BODY = "Pelvis"


def batch_forward_pose(model: mujoco.MjModel, qpos_seq: np.ndarray, body_names):
    """qpos_seq: (T, nq). Returns (pos, quat), each a dict body_name -> array,
    pos: (T, 3) world position, quat: (T, 4) world orientation (w,x,y,z).
    Uses mj_kinematics only (no dynamics/contacts) -- cheap, pose-only."""
    data = mujoco.MjData(model)
    T = qpos_seq.shape[0]
    body_ids = {name: model.body(name).id for name in body_names}
    pos = {name: np.zeros((T, 3)) for name in body_names}
    quat = {name: np.zeros((T, 4)) for name in body_names}
    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_kinematics(model, data)
        for name in body_names:
            bid = body_ids[name]
            pos[name][t] = data.xpos[bid]
            quat[name][t] = data.xquat[bid]
    return pos, quat


def quat_to_yaw(quat_wxyz: np.ndarray) -> np.ndarray:
    """quat_wxyz: (..., 4) MuJoCo convention (w,x,y,z) -> yaw angle (rad)
    around world z."""
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)
