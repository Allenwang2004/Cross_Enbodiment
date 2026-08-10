"""Batched quaternion helpers in torch, MuJoCo convention (w, x, y, z).

Everything here is shape-generic over leading dims: a quaternion tensor is
(..., 4) and a vector tensor is (..., 3). Nothing allocates an MjData, so all
of it is autograd-safe and runs on GPU.

The numpy counterpart for the (w,x,y,z) convention lives in
model/kinematics.py:48 quat_to_yaw -- kept consistent deliberately.
"""

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a (x) b. a, b: (..., 4) -> (..., 4)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=eps)


def quat_rot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by q. q: (..., 4), v: (..., 3) -> (..., 3).

    Uses the cross-product form (t = 2 q_vec x v; v' = v + q_w t + q_vec x t)
    rather than building a 3x3 -- fewer ops and no matmul on a broadcast shape.
    """
    qw = q[..., :1]
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """q: (..., 4) -> (..., 3, 3) rotation matrix (columns are the rotated
    basis vectors, i.e. the same layout as MuJoCo's data.xmat.reshape(3, 3))."""
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return m.reshape(*q.shape[:-1], 3, 3)


def axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """axis: (..., 3) UNIT vector, angle: (...) radians -> (..., 4).

    The model's hinge axes are exactly the canonical basis vectors (verified:
    every jnt_axis in assets/robots/*/robot.xml is one of [1,0,0]/[0,1,0]/
    [0,0,1]), so no renormalization is done here -- callers pass model axes."""
    half = 0.5 * angle
    return torch.cat([torch.cos(half).unsqueeze(-1), torch.sin(half).unsqueeze(-1) * axis], dim=-1)


def quat_log_map(q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Rotation-vector (exp-map) log of a unit quaternion. q: (..., 4) -> (..., 3).

    Sign-canonicalized to the w >= 0 hemisphere first, so the result is the
    minimal rotation and log(q) == -log(q_conj) holds. Used for orientation
    error terms, where ||log(q_ref^-1 (x) q)||^2 is the squared geodesic angle.
    """
    q = torch.where(q[..., :1] < 0, -q, q)
    w = q[..., 0].clamp(-1.0, 1.0)
    v = q[..., 1:]
    sin_half = v.norm(dim=-1).clamp(min=eps)
    angle = 2.0 * torch.atan2(sin_half, w)
    # angle / sin_half -> 2 as the rotation goes to zero; the clamp above keeps
    # the division finite and the limit is picked up by the where().
    scale = torch.where(sin_half > eps, angle / sin_half, torch.full_like(angle, 2.0))
    return v * scale.unsqueeze(-1)


def exp_map_to_quat(w: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Rotation vector -> unit quaternion. w: (..., 3) -> (..., 4). Inverse of
    quat_log_map. Used by RetargetNet's root-orientation correction."""
    theta = w.norm(dim=-1, keepdim=True)
    half = 0.5 * theta
    # sinc(half)/2 == sin(half)/theta, with the removable singularity at 0
    # handled by the where(); torch.sinc takes its argument in units of pi.
    small = theta < eps
    sin_over_theta = torch.where(small, torch.full_like(theta, 0.5), torch.sin(half) / theta.clamp(min=eps))
    return torch.cat([torch.cos(half), w * sin_over_theta], dim=-1)


def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) -> (...) yaw about world z. Mirrors model/kinematics.py:48."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_up_z(q: torch.Tensor) -> torch.Tensor:
    """World-z component of the body's own up axis. (..., 4) -> (...).

    up_z = 2 * (q_y*q_z + q_w*q_x)   -- NOT the textbook 1 - 2(qx^2 + qy^2).

    Copied verbatim (with this comment) from model/losses.py:203. This asset's
    Pelvis carries euler="90 0 0", so its rest quaternion is (0.7071, 0.7071,
    0, 0) and the body's local +Y -- not +Z -- maps to world up. Using the
    textbook formula here is a real bug that has already been paid for once.
    """
    w, x, y, z = q.unbind(-1)
    return 2.0 * (y * z + w * x)
