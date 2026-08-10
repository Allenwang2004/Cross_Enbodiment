"""Batched, differentiable forward kinematics in torch for this repo's humanoid.

Why this exists
---------------
The upper level needs d(reference geometry)/d(phi). model/kinematics.py:13
batch_forward_pose does the same job with mujoco.mj_kinematics, but it is a
per-frame Python loop and gives no gradient. MuJoCo has no autodiff (mujoco-mjx
is not installed, see proposal.md 6.1), so the reference-side FK is
reimplemented here. That is only tractable because the skeleton is fixed:
every assets/robots/<label>/robot.xml has nq=76, nv=75, nbody=25, njnt=70 with
identical names/types/axes/declaration order (scale_robot.py only changes
body_pos and geom sizes).

Three preconditions, all verified against every robot.xml in assets/robots/:
  1. model.jnt_pos is exactly 0 for every joint  -> MuJoCo's off-center-rotation
     correction in mj_kinematics is a no-op, so a joint is a pure quaternion
     post-multiply on its body's frame.
  2. model.qpos0[7:] is exactly 0                -> a hinge's angle IS its qpos
     entry (mj_kinematics uses qpos - qpos0).
  3. Only Pelvis has a non-identity body_quat, and Pelvis carries the free
     joint -- for which MuJoCo IGNORES body_pos/body_quat entirely and takes
     xpos/xquat straight from qpos[0:3]/qpos[3:7]. (That is exactly why the
     rest quaternion is (0.7071, 0.7071, 0, 0) and why the up-axis formula in
     quat.py:quat_up_z is what it is.)

Accuracy is asserted against mj_kinematics in tests/test_torch_kin.py -- see
proposal.md R3: if this file is silently wrong, the upper level optimizes a
geometry the simulator does not have and every loss curve still looks fine.
That test is the hard gate for the whole upper level.
"""

from typing import Dict, List, Optional, Sequence

import mujoco
import numpy as np
import torch
import torch.nn as nn

from model.bilevel.quat import axis_angle_to_quat, quat_mul, quat_normalize, quat_rot

# Box corner sign pattern, fixed once. Matches the itertools.product([1, -1],
# repeat=3) enumeration in scripts/qpos_retarget.py:114 (order is irrelevant --
# only the min over corners is used).
_BOX_SIGNS = np.array(
    [[sx, sy, sz] for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0)],
    dtype=np.float64,
)
_MAX_GEOM_PTS = 8  # a box needs 8; everything else is padded and masked off


class TorchKinematics(nn.Module):
    """Forward kinematics + exact lowest-geom-z for one MJCF, as a torch module.

    All model constants are registered buffers, so `.to(device)` / `.float()`
    move the whole thing. Stateless at call time: forward(qpos) allocates
    nothing that persists.

    qpos convention throughout: (..., 76) with [0:3] root world position,
    [3:7] root world quaternion (w,x,y,z), [7:76] the 69 hinge angles in
    declaration order.
    """

    def __init__(self, model: mujoco.MjModel, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.nbody = int(model.nbody)
        self.nq = int(model.nq)
        self.body_names: List[str] = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.nbody)
        ]
        self.body_index: Dict[str, int] = {n: i for i, n in enumerate(self.body_names)}

        self._check_preconditions(model)

        # ---- kinematic tree ----------------------------------------------
        self.register_buffer("body_parent", torch.as_tensor(model.body_parentid.copy(), dtype=torch.long))
        self.register_buffer("body_pos", torch.as_tensor(model.body_pos.copy(), dtype=dtype))
        self.register_buffer("body_quat", torch.as_tensor(model.body_quat.copy(), dtype=dtype))
        self.register_buffer("body_ipos", torch.as_tensor(model.body_ipos.copy(), dtype=dtype))
        self.register_buffer("body_mass", torch.as_tensor(model.body_mass.copy(), dtype=dtype))

        # Per-body joint plan, resolved once here so forward() is a plain loop.
        # free_body is the (single) body carrying the free joint.
        self.free_body: int = -1
        hinge_plan: List[List[int]] = [[] for _ in range(self.nbody)]  # body -> [joint ids]
        for j in range(model.njnt):
            b = int(model.jnt_bodyid[j])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.free_body = b
            else:
                hinge_plan[b].append(j)
        self._hinge_plan = hinge_plan

        max_h = max(len(v) for v in hinge_plan)
        axes = np.zeros((self.nbody, max_h, 3), dtype=np.float64)
        qadr = np.zeros((self.nbody, max_h), dtype=np.int64)
        for b, jids in enumerate(hinge_plan):
            for k, j in enumerate(jids):
                axes[b, k] = model.jnt_axis[j]
                qadr[b, k] = model.jnt_qposadr[j]
        self.register_buffer("hinge_axis", torch.as_tensor(axes, dtype=dtype))
        self.register_buffer("hinge_qadr", torch.as_tensor(qadr, dtype=torch.long))

        # PLAIN PYTHON copies of every index forward() needs.
        #
        # Reading them out of the buffers instead (int(self.body_parent[b]))
        # forces a device->host synchronization on EVERY body, ~96 per call.
        # Measured on this batch shape: 508 ms/call on GPU versus 10 ms on CPU,
        # for arithmetic that is identical -- the FK was 98% synchronization
        # stalls. With these it is ~15 ms on GPU. The loop body must stay free
        # of any .item()/int()/bool() on a device tensor.
        self._parent_py: List[int] = [int(v) for v in model.body_parentid]
        self._qadr_py: List[List[int]] = [
            [int(model.jnt_qposadr[j]) for j in jids] for jids in hinge_plan
        ]

        # Hinge qpos addresses in declaration order, i.e. the 69 entries of
        # qpos[7:]. Also the canonical ordering for jnt_range below.
        hinge_ids = [j for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        self.hinge_joint_ids = hinge_ids
        self.hinge_names: List[str] = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge_ids
        ]
        self.register_buffer(
            "hinge_lo", torch.as_tensor(model.jnt_range[hinge_ids, 0].copy(), dtype=dtype)
        )
        self.register_buffer(
            "hinge_hi", torch.as_tensor(model.jnt_range[hinge_ids, 1].copy(), dtype=dtype)
        )

        # ---- geometry (for min_geom_z) -----------------------------------
        self._build_geom_tables(model, dtype)

        # rest-pose root height, i.e. what scripts/qpos_retarget.py:83
        # load_root_rest_height reads out of skeleton.json. Taken straight from
        # qpos0 instead: verified identical, and 11 of 13 bodies have no
        # skeleton.json on disk.
        self.root_rest_height: float = float(model.qpos0[2])
        self.total_mass: float = float(model.body_mass.sum())

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _check_preconditions(model: mujoco.MjModel) -> None:
        if not np.allclose(model.jnt_pos, 0.0):
            raise ValueError(
                "TorchKinematics assumes every joint sits at its body origin "
                "(jnt_pos == 0); MuJoCo's off-center rotation correction is not "
                "implemented here. scripts/scale_robot.py:184 asserts the same."
            )
        if not np.allclose(model.qpos0[7:], 0.0):
            raise ValueError(
                "TorchKinematics assumes qpos0[7:] == 0 so that a hinge angle is "
                "its raw qpos entry."
            )
        n_free = int((model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).sum())
        if n_free != 1:
            raise ValueError(f"expected exactly one free joint, found {n_free}")

    def _build_geom_tables(self, model: mujoco.MjModel, dtype: torch.dtype) -> None:
        """Per-geom local sample points whose world z lower-bounds the geom.

        Same case analysis as scripts/qpos_retarget.py:99 _min_geom_z (and
        scale_robot.py:295 measure_min_z / check_retarget_ground_clearance.py:30
        min_geom_z -- three copies of it in the repo, this replaces all three
        for the differentiable path): box -> 8 corners, capsule -> the two cap
        centres minus the radius, sphere -> centre minus radius, anything else
        -> the centre. geom_rbound is NOT used; it is a bounding-sphere radius
        and is far too loose for the box feet.
        """
        pts: List[np.ndarray] = []
        rad: List[float] = []
        valid: List[np.ndarray] = []
        bodies: List[int] = []
        gpos: List[np.ndarray] = []
        gquat: List[np.ndarray] = []
        names: List[str] = []

        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if name == "floor":
                continue
            gtype = model.geom_type[gid]
            size = model.geom_size[gid]
            p = np.zeros((_MAX_GEOM_PTS, 3), dtype=np.float64)
            v = np.zeros(_MAX_GEOM_PTS, dtype=bool)
            r = 0.0
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                p[:8] = _BOX_SIGNS * size[None, :3]
                v[:8] = True
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                # MuJoCo capsules run along the geom's local z; size = [radius, half_length]
                p[0] = (0.0, 0.0, size[1])
                p[1] = (0.0, 0.0, -size[1])
                v[:2] = True
                r = float(size[0])
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                v[0] = True
                r = float(size[0])
            else:
                v[0] = True

            pts.append(p)
            valid.append(v)
            rad.append(r)
            bodies.append(int(model.geom_bodyid[gid]))
            gpos.append(model.geom_pos[gid].copy())
            gquat.append(model.geom_quat[gid].copy())
            names.append(name)

        self.geom_names = names
        self.register_buffer("geom_body", torch.as_tensor(np.array(bodies), dtype=torch.long))
        self.register_buffer("geom_pos", torch.as_tensor(np.array(gpos), dtype=dtype))
        self.register_buffer("geom_quat", torch.as_tensor(np.array(gquat), dtype=dtype))
        self.register_buffer("geom_pts", torch.as_tensor(np.array(pts), dtype=dtype))
        self.register_buffer("geom_radius", torch.as_tensor(np.array(rad), dtype=dtype))
        self.register_buffer("geom_valid", torch.as_tensor(np.array(valid), dtype=torch.bool))

    # ---------------------------------------------------------------- kinematics

    def forward(self, qpos: torch.Tensor):
        """qpos: (..., 76) -> (xpos, xquat), each (..., nbody, 3) / (..., nbody, 4).

        Body 0 is `world` and is returned as the identity frame so that indices
        line up with MuJoCo's own body ids (and with model/kinematics.py's
        model.body(name).id lookups).
        """
        if qpos.shape[-1] != self.nq:
            raise ValueError(f"expected qpos (..., {self.nq}), got {tuple(qpos.shape)}")
        lead = qpos.shape[:-1]
        dt, dev = qpos.dtype, qpos.device

        xpos: List[Optional[torch.Tensor]] = [None] * self.nbody
        xquat: List[Optional[torch.Tensor]] = [None] * self.nbody
        xpos[0] = torch.zeros(*lead, 3, dtype=dt, device=dev)
        ident = torch.zeros(*lead, 4, dtype=dt, device=dev)
        ident[..., 0] = 1.0
        xquat[0] = ident

        for b in range(1, self.nbody):
            if b == self.free_body:
                # MuJoCo ignores body_pos/body_quat for a free-joint body and
                # reads the world frame straight out of qpos (engine_core_smooth
                # mj_kinematics). It also normalizes the quaternion there, which
                # matters because RetargetNet's root correction can denormalize it.
                p = qpos[..., 0:3]
                q = quat_normalize(qpos[..., 3:7])
            else:
                # Every index here is a Python int (see _parent_py/_qadr_py):
                # touching a device tensor for an index would synchronize once
                # per body and dominate the whole call.
                par = self._parent_py[b]
                pq, pp = xquat[par], xpos[par]
                q = quat_mul(pq, self.body_quat[b].expand(*lead, 4))
                p = pp + quat_rot(pq, self.body_pos[b].expand(*lead, 3))
                for k, adr in enumerate(self._qadr_py[b]):
                    angle = qpos[..., adr]
                    axis = self.hinge_axis[b, k].expand(*lead, 3)
                    q = quat_mul(q, axis_angle_to_quat(axis, angle))
                    # no position update: jnt_pos == 0 (see _check_preconditions)
            xpos[b] = p
            xquat[b] = q

        return torch.stack(xpos, dim=-2), torch.stack(xquat, dim=-2)

    # ------------------------------------------------------------------ derived

    def com(self, xpos: torch.Tensor, xquat: torch.Tensor) -> torch.Tensor:
        """Whole-body centre of mass, (..., nbody, *) -> (..., 3).

        Equivalent to data.subtree_com[0] after mj_comPos (what
        model/kinematics.py:32 batch_com returns): the mass-weighted mean of
        every body's inertial-frame origin xipos = xpos + R * body_ipos.
        """
        xipos = xpos + quat_rot(xquat, self.body_ipos.expand_as(xpos))
        w = self.body_mass.unsqueeze(-1)
        return (xipos * w).sum(dim=-2) / self.body_mass.sum()

    def min_geom_z(self, xpos: torch.Tensor, xquat: torch.Tensor) -> torch.Tensor:
        """Exact lowest world-z over all non-floor geoms. (..., nbody, *) -> (...).

        Vectorized replacement for the Python geom loop in
        scripts/qpos_retarget.py:99 -- measured there at 113x the cost of the
        mj_kinematics call it follows. Here it is one broadcast over
        (..., ngeom, 8, 3).
        """
        gb = self.geom_body
        bq = xquat[..., gb, :]                                   # (..., G, 4)
        bp = xpos[..., gb, :]                                    # (..., G, 3)
        gq = quat_mul(bq, self.geom_quat.expand_as(bq))          # (..., G, 4)
        gp = bp + quat_rot(bq, self.geom_pos.expand_as(bp))      # (..., G, 3)

        pts = self.geom_pts.expand(*gq.shape[:-1], _MAX_GEOM_PTS, 3)   # (..., G, 8, 3)
        gq8 = gq.unsqueeze(-2).expand(*pts.shape[:-1], 4)              # (..., G, 8, 4)
        world = gp.unsqueeze(-2) + quat_rot(gq8, pts)
        z = world[..., 2] - self.geom_radius.unsqueeze(-1)             # (..., G, 8)
        z = torch.where(self.geom_valid.expand_as(z), z, torch.full_like(z, float("inf")))
        return z.flatten(start_dim=-2).min(dim=-1).values

    def body_pos_of(self, xpos: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
        """Gather named bodies' world positions. -> (..., len(names), 3)."""
        idx = torch.as_tensor([self.body_index[n] for n in names], device=xpos.device)
        return xpos.index_select(-2, idx)

    def body_quat_of(self, xquat: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
        idx = torch.as_tensor([self.body_index[n] for n in names], device=xquat.device)
        return xquat.index_select(-2, idx)

    # ---------------------------------------------------------------- utilities

    def normalized_ctrl(self, hinge_qpos: torch.Tensor) -> torch.Tensor:
        """Joint angles -> the ctrl that commands them, in [-1, 1] before clipping.

            a = 2*(q - lo)/(hi - lo) - 1

        This is EXACT, not an approximation: the actuators are
        <general biastype="affine"> position servos whose equilibrium angle is
        q* = -(gainprm[0]*ctrl + biasprm[0]) / biasprm[1], and solving that at
        ctrl = -+1 reproduces jnt_range to 4.4e-16 (measured on all 13 bodies).

        Used for two things: the feasibility penalty C in semantics.py (values
        outside [-1, 1] are exactly the frames MuJoCo would silently clamp --
        95.8% of reference frames have at least one) and the behaviour-cloning
        target in ppo.py.
        """
        return 2.0 * (hinge_qpos - self.hinge_lo) / (self.hinge_hi - self.hinge_lo) - 1.0

    def clamp_hinges(self, qpos: torch.Tensor) -> torch.Tensor:
        """Clamp qpos[..., 7:] into jnt_range, leaving the root untouched."""
        root, hinge = qpos[..., :7], qpos[..., 7:]
        return torch.cat([root, torch.clamp(hinge, self.hinge_lo, self.hinge_hi)], dim=-1)


def load_kinematics(xml_path, dtype: torch.dtype = torch.float64) -> TorchKinematics:
    return TorchKinematics(mujoco.MjModel.from_xml_path(str(xml_path)), dtype=dtype)
