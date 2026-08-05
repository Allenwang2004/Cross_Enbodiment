"""Foot-locked IK refinement on top of qpos_retarget.py's naive retargeting.

qpos_retarget.py copies joint angles unchanged and only rescales the root's
world position by a single scalar (source/target rest-pose height ratio).
That's exact for pure articulation shape (each limb chain is scaled
uniformly, so joint angles reproduce a geometrically similar, just smaller,
limb), but it does NOT preserve stance-phase foot placement: the SAME joint
angles at the SAME timesteps, combined with a differently-scaled root
trajectory, make the foot drift/slide during what should be a planted
stance phase. Measured on move-ego--90-2: naive retargeting increased mean
foot-slide-per-frame during nominal stance by ~5x versus the source motion
(0.0039 -> 0.0191 m/frame for the trailing foot) -- see the analysis this
script's PR/conversation is based on.

This script fixes that specific failure mode (footskate), not general
retargeting accuracy:
  1. Detect each foot's stance intervals from the ORIGIN motion (source
     skeleton) -- low height + low horizontal velocity.
  2. Naively retarget qpos (delegates to qpos_retarget.retarget_qpos).
  3. For each stance interval, lock that foot's target-skeleton world
     position to where it naively lands at the interval's first frame.
  4. For every frame in the interval, solve a damped-least-squares IK over
     that leg's 9 hip/knee/ankle DOF (minimal-change from the naive angles,
     per the DLS regularizer) to pull the foot back onto the locked point.
  5. Crossfade a few frames at interval boundaries so corrected and
     uncorrected angles don't snap.
Swing-phase frames are left as the naive copy -- footskate only matters
while a foot is supposed to be stationary on the ground.

This still isn't full biomechanical retargeting (stride timing/frequency
isn't Froude-rescaled, only within-stance drift is removed) -- it targets
the specific, measurable defect found in practice.

Usage (from project root):
    python3 scripts/qpos_retarget_ik.py --input_npz <robot_clip.npz> --output_npz <out.npz> \
        --source_xml assets/robots/adult/robot.xml --target_xml assets/robots/child/robot.xml \
        --source_skeleton_json assets/robots/adult/skeleton.json \
        --target_skeleton_json assets/robots/child/skeleton.json

    # batch mode:
    python3 scripts/qpos_retarget_ik.py --input_dir data/origin_motion --output_dir data/retargeted_motion_ik \
        --source_xml assets/robots/adult/robot.xml --target_xml assets/robots/child/robot.xml
"""

import argparse
import itertools
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from qpos_retarget import load_root_rest_height, retarget_qpos

FOOT_BODIES = ["L_Toe", "R_Toe"]
LEG_JOINTS = {
    "L_Toe": ["L_Hip_x", "L_Hip_y", "L_Hip_z", "L_Knee_x", "L_Knee_y", "L_Knee_z",
              "L_Ankle_x", "L_Ankle_y", "L_Ankle_z"],
    "R_Toe": ["R_Hip_x", "R_Hip_y", "R_Hip_z", "R_Knee_x", "R_Knee_y", "R_Knee_z",
              "R_Ankle_x", "R_Ankle_y", "R_Ankle_z"],
}


def foot_world_positions(model, qpos_seq, body_name):
    data = mujoco.MjData(model)
    body_id = model.body(body_name).id
    out = np.zeros((qpos_seq.shape[0], 3))
    for t in range(qpos_seq.shape[0]):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_kinematics(model, data)
        out[t] = data.xpos[body_id]
    return out


def detect_stance_intervals(model, qpos_seq, body_name, height_margin=0.015, speed_thresh=0.01):
    """Frames where `body_name` is near its own per-clip minimum height and
    not moving horizontally much -> merged into contiguous [start, end)
    intervals."""
    pos = foot_world_positions(model, qpos_seq, body_name)
    z = pos[:, 2]
    low = z < (z.min() + height_margin)

    xy_speed = np.zeros(len(pos))
    xy_speed[1:] = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    still = xy_speed < speed_thresh

    stance = low & still
    intervals = []
    start = None
    for t, s in enumerate(stance):
        if s and start is None:
            start = t
        elif not s and start is not None:
            intervals.append((start, t))
            start = None
    if start is not None:
        intervals.append((start, len(stance)))
    return [iv for iv in intervals if iv[1] - iv[0] >= 2]


def solve_leg_ik(model, data, qpos, foot_body_id, target_xyz, dof_indices, qpos_indices,
                  max_iters=50, tol=1e-5, damping=1e-2):
    q = qpos.copy()
    for _ in range(max_iters):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)  # mj_jac needs this in addition to mj_kinematics
        cur = data.xpos[foot_body_id].copy()
        err = target_xyz - cur
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, cur, foot_body_id)
        J = jacp[:, dof_indices]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        q[qpos_indices] += dq
    return q


def foot_lock_refine(model, qpos_naive, model_orig, qpos_orig):
    """qpos_naive: (T, nq) already naively retargeted onto `model`.
    Returns a corrected copy with stance-phase footskate removed."""
    data = mujoco.MjData(model)
    corrected = qpos_naive.copy()

    for foot_body in FOOT_BODIES:
        intervals = detect_stance_intervals(model_orig, qpos_orig, foot_body)
        if not intervals:
            continue

        foot_body_id = model.body(foot_body).id
        joint_names = LEG_JOINTS[foot_body]
        dof_indices = np.array([model.joint(n).dofadr[0] for n in joint_names])
        qpos_indices = np.array([model.joint(n).qposadr[0] for n in joint_names])

        naive_pos = foot_world_positions(model, qpos_naive, foot_body)

        for k, (start, end) in enumerate(intervals):
            target_xyz = naive_pos[start].copy()  # lock to first-frame landing spot

            for t in range(start, end):
                corrected[t] = solve_leg_ik(
                    model, data, corrected[t], foot_body_id, target_xyz,
                    dof_indices, qpos_indices,
                )

            # Fade a few frames at each boundary by interpolating the IK
            # *target position* (not the resulting joint angles) between the
            # naive foot position and the locked target, then re-solving IK
            # per frame -- interpolating joint angles directly would
            # overshoot in Cartesian space (the angle->position map is
            # nonlinear over a 9-DOF chain), which is what caused the
            # transition frames to slide *more* than either endpoint in
            # testing. Capped by the gap to the neighboring interval (and
            # halved, since both intervals fade into the same gap) so
            # adjacent fades never overlap and double-write frames.
            prev_end = intervals[k - 1][1] if k > 0 else 0
            next_start = intervals[k + 1][0] if k + 1 < len(intervals) else len(corrected)
            blend_before = max(0, min(3, (start - prev_end) // 2))
            blend_after = max(0, min(3, (next_start - end) // 2))

            for i, t in enumerate(range(max(0, start - blend_before), start)):
                w = (i + 1) / (blend_before + 1)  # 0 (far from start) -> 1 (at start)
                fade_target = (1 - w) * naive_pos[t] + w * target_xyz
                corrected[t] = solve_leg_ik(
                    model, data, qpos_naive[t], foot_body_id, fade_target,
                    dof_indices, qpos_indices,
                )
            for i, t in enumerate(range(end, min(len(corrected), end + blend_after))):
                w = 1 - (i + 1) / (blend_after + 1)  # 1 (at end) -> 0 (far from end)
                fade_target = (1 - w) * naive_pos[t] + w * target_xyz
                corrected[t] = solve_leg_ik(
                    model, data, qpos_naive[t], foot_body_id, fade_target,
                    dof_indices, qpos_indices,
                )

    return corrected


def retarget_one(qpos_orig, model_orig, model_target, scale):
    naive = retarget_qpos(qpos_orig, scale)
    return foot_lock_refine(model_target, naive, model_orig, qpos_orig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=Path)
    parser.add_argument("--output_npz", type=Path)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--source_xml", default="assets/robots/adult/robot.xml")
    parser.add_argument("--target_xml", default="assets/robots/child/robot.xml")
    parser.add_argument("--source_skeleton_json", default="assets/robots/adult/skeleton.json")
    parser.add_argument("--target_skeleton_json", default="assets/robots/child/skeleton.json")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    if bool(args.input_npz) == bool(args.input_dir):
        raise SystemExit("Pass exactly one of --input_npz/--output_npz or --input_dir/--output_dir")

    source_h = load_root_rest_height(args.source_skeleton_json)
    target_h = load_root_rest_height(args.target_skeleton_json)
    scale = target_h / source_h
    print(f"Root height scale ({args.source_skeleton_json} -> {args.target_skeleton_json}): {scale:.4f}")

    model_orig = mujoco.MjModel.from_xml_path(str(args.source_xml))
    model_target = mujoco.MjModel.from_xml_path(str(args.target_xml))

    if args.input_npz:
        qpos_orig = np.load(args.input_npz)["qpos"]
        out = retarget_one(qpos_orig, model_orig, model_target, scale)
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_npz, qpos=out, fps=args.fps)
        print(f"Wrote {out.shape} -> {args.output_npz}")
    else:
        npz_paths = sorted(args.input_dir.rglob("*.npz"))
        print(f"Found {len(npz_paths)} .npz files under {args.input_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for i, npz_path in enumerate(npz_paths):
            qpos_orig = np.load(npz_path)["qpos"]
            out = retarget_one(qpos_orig, model_orig, model_target, scale)
            out_path = args.output_dir / f"{npz_path.stem}.npz"
            np.savez(out_path, qpos=out, fps=args.fps)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(npz_paths)}")
        print(f"Wrote {len(npz_paths)} retargeted qpos files -> {args.output_dir}")


if __name__ == "__main__":
    main()
