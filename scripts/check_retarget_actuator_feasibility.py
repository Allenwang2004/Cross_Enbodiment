"""Check whether a retargeted reference motion (data/retargeted_motion, the
qpos_ref used as the D-loss target in model/losses.py) is even DYNAMICALLY
possible for the target body's actuators -- retargeting itself
(qpos_retarget.py/qpos_retarget_ik.py) is purely kinematic (mj_kinematics/
mj_forward only, never touches <actuator> or mj_step, see that script's
docstring), so nothing in the pipeline has ever verified that a retargeted
clip's implied accelerations are within the target skeleton's actuator
capability. If a clip requires more torque than robot_child.xml's
actuators can produce at some frame, D can never be driven to zero no
matter how well-trained the adapter/policy is -- the reference itself would
be asking for something physically impossible.

Method: per-frame inverse dynamics (mujoco.mj_inverse), NOT just a static
qpos=0 check (see torque_capability_check.py's --mode static) -- uses the
qvel/qacc IMPLIED BY THE RECORDED TRAJECTORY ITSELF (via
mujoco.mj_differentiatePos, which correctly handles the free joint's
quaternion -> angular-velocity conversion; a naive qpos[t+1]-qpos[t] finite
difference is invalid there since qpos's quaternion is 4-dim but qvel's
angular part is 3-dim), not an artificially-imposed target acceleration.
This sidesteps the floating-base mj_inverse pitfall found earlier in
torque_capability_check.py's --mode track (forcing an UNMEASURED free-joint
acceleration to 0 gives an inconsistent solve) -- here the free joint's
acceleration is whatever the recorded motion actually shows, so mj_inverse's
solve is self-consistent: "what force would the actuated joints need to
reproduce this exact recorded motion".

Usage (from project root):
    uv run scripts/check_retarget_actuator_feasibility.py
    uv run scripts/check_retarget_actuator_feasibility.py --limit 20
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def build_actuator_maps(model):
    n = model.nu
    dof_idx = np.zeros(n, dtype=int)
    forcerange = model.actuator_forcerange.copy()
    names = []
    for i in range(n):
        joint_id = model.actuator_trnid[i, 0]
        dof_idx[i] = model.jnt_dofadr[joint_id]
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
    return dof_idx, forcerange, names


def _smooth(x, window):
    """Simple centered moving-average filter along axis 0. Safe to apply to
    qvel/qacc (already in the flat tangent/dof space -- unlike qpos, whose
    quaternion component can't be box-filtered directly)."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded)[: x.shape[0]]


def compute_qvel_qacc(model, qpos_seq, dt, smooth_window=5):
    """qvel via mj_differentiatePos (correctly handles the free joint's
    quaternion -> angular-velocity conversion). qacc via differencing qvel
    -- but double-differentiating a 30fps-sampled trajectory hugely amplifies
    any high-frequency content (verified: even ORIGIN motion, which really
    did come from stepping this exact model's real physics via env.step(),
    showed ~34% of frames "infeasible" without smoothing -- a real physical
    trajectory can't be infeasible for the body it was recorded on, so that
    was numerical noise, not a real signal). Smoothing qvel before the
    second difference (not qpos directly, to avoid mangling the quaternion)
    trades away genuine brief transients (e.g. the instant of foot-ground
    impact) for a much lower false-positive rate on sustained/systematic
    infeasibility, which is what this check is actually trying to find."""
    T = qpos_seq.shape[0]
    qvel = np.zeros((T, model.nv))
    for t in range(T - 1):
        mujoco.mj_differentiatePos(model, qvel[t], dt, qpos_seq[t], qpos_seq[t + 1])
    if T > 1:
        qvel[-1] = qvel[-2]
    qvel = _smooth(qvel, smooth_window)
    qacc = np.zeros((T, model.nv))
    if T > 1:
        qacc[:-1] = np.diff(qvel, axis=0) / dt
        qacc[-1] = qacc[-2]
    return qvel, qacc


def check_clip(model, data, dof_idx, forcerange, qpos_seq, dt, smooth_window=5):
    qvel_seq, qacc_seq = compute_qvel_qacc(model, qpos_seq, dt, smooth_window)
    T = qpos_seq.shape[0]
    headroom = np.zeros((T, len(dof_idx)))

    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        data.qvel[:] = qvel_seq[t]
        data.qacc[:] = qacc_seq[t]
        mujoco.mj_inverse(model, data)
        required = np.abs(data.qfrc_inverse[dof_idx])
        available = forcerange[:, 1]
        headroom[t] = available / np.maximum(required, 1e-6)

    return headroom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retarget-dir", default="data/retargeted_motion")
    parser.add_argument("--xml", default="assets/robots/child/robot.xml")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--infeasible-threshold", type=float, default=1.0,
                         help="headroom below this counts as infeasible (1.0 = exactly at the limit)")
    parser.add_argument("--limit", type=int, default=None, help="only check the first N clips (sorted)")
    parser.add_argument("--smooth-window", type=int, default=5,
                         help="moving-average window (frames) applied to qvel before "
                              "the second finite difference for qacc -- see compute_qvel_qacc")
    parser.add_argument("--out", default="outputs/retarget_actuator_feasibility.csv")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    dof_idx, forcerange, names = build_actuator_maps(model)
    dt = 1.0 / args.fps

    clip_paths = sorted(Path(args.retarget_dir).glob("*.npz"))
    if args.limit:
        clip_paths = clip_paths[: args.limit]

    rows = []
    worst_joint_count = defaultdict(int)

    for i, clip_path in enumerate(clip_paths):
        task = clip_path.stem.rsplit("_", 1)[0]
        qpos_seq = np.load(clip_path)["qpos"]
        headroom = check_clip(model, data, dof_idx, forcerange, qpos_seq, dt, args.smooth_window)

        infeasible_mask = headroom < args.infeasible_threshold
        frac_frames_infeasible = float(infeasible_mask.any(axis=1).mean())
        worst_headroom = float(headroom.min())
        worst_joint = names[int(headroom.min(axis=0).argmin())]
        worst_joint_count[worst_joint] += 1

        rows.append({
            "task": task, "clip": clip_path.stem,
            "frac_frames_infeasible": frac_frames_infeasible,
            "worst_headroom": worst_headroom, "worst_joint": worst_joint,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(clip_paths)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")

    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    task_means = sorted(
        ((task, np.mean([r["frac_frames_infeasible"] for r in trs])) for task, trs in by_task.items()),
        key=lambda x: -x[1],
    )
    print(f"\n=== worst 15 tasks by mean fraction-of-frames-infeasible (n={len(by_task)} tasks) ===")
    for task, frac in task_means[:15]:
        print(f"  {task:30s} {frac:.1%}")

    all_frac = np.array([r["frac_frames_infeasible"] for r in rows])
    all_worst = np.array([r["worst_headroom"] for r in rows])
    print(f"\n=== overall ===")
    print(f"clips with ANY infeasible frame: {(all_frac > 0).mean():.1%}")
    print(f"mean fraction of frames infeasible (over clips with any): "
          f"{all_frac[all_frac > 0].mean():.1%}" if (all_frac > 0).any() else "n/a")
    print(f"worst-case headroom overall: {all_worst.min():.4f}x")
    print(f"\nmost frequent bottleneck joint (worst headroom) across clips:")
    for joint, count in sorted(worst_joint_count.items(), key=lambda x: -x[1])[:10]:
        print(f"  {joint:16s} {count} clips")


if __name__ == "__main__":
    main()
