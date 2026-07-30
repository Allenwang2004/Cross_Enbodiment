"""Physically simulate (not just kinematically render) a retargeted motion:
convert the retargeted qpos trajectory into per-frame actuator ctrl signals
via closed-form PD-servo inversion, then actually step MuJoCo physics
(mj_step, with real mass/inertia/contacts/gravity) instead of just setting
qpos and calling mj_forward for rendering (see render_qpos_playback.py,
which never touches the actuator model at all).

Why: naive qpos retargeting (qpos_retarget.py) is purely kinematic -- it
never asks whether the target body's actuators can actually produce/hold
that pose. This script is a standalone test of exactly that question,
without any of the confounds of adapter/action_head training (REINFORCE
variance, task-sampling noise, etc.) -- if PD-tracking alone can't keep the
body upright, that's strong evidence the actuator gains (currently an exact,
unscaled copy from robot.xml -- see scale_robot.py, which never touches
<actuator>) are a real bottleneck independent of any learning problem.

Math: these are MuJoCo `biastype="affine"` actuators --
    force = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*qpos + biasprm[2]*qvel
biasprm[1] < 0 acts as a spring constant, biasprm[2] < 0 as damping -- i.e.
each actuator is already a PD position servo whose equilibrium (force=0) is
    qpos_eq = -(gainprm[0]*ctrl + biasprm[0]) / biasprm[1]
Inverting for ctrl given a target angle:
    ctrl = (-biasprm[1]*qpos_target - biasprm[0]) / gainprm[0]
lets us convert the retargeted trajectory's joint angles directly into the
ctrl signal that makes each joint's PD servo target that angle -- clipped to
ctrlrange, then mj_step'd for real (action_repeat=15 substeps @ dt=1/450s,
matching humenv's HumEnv.step so 1 iteration = 1 recorded frame @ 30fps).

Usage (from project root):
    uv run scripts/pd_track_retarget.py --qpos data/retargeted_motion/crawl-0.4-0-d_0.npz \
        --xml assets/robots/robot_child.xml --out outputs/pd_track/crawl-0.4-0-d_0
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np

ACTION_REPEAT = 15  # matches humenv.env.HumEnv.action_repeat
PHYSICS_DT = 1.0 / 450.0  # matches humenv.env.HumEnv's simulation_dt


def build_actuator_targets_fn(model):
    """Returns qpos_idx (per actuator, the qpos address of its joint) and
    the (gain, bias0, bias1) coefficients needed to invert each actuator's
    affine PD servo for a target angle."""
    n = model.nu
    qpos_idx = np.zeros(n, dtype=int)
    gain = np.zeros(n)
    bias0 = np.zeros(n)
    bias1 = np.zeros(n)
    ctrlrange = model.actuator_ctrlrange.copy()
    for i in range(n):
        assert model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE, \
            f"actuator {i} isn't biastype=affine -- inversion formula doesn't apply"
        joint_id = model.actuator_trnid[i, 0]
        qpos_idx[i] = model.jnt_qposadr[joint_id]
        gain[i] = model.actuator_gainprm[i, 0]
        bias0[i] = model.actuator_biasprm[i, 0]
        bias1[i] = model.actuator_biasprm[i, 1]
    return qpos_idx, gain, bias0, bias1, ctrlrange


def targets_to_ctrl(qpos_target_frame, qpos_idx, gain, bias0, bias1, ctrlrange):
    target_angles = qpos_target_frame[qpos_idx]
    ctrl = (-bias1 * target_angles - bias0) / gain
    return np.clip(ctrl, ctrlrange[:, 0], ctrlrange[:, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpos", required=True, help="retargeted qpos .npz (key 'qpos'), joint angles used as PD targets")
    parser.add_argument("--xml", default="assets/robots/robot_child.xml")
    parser.add_argument("--out", required=True, help="output path prefix (no extension)")
    parser.add_argument("--camera", default="front_side")
    args = parser.parse_args()

    qpos_target = np.load(args.qpos)["qpos"]
    T = qpos_target.shape[0]

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.timestep = PHYSICS_DT
    data = mujoco.MjData(model)

    qpos_idx, gain, bias0, bias1, ctrlrange = build_actuator_targets_fn(model)

    data.qpos[:] = qpos_target[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=640, width=480)
    frames = []
    qpos_actual = np.zeros_like(qpos_target)

    for t in range(T):
        ctrl = targets_to_ctrl(qpos_target[t], qpos_idx, gain, bias0, bias1, ctrlrange)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data, nstep=ACTION_REPEAT)

        qpos_actual[t] = data.qpos.copy()
        renderer.update_scene(data, camera=args.camera)
        frames.append(renderer.render().copy())
    renderer.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(f"{args.out}.mp4", frames, fps=30)
    np.savez(f"{args.out}_actual.npz", qpos=qpos_actual)

    root_target = qpos_target[:, 2]
    root_actual = qpos_actual[:, 2]
    print(f"wrote {T} frames -> {args.out}.mp4, {args.out}_actual.npz")
    print(f"root z (pelvis height) -- target: mean={root_target.mean():.4f} min={root_target.min():.4f} max={root_target.max():.4f}")
    print(f"root z (pelvis height) -- actual (PD-tracked physics): mean={root_actual.mean():.4f} min={root_actual.min():.4f} max={root_actual.max():.4f}")
    print(f"root z tracking error (RMSE): {np.sqrt(np.mean((root_target - root_actual) ** 2)):.4f}")
    pose_rmse = np.sqrt(np.mean((qpos_target[:, 7:] - qpos_actual[:, 7:]) ** 2))
    print(f"joint-angle tracking error (RMSE, rad): {pose_rmse:.4f}")


if __name__ == "__main__":
    main()
