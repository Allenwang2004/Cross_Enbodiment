"""Drive a robot MJCF (position-actuated: `<general>` actuators whose
gainprm/biasprm already encode a per-joint PD relative to `ctrl`) with REAL
physics (mj_step: gravity, contacts, actuator forces) while tracking a qpos
trajectory produced by the retargeting pipeline (see `qpos_retarget_simple.py`).

Unlike `test_qpos_physics_playback.py` (which targets `mjcf/humanoid.xml`'s
raw `motor` actuators and needs a hand-rolled PD loop), these `<general>`
actuators are already position servos: for actuator i,
    force = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*qpos + biasprm[2]*qvel
so MuJoCo applies the qpos/qvel feedback internally each step -- we only need
to solve for the `ctrl` that makes this affine relation zero force at the
reference qpos (i.e. the quasi-static target), each frame:
    ctrl = -(biasprm[0] + biasprm[1]*target_qpos) / gainprm[0]

The free (root) joint has no actuator, same as test_qpos_physics_playback.py
-- its motion is whatever gravity/contacts produce, so the reference root
pose is only a target the body may fail to track (e.g. a headstand can
topple).

Run with:
    .venv/bin/python scripts/test_qpos_physics_playback_robot.py \
        --mjcf mjcf/robot_child.xml \
        --qpos-npz robot_child+headstand_0_Anim.npz \
        --output renders/robot_child+headstand_0_Anim_physics_playback.mp4
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np


def actuator_addresses(model):
    """Map each actuator to the qpos index of the single hinge joint it drives."""
    jnt_ids = model.actuator_trnid[:, 0]
    return model.jnt_qposadr[jnt_ids]


def target_ctrl(model, qpos_adr, target_qpos):
    """Invert each actuator's affine force law to find the ctrl that gives
    zero force at (target qpos, zero velocity) -- the quasi-static ctrl for
    that target, using the joint's own baked-in gain/bias."""
    target = target_qpos[qpos_adr]
    gain = model.actuator_gainprm[:, 0]
    bias0 = model.actuator_biasprm[:, 0]
    bias1 = model.actuator_biasprm[:, 1]
    ctrl = -(bias0 + bias1 * target) / gain
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    return np.clip(ctrl, lo, hi)


def execute(mjcf_path, qpos_npz_path, output_path, width, height, azimuth, default_fps):
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    camera = mujoco.MjvCamera()
    if Path(mjcf_path).stem.startswith("robot"):
        # robot*.xml has no <statistic> extent, so the default free camera
        # auto-scales to its 100x100 floor plane and renders the robot as a
        # speck. Frame it manually instead of touching the model file (see
        # render_qpos.py).
        camera.lookat = [0, -0.2, 0.9]
        camera.distance = 3.0
        camera.azimuth = 90
        camera.elevation = -10
    else:
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.azimuth = azimuth

    npz = np.load(qpos_npz_path)
    ref_qpos = npz["qpos"]
    # Raw (non-retargeted) robotmotion .npz files carry no fps of their own,
    # same as qpos_retarget_simple.py's --fps.
    fps = float(npz["fps"]) if "fps" in npz else default_fps
    n_frames = len(ref_qpos)
    duration = n_frames / fps

    qpos_adr = actuator_addresses(model)

    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    frame_interval = 1.0 / fps
    next_frame_time = 0.0

    with imageio.get_writer(output_path, fps=fps) as writer:
        while data.time < duration:
            frame_idx = min(int(data.time * fps), n_frames - 1)
            data.ctrl[:] = target_ctrl(model, qpos_adr, ref_qpos[frame_idx])
            mujoco.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)):
                raise RuntimeError(f"unstable (NaN) at t={data.time:.3f}s")
            if data.time >= next_frame_time:
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
                next_frame_time += frame_interval

    print(f"wrote {output_path}: {duration:.2f}s @ {fps}fps, "
          f"final root height = {data.qpos[2]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", default="mjcf/robot_child.xml")
    parser.add_argument("--qpos-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--azimuth",
        type=float,
        default=-90,
        help="Camera azimuth in degrees for non-robot models (ignored for "
        "robot*.xml, which uses a fixed manual camera).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Used only if the qpos .npz has no 'fps' field of its own "
        "(e.g. raw robotmotion clips, same convention as qpos_retarget_simple.py).",
    )
    args = parser.parse_args()
    execute(args.mjcf, args.qpos_npz, args.output, args.width, args.height, args.azimuth, args.fps)
