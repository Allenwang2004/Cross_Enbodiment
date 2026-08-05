"""Test whether a body's actuator design (gainprm/biasprm/forcerange) is
physically CAPABLE of standing/walking under its own control -- decoupled
from the frozen Metamotivo actor, any z0, AND any human/mocap reference
motion (no external qpos trajectory is used anywhere in this script).

1. --mode static: at the model's own rest pose (qpos=0), compute the exact
   generalized force needed to counteract gravity+coriolis at zero
   acceleration (mujoco.mj_inverse), and compare it against each actuator's
   forcerange. Pure statics -- no controller, no dynamics, no confounds from
   a bad/mismatched control law. If forcerange doesn't even cover the static
   holding torque, no controller could ever make this body stand.

2. --mode stand: physically simulate (real mj_step) a gravity-compensation
   + PD controller trying to hold the body's OWN rest pose (whatever qpos
   mj_resetData gives it -- not a human motion capture frame) starting from
   that same pose. This answers "can this actuator design keep the body
   standing at all", the prerequisite for walking, without needing an actual
   gait (walking itself needs real trajectory synthesis/whole-body balance
   control, out of scope here -- this only tests standing).

   Control law: required_force[joint] = qfrc_bias[joint] (gravity+coriolis
   at the CURRENT state, from mj_forward -- NOT mj_inverse with a guessed
   floating-base target acceleration, which was tried first and is WRONG:
   forcing the unactuated free joint's acceleration to 0 in mj_inverse's
   solve doesn't correspond to any real achievable force, so the "required"
   torque it back-computed for the actuated joints was inconsistent with
   what actually happens once simulated forward -- qfrc_bias sidesteps this
   by only asking "what force cancels gravity/coriolis right now", which is
   well-defined regardless of the free joint) + Kp*(target-qpos) + Kd*(-qvel).

Usage (from project root):
    uv run scripts/torque_capability_check.py --mode static --xml assets/robots/child/robot.xml
    uv run scripts/torque_capability_check.py --mode static --xml assets/robots/adult/robot.xml

    uv run scripts/torque_capability_check.py --mode stand \
        --xml assets/robots/child/robot.xml --seconds 10 \
        --out outputs/torque_check/stand_child
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np


def build_actuator_maps(model):
    n = model.nu
    qpos_idx = np.zeros(n, dtype=int)
    dof_idx = np.zeros(n, dtype=int)
    gain = np.zeros(n)
    bias0 = np.zeros(n)
    bias1 = np.zeros(n)
    bias2 = np.zeros(n)
    ctrlrange = model.actuator_ctrlrange.copy()
    forcerange = model.actuator_forcerange.copy()
    names = []
    for i in range(n):
        joint_id = model.actuator_trnid[i, 0]
        qpos_idx[i] = model.jnt_qposadr[joint_id]
        dof_idx[i] = model.jnt_dofadr[joint_id]
        gain[i] = model.actuator_gainprm[i, 0]
        bias0[i] = model.actuator_biasprm[i, 0]
        bias1[i] = model.actuator_biasprm[i, 1]
        bias2[i] = model.actuator_biasprm[i, 2]
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
    return {
        "qpos_idx": qpos_idx, "dof_idx": dof_idx, "gain": gain,
        "bias0": bias0, "bias1": bias1, "bias2": bias2,
        "ctrlrange": ctrlrange, "forcerange": forcerange, "names": names,
    }


def static_check(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    act = build_actuator_maps(model)

    # mj_resetData (not manually zeroing qpos) -- the free joint's qpos[3:7]
    # is a quaternion, whose identity is (1,0,0,0), not all-zero; an
    # all-zero quat is degenerate and made mj_inverse return garbage
    # (hundreds of thousands of N*m -- physically impossible) the first
    # time this was tried.
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    required = data.qfrc_inverse[act["dof_idx"]]
    available = act["forcerange"][:, 1]  # symmetric, [:,0] is -available

    margin = available - np.abs(required)
    ratio = available / np.maximum(np.abs(required), 1e-6)

    print(f"{xml_path}: static gravity-hold torque check at qpos=0")
    print(f"{'actuator':16s} {'required (N*m)':>15s} {'available (N*m)':>17s} {'margin':>10s} {'headroom x':>12s}")
    n_insufficient = 0
    for i, name in enumerate(act["names"]):
        flag = ""
        if margin[i] < 0:
            flag = "  <-- INSUFFICIENT"
            n_insufficient += 1
        print(f"{name:16s} {required[i]:15.3f} {available[i]:17.3f} {margin[i]:10.3f} {ratio[i]:11.2f}x{flag}")

    print(f"\n{n_insufficient}/{model.nu} actuators cannot even statically hold qpos=0 against gravity.")
    print(f"worst headroom: {ratio.min():.2f}x ({act['names'][ratio.argmin()]})")
    return ratio


def gravity_comp_pd_ctrl(model, data, act, qpos_target, kp, kd):
    """One control step: gravity/coriolis compensation (qfrc_bias, valid
    for a floating base since it doesn't assume any achievable acceleration
    on the unactuated free joint -- unlike mj_inverse with a guessed target
    qacc there, see module docstring) + PD correction toward qpos_target,
    only on the actuated (hinge) dof.

    Equilibrium condition (qacc=0) is qfrc_actuator + qfrc_passive =
    qfrc_bias, NOT qfrc_actuator = qfrc_bias -- qfrc_bias is purely
    Coriolis+gravity, it does NOT include each joint's own passive
    stiffness/damping spring (the `stiffness=`/`damping=` on every <joint>
    in robot.xml, a separate qfrc_passive term). Omitting it here first made
    even the original, well-provisioned body fall in <1s, which shouldn't
    happen with ~10x static torque headroom -- this was the bug."""
    mujoco.mj_forward(model, data)  # refresh qfrc_bias/qfrc_passive for the CURRENT qpos/qvel
    hinge_qpos_idx = act["qpos_idx"]
    hinge_dof_idx = act["dof_idx"]

    pos_err = qpos_target[hinge_qpos_idx] - data.qpos[hinge_qpos_idx]
    vel_err = -data.qvel[hinge_dof_idx]
    required_force = (data.qfrc_bias[hinge_dof_idx] - data.qfrc_passive[hinge_dof_idx]
                       + kp * pos_err + kd * vel_err)

    ctrl = (required_force - act["bias0"] - act["bias1"] * data.qpos[hinge_qpos_idx]
            - act["bias2"] * data.qvel[hinge_dof_idx]) / act["gain"]
    return np.clip(ctrl, act["ctrlrange"][:, 0], act["ctrlrange"][:, 1])


def stand_check(xml_path, out_prefix, kp, kd, seconds, camera="front_side"):
    """Start at the body's own rest pose (mj_resetData -- no external
    reference motion involved) and try to hold it there with
    gravity_comp_pd_ctrl. steps = seconds * 30 (one control decision per
    rendered frame @ 30fps, matching the rest of this repo's convention)."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 1.0 / 450.0
    data = mujoco.MjData(model)
    act = build_actuator_maps(model)
    action_repeat = 15

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    qpos_target = data.qpos.copy()  # hold exactly where it starts
    initial_root_z = qpos_target[2]

    renderer = mujoco.Renderer(model, height=640, width=480)
    frames = []
    root_z_trace = []
    T = int(seconds * 30)

    for t in range(T):
        ctrl = gravity_comp_pd_ctrl(model, data, act, qpos_target, kp, kd)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data, nstep=action_repeat)

        root_z_trace.append(data.qpos[2])
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()

    root_z_trace = np.array(root_z_trace)
    out_path = Path(out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(f"{out_prefix}.mp4", frames, fps=30)

    height_ratio = root_z_trace / initial_root_z
    print(f"{xml_path}: standing balance check ({seconds}s, own rest pose only, no external reference)")
    print(f"wrote {T} frames -> {out_prefix}.mp4")
    print(f"initial root z: {initial_root_z:.4f}")
    print(f"root z / initial -- mean={height_ratio.mean():.2%} min={height_ratio.min():.2%} "
          f"final={height_ratio[-1]:.2%}")
    fell = height_ratio < 0.5
    if fell.any():
        t_fell = np.argmax(fell) / 30.0
        print(f"FELL at t={t_fell:.2f}s (root z dropped below 50% of initial)")
    else:
        print("stayed upright (root z never dropped below 50% of initial) for the full duration")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "stand"], required=True)
    parser.add_argument("--xml", default="assets/robots/child/robot.xml")
    parser.add_argument("--out", default=None, help="required for --mode stand: output path prefix")
    parser.add_argument("--seconds", type=float, default=10.0, help="--mode stand duration")
    parser.add_argument("--kp", type=float, default=300.0)
    parser.add_argument("--kd", type=float, default=35.0)
    parser.add_argument("--camera", default="front_side")
    args = parser.parse_args()

    if args.mode == "static":
        static_check(args.xml)
    else:
        if args.out is None:
            parser.error("--mode stand requires --out")
        stand_check(args.xml, args.out, args.kp, args.kd, args.seconds, args.camera)


if __name__ == "__main__":
    main()
