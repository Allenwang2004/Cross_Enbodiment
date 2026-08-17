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
        --out outputs/torque_capability_check/stand_child
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.qpos_retarget import ground_correct_qpos  # noqa: E402


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


def phi0_root(qpos, model, adult_xml="assets/robots/adult/robot.xml"):
    """Retarget the reference root to this body: xyz scale, then ground contact.

    The xyz scale alone is NOT enough. Scaling the root by the rest-pelvis
    height ratio leaves the feet wherever they land -- typically above the floor
    -- so the body starts the episode in free fall and slides on touchdown, which
    looks like (and gets mistaken for) an actuator failure. The retargeting
    pipeline already solves this: scripts/qpos_retarget.py runs
    ground_correct_qpos after the scale, which is why data/retargeting_motion has
    a different z ratio (0.6187) from its xy ratio (0.6110). Same correction is
    applied here so the replay starts on the ground.

    Hinge angles are copied unchanged, so the action stream stays valid: jnt_range
    is identical across these bodies, hence the same ctrl commands the same
    target angle on each.
    """
    def rest_h(m):
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        return float(d.qpos[2])

    q = np.array(qpos, dtype=float, copy=True)
    h_src = rest_h(mujoco.MjModel.from_xml_path(adult_xml))
    if h_src:
        q[:, :3] *= rest_h(model) / h_src

    q, n_fixed, max_shift = ground_correct_qpos(model, q)
    if n_fixed:
        print(f"    ground correction: {n_fixed}/{len(q)} frames lifted, "
              f"max +{max_shift * 1000:.1f} mm")
    return q


def replay_actions(xml_path, action, qpos_ref, qvel_ref=None, *, root="pinned",
                   physics_dt=1.0 / 450.0, action_repeat=15):
    """Drive one body with a recorded ctrl stream and record what its actuators do.

    Why this beats inverse dynamics for a torque comparison: `action` is the
    ctrl that actually ran, so nothing here depends on a qacc differenced from
    30 Hz samples -- the term that was shown to be the whole error in the
    inverse-dynamics route. The torques read back are real forward-simulation
    actuator forces, already clamped by forcerange exactly as MuJoCo would.

    root:
      "pinned" — overwrite the free joint from the (retargeted) reference each
                 frame. Balance is assumed solved, so the bodies stay upright
                 and their torques stay comparable for the whole clip. This is
                 the default because an open-loop child falls within a second
                 and everything after that measures the fall, not the actuators.
      "free"   — nothing overwritten. Realistic, and the divergence is itself
                 the answer to "can this body take the adult's actions".

    NOTE the state is NOT resynced per frame. Doing that would make the
    comparison degenerate: with jnt_range identical across bodies, the same ctrl
    at the same pose gives an actuator force that depends only on the actuator
    parameters, so an unscaled child would report torques identical to the
    adult's by construction. The bodies have to actually diverge for the
    question to mean anything.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    act = build_actuator_maps(model)
    T = len(action)

    ref = phi0_root(qpos_ref, model)

    # Prefer the RECORDED velocities. Differencing them out of qpos is not a
    # harmless approximation here: this clip starts from rest (qvel_init is all
    # zeros) while the finite difference reads 6.4 rad/s, so a differenced start
    # kicks the body at frame 0 with a velocity that never existed -- enough to
    # bring the adult down at frame 81 instead of tracking its own recording
    # exactly for 150 frames. Only the root's linear velocity needs retargeting
    # (same xyz scale as the pose); joint velocities carry over unchanged
    # because the hinge angles do.
    dt_frame = physics_dt * action_repeat
    if qvel_ref is not None:
        rvel = np.array(qvel_ref[:T], dtype=float, copy=True)
        scale = ref[0, 2] / qpos_ref[0, 2] if qpos_ref[0, 2] else 1.0
        rvel[:, :3] *= scale
    else:
        rvel = np.zeros((T, model.nv))
        for t in range(min(T, len(ref) - 1)):
            mujoco.mj_differentiatePos(model, rvel[t], dt_frame,
                                       np.ascontiguousarray(ref[t]),
                                       np.ascontiguousarray(ref[t + 1]))

    limit = np.minimum(np.abs(act["forcerange"][:, 0]),
                       np.abs(act["forcerange"][:, 1]))
    force = np.zeros((T, model.nu))
    qout = np.zeros((T, model.nq))
    fell_at = -1

    saved = model.opt.timestep
    model.opt.timestep = physics_dt
    try:
        mujoco.mj_resetData(model, data)
        data.qpos[:] = ref[0]
        data.qvel[:] = rvel[0]
        mujoco.mj_forward(model, data)

        for t in range(T):
            if root == "pinned":
                data.qpos[:7] = ref[t, :7]
                data.qvel[:6] = rvel[t, :6]
            data.ctrl[:] = action[t]
            mujoco.mj_step(model, data, nstep=action_repeat)

            if not np.all(np.isfinite(data.qpos)):
                fell_at = t
                force[t:] = force[t - 1] if t else 0.0
                qout[t:] = qout[t - 1] if t else ref[0]
                break
            force[t] = data.actuator_force
            qout[t] = data.qpos
            # "fell" = pelvis below half its rest height; only meaningful when
            # the root is free, since pinning holds it up by construction.
            if root == "free" and fell_at < 0 and data.qpos[2] < 0.5 * ref[0, 2]:
                fell_at = t
    finally:
        model.opt.timestep = saved

    absf = np.abs(force)
    with np.errstate(invalid="ignore", divide="ignore"):
        util = np.where(limit > 0, absf / limit, np.nan)
    n = min(T, len(ref) - 1)
    return {
        "names": act["names"], "force": force, "util": util, "limit": limit,
        "peak": absf.max(axis=0), "sat_rate": (absf >= 0.98 * limit).mean(axis=0),
        "fell_at": fell_at, "qpos": qout, "mass": float(model.body_mass.sum()),
        "joint_err": float(np.abs(qout[:n, 7:] - ref[1:n + 1, 7:]).mean()),
    }


def render_bodies(xmls, qpos_seqs, out_path, *, fps=30.0, camera="front_side",
                  width=480, height=640):
    """One mp4, the bodies side by side, each rendered with its OWN model."""
    per_body = []
    for xml, seq in zip(xmls, qpos_seqs):
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
        frames = []
        for t in range(len(seq)):
            data.qpos[:] = seq[t]
            mujoco.mj_forward(model, data)
            try:
                renderer.update_scene(data, camera=camera)
            except Exception:
                renderer.update_scene(data)
            frames.append(renderer.render().copy())
        renderer.close()
        per_body.append(frames)

    n = min(len(f) for f in per_body)
    out = [np.concatenate([f[i] for f in per_body], axis=1) for i in range(n)]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, out, fps=fps)
    print(f"video -> {out_path}  ({n} frames, "
          f"{' | '.join(Path(x).parent.name for x in xmls)})")


def replay_check(action_npz, xmls, root, out_csv=None, video=None):
    blob = np.load(action_npz, allow_pickle=True)
    if "action" not in blob.files:
        raise SystemExit(f"{action_npz} has no 'action' key")
    action = blob["action"]
    qpos_ref = np.vstack([blob["qpos_init"], blob["qpos"]])
    qvel_ref = (np.vstack([blob["qvel_init"], blob["qvel"]])
                if "qvel" in blob.files and "qvel_init" in blob.files else None)

    print(f"action stream : {action_npz}")
    print(f"  {len(action)} frames x {action.shape[1]} actuators, "
          f"{100 * (np.abs(action) >= 0.999).mean():.1f}% of entries at ±1")
    print(f"  root {root}   (pinned = free joint driven by the retargeted reference)")
    print(f"  velocities: {'recorded' if qvel_ref is not None else 'DIFFERENCED (no qvel in file)'}\n")

    results = {}
    for xml in xmls:
        r = replay_actions(xml, action, qpos_ref, qvel_ref, root=root)
        results[xml] = r
        fell = ("survived" if r["fell_at"] < 0 else f"fell/diverged @ frame {r['fell_at']}")
        print(f"{Path(xml).parent.name}/{Path(xml).name}   ({r['mass']:.1f} kg)")
        print(f"  peak torque   : median {np.median(r['peak']):7.2f}   "
              f"max {r['peak'].max():8.2f} N·m")
        print(f"  force util    : mean {np.nanmean(r['util']):.1%}   "
              f"p99 {np.nanpercentile(r['util'], 99):.1%}   "
              f"max {np.nanmax(r['util']):.1%}")
        print(f"  saturation    : mean {r['sat_rate'].mean():.2%}   "
              f"worst joint {r['sat_rate'].max():.1%} of frames")
        print(f"  joint tracking: {r['joint_err']:.4f} rad     {fell}\n")

    xs = list(results)
    if len(xs) >= 2:
        base = results[xs[0]]
        print(f"=== per-joint peak-torque ratio vs {Path(xs[0]).parent.name} ===")
        for x in xs[1:]:
            rat = results[x]["peak"] / np.maximum(base["peak"], 1e-9)
            print(f"{Path(x).parent.name:>14}: median {np.median(rat):.3f}   "
                  f"IQR {np.percentile(rat, 25):.3f}–{np.percentile(rat, 75):.3f}")

    if video:
        render_bodies(xmls, [results[x]["qpos"] for x in results], video)

    if out_csv:
        import csv as _csv
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["actuator"] + [f"{k}_{Path(x).parent.name}"
                                       for x in xs for k in ("peak", "sat", "limit")])
            for i, nm in enumerate(results[xs[0]]["names"]):
                row = [nm]
                for x in xs:
                    r = results[x]
                    row += [f"{r['peak'][i]:.4f}", f"{r['sat_rate'][i]:.4f}",
                            f"{r['limit'][i]:.4f}"]
                w.writerow(row)
        print(f"\nper-joint table -> {out_csv}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "stand", "replay"], required=True)
    parser.add_argument("--xml", default="assets/robots/child/robot.xml")
    parser.add_argument("--out", default=None, help="required for --mode stand: output path prefix")
    parser.add_argument("--seconds", type=float, default=10.0, help="--mode stand duration")
    parser.add_argument("--kp", type=float, default=300.0)
    parser.add_argument("--kd", type=float, default=35.0)
    parser.add_argument("--camera", default="front_side")
    parser.add_argument("--action", default="data/origin_action/move-ego-0-0/move-ego-0-0_0.npz",
                        help="--mode replay: recorded ctrl stream to run on every body")
    parser.add_argument("--xmls", nargs="+",
                        default=["assets/robots/adult/robot.xml",
                                 "assets/robots/child/robot.xml",
                                 "assets/robot_torque/robot_torque.xml"],
                        help="--mode replay: the bodies to run it on")
    parser.add_argument("--video", default=None,
                        help="--mode replay: write a side-by-side mp4 of the bodies")
    parser.add_argument("--root", choices=["pinned", "free"], default="pinned",
                        help="--mode replay: whether the free joint follows the reference")
    args = parser.parse_args()

    if args.mode == "static":
        static_check(args.xml)
    elif args.mode == "replay":
        replay_check(args.action, args.xmls, args.root, out_csv=args.out,
                     video=args.video)
    else:
        if args.out is None:
            parser.error("--mode stand requires --out")
        stand_check(args.xml, args.out, args.kp, args.kd, args.seconds, args.camera)


if __name__ == "__main__":
    main()
