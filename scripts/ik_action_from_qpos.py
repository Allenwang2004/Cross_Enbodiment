#!/usr/bin/env python3
"""ik_action_from_qpos.py — recover the ctrl stream that commands a pose-only
clip, so a qpos trajectory can be fed to anything that expects an action record.

The gap this fills
------------------
data/retargeting_motion stores `qpos` and `fps` and nothing else. Everything
downstream that reasons about a policy -- ik_z_from_action.py above all -- needs
`action` (plus the state each action was chosen from). There is no recorded
action for a retargeted clip, because no policy ever produced it: the clip is
data/origin_motion with the root rescaled, not a rollout. So the action has to
be solved for, and this file solves it.

Why this is closed form and not an optimisation
-----------------------------------------------
Every actuator on every body in this repo is an affine position servo
(gaintype=fixed, biastype=affine), so its generalised force is

    f = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*(gear*q) + biasprm[2]*(gear*qdot)

and the angle it holds at equilibrium is a bijection in ctrl. Inverting it,

    ctrl(q) = -(biasprm[0] + biasprm[1]*gear*q) / gainprm[0]

is the exact ctrl that COMMANDS angle q -- one line, no solver, no residual.
This is the same identity pd_track_bodies.py drives its servo with, stated there
in jnt_range form; the two agree to 4.4e-16 on adult, child and robot_torque
alike, and this file asserts that agreement per actuator rather than assuming
it (--range-tol). The jnt_range form is the one to quote; the coefficient form
above is the one to compute with, because it stays correct if a rescaled MJCF
ever moves the servo's neutral angle away from the joint's midpoint.

The t+1 shift is not cosmetic
-----------------------------
metamotivo_motion_rollout.py:rollout_once appends qpos AFTER env.step, so in
data/origin_action `qpos[t]` is the RESULT of `action[t]`, and `action[0]` was
chosen from the separately stored `qpos_init`. ik_z_from_action.py:acting_states
rebuilds that pairing and would silently shift a whole clip by one control step
if the record disagreed. So the record written here uses the same convention:

    action[t]  = ctrl commanding q_ref[t+1]      <- targets the next frame
    qpos[t]    = q_ref[t+1]                      <- the state it produces
    qpos_init  = q_ref[0]                        <- the state action[0] acts from

which costs one frame: a T-frame clip becomes a T-1-step record.

qvel is reconstructed, and it is the weak link
----------------------------------------------
The record needs velocities -- ik_z_from_action.py's docstring is explicit that
144 of the 358 obs features are velocities and that faking them poisons the obs
BatchNorm -- but a pose-only clip does not carry them. They are recovered by
central differencing through mj_differentiatePos (which handles the root
quaternion; plain np.diff on qpos does not). Measured against the true qvel
recorded alongside the same poses in data/origin_action:

    clip              root_lin corr   joint corr   joint rel.err
    move-ego-0-2_0        0.999          0.946         0.30
    jump-2_0              0.998          0.875         0.40
    crouch-0_0              --           0.801         0.46

The root is essentially exact; the joints are not, because qpos is sampled once
per control step (15 physics steps at 1/450 s) while the servo oscillates
within it, so the interval-average difference is not the instantaneous velocity
at the endpoint. Nothing here can fix that -- the information is not in the file
-- but it is the accuracy ceiling for anything computed from these obs, and it
should be quoted rather than discovered later.

Saturation is a property of the reference, not of the inverse
-------------------------------------------------------------
ctrl in [-1, 1] spans exactly jnt_range, so an angle outside jnt_range has no
ctrl at all and gets clipped to the limit. That happens, and not rarely: MuJoCo
joint limits are soft constraints, so the source rollout penetrates them and the
recorded qpos leaves the range on 3.5% (move-ego-0-2) to 11.6% (crawl-0.4-0-d)
of joint-frames. The clipped cells concentrate on the joints with the tightest
ranges -- R_Toe_z is +-0.08 rad, so a 0.02 rad overshoot already reads as
|ctrl| 1.25 -- which is why the report quotes the overshoot in RADIANS as well
as the ctrl saturation count. Judge it there: a few hundredths of a radian is
the reference brushing its limit, a few tenths would mean the clip is not
representable on this body at all.

What this does NOT do
---------------------
It does not simulate. The returned action is what the servo is COMMANDED with,
not proof the body reaches the commanded pose: whether a 38 kg child with
torque-scaled actuators actually tracks it is a dynamics question, and
pd_track_bodies.py is the file that answers it.

A consequence specific to retargeting: qpos_retarget.py rescales the root and
copies every hinge angle verbatim, and all bodies share one jnt_range, so the
action recovered from data/retargeting_motion is bit-identical to the action
recovered from data/origin_motion. The command carries no body information --
only the outcome under it does.

Usage (from project root):
    uv run scripts/ik_action_from_qpos.py
    uv run scripts/ik_action_from_qpos.py --limit 5
    uv run scripts/ik_action_from_qpos.py \
        --clip data/retargeting_motion/jump-2/jump-2_0.npz

    # then the z stage, which consumes exactly the record written here:
    uv run scripts/ik_z_from_action.py --action-dir data/ik_retargeting_action \
        --limit 0 --save-z data/ik_retargeting_z

Writes <out-dir>/<task>/<task>_<trial>.npz (action, qpos, qvel, qpos_init,
qvel_init, resets, fps, source, xml) and <out>/per_clip.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _abs(path: str) -> Path:
    """Relative paths mean project-root-relative, so the script runs from anywhere."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def servo_inverse(model, range_tol: float = 1e-6):
    """ctrl = slope*q + intercept, per actuator, from the servo's own coefficients.

    Returns (qposadr, slope, intercept, lo, hi), each (nu,), with lo/hi the driven
    joint's own range -- carried along because saturation is only interpretable in
    radians (a 0.02 rad overshoot on R_Toe_z, range +-0.08, reads as |ctrl| 1.25).
    Raises if an actuator is not an affine position servo on a single hinge -- the
    inverse is only closed form for that case, and silently mis-inverting a tendon
    or a general gain would produce plausible-looking garbage.
    """
    nu = model.nu
    qposadr = np.empty(nu, dtype=int)
    slope = np.empty(nu)
    intercept = np.empty(nu)
    lo_out = np.empty(nu)
    hi_out = np.empty(nu)

    for i in range(nu):
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
            raise SystemExit(f"actuator {i} has trntype {model.actuator_trntype[i]}, "
                             f"expected mjTRN_JOINT")
        if model.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED:
            raise SystemExit(f"actuator {i} has gaintype {model.actuator_gaintype[i]}, "
                             f"expected mjGAIN_FIXED")
        if model.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_AFFINE:
            raise SystemExit(f"actuator {i} has biastype {model.actuator_biastype[i]}, "
                             f"expected mjBIAS_AFFINE (this is not a position servo)")

        j = model.actuator_trnid[i, 0]
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise SystemExit(f"actuator {i} drives joint type {model.jnt_type[j]}, "
                             f"expected a hinge")

        kp_ctrl = model.actuator_gainprm[i, 0]
        b0 = model.actuator_biasprm[i, 0]
        kq = model.actuator_biasprm[i, 1]
        gear = model.actuator_gear[i, 0]
        if kp_ctrl == 0.0:
            raise SystemExit(f"actuator {i} has gainprm[0] == 0; ctrl does nothing")

        qposadr[i] = model.jnt_qposadr[j]
        slope[i] = -kq * gear / kp_ctrl
        intercept[i] = -b0 / kp_ctrl

        # Cross-check against the jnt_range form quoted elsewhere in the repo:
        # ctrl in [-1, 1] should map onto the joint's own limits.
        lo, hi = model.jnt_range[j]
        lo_out[i], hi_out[i] = lo, hi
        c_lo = slope[i] * lo + intercept[i]
        c_hi = slope[i] * hi + intercept[i]
        err = max(abs(c_lo - model.actuator_ctrlrange[i, 0]),
                  abs(c_hi - model.actuator_ctrlrange[i, 1]))
        if err > range_tol:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            raise SystemExit(
                f"actuator {i} ({name}): the servo's neutral angle disagrees with "
                f"jnt_range by {err:.3e} (> --range-tol {range_tol:g}). ctrl no "
                f"longer spans exactly [lo, hi], so anything that assumes the "
                f"jnt_range form -- pd_track_bodies.py, ppo.py's BC target -- is "
                f"reading this body wrong.")

    return qposadr, slope, intercept, lo_out, hi_out


def central_diff_qvel(model, qpos: np.ndarray, dt: float) -> np.ndarray:
    """(T, nq) -> (T, nv), central differences via mj_differentiatePos.

    mj_differentiatePos is what makes this correct at the root: qpos carries a
    unit quaternion and qvel carries an angular velocity in the body frame, so
    (qpos[t+1] - qpos[t]) / dt is not a velocity in any frame. Endpoints fall
    back to one-sided differences.
    """
    T = len(qpos)
    out = np.zeros((T, model.nv))
    scratch = np.zeros(model.nv)
    for t in range(T):
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        if a == b:
            continue
        mujoco.mj_differentiatePos(model, scratch, dt * (b - a), qpos[a], qpos[b])
        out[t] = scratch
    return out


def solve_clip(model, qpos: np.ndarray, fps: float, inv):
    """One clip: (T, nq) poses -> the T-1-step action record described up top."""
    qposadr, slope, intercept, lo, hi = inv
    qvel = central_diff_qvel(model, qpos, 1.0 / fps)

    q = qpos[:, qposadr]                                # (T, nu), the driven angles
    raw = q * slope + intercept                         # unclipped ctrl
    ctrl = np.clip(raw, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    n_sat = int((raw != ctrl).sum())
    # Saturation says the REFERENCE leaves jnt_range, which it can: MuJoCo joint
    # limits are soft constraints, so a rollout penetrates them and no ctrl can
    # command the angle back. Report the overshoot in radians, where it is small.
    overshoot = float(np.maximum(np.maximum(lo - q, q - hi), 0.0).max())

    rec = {
        "action": ctrl[1:].astype(np.float32),          # action[t] commands qpos[t+1]
        "qpos": qpos[1:].astype(np.float64),
        "qvel": qvel[1:].astype(np.float64),
        "qpos_init": qpos[0].astype(np.float64),
        "qvel_init": qvel[0].astype(np.float64),
        # rollout_once writes this; ik_z_from_action.py reads it unconditionally
        # to find spliced episodes. A solved clip is one unbroken chain.
        "resets": np.zeros(0, dtype=np.int64),
        "fps": np.float64(fps),
    }
    return rec, n_sat, float(np.abs(raw).max()), overshoot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion-dir", default="data/retargeting_motion",
                   help="clips holding qpos only, as <task>/<task>_<trial>.npz")
    p.add_argument("--clip", default=None, help="a single .npz instead of --motion-dir")
    p.add_argument("--out-dir", default="data/ik_retargeting_action",
                   help="where the action records go; the layout mirrors --motion-dir")
    p.add_argument("--xml", default="assets/robot_torque/robot_torque.xml",
                   help="the body the qpos is expressed on. Only its actuator "
                        "coefficients and nq are read, and all bodies here share "
                        "one jnt_range, so this changes nothing about the answer "
                        "-- it is the consistency check, not a parameter.")
    p.add_argument("--fps", type=float, default=30.0,
                   help="fallback control rate for clips that do not store 'fps'")
    p.add_argument("--limit", type=int, default=0, help="0 = every clip")
    p.add_argument("--range-tol", type=float, default=1e-6,
                   help="how far the servo's ctrl<->jnt_range identity may drift "
                        "before this refuses to run (measured 4.4e-16)")
    p.add_argument("--out", default="outputs/ik_action_from_qpos",
                   help="directory for per_clip.csv")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(str(_abs(args.xml)))
    inv = servo_inverse(model, args.range_tol)

    paths = ([_abs(args.clip)] if args.clip
             else sorted(_abs(args.motion_dir).glob("*/*.npz")))
    if args.limit and not args.clip:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no clips found under {args.motion_dir}")

    out_root = _abs(args.out_dir)
    rows = []
    skipped = []

    for i, path in enumerate(paths):
        src = np.load(path)
        if "qpos" not in src.files:
            skipped.append((path, "no 'qpos' key"))
            continue
        qpos = np.asarray(src["qpos"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != model.nq:
            skipped.append((path, f"qpos {qpos.shape} does not match nq={model.nq}"))
            continue
        if len(qpos) < 2:
            skipped.append((path, f"only {len(qpos)} frame(s); the t+1 shift needs 2"))
            continue

        fps = float(src["fps"]) if "fps" in src.files else args.fps
        rec, n_sat, max_raw, overshoot = solve_clip(model, qpos, fps, inv)

        dst_dir = out_root / path.parent.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / path.name
        src_name = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        np.savez(dst, source=src_name, xml=args.xml, **rec)

        n_cells = rec["action"].size
        clip = f"{path.parent.name}/{path.stem}"
        rows.append({"clip": clip, "T_in": len(qpos), "T_out": len(rec["action"]),
                     "fps": fps, "n_sat": n_sat,
                     "frac_sat": f"{n_sat / n_cells:.6f}",
                     "max_abs_ctrl_unclipped": f"{max_raw:.4f}",
                     "max_overshoot_rad": f"{overshoot:.4f}"})
        sat = (f"  {100 * n_sat / n_cells:5.2f}% ctrl clipped "
               f"(worst {overshoot:.3f} rad outside jnt_range)") if n_sat else ""
        print(f"[{i+1}/{len(paths)}] {clip:36s} {len(qpos):4d} -> "
              f"{len(rec['action']):4d} steps @ {fps:g} fps{sat}")

    if not rows:
        raise SystemExit("nothing written")

    out_dir = _abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_clip.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    total_sat = sum(r["n_sat"] for r in rows)
    total_cells = sum(r["T_out"] for r in rows) * model.nu
    worst = max(rows, key=lambda r: float(r["max_overshoot_rad"]))
    print(f"\n{len(rows)} clips -> {out_root}")
    print(f"  steps           {sum(r['T_out'] for r in rows)} "
          f"(one frame per clip is consumed by the action[t] -> qpos[t+1] shift)")
    # Saturation means the reference pose is OUTSIDE the joint's own range, so no
    # ctrl can command it -- MuJoCo's soft limits let the source rollout penetrate
    # them. It is a property of the reference, not a failure of the inverse.
    print(f"  ctrl saturated  {total_sat} / {total_cells} "
          f"({100 * total_sat / total_cells:.2f}%), worst overshoot "
          f"{worst['max_overshoot_rad']} rad on {worst['clip']}")
    if skipped:
        print(f"  skipped         {len(skipped)}")
        for path, why in skipped[:10]:
            print(f"      {path}: {why}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
