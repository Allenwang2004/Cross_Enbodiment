"""Give every body the actuator strength it actually needs, measured not predicted.

The problem (scripts/audit_bodies.py)
-------------------------------------
Only 8 of 13 bodies can generate enough torque to hold their own rest pose:

    heavy 0.03   short_stocky 0.04   short_limbed 0.09   giant 0.48   pear_shaped 0.96

They ship with the ADULT's gainprm/biasprm/forcerange despite masses of 38-210 kg.

Why scale_robot.py's load model does not fix it
-----------------------------------------------
`compute_joint_loads` (scale_robot.py:103) predicts each joint's load as
mass x lever arm in the REST pose, then scales the actuator by the ratio to the
adult's. Measured (scripts/regen_bodies.py + re-audit), that makes things worse
overall: it drives every body's headroom toward the adult's own ratio rather
than granting margin, so heavy only reaches 0.11 (still 10x short) while child
falls 7.54 -> 3.00 and petite 1.41 -> 0.65, i.e. it costs two usable bodies and
recovers none. The rest pose is simply not where the torque demand lives.

What this does instead
----------------------
Measure the demand. For a sample of real reference poses (the p=0 retarget of
actual clips -- exactly what the policy will be asked to track), compute the
static torque each actuator must supply, and size the actuator so that this body
has the SAME headroom the adult has on the same motion:

    tau_b[j]  = high percentile of |qfrc_inverse[j]| over reference poses
    h_adult[j] = forcerange_adult[j] / tau_adult[j]        (the adult's own margin)
    forcerange_b[j] = tau_b[j] * h_adult[j]

plus a floor ensuring the rest pose is held with at least `--min-rest` headroom.

This is a modelling decision, and a defensible one: a heavier human really does
have proportionally stronger muscles, and normalizing every body to the adult's
margin is the right premise for cross-embodiment work -- morphology should
differ by GEOMETRY, not by some bodies being crippled. It is stated here rather
than buried because it changes the physics, not just the numbers.

Preserving the actuator identity
--------------------------------
gainprm[0], biasprm[0] and biasprm[1] are ALL scaled by the same per-joint k.
Since the servo's equilibrium angle is q* = -(g*ctrl + b0)/b1, scaling all three
leaves q* untouched, so `ctrl in [-1,1] <=> qpos in jnt_range` still holds
exactly -- the identity that model/bilevel's BC target and feasibility penalty
both depend on (verified to 4.4e-16, and re-checked by
model/bilevel/tests/test_torch_kin.py after this runs).

Run:
    uv run scripts/calibrate_actuators.py
    uv run scripts/calibrate_actuators.py --min-rest 3.0 --percentile 90
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def actuated_dofs(model):
    dof = [int(model.jnt_dofadr[int(model.actuator_trnid[i, 0])]) for i in range(model.nu)]
    return np.array(dof)


def static_torques(model, poses):
    """|qfrc_inverse| at the actuated DOFs for each motionless pose. -> (T, nu)."""
    data = mujoco.MjData(model)
    dof = actuated_dofs(model)
    out = np.empty((len(poses), model.nu))
    for t, q in enumerate(poses):
        data.qpos[:] = q
        data.qvel[:] = 0
        data.qacc[:] = 0
        mujoco.mj_inverse(model, data)
        out[t] = np.abs(data.qfrc_inverse[dof])
    return out


def retarget_naive(frames, model, src_rest_h):
    """p = 0: root scaled by the rest-height ratio, hinges copied then clamped.
    Identical to model/bilevel/retarget.py at u = 0."""
    q = frames.copy()
    q[:, 0:3] *= float(model.qpos0[2]) / src_rest_h
    np.clip(q[:, 7:], model.jnt_range[1:, 0], model.jnt_range[1:, 1], out=q[:, 7:])
    return q


def _set_attr(tag: str, attr: str, value: float) -> str:
    """Set attr=value on an XML tag, inserting it if absent."""
    if re.search(rf'\b{attr}="[^"]*"', tag):
        return re.sub(rf'\b{attr}="[^"]*"', f'{attr}="{value:.10g}"', tag)
    return re.sub(r"(\s*/?>)$", f' {attr}="{value:.10g}"\\1', tag, count=1)


def rewrite_joints(xml_text: str, k: np.ndarray, names: list, model) -> str:
    """Scale each joint's armature, damping and passive stiffness by the same k
    as its actuator.

    THIS IS NOT OPTIONAL. The servo's stiffness is biasprm[1], so scaling the
    actuator by k scales Kp by k, and the explicit integrator needs
    dt * sqrt(Kp / armature) < 2 to stay stable. Measured with armature left
    alone at dt = 1/450:

        adult          0.34   (original, fine)
        heavy   calib  1.85   marginal
        athletic calib 2.32   UNSTABLE
        child   calib  4.83   UNSTABLE

    and the whole training run degraded accordingly -- pose_err went from 0.05
    to 8126 and the termination rate from 0.34 to 0.86.

    Scaling armature by the same k holds sqrt(Kp/I) exactly constant, and
    scaling damping by k holds the damping ratio c/(2*sqrt(Kp*I)) constant too.
    Physically this is just "make the actuator AND its rotor k times bigger",
    which is the honest model of a stronger joint anyway.

    The joint's PASSIVE stiffness is deliberately left alone. It is a separate
    modelling choice from the actuator, and scaling it up would put a k-times
    stronger spring in the actuator's way -- it shows up in qfrc_inverse and
    directly cancels part of the strength being added (measured: it cost ~20%
    of the headroom gain). Leaving it fixed also keeps sqrt((k*Kp_act +
    Kp_pass)/(k*I)) -> sqrt(Kp_act/I), so stability still holds.
    """
    by_name = dict(zip(names, k))
    # actuator name == joint name in this asset family
    jid = {n: model.joint(n).id for n in names}

    def fix(m):
        tag = m.group(0)
        nm = re.search(r'name="([^"]+)"', tag)
        if nm is None or nm.group(1) not in by_name:
            return tag
        name = nm.group(1)
        s = float(by_name[name])
        j = jid[name]
        dof = int(model.jnt_dofadr[j])
        tag = _set_attr(tag, "armature", float(model.dof_armature[dof]) * s)
        tag = _set_attr(tag, "damping", float(model.dof_damping[dof]) * s)
        return tag

    return re.sub(r'<joint\b[^>]*?/>', fix, xml_text)


def rewrite_actuators(xml_text: str, k: np.ndarray, names: list) -> str:
    """Scale gainprm[0], biasprm[0..2] and forcerange by k, per actuator.

    All of gainprm[0], biasprm[0] and biasprm[1] scale together so that the
    servo's equilibrium angle q* = -(g*ctrl + b0)/b1 is unchanged, i.e.
    `ctrl in [-1,1] <=> qpos in jnt_range` still holds exactly.
    """
    by_name = dict(zip(names, k))

    def fix(m):
        tag = m.group(0)
        name = re.search(r'name="([^"]+)"', tag).group(1)
        s = float(by_name[name])
        g = float(re.search(r'gainprm="([^"]+)"', tag).group(1))
        b = [float(x) for x in re.search(r'biasprm="([^"]+)"', tag).group(1).split()]
        f = [float(x) for x in re.search(r'forcerange="([^"]+)"', tag).group(1).split()]
        tag = re.sub(r'gainprm="[^"]+"', f'gainprm="{g * s:.10g}"', tag)
        tag = re.sub(r'biasprm="[^"]+"',
                     f'biasprm="{b[0] * s:.10g} {b[1] * s:.10g} {b[2] * s:.10g}"', tag)
        tag = re.sub(r'forcerange="[^"]+"',
                     f'forcerange="{f[0] * s:.10g} {f[1] * s:.10g}"', tag)
        return tag

    return re.sub(r"<general\b[^/]*/>", fix, xml_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=REPO_ROOT / "assets" / "robots")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "robots_calib")
    ap.add_argument("--source-body", default="adult")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--percentile", type=float, default=90.0,
                    help="torque percentile over reference poses to size against")
    ap.add_argument("--min-rest", type=float, default=3.0,
                    help="floor on rest-pose headroom (forcerange / static torque at qpos0)")
    ap.add_argument("--max-k", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-prefixes", nargs="*", default=None,
                    help="restrict the reference-pose sample to these task-name "
                         "prefixes (e.g. `move jump`). WHAT THIS FIXES: the p90 "
                         "band the actuator is sized against is dominated by "
                         "GROUND-LEVEL poses -- measured over all 54 tasks, the "
                         "top-10%-torque poses have a median pelvis height of "
                         "0.216 m against 0.885 m for the rest, and 7.4 clamped "
                         "joints against 4.1. Those are crawl/lieonground/"
                         "sitonground/split frames whose torque is an artifact of "
                         "the p=0 retarget pinning joints on their limits, not a "
                         "demand the motion actually makes. Sizing a walking "
                         "robot against them is what drives k into the --max-k "
                         "ceiling (short_stocky 36/69 actuators, short_limbed "
                         "25/69). Default None = all tasks, the shipped behaviour.")
    args = ap.parse_args()

    # ---- reference poses, from real clips -------------------------------
    rng = np.random.default_rng(args.seed)
    clips = sorted((REPO_ROOT / "data" / "origin_motion").rglob("*.npz"))
    if args.task_prefixes:
        clips = [c for c in clips
                 if any(c.parent.name.startswith(p) for p in args.task_prefixes)]
    if not clips:
        raise SystemExit(f"no clips under data/origin_motion "
                         f"matching {args.task_prefixes}")
    # One frame per distinct clip while the pool allows it -- which keeps the
    # unfiltered default bit-identical to the shipped calibration. A filtered
    # pool can be smaller than --frames, so fall back to sampling clips with
    # replacement rather than silently shrinking the sample.
    replace = len(clips) < args.frames
    picks = rng.choice(len(clips), size=args.frames, replace=replace)
    frames = np.stack([
        (lambda q: q[rng.integers(0, len(q))])(np.load(clips[i])["qpos"]) for i in picks
    ])
    print(f"reference poses: {args.frames} from {len(clips)} clips"
          + (f" matching {args.task_prefixes}" if args.task_prefixes else " (all tasks)")
          + (" [clips sampled with replacement]" if replace else ""))

    src_dir = args.src / args.source_body
    src_model = mujoco.MjModel.from_xml_path(str(src_dir / "robot.xml"))
    src_rest_h = float(src_model.qpos0[2])
    src_xml_text = (src_dir / "robot.xml").read_text()
    names = [mujoco.mj_id2name(src_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(src_model.nu)]
    F_adult = np.abs(src_model.actuator_forcerange[:, 1]).copy()

    # What the sample actually looks like, since the whole calibration is an
    # order statistic over it. The two rows that matter are pelvis height and
    # clamped-joint count in the tail band: if the top decile is much lower and
    # much more clamped than the rest, the actuator is being sized against
    # retarget artifacts rather than against the motion.
    src_poses = retarget_naive(frames, src_model, src_rest_h)
    T_src = static_torques(src_model, src_poses)
    n_clamped = ((frames[:, 7:] < src_model.jnt_range[1:, 0])
                 | (frames[:, 7:] > src_model.jnt_range[1:, 1])).sum(1)
    per_pose = T_src.max(1)
    hi = per_pose >= np.percentile(per_pose, args.percentile)
    print(f"  pose sample     {'below p' + str(int(args.percentile)):>18}{'top band':>12}")
    print(f"    max torque (Nm){np.median(per_pose[~hi]):>18.1f}{np.median(per_pose[hi]):>12.1f}")
    print(f"    pelvis z (m)   {np.median(src_poses[~hi, 2]):>18.3f}"
          f"{np.median(src_poses[hi, 2]):>12.3f}")
    print(f"    clamped joints {n_clamped[~hi].mean():>18.1f}{n_clamped[hi].mean():>12.1f}")

    tau_adult = np.percentile(T_src, args.percentile, axis=0)
    # Floor the adult's demand before dividing by it. Several joints are almost
    # unloaded on the adult (Torso_x, the near-locked knee_y/z), so a per-joint
    # headroom of F_adult/tau_adult is astronomically large there, and sizing
    # another body as tau_b * that headroom amplifies noise into 50x scale
    # factors -- measured: athletic (1.4x the adult's mass) came out needing
    # 18.7x the torque. The ratio of DEMANDS is the physically meaningful
    # quantity; the floor keeps it finite.
    tau_floor = np.percentile(tau_adult, 25)
    tau_adult_ref = np.maximum(tau_adult, tau_floor)
    print(f"adult demand at p{args.percentile:.0f}: median {np.median(tau_adult):.1f} Nm, "
          f"floor {tau_floor:.1f} Nm, margin median {np.median(F_adult / tau_adult_ref):.2f}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    bodies = sorted(p for p in args.src.iterdir() if (p / "robot.xml").exists())
    print(f"{'body':<14}{'k median':>10}{'k max':>9}{'worst actuator':>18}")
    print("-" * 52)

    for body in bodies:
        dst = args.out / body.name
        dst.mkdir(parents=True, exist_ok=True)
        for extra in ("parameter.json", "skeleton.json"):
            if (body / extra).exists():
                shutil.copy2(body / extra, dst / extra)

        if body.name == args.source_body:
            shutil.copy2(body / "robot.xml", dst / "robot.xml")
            print(f"{body.name:<14}{1.0:>10.2f}{1.0:>9.2f}{'(reference)':>18}")
            continue

        model = mujoco.MjModel.from_xml_path(str(body / "robot.xml"))
        poses = retarget_naive(frames, model, src_rest_h)
        tau = np.percentile(static_torques(model, poses), args.percentile, axis=0)

        # Scale the adult's actuator by how much more torque THIS body demands
        # at the same joint on the same motion...
        F_needed = F_adult * (tau / tau_adult_ref)
        # ...and separately guarantee the rest pose is comfortably holdable,
        # which is the hard gate in audit_bodies.py and the one the load model
        # in scale_robot.py misses entirely.
        tau_rest = static_torques(model, model.qpos0[None, :])[0]
        F_rest = tau_rest * args.min_rest
        F_target = np.maximum(F_needed, F_rest)

        # Scale relative to what THIS body's xml already has, not to the adult's
        # -- `elderly` was rescaled once already (commit 44da9cf) and would be
        # double-counted otherwise.
        F_body = np.abs(model.actuator_forcerange[:, 1])
        s = np.clip(F_target / np.maximum(F_body, 1e-9), 1e-3, args.max_k)
        worst = names[int(s.argmax())]
        print(f"{body.name:<14}{np.median(s):>10.2f}{s.max():>9.2f}{worst:>18}")

        text = (body / "robot.xml").read_text()
        text = rewrite_actuators(text, s, names)
        text = rewrite_joints(text, s, names, model)   # keeps dt*sqrt(Kp/I) invariant
        (dst / "robot.xml").write_text(text)
        par = json.loads((dst / "parameter.json").read_text())
        par["scale_actuators"] = "calibrated"
        par["actuator_scale_median"] = float(np.median(s))
        (dst / "parameter.json").write_text(json.dumps(par, indent=2) + "\n")

    print(f"\nwrote {args.out}")
    print(f"verify with:  uv run scripts/audit_bodies.py --robots {args.out}")
    print(f"              uv run model/bilevel/tests/test_torch_kin.py   "
          f"(the ctrl<->qpos identity must survive)")


if __name__ == "__main__":
    main()
