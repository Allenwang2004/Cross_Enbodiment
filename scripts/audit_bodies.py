"""Stage 0 / R1: can each body's actuators actually hold the retargeted motion?

Why this must run before anything else
--------------------------------------
12 of the 13 bodies in assets/robots/ ship with the ADULT's actuator gains AND
the adult's forcerange, while their masses span 38 kg (child) to 210 kg
(heavy) -- only `elderly` was ever rescaled. scripts/scale_robot.py has the fix
(`compute_joint_loads`, :103, a per-joint load model added in commit 44da9cf)
but the 11 bodies generated in 9495329 were all produced with
--no-actuator-scale, and their parameter.json records "scale_actuators": false.

A body whose actuators cannot hold its own reference poses contributes nothing
but gradient noise for the whole run. proposal.md R1 makes picking the 10
training bodies conditional on this audit rather than on the label list.

What is measured
----------------
For N reference frames drawn from real clips (naively retargeted onto the body,
i.e. exactly what phi=0 produces):
    qvel = qacc = 0, then mj_inverse  ->  the static torque needed to HOLD the pose
    headroom_j = forcerange_j / |qfrc_inverse_j|      (>= 1 means feasible)
This is the quasi-static lower bound: a frame that fails it cannot be held even
motionless, let alone tracked. Same method as
scripts/check_retarget_actuator_feasibility.py, run across all bodies.

Also reports gravity headroom: the torque to hold the REST pose, which isolates
"this body cannot even stand" from "this pose is hard".

Run:
    uv run scripts/audit_bodies.py
    uv run scripts/audit_bodies.py --frames 400 --top 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def actuated_dofs(model):
    """DOF index and positive force limit for each actuator."""
    dof, lim = [], []
    for i in range(model.nu):
        j = int(model.actuator_trnid[i, 0])
        dof.append(int(model.jnt_dofadr[j]))
        lim.append(abs(float(model.actuator_forcerange[i, 1])) or np.inf)
    return np.array(dof), np.array(lim)


def static_torque(model, data, qpos):
    """|qfrc_inverse| at the actuated DOFs for a motionless pose."""
    data.qpos[:] = qpos
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_inverse(model, data)
    return np.abs(data.qfrc_inverse)


def audit(body_dir: Path, frames: np.ndarray, src_rest_h: float, verbose=True):
    model = mujoco.MjModel.from_xml_path(str(body_dir / "robot.xml"))
    data = mujoco.MjData(model)
    dof, lim = actuated_dofs(model)
    rest_h = float(model.qpos0[2])
    mass = float(model.body_mass.sum())

    # Naive (phi = 0) retarget: root scaled by the rest-height ratio, hinges
    # copied, then clamped into this body's own jnt_range -- exactly what
    # model/bilevel/retarget.py produces at u = 0.
    q = frames.copy()
    q[:, 0:3] *= rest_h / src_rest_h
    np.clip(q[:, 7:], model.jnt_range[1:, 0], model.jnt_range[1:, 1], out=q[:, 7:])

    headroom = np.empty((len(q), model.nu))
    for t in range(len(q)):
        tau = static_torque(model, data, q[t])[dof]
        headroom[t] = lim / np.maximum(tau, 1e-9)

    worst_per_frame = headroom.min(axis=1)
    infeasible = float((worst_per_frame < 1.0).mean())

    # Rest pose alone: "can this body stand up at all?"
    tau_rest = static_torque(model, data, model.qpos0)[dof]
    rest_headroom = float((lim / np.maximum(tau_rest, 1e-9)).min())

    worst_j = int(headroom.mean(axis=0).argmin())
    worst_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, worst_j)

    par = json.loads((body_dir / "parameter.json").read_text())
    return {
        "body": body_dir.name,
        "mass": mass,
        "rest_h": rest_h,
        "scaled_actuators": bool(par.get("scale_actuators", False)),
        "split": par.get("split", "-"),
        "frac_infeasible": infeasible,
        "median_headroom": float(np.median(worst_per_frame)),
        "rest_headroom": rest_headroom,
        "worst_joint": worst_name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300, help="reference frames sampled per body")
    ap.add_argument("--top", type=int, default=10, help="how many bodies to recommend for training")
    ap.add_argument("--source", default="adult")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robots", type=Path, default=REPO_ROOT / "assets" / "robots")
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "outputs" / "body_audit.json")
    args = ap.parse_args()

    robots = args.robots
    src_model = mujoco.MjModel.from_xml_path(str(robots / args.source / "robot.xml"))
    src_rest_h = float(src_model.qpos0[2])

    # Sample reference frames from the real source motion, so the audit reflects
    # the poses actually asked for rather than random configurations.
    rng = np.random.default_rng(args.seed)
    clips = sorted((REPO_ROOT / "data" / "origin_motion").rglob("*.npz"))
    if not clips:
        raise SystemExit("no clips under data/origin_motion")
    picks = rng.choice(len(clips), size=min(args.frames, len(clips)), replace=False)
    frames = np.stack([
        np.load(clips[i])["qpos"][rng.integers(0, np.load(clips[i])["qpos"].shape[0])]
        for i in picks
    ])

    bodies = sorted(p for p in robots.iterdir() if (p / "robot.xml").exists())
    print(f"static feasibility of the phi=0 reference, {len(frames)} frames x {len(bodies)} bodies")
    print(f"(headroom = forcerange / |static torque|; >= 1 means the pose can be HELD)\n")
    print(f"{'body':<14}{'mass':>7}{'rest_h':>8}{'scaled':>7}{'infeas':>8}"
          f"{'med.head':>10}{'rest.head':>10}  worst joint")
    print("-" * 88)

    rows = [audit(p, frames, src_rest_h) for p in bodies]
    for r in sorted(rows, key=lambda r: r["frac_infeasible"]):
        print(f"{r['body']:<14}{r['mass']:>7.1f}{r['rest_h']:>8.3f}"
              f"{str(r['scaled_actuators']):>7}{r['frac_infeasible']:>8.2f}"
              f"{r['median_headroom']:>10.3f}{r['rest_headroom']:>10.3f}  {r['worst_joint']}")

    # rest_headroom is the HARD gate, not frac_infeasible.
    #
    # frac_infeasible is conservative by construction -- it asks whether each
    # pose could be held motionless, while a real rollout carries momentum
    # through it. `adult`, the body the frozen policy was actually trained on,
    # scores 0.58 on it, and the repo's own earlier audit
    # (outputs/retarget_actuator_feasibility.csv) reports 0.91 for child. So a
    # high frac_infeasible is normal and not disqualifying.
    #
    # rest_headroom < 1 is different in kind: the body cannot generate enough
    # torque to hold its own rest pose against gravity. It cannot stand still,
    # so it certainly cannot track, and it will contribute nothing but gradient
    # noise for the entire run.
    feasible = [r for r in rows if r["rest_headroom"] >= 1.0]
    infeasible = [r for r in rows if r["rest_headroom"] < 1.0]
    feasible.sort(key=lambda r: -r["rest_headroom"])
    infeasible.sort(key=lambda r: -r["rest_headroom"])

    keep = [r["body"] for r in feasible[:args.top]]
    drop = [r["body"] for r in feasible[args.top:]] + [r["body"] for r in infeasible]

    print(f"\ncan hold their own rest pose (usable): {len(feasible)}")
    print(f"  -> train_bodies: {keep}")
    if len(feasible) < args.top:
        print(f"\n  *** ONLY {len(feasible)} USABLE BODIES, {args.top} REQUESTED ***")
        print("      The remaining bodies cannot stand up under their own weight with the")
        print("      actuators they ship with. This is an asset problem, not a training one.")
    print(f"\ncannot hold their own rest pose (rejected): "
          f"{[(r['body'], round(r['rest_headroom'], 3)) for r in infeasible]}")

    unscaled = [r["body"] for r in rows if not r["scaled_actuators"]]
    if unscaled:
        print(
            f"\nNOTE: {len(unscaled)} bodies still carry the adult's actuators "
            f"({', '.join(unscaled)}).\n"
            "      scripts/scale_robot.py can regenerate them WITH load-based gain scaling\n"
            "      (drop --no-actuator-scale); compute_joint_loads at :103 exists for exactly\n"
            "      this. Doing so invalidates comparisons against the existing\n"
            "      outputs/{baseline,eval}/report.json numbers, so it is a deliberate choice."
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(
        {"rows": rows, "recommended": keep, "rejected": drop}, indent=2
    ))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
