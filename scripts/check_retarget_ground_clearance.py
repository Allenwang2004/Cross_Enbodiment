"""Measure ground penetration/clearance of retargeted motion (data/retargeted_motion,
robot_child.xml) against the origin motion it came from (data/origin_motion, robot.xml),
to check whether qpos_retarget.py's naive single-scalar root-height rescale actually
keeps the target body flush with the ground across a whole clip, not just at rest pose.

Per frame, computes the exact lowest point (z) over all non-floor geoms (box corners /
capsule caps / sphere surface -- geom_rbound is not tight for box geoms, see
scale_robot.py's measure_min_z, which this reuses the same math as). Negative = geometry
is below the floor plane (penetration); the origin motion's own numbers (using robot.xml)
are the "normal" contact-solver penetration baseline to compare against, since MuJoCo's
soft contact always allows a few mm/cm of interpenetration.

Usage (from project root):
    uv run scripts/check_retarget_ground_clearance.py
    uv run scripts/check_retarget_ground_clearance.py --limit 50   # quick subset
"""

import argparse
import itertools
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def min_geom_z(model, data):
    lo = np.inf
    for gid in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "floor":
            continue
        gtype = model.geom_type[gid]
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            for signs in itertools.product([1, -1], repeat=3):
                z = (pos + mat @ (np.array(signs) * size))[2]
                lo = min(lo, z)
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            c1 = (pos + mat @ np.array([0, 0, size[1]]))[2] - size[0]
            c2 = (pos + mat @ np.array([0, 0, -size[1]]))[2] - size[0]
            lo = min(lo, c1, c2)
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            lo = min(lo, pos[2] - size[0])
        else:
            lo = min(lo, pos[2])
    return lo


def clip_min_z_trace(model, data, qpos_seq):
    out = np.empty(qpos_seq.shape[0])
    for t in range(qpos_seq.shape[0]):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_kinematics(model, data)
        out[t] = min_geom_z(model, data)
    return out


def task_of(stem):
    # "<reward_name>_<trial>" -> "<reward_name>"
    return stem.rsplit("_", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retarget-dir", default="data/retargeted_motion")
    parser.add_argument("--origin-dir", default="data/origin_motion")
    parser.add_argument("--retarget-xml", default="assets/robots/child/robot.xml")
    parser.add_argument("--origin-xml", default="assets/robots/adult/robot.xml")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N files (sorted)")
    parser.add_argument("--out", default="outputs/retarget_ground_clearance.csv")
    args = parser.parse_args()

    model_r = mujoco.MjModel.from_xml_path(args.retarget_xml)
    data_r = mujoco.MjData(model_r)
    model_o = mujoco.MjModel.from_xml_path(args.origin_xml)
    data_o = mujoco.MjData(model_o)

    retarget_files = sorted(Path(args.retarget_dir).glob("*.npz"))
    if args.limit:
        retarget_files = retarget_files[: args.limit]

    rows = []
    missing_origin = 0
    for i, rf in enumerate(retarget_files):
        stem = rf.stem
        task = task_of(stem)
        of = Path(args.origin_dir) / task / f"{stem}.npz"

        qpos_r = np.load(rf)["qpos"]
        mins_r = clip_min_z_trace(model_r, data_r, qpos_r)

        if of.exists():
            qpos_o = np.load(of)["qpos"]
            mins_o = clip_min_z_trace(model_o, data_o, qpos_o)
            origin_mean = float(mins_o.mean())
            origin_min = float(mins_o.min())
        else:
            missing_origin += 1
            origin_mean = origin_min = float("nan")

        rows.append({
            "task": task, "trial": stem,
            "retarget_mean": float(mins_r.mean()), "retarget_min": float(mins_r.min()),
            "origin_mean": origin_mean, "origin_min": origin_min,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(retarget_files)}")

    if missing_origin:
        print(f"WARNING: {missing_origin} retargeted files had no matching origin_motion file")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")

    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    print(f"\n=== per-task retarget_mean (n={len(by_task)} tasks) ===")
    task_means = sorted(
        ((task, float(np.mean([r["retarget_mean"] for r in trs]))) for task, trs in by_task.items()),
        key=lambda x: x[1],
    )
    for task, m in task_means[:15]:
        print(f"  {task:30s} {m:+.4f}  (worst 15)")
    print("  ...")
    for task, m in task_means[-5:]:
        print(f"  {task:30s} {m:+.4f}  (best 5)")

    all_r_mean = np.array([r["retarget_mean"] for r in rows])
    all_o_mean = np.array([r["origin_mean"] for r in rows if not np.isnan(r["origin_mean"])])
    print(f"\n=== overall ===")
    print(f"retarget_mean: mean={all_r_mean.mean():+.4f} median={np.median(all_r_mean):+.4f} "
          f"p10={np.percentile(all_r_mean, 10):+.4f} p90={np.percentile(all_r_mean, 90):+.4f}")
    if len(all_o_mean):
        print(f"origin_mean:   mean={all_o_mean.mean():+.4f} median={np.median(all_o_mean):+.4f}")
    frac_bad = float(np.mean(all_r_mean < -0.01))
    print(f"fraction of clips with retarget_mean < -0.01m: {frac_bad:.1%}")


if __name__ == "__main__":
    main()
