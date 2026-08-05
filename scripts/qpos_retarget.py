"""Direct qpos-space retargeting between two MJCF models that share the exact
same body/joint topology (same body names, same joint types/axes/declaration
order -- just different segment lengths/offsets, i.e. `body_pos`). This is
NOT a general-purpose retargeter: it only works because `robot.xml` and
`robot_child.xml`/`robot_elderly.xml` were built as scaled variants of the
same skeleton (confirmed via direct MuJoCo inspection -- 0 diffs in
body/joint names, types, axes, or declaration order between them; only
`body_pos` differs, non-uniformly).

Because the topology is identical, every non-root qpos value is already a
joint-local hinge angle (relative to its own parent, per that body's own
`axis=` declarations) -- copying it unchanged from source to target
skeleton reproduces the same *articulation* regardless of segment length
differences. Only the free-joint root (qpos[0:3], the pelvis's world
position) needs adjusting, since it's an absolute world-space translation
recorded for the SOURCE skeleton's proportions -- left unscaled, a smaller
skeleton's feet would float above the ground (or a larger skeleton's would
clip into it). It's rescaled uniformly by the ratio of each skeleton's own
rest-pose (qpos=0) pelvis height, which keeps the motion's overall path
shape and grounding proportional to the resulting skeleton's size.

This does NOT do proper IK-based retargeting (no per-limb correction for
foot/hand placement -- e.g. exact foot-ground contact timing can drift
slightly since only the root height is corrected, not each leg's reach).
For that, use the qpos_to_world_pose.py -> world_pose_to_fbx.py -> Unreal
Engine IK Retargeter pipeline instead, which supports true cross-topology
retargeting. For a same-topology proportion change like robot->robot_child,
this is a much cheaper approximation of what that pipeline's own IK-disabled
FK-chain retargeting already does.

The single rest-pose height ratio used for the root z scale is only exact
when the body is actually in a standing-like pose. For non-standing clips
(crawl, headstand, lieonground, split, crouch, ...) the ratio between "root
height at rest" and "root height in this specific pose" is NOT the same
across skeletons whose limb segments are scaled non-uniformly (this repo's
robot_child.xml scales legs/arms/torso/head independently -- see
assets/robots/child/parameter.json), so the naive scale alone
systematically drives the target skeleton's lowest geometry below the floor.
Measured across all 540 clips in data/retargeted_motion: naive scaling alone
put 83.7% of clips' mean lowest-point z below -0.01m (vs +0.0065m for the
unretargeted origin motion), worst on non-standing tasks (headstand -0.082,
crawl-0.4-2-d -0.056) -- see scripts/check_retarget_ground_clearance.py.

`--ground-correct` (on by default) fixes this with a per-frame vertical-only
correction on top of the naive scale: after naively retargeting, run forward
kinematics on the TARGET skeleton for each frame and find the exact lowest
point across all its geoms (box corners / capsule caps / sphere surface,
not the loose geom_rbound sphere). If that point is below `--tolerance`
(matching MuJoCo's own few-mm/cm soft-contact penetration, so corrected
frames still look/behave normally under the contact solver), shift qpos[:,2]
(root world z only -- x/y and all joint angles are untouched) up by exactly
enough to bring it to `--tolerance`. Frames that are already at or above
tolerance (e.g. genuine mid-air frames in a jump or rotate flip) are left
alone -- this only ever pulls the body UP out of the floor, never pushes a
legitimately airborne pose down.

Run with (this repo's own venv):
    python3 scripts/qpos_retarget.py --input_npz <robot_clip.npz> \
        --output_npz <robot_child_clip.npz> \
        [--source_skeleton_json assets/robots/adult/skeleton.json] \
        [--target_skeleton_json assets/robots/child/skeleton.json] \
        [--target_xml assets/robots/child/robot.xml]

    # batch mode:
    python3 scripts/qpos_retarget.py --input_dir <robotmotion_dir> \
        --output_dir <robot_child_qpos_dir> \
        [--source_skeleton_json assets/robots/adult/skeleton.json] \
        [--target_skeleton_json assets/robots/child/skeleton.json] \
        [--target_xml assets/robots/child/robot.xml]
"""
import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def load_root_rest_height(skeleton_json_path):
    with open(skeleton_json_path) as f:
        data = json.load(f)
    bodies = {b["name"]: b for b in data["bodies"]}
    root_name = next(b["name"] for b in data["bodies"] if b["parent"] == "world")
    return bodies[root_name]["world_pos"][2]


def retarget_qpos(qpos, scale):
    out = qpos.copy()
    out[:, 0:3] *= scale  # root (free joint) world position only; quat (3:7)
    # and every hinge angle (7:) are joint-local and topology-identical, so
    # they carry over unchanged.
    return out


def _min_geom_z(model, data):
    """Exact lowest world-z point over all non-floor geoms for the CURRENT
    data.qpos (caller must have already run mj_kinematics). Same math as
    scale_robot.py's measure_min_z / check_retarget_ground_clearance.py's
    min_geom_z -- geom_rbound is a bounding-sphere radius and isn't tight
    for box geoms (the feet here are boxes)."""
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
                lo = min(lo, (pos + mat @ (np.array(signs) * size))[2])
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            c1 = (pos + mat @ np.array([0, 0, size[1]]))[2] - size[0]
            c2 = (pos + mat @ np.array([0, 0, -size[1]]))[2] - size[0]
            lo = min(lo, c1, c2)
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            lo = min(lo, pos[2] - size[0])
        else:
            lo = min(lo, pos[2])
    return lo


def ground_correct_qpos(target_model, qpos, tolerance=-0.005):
    """Per-frame: if the target skeleton's lowest geometry point is below
    `tolerance`, lift the whole pose (root z only) so it's exactly at
    `tolerance`. Never pushes an already-clear/airborne frame down."""
    data = mujoco.MjData(target_model)
    out = qpos.copy()
    n_corrected = 0
    max_shift = 0.0
    for t in range(out.shape[0]):
        data.qpos[:] = out[t]
        mujoco.mj_kinematics(target_model, data)
        min_z = _min_geom_z(target_model, data)
        if min_z < tolerance:
            shift = tolerance - min_z
            out[t, 2] += shift
            n_corrected += 1
            max_shift = max(max_shift, shift)
    return out, n_corrected, max_shift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=Path)
    parser.add_argument("--output_npz", type=Path)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--source_skeleton_json", default="assets/robots/adult/skeleton.json")
    parser.add_argument("--target_skeleton_json", default="assets/robots/child/skeleton.json")
    parser.add_argument("--target_xml", default="assets/robots/child/robot.xml",
                         help="target MJCF, used for --ground-correct's forward kinematics")
    parser.add_argument("--ground-correct", dest="ground_correct", action="store_true", default=True,
                         help="per-frame lift the pose so it never sinks below --tolerance (default: on)")
    parser.add_argument("--no-ground-correct", dest="ground_correct", action="store_false",
                         help="disable ground correction, output the pure naive-scale retarget")
    parser.add_argument("--tolerance", type=float, default=-0.005,
                         help="min allowed lowest-geom z before correction kicks in (m); "
                              "matches MuJoCo's normal soft-contact penetration range")
    parser.add_argument("--fps", type=float, default=30.0,
                         help="robotmotion .npz files carry no fps of their own; render_qpos.py "
                              "expects one in the output, so it's passed through explicitly.")
    args = parser.parse_args()

    if bool(args.input_npz) == bool(args.input_dir):
        raise SystemExit("Pass exactly one of --input_npz/--output_npz or --input_dir/--output_dir")

    source_h = load_root_rest_height(args.source_skeleton_json)
    target_h = load_root_rest_height(args.target_skeleton_json)
    scale = target_h / source_h
    print(f"Root height scale ({args.source_skeleton_json} -> {args.target_skeleton_json}): {scale:.4f}")

    target_model = mujoco.MjModel.from_xml_path(args.target_xml) if args.ground_correct else None

    def process(qpos):
        out = retarget_qpos(qpos, scale)
        if args.ground_correct:
            out, n_corrected, max_shift = ground_correct_qpos(target_model, out, args.tolerance)
            return out, n_corrected, max_shift
        return out, 0, 0.0

    if args.input_npz:
        qpos = np.load(args.input_npz)["qpos"]
        out, n_corrected, max_shift = process(qpos)
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_npz, qpos=out, fps=args.fps)
        print(f"Wrote {out.shape} -> {args.output_npz}"
              + (f" (ground-corrected {n_corrected}/{out.shape[0]} frames, max shift {max_shift:.4f}m)"
                 if args.ground_correct else ""))
    else:
        npz_paths = sorted(args.input_dir.rglob("*.npz"))
        print(f"Found {len(npz_paths)} .npz files under {args.input_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        total_corrected = 0
        for i, npz_path in enumerate(npz_paths):
            qpos = np.load(npz_path)["qpos"]
            out, n_corrected, max_shift = process(qpos)
            total_corrected += n_corrected
            out_path = args.output_dir / f"{npz_path.stem}.npz"
            np.savez(out_path, qpos=out, fps=args.fps)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(npz_paths)}")
        print(f"Wrote {len(npz_paths)} retargeted qpos files -> {args.output_dir}"
              + (f" ({total_corrected} frames ground-corrected total)" if args.ground_correct else ""))


if __name__ == "__main__":
    main()
