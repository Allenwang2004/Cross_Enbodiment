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

Run with (this repo's own venv):
    python3 scripts/qpos_retarget_simple.py --input_npz <robot_clip.npz> \
        --output_npz <robot_child_clip.npz> \
        [--source_skeleton_json assets/robot/robot.json] \
        [--target_skeleton_json assets/robot/robot_child.json]

    # batch mode:
    python3 scripts/qpos_retarget_simple.py --input_dir <robotmotion_dir> \
        --output_dir <robot_child_qpos_dir> \
        [--source_skeleton_json assets/robot/robot.json] \
        [--target_skeleton_json assets/robot/robot_child.json]
"""
import argparse
import json
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=Path)
    parser.add_argument("--output_npz", type=Path)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--source_skeleton_json", default="assets/robot/robot.json")
    parser.add_argument("--target_skeleton_json", default="assets/robot/robot_child.json")
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

    if args.input_npz:
        qpos = np.load(args.input_npz)["qpos"]
        out = retarget_qpos(qpos, scale)
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_npz, qpos=out, fps=args.fps)
        print(f"Wrote {out.shape} -> {args.output_npz}")
    else:
        npz_paths = sorted(args.input_dir.rglob("*.npz"))
        print(f"Found {len(npz_paths)} .npz files under {args.input_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for npz_path in npz_paths:
            qpos = np.load(npz_path)["qpos"]
            out = retarget_qpos(qpos, scale)
            out_path = args.output_dir / f"{npz_path.stem}.npz"
            np.savez(out_path, qpos=out, fps=args.fps)
        print(f"Wrote {len(npz_paths)} retargeted qpos files -> {args.output_dir}")


if __name__ == "__main__":
    main()
