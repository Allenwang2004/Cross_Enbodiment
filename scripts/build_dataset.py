"""DEPRECATED -- the bilevel path does not use this script.

model/bilevel/data.py reads data/origin_motion/, data/z/ and
assets/robots_calib/*/parameter.json directly: no manifest, no build step, and
no 78 MB duplicate on disk. Two of this script's premises are also gone:

  - `retargeted_motion` has been REMOVED (retargeting is now a runtime,
    differentiable function of phi -- that is the whole point of the redesign),
    so the files it copies and the manifest key it writes no longer exist.
    Regenerate them with scripts/qpos_retarget.py first if you want them back.
  - MORPHOLOGY_SRC at :44 points at assets/robots/robot_child_parameter.json,
    which the 9495329 asset restructure removed, so a fresh run crashes there.

Kept only so the legacy model/train.py baseline can be reconstituted.

Original description
--------------------
Build datasets/crossenbodiment-1-datasets from what's already generated.
Each row (indexed by reward_name/trial, enumerated from data/z/ -- the only
field every row must have, see CrossEmbodimentDataset) bundles:
  - origin_z          the z latent used for that rollout
                       (data/z/<reward_name>/<reward_name>_<trial>.npy)
  - retargeted_motion  the origin qpos trajectory retargeted onto the target
                       body via scripts/qpos_retarget.py
                       (data/retargeted_motion/<reward_name>_<trial>.npz)
                       -- optional, None if missing (e.g. rows produced by
                       scripts/metamotivo_motion_rollout.py --z-only, which
                       skips the qpos rollout entirely and so never has one).
  - morphology         body-shape scale parameters -- currently fixed to
                       assets/robots/child/parameter.json for every
                       row, even though the origin rollout was generated on
                       the baseline body. That mismatch is intentional per
                       current instructions, not a bug -- just be aware of
                       it before training on this.

origin_motion (the un-retargeted qpos) is intentionally NOT included in the
built dataset -- when present, it only exists as an intermediate used to
produce retargeted_motion and origin_z, both of which are already in the
manifest.

This only builds a manifest + copies files into a self-contained dataset
folder; it does not regenerate any rollouts or retargeting.

Usage (from project root, after scripts/qpos_retarget.py has populated
data/retargeted_motion/):
    uv run scripts/build_dataset.py
    uv run scripts/build_dataset.py --mode symlink   # save disk instead of copying
"""

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
NPZ_DIR = DATA_DIR / "origin_motion"
Z_DIR = DATA_DIR / "z"
RETARGET_DIR = DATA_DIR / "retargeted_motion"
MORPHOLOGY_SRC = REPO_ROOT / "assets" / "robots" / "robot_child_parameter.json"
MORPHOLOGY_LABEL = "child"

DATASET_DIR = REPO_ROOT / "datasets" / "crossenbodiment-1-datasets"
MANIFEST_PATH = DATASET_DIR / "manifest.jsonl"


def _copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def build(mode="copy"):
    if not Z_DIR.exists():
        raise SystemExit(
            f"{Z_DIR} not found -- run "
            "scripts/metamotivo_motion_rollout.py --tasks-file ... first"
        )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    morph_dst = DATASET_DIR / "morphology" / f"{MORPHOLOGY_LABEL}.json"
    _copy_or_link(MORPHOLOGY_SRC, morph_dst, mode)

    rows = []
    row_id = 0
    n_missing_retarget = 0
    # Enumerated from Z_DIR (origin_z is the only field every row must have --
    # see CrossEmbodimentDataset, which never requires origin_motion) rather
    # than NPZ_DIR, since --z-only rollouts (scripts/metamotivo_motion_rollout.py)
    # only ever produce a z, no qpos .npz.
    for task_dir in sorted(Z_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        reward_name = task_dir.name
        for z_path in sorted(task_dir.glob(f"{reward_name}_*.npy")):
            trial = int(z_path.stem.rsplit("_", 1)[-1])

            origin_z_dst = DATASET_DIR / "origin_z" / reward_name / z_path.name
            _copy_or_link(z_path, origin_z_dst, mode)

            retarget_src = RETARGET_DIR / f"{reward_name}_{trial}.npz"
            retargeted_motion = None
            if retarget_src.exists():
                retarget_dst = DATASET_DIR / "retargeted_motion" / reward_name / retarget_src.name
                _copy_or_link(retarget_src, retarget_dst, mode)
                retargeted_motion = str(retarget_dst.relative_to(DATASET_DIR))
            else:
                n_missing_retarget += 1

            rows.append({
                "id": row_id,
                "reward_name": reward_name,
                "trial": trial,
                "origin_z": str(origin_z_dst.relative_to(DATASET_DIR)),
                "retargeted_motion": retargeted_motion,
                "morphology": str(morph_dst.relative_to(DATASET_DIR)),
                "morphology_label": MORPHOLOGY_LABEL,
            })
            row_id += 1

    with open(MANIFEST_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} rows -> {MANIFEST_PATH}")
    if n_missing_retarget:
        print(f"WARNING: {n_missing_retarget}/{len(rows)} rows have no retargeted_motion "
              f"(no matching file in {RETARGET_DIR})")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy",
                         help="copy files into the dataset folder (portable, "
                              "default) or symlink them (saves disk, breaks "
                              "if the source files move)")
    args = parser.parse_args()
    build(mode=args.mode)


if __name__ == "__main__":
    main()
