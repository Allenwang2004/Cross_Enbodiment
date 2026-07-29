"""
Build datasets/crossenbodiment-1-datasets from what's already generated.
Each row (indexed by reward_name/trial, enumerated from
data/origin_motion/) bundles:
  - origin_z          the z latent used for that rollout
                       (data/z/<reward_name>/<reward_name>_<trial>.npy)
  - retargeted_motion  the origin qpos trajectory retargeted onto the target
                       body via scripts/qpos_retarget.py
                       (data/retargeted_motion/<reward_name>_<trial>.npz)
                       -- 1:1 matched with the origin rollout by filename, so
                       every row gets one.
  - morphology         body-shape scale parameters -- currently fixed to
                       assets/robots/robot_child_parameter.json for every
                       row, even though the origin rollout was generated on
                       the baseline body. That mismatch is intentional per
                       current instructions, not a bug -- just be aware of
                       it before training on this.

origin_motion (the un-retargeted qpos) is intentionally NOT included in the
built dataset -- it only exists as an intermediate used to produce
retargeted_motion and origin_z, both of which are already in the manifest.

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
    if not NPZ_DIR.exists():
        raise SystemExit(
            f"{NPZ_DIR} not found -- run "
            "scripts/metamotivo_rollout.py --tasks-file ... first"
        )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    morph_dst = DATASET_DIR / "morphology" / f"{MORPHOLOGY_LABEL}.json"
    _copy_or_link(MORPHOLOGY_SRC, morph_dst, mode)

    rows = []
    row_id = 0
    n_missing_retarget = 0
    for task_dir in sorted(NPZ_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        reward_name = task_dir.name
        for npz_path in sorted(task_dir.glob(f"{reward_name}_*.npz")):
            trial = int(npz_path.stem.rsplit("_", 1)[-1])
            z_path = Z_DIR / reward_name / f"{reward_name}_{trial}.npy"
            if not z_path.exists():
                print(f"WARNING: missing z for {npz_path}, skipping")
                continue

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
