"""Reads datasets/crossenbodiment-1-datasets/manifest.jsonl (see
scripts/build_dataset.py).

NOTE -- retargeted_motion has been REMOVED from this dataset.
The bilevel system (model/bilevel/) produces the reference motion at runtime as
a differentiable function of phi, so a pre-baked copy is not just redundant, it
is the thing the design exists to replace. The 540 baked .npz files (78 MB here
plus 78 MB in data/) were deleted along with the manifest field.

Consequence for this LEGACY path: __getitem__ now always returns
qpos_ref=None, so losses.functional_equivalence returns 0.0 and the D term in
model/train.py's objective is identically zero. What remains being optimized is
lambda_z * ||z_beta - z0||^2 + lambda_phys * L_phys. That is a real change to
the baseline's behaviour, and the constructor says so out loud rather than
letting the loss quietly collapse (990 of the original 1530 rows already had
retargeted_motion=None, so a silent D=0 was always easy to miss here).

To run the old baseline as it was, regenerate the references first:
    uv run scripts/qpos_retarget.py --input_dir data/origin_motion \\
        --output_dir data/retargeted_motion \\
        --target_xml assets/robots/child/robot.xml
    uv run scripts/build_dataset.py
"""

import json
from pathlib import Path

import numpy as np

# Must match scripts/scale_robot.py's AXES order -- beta is this 8-dim
# vector read straight out of a robot_<label>_parameter.json (NOT
# robot_<label>.json -- that name now means the skeleton-export schema with
# a "bodies" list, see scripts/export_skeleton_json.py).
BETA_AXES = ["leg_scale", "arm_scale", "torso_scale", "head_scale",
             "leg_girth", "arm_girth", "torso_girth", "head_girth"]


def load_beta(morphology_json_path) -> np.ndarray:
    d = json.loads(Path(morphology_json_path).read_text())
    return np.array([d[a] for a in BETA_AXES], dtype=np.float32)


def load_task_list(path) -> list:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


class CrossEmbodimentDataset:
    def __init__(self, dataset_dir, task_filter=None):
        """task_filter: optional iterable of reward_name strings (e.g. from
        datasets/crossenbodiment-1-datasets/splits/train_tasks.txt or
        test_tasks.txt, see scripts/split_tasks.py) -- only rows whose
        reward_name is in this set are kept. Use this to keep train/test
        task splits from leaking into each other."""
        self.dataset_dir = Path(dataset_dir)
        manifest_path = self.dataset_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found -- run scripts/build_dataset.py first"
            )
        self.rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        if task_filter is not None:
            task_filter = set(task_filter)
            self.rows = [r for r in self.rows if r["reward_name"] in task_filter]
        if not self.rows:
            raise ValueError(f"no rows in {manifest_path} (after task_filter)")

        if not any(r.get("retargeted_motion") for r in self.rows):
            print(
                "WARNING: this dataset has no retargeted_motion -- it was removed in favour of "
                "model/bilevel's runtime retargeting.\n"
                "         qpos_ref will be None for every row, so functional_equivalence "
                "returns 0.0 and the\n"
                "         D term of model/train.py's loss is identically zero. See this "
                "module's docstring to regenerate."
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        z0 = np.load(self.dataset_dir / row["origin_z"]).reshape(-1).astype(np.float32)
        beta = load_beta(self.dataset_dir / row["morphology"])

        # Kept as an optional field so an older manifest (or a regenerated one)
        # still works; the current dataset has none, see the module docstring.
        qpos_ref = None
        rel = row.get("retargeted_motion")
        if rel:
            qpos_ref = np.load(self.dataset_dir / rel)["qpos"]

        return {
            "reward_name": row["reward_name"],
            "trial": row["trial"],
            "z0": z0,
            "beta": beta,
            "qpos_ref": qpos_ref,
        }
