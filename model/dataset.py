"""Reads datasets/crossenbodiment-1-datasets/manifest.jsonl (see
scripts/build_dataset.py). retargeted_motion is optional per row -- rows
without it yet just get D=0 during training (see losses.functional_equivalence)."""

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

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        z0 = np.load(self.dataset_dir / row["origin_z"]).reshape(-1).astype(np.float32)
        beta = load_beta(self.dataset_dir / row["morphology"])

        qpos_ref = None
        if row["retargeted_motion"] is not None:
            qpos_ref = np.load(self.dataset_dir / row["retargeted_motion"])["qpos"]

        return {
            "reward_name": row["reward_name"],
            "trial": row["trial"],
            "z0": z0,
            "beta": beta,
            "qpos_ref": qpos_ref,
        }
