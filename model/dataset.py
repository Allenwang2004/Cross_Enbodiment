"""Reads data/crossenbodiment-1-datasets/manifest.jsonl (see
scripts/build_dataset.py). retargeted_motion is optional per row -- rows
without it yet just get D=0 during training (see losses.functional_equivalence)."""

import json
from pathlib import Path

import numpy as np

# Must match scripts/scale_robot.py's AXES order -- beta is this 8-dim
# vector read straight out of a robot_<label>.json.
BETA_AXES = ["leg_scale", "arm_scale", "torso_scale", "head_scale",
             "leg_girth", "arm_girth", "torso_girth", "head_girth"]


def load_beta(morphology_json_path) -> np.ndarray:
    d = json.loads(Path(morphology_json_path).read_text())
    return np.array([d[a] for a in BETA_AXES], dtype=np.float32)


class CrossEmbodimentDataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        manifest_path = self.dataset_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found -- run scripts/build_dataset.py first"
            )
        self.rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"no rows in {manifest_path}")

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
