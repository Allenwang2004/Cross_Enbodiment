import sys
sys.path.insert(0, "scripts")
import os
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np
import mujoco
import imageio

from render_qpos_playback import render_qpos

video_dir = Path("outputs/robot_motion_video")
retarget_npz_dir = Path("data/retargeted_motion")
out_dir = Path("outputs/retarget_robot_motion_video")
out_dir.mkdir(parents=True, exist_ok=True)

tasks = sorted(p.stem for p in video_dir.glob("*.mp4"))
print(f"{len(tasks)} tasks")

missing = []
for i, task in enumerate(tasks):
    npz_path = retarget_npz_dir / f"{task}_0.npz"
    if not npz_path.exists():
        missing.append(task)
        continue
    qpos = np.load(npz_path)["qpos"]
    frames = render_qpos("assets/robots/robot_child.xml", qpos)
    out_path = out_dir / f"{task}.mp4"
    imageio.mimsave(out_path, frames, fps=30)
    print(f"[{i+1}/{len(tasks)}] {task}: {len(frames)} frames -> {out_path}")

if missing:
    print(f"MISSING retargeted npz for: {missing}")