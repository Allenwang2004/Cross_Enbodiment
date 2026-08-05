"""Render a qpos-only .npz (kinematic playback, no physics) into an mp4.
Optionally render a second qpos file (e.g. the retargeted version) side by
side for direct visual comparison.

Usage (from project root):
    cd /home/allen19/crossenbodiment
    uv run scripts/render_qpos_playback.py \
        --qpos data/origin_motion/move-ego--90-2/move-ego--90-2_0.npz --xml assets/robots/adult/robot.xml \
        --qpos2 data/origin_motion/move-ego--90-2/move-ego--90-2_0.npz --xml2 assets/robots/child/robot.xml \
        --out outputs/robot_video/compare.mp4 2>&1 | grep -v "EGLError\|Exception ignored\|Traceback\|File \"\|renderer.py\|__init__.py\|error.py"
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np


def render_qpos(xml_path, qpos_seq, width=480, height=640, camera="front_side"):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = []
    for t in range(qpos_seq.shape[0]):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpos", required=True, help="path to a qpos .npz (key 'qpos')")
    parser.add_argument("--xml", required=True, help="MJCF the qpos was generated/retargeted for")
    parser.add_argument("--qpos2", default=None, help="optional second qpos .npz for side-by-side")
    parser.add_argument("--xml2", default=None, help="MJCF for --qpos2 (defaults to --xml)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera", default="front_side")
    args = parser.parse_args()

    qpos1 = np.load(args.qpos)["qpos"]
    frames1 = render_qpos(args.xml, qpos1, camera=args.camera)

    if args.qpos2:
        qpos2 = np.load(args.qpos2)["qpos"]
        xml2 = args.xml2 or args.xml
        frames2 = render_qpos(xml2, qpos2, camera=args.camera)
        n = min(len(frames1), len(frames2))
        frames = [np.concatenate([frames1[i], frames2[i]], axis=1) for i in range(n)]
    else:
        frames = frames1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
