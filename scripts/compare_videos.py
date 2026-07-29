"""Combine two already-rendered .mp4 files side by side (or stacked) for
direct visual comparison. Unlike render_qpos_playback.py (which renders qpos
from scratch), this just reads and stacks frames from two existing videos.

Usage (from project root):
    uv run scripts/compare_videos.py --video1 a.mp4 --video2 b.mp4 --out compare.mp4
    uv run scripts/compare_videos.py --video1 a.mp4 --video2 b.mp4 --out compare.mp4 --layout vertical
"""

import argparse
from pathlib import Path

import imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", required=True)
    parser.add_argument("--video2", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=None,
                         help="defaults to video1's own fps")
    parser.add_argument("--layout", choices=["horizontal", "vertical"], default="horizontal")
    args = parser.parse_args()

    reader1 = imageio.get_reader(args.video1)
    reader2 = imageio.get_reader(args.video2)

    fps = args.fps or reader1.get_meta_data().get("fps", 30.0)
    axis = 1 if args.layout == "horizontal" else 0

    frames = []
    n1 = n2 = 0
    for f1, f2 in zip(reader1, reader2):
        n1 += 1
        n2 += 1
        h = min(f1.shape[0], f2.shape[0])
        w = min(f1.shape[1], f2.shape[1])
        frames.append(np.concatenate([f1[:h, :w], f2[:h, :w]], axis=axis))

    reader1.close()
    reader2.close()

    if len(frames) < max(n1, n2):
        print(f"NOTE: video1 and video2 have different lengths -- truncated "
              f"to the shorter one ({len(frames)} frames)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=fps)
    print(f"wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
