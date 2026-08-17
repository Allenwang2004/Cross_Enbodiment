"""Reverse-infer z from a qpos motion (Metamotivo's tracking_inference /
goal_inference), and optionally compare it against a known origin z.

qpos -> obs conversion is done by loading the qpos into a HumEnv instance
(env.unwrapped.set_physics) and reading env.unwrapped.get_obs()["proprio"] --
tracking_inference/goal_inference operate on this obs feature vector, not
raw qpos.

Usage (from project root):
    # just see what z looks like, no comparison
    uv run scripts/infer_z_from_qpos.py --qpos data/origin_motion/crawl-0.4-0-d/crawl-0.4-0-d_0.npz \
        --xml assets/robots/adult/robot.xml

    # compare against the z that actually generated this motion
    uv run scripts/infer_z_from_qpos.py --qpos data/origin_motion/move-ego--90-2/move-ego--90-2_0.npz \
        --xml assets/robots/adult/robot.xml \
        --z0 data/z/move-ego--90-2/move-ego--90-2_0.npy \
        --out outputs/infer_z_from_qpos/move-ego--90-2_0

    # same motion, but retargeted onto a different body -- see how much
    # cross-embodiment inference degrades vs the same-body case above
    uv run scripts/infer_z_from_qpos.py --qpos data/retargeted_motion/move-ego--90-2_0.npz \
        --xml assets/robots/child/robot.xml \
        --z0 data/z/move-ego--90-2/move-ego--90-2_0.npy \
        --out outputs/infer_z_from_qpos/move-ego--90-2_0_child

Writes (if --out is given):
    <out>_z_track.npy   (T, 256) -- per-step tracking_inference z
    <out>_z_goal.npy    (1, 256) -- goal_inference z from the last frame
    <out>_cosine.png    plot of cosine(z_track_t, z0) over time (only if --z0 given)
    <out>_cosine.csv    the raw per-step cosine values (only if --z0 given)
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from humenv import make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel


def qpos_to_obs_seq(env, qpos_seq, device):
    nv = env.unwrapped.model.nv
    obs_hist = []
    for t in range(qpos_seq.shape[0]):
        env.unwrapped.set_physics(qpos=qpos_seq[t], qvel=np.zeros(nv))
        obs_hist.append(env.unwrapped.get_obs()["proprio"].copy())
    return torch.tensor(np.stack(obs_hist), dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpos", required=True, help="path to a qpos .npz (key 'qpos')")
    parser.add_argument("--xml", required=True, help="skeleton the qpos is expressed on")
    parser.add_argument("--z0", default=None, help="optional known z (.npy) to compare against")
    parser.add_argument("--out", default=None, help="output path prefix (no extension)")
    parser.add_argument("--model", default="facebook/metamotivo-M-1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    qpos_seq = np.load(args.qpos)["qpos"]

    model = FBcprModel.from_pretrained(args.model).to(args.device)
    model.eval()

    env, _ = make_humenv(num_envs=1, task=None, xml=args.xml, state_init="Default")
    obs_seq = qpos_to_obs_seq(env, qpos_seq, args.device)
    env.close()

    with torch.no_grad():
        z_track = model.tracking_inference(next_obs=obs_seq)
        z_goal = model.goal_inference(next_obs=obs_seq[-1:])

    print(f"{qpos_seq.shape[0]} frames -> z_track {tuple(z_track.shape)}, z_goal {tuple(z_goal.shape)}")

    cosine = None
    if args.z0 is not None:
        z0 = torch.tensor(np.load(args.z0), dtype=torch.float32, device=args.device).reshape(1, -1)
        cosine = torch.nn.functional.cosine_similarity(z_track, z0.expand_as(z_track)).cpu().numpy()
        cos_goal = torch.nn.functional.cosine_similarity(z_goal, z0).item()
        print(f"cosine(z_track_t, z0): mean={cosine.mean():.4f} min={cosine.min():.4f} "
              f"max={cosine.max():.4f} (t=0: {cosine[0]:.4f}, t=-1: {cosine[-1]:.4f})")
        print(f"cosine(z_goal, z0): {cos_goal:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(f"{args.out}_z_track.npy", z_track.cpu().numpy())
        np.save(f"{args.out}_z_goal.npy", z_goal.cpu().numpy())
        print(f"wrote {args.out}_z_track.npy, {args.out}_z_goal.npy")

        if cosine is not None:
            import csv
            with open(f"{args.out}_cosine.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "cosine_to_z0"])
                for t, c in enumerate(cosine):
                    w.writerow([t, c])

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 4))
            plt.plot(cosine)
            plt.axhline(cos_goal, color="orange", linestyle="--", label=f"goal_inference ({cos_goal:.3f})")
            plt.xlabel("timestep")
            plt.ylabel("cosine(z_track_t, z0)")
            plt.title(Path(args.qpos).name)
            plt.ylim(-1, 1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{args.out}_cosine.png", dpi=120)
            print(f"wrote {args.out}_cosine.png, {args.out}_cosine.csv")


if __name__ == "__main__":
    main()
