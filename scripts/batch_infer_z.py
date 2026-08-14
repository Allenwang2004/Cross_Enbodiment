"""Batch qpos -> z inference over a whole motion tree, mirroring its layout.

This is the batch driver for `infer_z_from_qpos.py`: same conversion (load
qpos into a HumEnv instance for the given skeleton, read
`get_obs()["proprio"]`, run Metamotivo's `tracking_inference`), but the
FB-CPR model and the HumEnv instance are built ONCE and reused across all
clips. Per-clip re-instantiation is what makes the single-clip script
unusable for the 540-clip tree (model load dominates by orders of
magnitude).

The env is created with `--xml`, so the obs are the ones the TARGET body
actually produces -- inferring z for a retargeted child motion means the
proprio features come off the child skeleton, not the adult's.

Input layout  <input_dir>/<task>/<task>_<trial>.npz   (key "qpos")
Output layout <output_dir>/<task>/<task>_<trial>.npy  ((T, 256) z_track)

matching `data/z/`'s <task>/<task>_<trial> naming (that one is (1, 256)
reward-inferred z; this one is per-frame tracking-inferred z).

Usage (from project root):
    uv run scripts/batch_infer_z.py \
        --input_dir amass_origin_motion \
        --output_dir amass_z_inference \
        --xml assets/robots/adult/robot.xml \
        [--z0_dir data/z]   # also report cosine against the origin z

With --z0_dir, a per-clip summary (mean/min/max cosine of z_track_t against
the source clip's z0) is written to <output_dir>/cosine_summary.csv -- that
is the number that says how much the body swap degraded the inference.
"""

import argparse
import csv
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
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--xml", required=True, help="skeleton the input qpos is expressed on")
    parser.add_argument("--method", choices=["tracking", "goal"], default="tracking",
                        help="tracking: B(s) averaged over the next cfg.seq_length (8) frames, "
                             "then projected. goal: B(s) projected per frame, no averaging at all "
                             "-- one independent goal-reaching z per step.")
    parser.add_argument("--z0_dir", type=Path, default=None,
                        help="optional <task>/<clip>.npy tree of source z, for cosine reporting")
    parser.add_argument("--model", default="facebook/metamotivo-M-1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    npz_paths = sorted(args.input_dir.rglob("*.npz"))
    if not npz_paths:
        raise SystemExit(f"no .npz under {args.input_dir}")
    print(f"Found {len(npz_paths)} clips under {args.input_dir}")

    model = FBcprModel.from_pretrained(args.model).to(args.device)
    model.eval()
    env, _ = make_humenv(num_envs=1, task=None, xml=args.xml, state_init="Default")

    rows = []
    try:
        for i, npz_path in enumerate(npz_paths):
            qpos = np.load(npz_path)["qpos"]
            obs_seq = qpos_to_obs_seq(env, qpos, args.device)
            with torch.no_grad():
                infer = (model.tracking_inference if args.method == "tracking"
                         else model.goal_inference)
                z_track = infer(next_obs=obs_seq).cpu().numpy()

            rel = npz_path.relative_to(args.input_dir).with_suffix(".npy")
            out_path = args.output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, z_track)

            if args.z0_dir is not None:
                z0_path = args.z0_dir / rel
                if z0_path.exists():
                    z0 = np.load(z0_path).reshape(1, -1)
                    cos = (z_track @ z0.T).ravel() / (
                        np.linalg.norm(z_track, axis=1) * np.linalg.norm(z0) + 1e-12)
                    rows.append([str(rel.with_suffix("")), z_track.shape[0],
                                 float(cos.mean()), float(cos.min()), float(cos.max())])
                else:
                    print(f"  (no z0 for {rel})")

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(npz_paths)}")
    finally:
        env.close()

    print(f"Wrote {len(npz_paths)} z files -> {args.output_dir}")

    if rows:
        summary = args.output_dir / "cosine_summary.csv"
        with open(summary, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip", "T", "cos_mean", "cos_min", "cos_max"])
            w.writerows(rows)
        means = np.array([r[2] for r in rows])
        print(f"cosine(z_track_t, z0) over {len(rows)} clips: "
              f"mean={means.mean():.4f} p10={np.percentile(means, 10):.4f} "
              f"p90={np.percentile(means, 90):.4f} -> {summary}")


if __name__ == "__main__":
    main()
