"""Behavioural check on the map learned by fit_z_map.py.

Cosine to z0 is only a proxy: FB's reward is <B(s), z>, so any component of z
orthogonal to the span of B does not change the induced policy. A higher cosine
therefore does not guarantee the actor actually does the task. This script
closes that loop -- it drives the frozen actor on the adult body with each z
variant and scores the resulting trajectory with the real humenv task reward.

Per held-out task, three rollouts:
    z0      data/z/<task>/<task>_<trial>.npy   -- upper bound, this generated the clip
    raw     project_z(mean_t z_track_t)        -- current baseline, no map
    mapped  project_z(mean_z @ W)              -- what fit_z_map.py produces

The map is the one fitted WITHOUT these tasks (W_heldout), so this is a genuine
generalisation test rather than self-validation.

Usage (from project root):
    python3 scripts/eval_z_map_rollout.py
    python3 scripts/eval_z_map_rollout.py --steps 300 --trial 0
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
# see fit_z_map.py: unpinned OpenBLAS thrashes badly on small matrices
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from humenv import make_humenv
from humenv.env import make_from_name
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.wrappers.humenvbench import relabel


def project_z(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True) * np.sqrt(v.shape[-1])


def apply_map(v, W):
    """Keep in sync with fit_z_map.py:apply_map. W was fitted on unit vectors;
    normalising first is redundant for a linear map followed by project_z, but
    makes the contract explicit."""
    v = np.asarray(v, dtype=np.float64)
    return project_z((v / np.linalg.norm(v)) @ W)


def rollout(model, env, z, steps, seed, device):
    """Constant-z rollout. Mirrors rollout_z_on_body.py:134-159 but without
    rendering; keeps frames-worth of qpos/qvel/action for relabelling."""
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    z_t = torch.tensor(np.asarray(z, dtype=np.float32), device=device).reshape(1, -1)
    qpos, qvel, act = [], [], []
    diverged = False
    for _ in range(steps):
        obs_t = torch.tensor(obs["proprio"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = model.act(obs_t, z_t, mean=True)
        a = action.cpu().numpy().ravel()
        try:
            obs, _, terminated, truncated, info = env.step(a)
        except ValueError:
            # humenv raises on mjWARN_BADQACC; keep what we have (see
            # rollout_z_on_body.py:142-151) rather than losing the run.
            diverged = True
            break
        act.append(a.copy())
        qpos.append(info["qpos"].copy())
        qvel.append(info["qvel"].copy())
        if terminated or truncated:
            break
    return np.stack(qpos), np.stack(qvel), np.stack(act).astype(np.float32), diverged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="outputs/z_map/W.npz")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--tree", default="z_inference")
    ap.add_argument("--xml", default="assets/robots/adult/robot.xml")
    ap.add_argument("--out-dir", default="outputs/z_map")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--model", default="facebook/metamotivo-M-1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relabel-workers", type=int, default=8)
    args = ap.parse_args()

    # thread count changes float32 reduction order and so changes the
    # trajectory; pinned like rollout_z_on_body.py:79-81
    torch.set_num_threads(1)

    blob = np.load(args.map, allow_pickle=True)
    W = blob["W_heldout"]
    held_out = [str(t) for t in blob["held_out_tasks"]]
    print(f"map: lam={blob['lam']} alpha={blob['alpha']} split_seed={blob['split_seed']}")
    print(f"{len(held_out)} held-out tasks, trial {args.trial}, {args.steps} steps on {args.xml}\n")

    root = Path(args.data_root)
    model = FBcprModel.from_pretrained(args.model).to(args.device)
    model.eval()
    env, _ = make_humenv(num_envs=1, task=None, xml=args.xml, state_init="Default")

    rows = []
    for i, task in enumerate(held_out):
        name = f"{task}_{args.trial}"
        z0_path = root / "z" / task / f"{name}.npy"
        zi_path = root / args.tree / task / f"{name}.npy"
        if not (z0_path.exists() and zi_path.exists()):
            print(f"[{i+1}/{len(held_out)}] {task}: missing z, skipped")
            continue

        mean_z = project_z(np.load(zi_path).mean(axis=0))
        variants = {
            "z0": project_z(np.load(z0_path).reshape(-1)),
            "raw": mean_z,
            "mapped": apply_map(mean_z, W),
        }
        reward_fn = make_from_name(task)
        row = {"task": task}
        msg = []
        for vname, z in variants.items():
            qpos, qvel, act, diverged = rollout(model, env, z, args.steps, args.seed, args.device)
            r = relabel(env, qpos=qpos, qvel=qvel, action=act,
                        reward_fn=reward_fn, max_workers=args.relabel_workers)
            score = float(np.asarray(r).mean())
            row[vname] = score
            row[vname + "_diverged"] = diverged
            msg.append(f"{vname} {score:.4f}" + ("!" if diverged else ""))
        rows.append(row)
        print(f"[{i+1}/{len(held_out)}] {task:34s} " + "  ".join(msg))
    env.close()

    if not rows:
        raise SystemExit("no tasks evaluated")

    # Normalise within task so different reward scales are comparable. Tasks
    # where even z0 scores ~0 carry no signal and are excluded from the summary.
    EPS = 1e-3
    usable = [r for r in rows if r["z0"] > EPS]
    print(f"\n{len(usable)}/{len(rows)} tasks have a usable z0 score (> {EPS})")
    print(f"{'variant':10s} {'mean raw':>10s} {'mean / z0':>12s}")
    summary = {}
    for v in ["z0", "raw", "mapped"]:
        raw_m = float(np.mean([r[v] for r in usable]))
        rel_m = float(np.mean([r[v] / r["z0"] for r in usable]))
        summary[v] = (raw_m, rel_m)
        print(f"{v:10s} {raw_m:10.4f} {rel_m:12.3f}")
    wins = sum(1 for r in usable if r["mapped"] > r["raw"])
    print(f"\nmapped beats raw on {wins}/{len(usable)} tasks")

    # The headline mean hides the actual structure: the map helps exactly where
    # the untransformed tracking embedding fails. A pose task's tracking z is
    # already a serviceable reward z (both describe a fixed point), so there is
    # nothing to fix and the map only adds error; a locomotion task needs a
    # limit cycle, which is where the goal-like tracking z breaks down.
    THRESH = 0.75
    for label, sel in [("raw already works (raw/z0 >= %.2f)" % THRESH,
                        lambda r: r["raw"] / r["z0"] >= THRESH),
                       ("raw fails      (raw/z0 <  %.2f)" % THRESH,
                        lambda r: r["raw"] / r["z0"] < THRESH)]:
        grp = [r for r in usable if sel(r)]
        if not grp:
            continue
        rr = np.mean([r["raw"] / r["z0"] for r in grp])
        mm = np.mean([r["mapped"] / r["z0"] for r in grp])
        w = sum(1 for r in grp if r["mapped"] > r["raw"])
        print(f"  {label}: n={len(grp):2d}  raw {rr:.3f} -> mapped {mm:.3f}  "
              f"({w}/{len(grp)} improved)")

    print(f"\n{'task family':16s} {'n':>3s} {'raw/z0':>8s} {'mapped/z0':>10s}")
    fams = {}
    for r in usable:
        fams.setdefault(r["task"].split("-")[0], []).append(r)
    for fam, grp in sorted(fams.items()):
        print(f"  {fam:14s} {len(grp):3d} "
              f"{np.mean([r['raw'] / r['z0'] for r in grp]):8.3f} "
              f"{np.mean([r['mapped'] / r['z0'] for r in grp]):10.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    order = sorted(usable, key=lambda r: -r["z0"])
    labels = [r["task"] for r in order]
    xs = np.arange(len(order))
    width = 0.27
    plt.figure(figsize=(max(9, 0.62 * len(order)), 5.8))
    for k, (v, color) in enumerate(zip(["z0", "raw", "mapped"],
                                       ["#55A868", "#999999", "#DD8452"])):
        vals = [r[v] / r["z0"] for r in order]
        plt.bar(xs + (k - 1) * width, vals, width,
                label=f"{v} (mean {summary[v][1]:.2f})", color=color)
    plt.axhline(1.0, color="#55A868", linestyle="--", alpha=0.6)
    plt.xticks(xs, labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("humenv task reward / $z_0$ reward")
    plt.title(f"Rollout on {Path(args.xml).parent.name} body, held-out tasks\n"
              f"does the recovered $\\hat{{z}}$ actually do the task?")
    plt.legend(fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "rollout_reward.png", dpi=200)
    plt.close()
    print(f"wrote {out_dir / 'rollout_reward.png'}")


if __name__ == "__main__":
    main()
