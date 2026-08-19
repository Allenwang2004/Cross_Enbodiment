#!/usr/bin/env python3
"""infer_z_from_action.py — recover z from an (obs, action) stream by inverting
the frozen actor.

The gap this fills
------------------
Every z metamotivo can produce goes through the backward map B, and B's first
layer is Linear(obs_dim, hidden) -- there is no place to put an action:

    reward_inference(next_obs, reward)     -> project_z(sum wr * B(next_obs))
    goal_inference(next_obs)               -> project_z(B(next_obs))
    tracking_inference(next_obs)           -> project_z(movavg(B(next_obs)))

So "which z produced these actions" has no amortised answer and has to be
solved. The actor is the only network that sees an action and a z at the same
time, and it is where the question actually lives.

Why this is a least-squares problem
-----------------------------------
The actor returns TruncatedNormal(mu, std) with std = cfg.actor_std = 0.2, a
constant that does not depend on the state (nn_models.py:233 -- std is passed
in, not predicted). A Gaussian log-likelihood with fixed variance is the squared
error up to an additive constant, so maximum likelihood over the clip is

    z* = argmin_z  sum_t || a_t - tanh(policy(obs_t, z)) ||^2     s.t. ||z|| = 16

69 * T residuals for 256 unknowns -- at T=300 that is 20700 equations, heavily
overdetermined. The sphere constraint is not optional: norm_z is True for this
checkpoint, so project_z rescales every z the actor was ever trained on to
sqrt(256) = 16, and an unconstrained solution lands somewhere the actor has
never seen.

Three things that must be done exactly this way
-----------------------------------------------
  * model._actor, not model.act/model.actor. Both public paths are
    @torch.no_grad()-wrapped, so gradients never reach z. The frozen parameters
    keep requires_grad=False, which does NOT block the gradient from flowing
    back to an input.
  * project_z inside the graph, not after each optimiser step. Projecting
    afterwards lets the iterate drift off the sphere and be yanked back, so the
    step the optimiser took is not the step it is scored on.
  * obs must carry real velocities. 144 of the 358 features are
    local_body_vel + local_body_ang_vel, and zeroing them does not read as
    "no information" -- the obs BatchNorm turns them into a fixed negative
    offset, i.e. a pose that is confidently standing still.

Exact in the limit, and the cost is iterations rather than accuracy
-------------------------------------------------------------------
Substituting the true z0 gives MSE 8.3e-13, so the forward map is deterministic
(rollout_once calls act with mean=True) and the global minimum is the answer,
not an approximation of it. The optimisation walks to it monotonically -- on
move-ego--90-2_0, from the tracking_inference seed:

     400 steps   MSE 2.2e-05   cos(z,z0) 0.9788
    1000 steps   MSE 4.3e-06             0.9914
    2000 steps   MSE 6.1e-07             0.9979
    4000 steps   MSE 2.8e-08             0.9998
    8000 steps   MSE 2.7e-09             0.99997

LBFGS from the 8000-step point moves nothing (2.734e-9 vs 2.735e-9), so 1e-9 is
where float32 flattens out, not a competing optimum. --steps is therefore an
accuracy dial, and the default trades the last two digits for wall time.

There is one basin, not many. Random seeds reach cos 0.906..0.932 after 8000
steps (mutually 0.879..0.892) -- far worse than the tracking seed at the same
budget, but converging on the same point, and their MSE ranks their cosine
exactly. An earlier 600-step run of the same test read as separate local minima
at cos 0.46..0.55; that was under-training, and any conclusion drawn from a loss
that is still falling would repeat the mistake.

Two practical consequences. Seed from tracking_inference -- it is worth
thousands of iterations, not correctness. And trust the final MSE: it tracks
cosine monotonically across every run above, so on clips with no known z0 it is
the only quality signal available, and the one --mse-warn thresholds.

Reading the output
------------------
data/origin_action stores the z that generated each clip alongside the actions,
so cos(inverted, z0) has a known-good answer here: those actions came out of
model.act(obs, z0) with that exact z0. A low cosine on this data means the
optimisation is wrong, not that the motion is hard. cos(tracking, z0) is printed
beside it as the baseline this is trying to beat -- it is ~0.50 on adult clips
(data/z_inference/cosine_summary.csv).

Usage (from project root):
    uv run scripts/infer_z_from_action.py --limit 10
    uv run scripts/infer_z_from_action.py --clip data/origin_action/jump-2/jump-2_0.npz
    uv run scripts/infer_z_from_action.py --limit 50 --steps 600 --device cuda

Writes <out>/per_clip.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def qpos_to_obs(env, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """(T, nq) + (T, nv) -> (T, 358) proprio, the same conversion the rollout saw."""
    e = env.unwrapped
    out = np.empty((len(qpos), 358))
    for t in range(len(qpos)):
        e.set_physics(qpos=qpos[t], qvel=qvel[t])
        out[t] = e.get_obs()["proprio"]
    return out


def acting_states(rec) -> tuple[np.ndarray, np.ndarray]:
    """The states the recorded actions were actually computed FROM.

    metamotivo_motion_rollout.py:rollout_once appends qpos AFTER env.step, so
    qpos[t] is the result of action[t], not its input -- action[t] was chosen
    from qpos[t-1], and action[0] from the pre-rollout qpos_init that the record
    stores for exactly this reason. Pairing action[t] with qpos[t] shifts the
    whole clip by one control step and silently turns an exactly solvable
    problem into an approximate one.
    """
    qpos = np.concatenate([rec["qpos_init"][None], rec["qpos"][:-1]])
    qvel = np.concatenate([rec["qvel_init"][None], rec["qvel"][:-1]])
    return qpos, qvel


def invert_actor(model, obs_t: torch.Tensor, act_t: torch.Tensor, *,
                 steps: int = 400, lr: float = 3e-2, z_init: torch.Tensor | None = None):
    """Least-squares z reproducing act_t from obs_t through the frozen actor.

    Returns (z, loss_first, loss_best). The returned z is the best iterate seen,
    not the last: Adam on a sphere-constrained non-convex objective can step past
    a good point, and there is no reason to hand back a worse answer than one
    already evaluated.
    """
    if z_init is None:
        with torch.no_grad():
            z_init = model.tracking_inference(next_obs=obs_t).mean(0, keepdim=True)
    z = torch.nn.Parameter(model.project_z(z_init.detach().clone()))
    opt = torch.optim.Adam([z], lr=lr)

    # eval-mode BatchNorm: a fixed affine map, so hoist it out of the loop.
    obs_norm = model._normalize(obs_t)
    T = obs_t.shape[0]

    first = None
    best = float("inf")
    best_z = z.detach().clone()
    for i in range(steps):
        zp = model.project_z(z).expand(T, -1)
        mu = model._actor(obs_norm, zp, model.cfg.actor_std).mean
        loss = ((mu - act_t) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        val = loss.detach().item()
        if i == 0:
            first = val
        if val < best:
            best, best_z = val, z.detach().clone()

    with torch.no_grad():
        return model.project_z(best_z), first, best


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--action-dir", default="data/origin_action",
                   help="clips holding action + qpos + qvel (+ z, if scoring)")
    p.add_argument("--clip", default=None, help="a single .npz instead of --action-dir")
    p.add_argument("--z-dir", default="data/z", help="reference z0 for scoring")
    p.add_argument("--steps", type=int, default=2000,
                   help="accuracy dial, not a convergence criterion: 400 -> cos "
                        "0.979, 2000 -> 0.998, 8000 -> 0.99997 (see docstring)")
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--limit", type=int, default=10, help="0 = every clip")
    p.add_argument("--mse-warn", type=float, default=1e-5,
                   help="flag clips whose final MSE stays above this: at "
                        "--steps 2000 a healthy clip reaches ~1e-6..1e-7, so a "
                        "higher value means that clip stopped converging")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model", default="facebook/metamotivo-M-1")
    p.add_argument("--save-z", default=None,
                   help="directory to write <task>/<task>_<trial>.npy")
    p.add_argument("--out", default="outputs/infer_z_from_action")
    args = p.parse_args()

    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel

    env, _ = make_humenv(num_envs=1)
    model = FBcprModel.from_pretrained(args.model).to(args.device)

    paths = ([Path(args.clip)] if args.clip
             else sorted(Path(args.action_dir).glob("*/*.npz")))
    if args.limit and not args.clip:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit("no clips found")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, path in enumerate(paths):
        rec = np.load(path)
        missing = [k for k in ("qpos", "qvel", "action", "qpos_init", "qvel_init")
                   if k not in rec.files]
        if missing:
            raise SystemExit(f"{path} lacks {missing}; this needs the replay "
                             f"record written by metamotivo_motion_rollout.py, "
                             f"not the qpos-only origin_motion clip")
        # rollout_once records resets as t+1, so a value equal to the stream
        # length fired after the last recorded action and splices nothing. Only
        # an earlier one joins two episodes into one action stream.
        mid = [int(r) for r in np.atleast_1d(rec["resets"]) if 0 < r < len(rec["action"])]
        if mid:
            print(f"  note: {path.stem} reset mid-rollout at {mid}; the state "
                  f"chain breaks there and those frames are inconsistent")

        obs = qpos_to_obs(env, *acting_states(rec))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device)
        act_t = torch.as_tensor(rec["action"], dtype=torch.float32, device=args.device)

        z_hat, l0, l1 = invert_actor(model, obs_t, act_t,
                                     steps=args.steps, lr=args.lr)

        with torch.no_grad():
            z_track = model.project_z(
                model.tracking_inference(next_obs=obs_t).mean(0, keepdim=True))

        # z0: prefer the copy stored in the record itself (exactly the vector
        # this rollout used); fall back to data/z for records without it.
        if "z" in rec.files:
            z0 = torch.as_tensor(rec["z"], dtype=torch.float32, device=args.device)
        else:
            z0_path = Path(args.z_dir) / path.parent.name / f"{path.stem}.npy"
            z0 = (torch.as_tensor(np.load(z0_path), dtype=torch.float32,
                                  device=args.device) if z0_path.exists() else None)

        c_inv = cos(z_hat, z0) if z0 is not None else float("nan")
        c_trk = cos(z_track, z0) if z0 is not None else float("nan")

        clip = f"{path.parent.name}/{path.stem}"
        # Not a local minimum -- there is only one basin (see docstring). A high
        # final MSE means this clip was still descending when --steps ran out,
        # so the fix is more iterations, not a different seed.
        flag = "  !! still converging, raise --steps" if l1 > args.mse_warn else ""
        rows.append({"clip": clip, "T": obs.shape[0],
                     "mse_start": f"{l0:.5f}", "mse_end": f"{l1:.7f}",
                     "cos_inverted": f"{c_inv:.4f}", "cos_tracking": f"{c_trk:.4f}",
                     "suspect": int(bool(flag))})
        print(f"[{i+1}/{len(paths)}] {clip:36s} MSE {l0:.4f}->{l1:.2e}  "
              f"cos(inv,z0) {c_inv:+.4f}  cos(track,z0) {c_trk:+.4f}{flag}")

        if args.save_z:
            d = Path(args.save_z) / path.parent.name
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{path.stem}.npy", z_hat.cpu().numpy())

    csv_path = out_dir / "per_clip.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ci = np.array([float(r["cos_inverted"]) for r in rows])
    ct = np.array([float(r["cos_tracking"]) for r in rows])
    m0 = np.array([float(r["mse_start"]) for r in rows])
    m1 = np.array([float(r["mse_end"]) for r in rows])
    n_bad = sum(r["suspect"] for r in rows)
    print(f"\n{len(rows)} clips")
    print(f"  action MSE      {m0.mean():.4f} -> {m1.mean():.2e} "
          f"(worst clip {m1.max():.2e})")
    if n_bad:
        print(f"  {n_bad} clip(s) above --mse-warn {args.mse_warn:g}: still "
              f"descending at --steps {args.steps}, rerun them with more")
    print(f"  cos(inverted,z0)  mean {np.nanmean(ci):+.4f}  "
          f"min {np.nanmin(ci):+.4f}  max {np.nanmax(ci):+.4f}")
    print(f"  cos(tracking,z0)  mean {np.nanmean(ct):+.4f}   <- baseline to beat")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
