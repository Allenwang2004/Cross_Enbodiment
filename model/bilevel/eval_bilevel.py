"""Held-out evaluation on the body x task grid.

Resolves the open question the proposal left standing: which dimension is held
out. The answer is BOTH, evaluated as four quadrants, because they measure
different things and only the last one is generalization in the sense the
project cares about:

    seen body  x seen task     training distribution -- a sanity floor
    seen body  x unseen task   does the adapter generalize across MOTIONS?
    unseen body x seen task    does it generalize across MORPHOLOGIES?
    unseen body x unseen task  both at once -- the honest headline number

Bodies split by BilevelConfig.train_bodies / heldout_bodies (mirrored into each
parameter.json by scripts/write_body_splits.py); tasks by the existing
datasets/.../splits/{train,test}_tasks.txt, so the task axis stays identical to
the one model/evaluate.py and model/baseline.py used and the numbers remain
comparable to outputs/{baseline,eval}/report.json.

Two things this does differently from training, deliberately:

  f_max = 0        The external root wrench is a training crutch (proposal.md
                   R5). Every reported number is taken without it, from
                   iteration 1 -- otherwise the policy is being scored while
                   leaning on something the deliverable does not have.

  long rollouts    Training optimizes 24-step windows (0.8 s). That cannot see
                   whether the policy is stitching locally-good, globally
                   incoherent motion (proposal.md R6), so evaluation runs
                   hundreds of steps from a single RSI.

Scoring uses model/losses.py UNCHANGED -- functional_equivalence and
physics_penalty are whole-trajectory numpy functions, which is the wrong shape
for a per-step reward but exactly right here, and keeping them untouched is
what makes the comparison against the old baseline meaningful.

D is reported against BOTH references:
    D_phi   vs the phi-adjusted reference the upper level produced
    D_0     vs the naive phi=0 retarget
This is the decisive anti-degeneracy test from proposal.md 3.5. If D_phi falls
while D_0 does not, the upper level closed the gap by moving the reference
rather than by the robot improving.

Usage:
    uv run model/bilevel/eval_bilevel.py --ckpt model/bilevel/checkpoints/stage2_001000.pt
    uv run model/bilevel/eval_bilevel.py --ckpt ... --clips 20 --horizon 299
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from model import losses
from model.bilevel import rewards as R
from model.bilevel.config import BilevelConfig
from model.bilevel.data import WindowDataset, load_splits
from model.bilevel.policy import LowerPolicy, build_value_net, load_frozen_model
from model.bilevel.retarget import U_DIM, RetargetNet, Retargeter
from model.bilevel.rollout import Collector, sample_rsi_noise
from model.bilevel.sim.pool import SimPool
from model.bilevel.semantics import frac_illegal_frames

QUADRANTS = [
    ("seen_body_seen_task", True, True),
    ("seen_body_unseen_task", True, False),
    ("unseen_body_seen_task", False, True),
    ("unseen_body_unseen_task", False, False),
]

D_WEIGHTS = {"root": 1.0, "ee": 1.0, "contact": 1.0, "pose": 1.0, "velocity": 1.0}


def _eval_config(ckpt_cfg, horizon: int, n_envs: int, n_workers: int) -> BilevelConfig:
    """A config for evaluation: long horizon, no wrench, no RSI noise."""
    import copy

    cfg = copy.deepcopy(ckpt_cfg)
    cfg.horizon = horizon
    cfg.n_envs = n_envs
    cfg.n_workers = n_workers
    cfg.enable_time_warp = False
    cfg.wrench_enabled = False      # f_max = 0 -- see module docstring
    cfg.rsi_sigma_max = 0.0         # evaluate from the reference pose exactly
    return cfg


@torch.no_grad()
def eval_body(cfg, body: str, tasks, policy, value_net, net, source_rest_h,
              n_clips: int, seed: int, verbose=True):
    """Roll out `n_clips` full-length clips on one body. -> list of per-clip dicts.

    One body at a time: a MuJoCo sim is bound to one MjModel, so evaluating a
    single morphology across all slots avoids the fixed slot->body assignment
    that training relies on (data.py:body_assignment).
    """
    ds = WindowDataset(cfg, bodies=[body], tasks=tasks, verbose=False)
    if not ds.clips:
        return []
    spec = ds.bodies[0]
    spec.kin.to(cfg.device)
    ds.source.kin.to(cfg.device)

    retargeters = {0: Retargeter(cfg, net, spec.kin, source_rest_h).to(cfg.device).double()}
    fk_model = mujoco.MjModel.from_xml_path(str(spec.xml_path))

    rng = np.random.default_rng(seed)
    n = min(cfg.n_envs, len(ds.clips))
    pick = rng.choice(len(ds.clips), size=min(n_clips, len(ds.clips)), replace=False)

    rows = []
    for lo in range(0, len(pick), n):
        chunk = pick[lo:lo + n]
        # The pool has exactly n slots, so a short chunk is cycled to fill them.
        # Only the first len(chunk) rows are scored; the rest are duplicates.
        idx = np.resize(chunk, n)
        body_idx = np.zeros(n, dtype=np.int64)
        t0 = np.zeros(n, dtype=np.int64)          # from the top of every clip
        batch = ds.build_batch(idx, body_idx, t0)

        pool = SimPool(cfg, [str(spec.xml_path)] * n, [spec.leg_len] * n,
                       obs_dim=policy.obs_dim)
        try:
            coll = Collector(cfg, ds, pool, policy, value_net, retargeters, cfg.device)
            u_env = net(torch.as_tensor(batch["beta"], dtype=torch.float64,
                                        device=cfg.device))
            ep = coll.collect(
                it=10 ** 9,                                  # past every anneal -> f_max = 0
                batch=batch, u_per_env=u_env,
                rsi_noise=np.zeros((n, 69), dtype=np.float32),
                action_noise=torch.zeros(n, cfg.horizon, 75, device=cfg.device),
                deterministic=True,
            )
        finally:
            pool.close()

        # phi = 0 reference, for the decisive D_0 comparison
        u0 = torch.zeros(n, U_DIM, dtype=torch.float64, device=cfg.device)
        src = torch.as_tensor(batch["src_qpos"], dtype=torch.float64, device=cfg.device)
        beta = torch.as_tensor(batch["beta"], dtype=torch.float64, device=cfg.device)
        raw0, ref0, _ = retargeters[0](src, beta, u_override=u0, n_out=cfg.horizon + 1)
        raw_phi, _, _ = retargeters[0](src, beta, u_override=u_env, n_out=cfg.horizon + 1)

        tau = ep["qpos"].transpose(0, 1).cpu().numpy()        # (n, H, 76)
        valid = ep["valid"].transpose(0, 1).cpu().numpy()
        ref_phi = ep["ref"].cpu().numpy()[:, 1:]
        ref_0 = ref0.cpu().numpy()[:, 1:]
        terms = ep["terms"].transpose(0, 1).cpu().numpy()
        i = R.TERM_IDX

        for k in range(len(chunk)):
            alive = int(valid[k].sum())
            if alive < 2:
                traj = tau[k, :2]
            else:
                traj = tau[k, :alive]
            d_phi, _ = losses.functional_equivalence(fk_model, traj, ref_phi[k, :len(traj)], D_WEIGHTS)
            d_0, _ = losses.functional_equivalence(fk_model, traj, ref_0[k, :len(traj)], D_WEIGHTS)
            l_phys, _ = losses.physics_penalty(fk_model, traj)
            w = valid[k][:, None]
            denom = max(1.0, float(valid[k].sum()))
            rows.append({
                "body": body,
                "task": ds.clips[idx[k]].task,
                "trial": ds.clips[idx[k]].trial,
                "survived": alive / cfg.horizon,
                "D_phi": float(d_phi),
                "D_0": float(d_0),
                "L_phys": float(l_phys),
                "r_track": float(sum(
                    getattr(cfg, f"w_{c}") * (terms[k][:, i[f"r_{c}"]] * valid[k]).sum() / denom
                    for c in ("pose", "vel", "ee", "root", "com")
                )),
                "pose_err": float((terms[k][:, i["pose_err"]] * valid[k]).sum() / denom),
            })

    frac_ill_phi = float(frac_illegal_frames(spec.kin, raw_phi))
    frac_ill_0 = float(frac_illegal_frames(spec.kin, raw0))
    for r in rows:
        r["frac_illegal_phi"] = frac_ill_phi
        r["frac_illegal_0"] = frac_ill_0
    if verbose:
        print(f"    {body:<14} {len(rows):3d} clips   "
              f"survived {np.mean([r['survived'] for r in rows]):.2f}   "
              f"D_phi {np.mean([r['D_phi'] for r in rows]):7.2f}   "
              f"D_0 {np.mean([r['D_0'] for r in rows]):7.2f}   "
              f"L_phys {np.mean([r['L_phys'] for r in rows]):6.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=299,
                    help="rollout length in control steps; clips shorter than "
                         "horizon+1 frames are skipped (coverage is reported)")
    ap.add_argument("--clips", type=int, default=20, help="clips per (body, task-split) cell")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "bilevel_eval")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base_cfg = ck["cfg"]
    if args.device:
        base_cfg.device = args.device
    cfg = _eval_config(base_cfg, args.horizon, args.envs, args.workers)
    dev = torch.device(cfg.device)

    train_tasks, test_tasks = load_splits(cfg)
    if train_tasks is None:
        raise SystemExit("no task split found; evaluation needs one")

    frozen = load_frozen_model(cfg)
    beta_dim = 8
    policy = LowerPolicy(cfg, frozen, beta_dim).to(dev)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    value_net = build_value_net(cfg, frozen, beta_dim).to(dev)
    value_net.load_state_dict(ck["value"])
    net = RetargetNet(beta_dim, cfg.retarget_hidden_dims).to(dev).double()
    net.load_state_dict(ck["retarget_net"])
    net.eval()

    src_model = mujoco.MjModel.from_xml_path(
        str(REPO_ROOT / cfg.robots_dir / cfg.source_body / "robot.xml"))
    src_rest_h = float(src_model.qpos0[2])
    R.set_adult_leg_len(sum(
        float(np.linalg.norm(src_model.body_pos[src_model.body(b).id]))
        for b in ("L_Knee", "L_Ankle")
    ))

    print(f"checkpoint : {args.ckpt}  (stage {ck.get('stage', '?')}, iter {ck.get('iter', '?')})")
    print(f"rollout    : {args.horizon} steps @ 30 Hz = {args.horizon / 30:.1f} s, "
          f"f_max = 0, deterministic")
    print(f"bodies     : train {cfg.train_bodies}")
    print(f"             test  {cfg.heldout_bodies}")
    print(f"tasks      : {len(train_tasks)} train / {len(test_tasks)} test\n")

    all_rows = []
    t0 = time.perf_counter()
    for name, seen_body, seen_task in QUADRANTS:
        bodies = cfg.train_bodies if seen_body else cfg.heldout_bodies
        tasks = train_tasks if seen_task else test_tasks
        print(f"  {name}")
        for body in bodies:
            rows = eval_body(cfg, body, tasks, policy, value_net, net, src_rest_h,
                             n_clips=args.clips, seed=args.seed)
            for r in rows:
                r["quadrant"] = name
            all_rows.extend(rows)
        print()

    # ---- summary ---------------------------------------------------------
    print(f"{'quadrant':<26}{'n':>5}{'survive':>9}{'D_phi':>9}{'D_0':>9}"
          f"{'L_phys':>9}{'r_track':>9}{'illegal':>9}")
    print("-" * 85)
    summary = {}
    by_q = defaultdict(list)
    for r in all_rows:
        by_q[r["quadrant"]].append(r)
    for name, _, _ in QUADRANTS:
        rs = by_q.get(name, [])
        if not rs:
            continue
        agg = {k: float(np.mean([r[k] for r in rs]))
               for k in ("survived", "D_phi", "D_0", "L_phys", "r_track",
                         "frac_illegal_phi", "frac_illegal_0")}
        agg["n"] = len(rs)
        summary[name] = agg
        print(f"{name:<26}{len(rs):>5}{agg['survived']:>9.2f}{agg['D_phi']:>9.2f}"
              f"{agg['D_0']:>9.2f}{agg['L_phys']:>9.2f}{agg['r_track']:>9.3f}"
              f"{agg['frac_illegal_phi']:>9.3f}")

    hl = summary.get("unseen_body_unseen_task")
    if hl:
        print(f"\nheadline (unseen body x unseen task): D_phi {hl['D_phi']:.2f}  "
              f"L_phys {hl['L_phys']:.2f}  survived {hl['survived']:.2f}")

    # The decisive anti-degeneracy check (proposal.md 3.5).
    seen = summary.get("seen_body_seen_task")
    if seen:
        print(f"\nupper-level honesty check on the training quadrant:")
        print(f"  frac_illegal  phi=0 {seen['frac_illegal_0']:.3f} -> phi {seen['frac_illegal_phi']:.3f}"
              f"   (target < 0.2)")
        print(f"  D against     phi=0 ref {seen['D_0']:.2f}   phi ref {seen['D_phi']:.2f}")
        if seen["D_phi"] < seen["D_0"] * 0.5:
            print("  WARNING: D_phi is far below D_0 -- the gap may be closing because the")
            print("           reference moved toward the robot, not because the robot improved.")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.ckpt.stem}_eval.json"
    out.write_text(json.dumps({
        "checkpoint": str(args.ckpt), "iter": ck.get("iter"), "stage": ck.get("stage"),
        "horizon": args.horizon, "summary": summary, "rows": all_rows,
    }, indent=2))
    print(f"\n{(time.perf_counter() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
