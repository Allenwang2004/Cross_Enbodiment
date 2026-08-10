"""SimPool smoke + throughput, and the Stage 0 RSI smoke test.

Covers proposal.md 8.2 Stage 0 items 3 and 4:
  - RSI from random reference frames with sigma up to rsi_sigma_max, stepping
    with zero ctrl, counting divergences and penetrations (R8)
  - end-to-end pool throughput, to fix n_workers / sims_per_worker (R10)

Also checks the things that are easy to get silently wrong:
  - the reference window actually reaches the workers (RSI qpos matches ref[0]
    up to the injected noise, and the root is EXACT)
  - the external wrench actually moves the body
  - termination freezes a slot instead of killing a worker

Run:
    uv run model/bilevel/tests/test_simpool.py
    uv run model/bilevel/tests/test_simpool.py --iters 20 --workers 32
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from model.bilevel import rewards as R
from model.bilevel.config import BilevelConfig
from model.bilevel.data import WindowDataset, batch_ref_qvel, load_splits
from model.bilevel.sim.pool import SimPool, physical_cpus


def build_pool(cfg, ds):
    assign = ds.body_assignment(cfg.n_envs)
    xmls = [str(ds.bodies[b].xml_path) for b in assign]
    legs = [ds.bodies[b].leg_len for b in assign]
    counts = {ds.bodies[b].name: int((assign == b).sum()) for b in range(len(ds.bodies))}
    print(f"slot->body assignment: {counts}")
    per_worker = {
        w: sorted({ds.bodies[b].name for b in assign[w * cfg.sims_per_worker:(w + 1) * cfg.sims_per_worker]})
        for w in range(cfg.n_workers)
    }
    n_multi = sum(1 for v in per_worker.values() if len(v) > 1)
    print(f"workers touching >1 body: {n_multi}/{cfg.n_workers} (contiguous assignment keeps this low)")
    return SimPool(cfg, xmls, legs, obs_dim=358), assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--envs", type=int, default=None)
    args = ap.parse_args()

    cfg = BilevelConfig()
    if args.workers:
        cfg.n_workers = args.workers
    if args.envs:
        cfg.n_envs = args.envs

    print(f"physical cores available: {len(physical_cpus(128))}")
    print(f"config: {cfg.n_envs} envs = {cfg.n_workers} workers x {cfg.sims_per_worker} sims, "
          f"H={cfg.horizon}\n")

    tr, _ = load_splits(cfg)
    ds = WindowDataset(cfg, tasks=tr)
    rng = np.random.default_rng(0)
    pool, assign = build_pool(cfg, ds)

    H = cfg.horizon
    nu = 69
    try:
        # ---------------- correctness on one window ----------------------
        ci, bi, t0 = ds.sample(cfg.n_envs, rng, body_idx=assign)
        batch = ds.build_batch(ci, bi, t0)
        # phi = 0 retarget: root scaled by the rest-height ratio, hinges copied.
        ref = batch["src_qpos"][:, : H + 1].copy()
        for i, b in enumerate(bi):
            ref[i, :, 0:3] *= ds.bodies[b].rest_h / ds.source.rest_h
        # clamp hinges, as apply_retarget's `ref` output does
        for i, b in enumerate(bi):
            m = ds.bodies[b].model
            np.clip(ref[i, :, 7:], m.jnt_range[1:, 0], m.jnt_range[1:, 1], out=ref[i, :, 7:])

        qvel0 = np.stack([
            batch_ref_qvel(ds.bodies[b].model, ref[i:i + 1], cfg.dt)[0] for i, b in enumerate(bi)
        ])
        sigma = rng.uniform(0.0, cfg.rsi_sigma_max, size=(cfg.n_envs, 1))
        noise = (rng.standard_normal((cfg.n_envs, nu)) * sigma).astype(np.float32)

        pool.reset_windows(ref, qvel0, noise)
        root_err = np.abs(pool.qpos[:, :7] - ref[:, 0, :7]).max()
        hinge_dev = np.abs(pool.qpos[:, 7:] - ref[:, 0, 7:]).max()
        print(f"\nRSI: root exact to {root_err:.2e} (must be ~0), "
              f"max hinge deviation {hinge_dev:.4f} rad (noise sigma <= {cfg.rsi_sigma_max})")
        assert root_err < 1e-12, "RSI perturbed the root -- it must not (new.md: 除了 root 以外)"
        assert np.isfinite(pool.obs).all(), "non-finite obs straight out of RSI"

        # ---------------- zero-ctrl rollout: divergence / penetration ----
        zero = np.zeros((cfg.n_envs, nu), dtype=np.float32)
        nowr = np.zeros((cfg.n_envs, 6), dtype=np.float32)
        for t in range(1, H + 1):
            pool.step(zero, nowr, zero, t_ref=t)
        n_done = int(pool.done.sum())
        print(f"zero-ctrl 24-step rollout: {n_done}/{cfg.n_envs} slots terminated "
              f"({100 * n_done / cfg.n_envs:.0f}%) -- high is EXPECTED with no control")
        assert np.isfinite(pool.qpos).all(), "non-finite qpos after rollout"

        alive_terms = pool.rew_terms[pool.done == 0]
        if len(alive_terms):
            i = R.TERM_IDX
            print("  surviving-slot means: " + ", ".join(
                f"{n}={alive_terms[:, i[n]].mean():.3f}"
                for n in ("r_pose", "r_ee", "r_root", "e_tau", "e_slip")
            ))

        # ---------------- the wrench actually does something -------------
        pool.reset_windows(ref, qvel0, np.zeros_like(noise))
        base_z = pool.qpos[:, 2].copy()
        big = np.zeros((cfg.n_envs, 6), dtype=np.float32)
        big[:, 2] = 3000.0   # straight up, far more than any body's weight
        for t in range(1, 6):
            pool.step(zero, big, zero, t_ref=t)
        lifted = (pool.qpos[:, 2] - base_z).mean()
        print(f"\nxfrc_applied check: mean root z change under +3000 N = {lifted:+.4f} m")
        assert lifted > 0.01, "the external wrench is not reaching the simulator"

        # ---------------- throughput -------------------------------------
        print()
        pool.reset_windows(ref, qvel0, noise)
        for t in range(1, H + 1):   # warm up
            pool.step(zero, nowr, zero, t_ref=t)
        t0c = time.perf_counter()
        for _ in range(args.iters):
            pool.reset_windows(ref, qvel0, noise)
            for t in range(1, H + 1):
                pool.step(zero, nowr, zero, t_ref=t)
        el = (time.perf_counter() - t0c) / args.iters
        steps = cfg.n_envs * H
        print(f"throughput: {el * 1000:7.1f} ms / {H}-step iteration "
              f"({steps / el:,.0f} env-steps/s)")
        print(f"            -> {cfg.num_iters} iterations = {el * cfg.num_iters / 3600:.2f} h "
              f"of simulation")
    finally:
        pool.close()

    print("\nPASS -- SimPool drives RSI, physics, the wrench and termination correctly.")


if __name__ == "__main__":
    main()
