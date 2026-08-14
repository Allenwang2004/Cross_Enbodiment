"""Per-joint actuator search: pick k_j for each of the 69 actuators, not one
global scalar.

calibrate_actuators.py already chooses k per joint; a search that only tunes a
single multiplier per body answers a different (and easier) question. This does
the per-joint version.

The trick that makes it cheap: one PD-tracking rollout at a uniform scale s
yields the tracking error of ALL 69 joints at that s. So a single sweep over the
s grid fills a (grid x 69) error table per body, and each joint's own k_j can be
read straight off its own column. No 69-dimensional search is needed for the
first pass.

    THE APPROXIMATION, STATED: joints are not independent. A stiffer hip changes
    the load the knee sees, so a column read off a UNIFORM sweep is a
    first-order estimate of what that joint wants. Stage 2 therefore rebuilds
    the per-joint vector, runs it, and compares against the uniform baseline --
    if the coupling mattered more than the per-joint gain, that shows up as the
    combined vector underperforming its own prediction. Round 2 repeats the
    sweep around the stage-1 vector (coordinate descent), which is where the
    coupling gets absorbed.

Selection is the knee rule, per joint: the smallest k_j reaching within
--tol of that joint's best achievable error. Plain argmin is degenerate --
error falls monotonically in stiffness, so argmin is always the top of the grid
(measured, every body, every joint).

Usage:
    uv run scripts/search_actuator_per_joint.py --task-prefixes move jump
    uv run scripts/search_actuator_per_joint.py --rounds 2 --write-tree assets/robots_pj
"""

import argparse
import csv
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model.bilevel.config import BilevelConfig
from pd_track_bodies import bc_target, phi0_retarget

_G: Dict = {}


def _init(base, timestep, repeat, src_h, root):
    _G.update(base=base, timestep=timestep, repeat=repeat, src_h=src_h, root=root, cache={})


def build_model(base, body, k, timestep):
    """k is (69,) -- one factor per actuator. All of gainprm[0], biasprm[0..2]
    and forcerange scale together so q* is preserved, and armature/damping too
    so dt*sqrt(Kp/armature) stays invariant."""
    m = mujoco.MjModel.from_xml_path(str(REPO_ROOT / base / body / "robot.xml"))
    m.opt.timestep = timestep
    dof = np.array([int(m.jnt_dofadr[int(m.actuator_trnid[i, 0])]) for i in range(m.nu)])
    m.actuator_gainprm[:, 0] *= k
    m.actuator_biasprm[:, 0:3] *= k[:, None]
    m.actuator_forcerange *= k[:, None]
    m.dof_armature[dof] *= k
    m.dof_damping[dof] *= k
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return m


def roll(job):
    """-> per-JOINT mean |error|, plus the scalar summaries."""
    body, tag, k, path = job
    key = (body, tag)
    if key not in _G["cache"]:
        _G["cache"] = {key: build_model(_G["base"], body, k, _G["timestep"])}
    m = _G["cache"][key]
    d = mujoco.MjData(m)
    ref = phi0_retarget(np.load(path)["qpos"], m, _G["src_h"])
    a = bc_target(ref, m)
    T, rep = a.shape[0], _G["repeat"]
    dt = _G["timestep"] * rep
    frange = np.abs(m.actuator_forcerange[:, 1])

    rvel = np.zeros((T, 6))
    v = np.zeros(m.nv)
    for t in range(T):
        mujoco.mj_differentiatePos(m, v, dt, np.ascontiguousarray(ref[t]),
                                   np.ascontiguousarray(ref[t + 1]))
        rvel[t] = v[:6]
    mujoco.mj_resetData(m, d)
    d.qpos[:] = ref[0]
    d.qvel[:6] = rvel[0]
    mujoco.mj_forward(m, d)

    q = np.empty((T, 69))
    f = np.empty((T, 69))
    for t in range(T):
        if _G["root"] == "driven":
            d.qpos[:7] = ref[t, :7]
            d.qvel[:6] = rvel[t]
        d.ctrl[:] = a[t]
        mujoco.mj_step(m, d, nstep=rep)
        q[t] = d.qpos[7:]
        f[t] = d.actuator_force
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    err = np.abs(q - ref[1:, 7:])
    ratio = np.abs(f) / np.maximum(frange, 1e-9)
    jerk = lambda x: float((np.diff(x, n=2, axis=0) ** 2).mean())
    return {"body": body, "tag": tag,
            "err_j": err.mean(0),                       # (69,)
            "pose_err": float(err.mean()),
            "sat_frac": float((ratio >= 0.999).mean()),
            "jerk_ratio": jerk(q) / max(jerk(ref[1:, 7:]), 1e-12)}


def sweep(pool, bodies, clips, grid, base_k):
    """Uniform sweep around base_k -> {body: (n_grid, 69) error table}."""
    jobs = [(b, float(s), base_k[b] * s, p) for b in bodies for s in grid for (_, p) in clips]
    acc: Dict = {b: {float(s): [] for s in grid} for b in bodies}
    scal: Dict = {b: {float(s): [] for s in grid} for b in bodies}
    for r in pool.imap_unordered(roll, jobs, chunksize=len(clips)):
        acc[r["body"]][r["tag"]].append(r["err_j"])
        scal[r["body"]][r["tag"]].append((r["pose_err"], r["sat_frac"], r["jerk_ratio"]))
    tab = {b: np.stack([np.mean(acc[b][float(s)], 0) for s in grid]) for b in bodies}
    sc = {b: {float(s): np.mean(scal[b][float(s)], 0) for s in grid} for b in bodies}
    return tab, sc


def pick(tab, grid, tol):
    """Per joint: smallest scale within `tol` of that joint's own best error."""
    best = tab.min(0)                                    # (69,)
    ok = tab <= best[None, :] * (1.0 + tol)              # (n_grid, 69)
    return np.array(grid)[ok.argmax(0)]                  # first True per column


def evaluate(pool, bodies, clips, kmap, tag):
    jobs = [(b, tag, kmap[b], p) for b in bodies for (_, p) in clips]
    out: Dict = {b: [] for b in bodies}
    for r in pool.imap_unordered(roll, jobs, chunksize=len(clips)):
        out[r["body"]].append((r["pose_err"], r["sat_frac"], r["jerk_ratio"]))
    return {b: np.mean(v, 0) for b, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="assets/robots")
    ap.add_argument("--bodies", nargs="*", default=None)
    ap.add_argument("--task-prefixes", nargs="*", default=["move", "jump"])
    ap.add_argument("--split", default="datasets/crossenbodiment-1-datasets/splits/train_tasks.txt")
    ap.add_argument("--clips-per-task", type=int, default=2)
    ap.add_argument("--s-min", type=float, default=0.125)
    ap.add_argument("--s-max", type=float, default=64.0)
    ap.add_argument("--s-steps", type=int, default=13)
    ap.add_argument("--tol", type=float, default=0.10, help="knee tolerance per joint")
    ap.add_argument("--rounds", type=int, default=2, help="coordinate-descent rounds")
    ap.add_argument("--root", choices=["driven", "free"], default="driven")
    ap.add_argument("--timestep", type=float, default=1.0 / 450.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--write-tree", default=None)
    ap.add_argument("--out", default="outputs/actuator_search_pj")
    args = ap.parse_args()

    cfg = BilevelConfig()
    bodies = args.bodies or (list(cfg.train_bodies) + list(cfg.heldout_bodies))
    tasks = {l.strip() for l in (REPO_ROOT / args.split).read_text().splitlines() if l.strip()}
    tasks = {t for t in tasks if any(t.startswith(p) for p in args.task_prefixes)}
    clips = []
    for t in sorted(tasks):
        clips += [(t, str(p)) for p in
                  sorted((REPO_ROOT / "data" / "origin_motion" / t).glob("*.npz"))[:args.clips_per_task]]
    grid = np.unique(np.round(np.geomspace(args.s_min, args.s_max, args.s_steps), 4))
    src_h = float(mujoco.MjModel.from_xml_path(
        str(REPO_ROOT / args.base / cfg.source_body / "robot.xml")).qpos0[2])
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(bodies)} bodies x {len(grid)} scales x {len(clips)} clips, "
          f"{args.rounds} round(s), per-joint knee tol {args.tol}\n")
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init,
                  initargs=(args.base, args.timestep, cfg.action_repeat, src_h, args.root)) as pool:
        k = {b: np.ones(69) for b in bodies}
        base_uniform = None
        history = []
        for rnd in range(1, args.rounds + 1):
            tab, sc = sweep(pool, bodies, clips, grid, k)
            if rnd == 1:
                # the best UNIFORM scale, as the baseline the per-joint vector must beat
                base_uniform = {b: float(np.array(grid)[
                    (tab[b].mean(1) <= tab[b].mean(1).min() * (1 + args.tol)).argmax()]) for b in bodies}
            newk = {b: k[b] * pick(tab[b], grid, args.tol) for b in bodies}
            got = evaluate(pool, bodies, clips, newk, f"r{rnd}")
            history.append({b: {"pose_err": float(got[b][0]), "sat": float(got[b][1]),
                                "jerk": float(got[b][2]),
                                "k_median": float(np.median(newk[b])),
                                "k_min": float(newk[b].min()), "k_max": float(newk[b].max())}
                            for b in bodies})
            k = newk
            print(f"round {rnd} done ({time.time()-t0:.0f}s)")

        uni = evaluate(pool, bodies, clips, {b: np.full(69, base_uniform[b]) for b in bodies}, "uni")
        one = evaluate(pool, bodies, clips, {b: np.ones(69) for b in bodies}, "raw")

    print("\n" + "=" * 104)
    print("PER-JOINT vs BEST UNIFORM vs RAW   (pose_err rad, on the search tasks)")
    print("=" * 104)
    print(f"{'body':<14}{'raw':>9}{'best uniform':>15}{'s_uni':>8}"
          + "".join(f"{'per-joint r'+str(i+1):>15}" for i in range(args.rounds))
          + f"{'k range (final)':>22}")
    for b in bodies:
        h = [history[i][b] for i in range(args.rounds)]
        print(f"{b:<14}{one[b][0]:>9.4f}{uni[b][0]:>15.4f}{base_uniform[b]:>8.3g}"
              + "".join(f"{x['pose_err']:>15.4f}" for x in h)
              + f"{h[-1]['k_min']:>10.2f} - {h[-1]['k_max']:<9.1f}")

    imp = np.mean([(uni[b][0] - history[-1][b]["pose_err"]) / uni[b][0] for b in bodies])
    print(f"\nper-joint beats the best uniform scale by {imp*100:+.1f}% on average")
    print(f"{'':2}(if this is ~0, the per-joint freedom is not buying anything and the")
    print(f"{'':2} single-scalar answer was the right level of detail after all)")

    np.savez(out_dir / "k_per_joint.npz", **{b: k[b] for b in bodies})
    (out_dir / "summary.json").write_text(json.dumps(
        {"grid": grid.tolist(), "tol": args.tol, "rounds": args.rounds,
         "best_uniform": base_uniform,
         "raw": {b: float(one[b][0]) for b in bodies},
         "uniform": {b: float(uni[b][0]) for b in bodies},
         "per_joint": history}, indent=2))
    if args.write_tree:
        write_tree(args.base, args.write_tree, cfg, k)
    print(f"\nwrote {out_dir}/  ({(time.time()-t0)/60:.1f} min)")


def write_tree(base, dest, cfg, k):
    from calibrate_actuators import rewrite_actuators, rewrite_joints
    out = REPO_ROOT / dest
    out.mkdir(parents=True, exist_ok=True)
    for src in sorted((REPO_ROOT / base).iterdir()):
        if not (src / "robot.xml").exists():
            continue
        d = out / src.name
        d.mkdir(exist_ok=True)
        for e in ("parameter.json", "skeleton.json"):
            if (src / e).exists():
                shutil.copy2(src / e, d / e)
        if src.name not in k:
            shutil.copy2(src / "robot.xml", d / "robot.xml")
            continue
        m = mujoco.MjModel.from_xml_path(str(src / "robot.xml"))
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        text = rewrite_joints(rewrite_actuators((src / "robot.xml").read_text(), k[src.name], names),
                              k[src.name], names, m)
        (d / "robot.xml").write_text(text)
        par = json.loads((d / "parameter.json").read_text())
        par["scale_actuators"] = "searched_per_joint"
        par["actuator_scale_median"] = float(np.median(k[src.name]))
        (d / "parameter.json").write_text(json.dumps(par, indent=2) + "\n")
    print(f"wrote tree -> {out}")


if __name__ == "__main__":
    main()
