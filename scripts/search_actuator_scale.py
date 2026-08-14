"""Search for each body's actuator strength instead of predicting it.

calibrate_actuators.py picks k from a formula: a percentile of the static torque
demand, divided by the adult's own margin. That formula turned out to be very
sensitive to which poses go into the percentile (measured: the p90 band is
dominated by ground-level frames whose torque is an artifact of the p=0
retarget clamping joints onto their limits), and its --max-k backstop was
binding on half the actuators of the heaviest bodies.

This measures instead. For a grid of global multipliers `s` on the base tree's
actuators, it runs the PD tracking test of pd_track_bodies.py and reports what
actually happens, then picks s per body by an explicit rule.

    THE OBJECTIVE CANNOT BE "MINIMIZE TRACKING ERROR". A position servo's
    steady-state error is ~load/Kp, so error falls monotonically in s and the
    argmin is s = infinity. Measured on this asset family, that is not a
    theoretical worry: the all-task calibration is the strongest of the three
    trees, has the lowest mean pose error, AND produces motion 1.81x jerkier
    than the reference while falling soonest in full-body replay.

So the rule is constrained:

    minimize   pose_err
    subject to jerk_ratio <= --max-jerk   (not stiffer than the reference motion)
               sat_frac   <= --max-sat    (motors not pinned at forcerange)

The jerk constraint is what makes the problem well-posed; it is the measurable
signature of an over-stiff servo. Both the constrained pick and the degenerate
unconstrained argmin are reported, so the difference is visible rather than
asserted.

`s` is a single scalar per body, applied to every actuator, so the per-joint
SHAPE of the base tree is kept and only its overall level is searched. That
keeps the search 1-D per body (cheap, and the curve can be read by eye) and
answers the question the formula got wrong, which was magnitude, not shape.

All of gainprm[0], biasprm[0..2] and forcerange scale by s together, so the
servo's equilibrium angle q* = -(g*ctrl+b0)/b1 is untouched and
`ctrl in [-1,1] <=> qpos in jnt_range` still holds. armature and damping scale
by s too, which holds dt*sqrt(Kp/armature) exactly invariant -- without that,
large s is unstable rather than merely stiff (calibrate_actuators.py:117).

Search on TRAIN tasks; validate the result on the held-out split with
pd_track_bodies.py, or the number is fitted to its own test.

Usage (from project root):
    uv run scripts/search_actuator_scale.py --task-prefixes move jump
    uv run scripts/search_actuator_scale.py --write-tree assets/robots_searched

Writes under --out (default outputs/actuator_search/):
    curve.csv     every (body, s) with all metrics -- the full trade-off curve
    chosen.json   s* per body under the rule, plus the unconstrained argmin
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from model.bilevel.config import BilevelConfig
from pd_track_bodies import bc_target, phi0_retarget

_G: Dict = {}


def _init_worker(base: str, timestep: float, repeat: int, src_rest_h: float,
                 root_mode: str):
    _G.update(base=base, timestep=timestep, repeat=repeat, src_rest_h=src_rest_h,
              root_mode=root_mode, cache={})


def scaled_model(base: str, body: str, s: float, timestep: float) -> mujoco.MjModel:
    """The base body with every actuator (and its rotor) s times bigger.

    Verified equivalent to writing the scaled XML and recompiling: max|qpos
    diff| over a 300-step tracking run is 4e-10, which is the XML writer's
    10-significant-digit rounding, not a modelling difference.
    """
    m = mujoco.MjModel.from_xml_path(str(REPO_ROOT / base / body / "robot.xml"))
    m.opt.timestep = timestep
    dof = np.array([int(m.jnt_dofadr[int(m.actuator_trnid[i, 0])]) for i in range(m.nu)])
    m.actuator_gainprm[:, 0] *= s
    m.actuator_biasprm[:, 0:3] *= s      # b0, b1 with b2 -- q* is preserved
    m.actuator_forcerange *= s
    m.dof_armature[dof] *= s             # holds dt*sqrt(Kp/armature) invariant
    m.dof_damping[dof] *= s
    mujoco.mj_setConst(m, mujoco.MjData(m))   # refresh the derived inertia terms
    return m


def run_one(job: Tuple[str, float, str, str]) -> Dict:
    body, s, task, path = job
    key = (body, s)
    if key not in _G["cache"]:
        _G["cache"] = {key: scaled_model(_G["base"], body, s, _G["timestep"])}
    m = _G["cache"][key]
    d = mujoco.MjData(m)

    ref = phi0_retarget(np.load(path)["qpos"], m, _G["src_rest_h"])
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

    driven = _G["root_mode"] == "driven"
    q = np.empty((T, 69))
    f = np.empty((T, 69))
    root = np.empty((T, 7))
    for t in range(T):
        if driven:                       # balance assumed solved; joints on their own
            d.qpos[:7] = ref[t, :7]
            d.qvel[:6] = rvel[t]
        d.ctrl[:] = a[t]
        mujoco.mj_step(m, d, nstep=rep)
        q[t] = d.qpos[7:]
        f[t] = d.actuator_force
        root[t] = d.qpos[:7]

    blew = bool(~np.isfinite(q).all() or np.abs(q).max() > 1e3)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    root = np.nan_to_num(root, nan=0.0, posinf=0.0, neginf=0.0)
    err = q - ref[1:, 7:]
    ratio = np.abs(f) / np.maximum(frange, 1e-9)
    jerk = lambda x: float((np.diff(x, n=2, axis=0) ** 2).mean())

    # Survival, and it only means anything with a free root. The driven root is
    # glued to the reference, so the body CANNOT fall -- which is exactly why
    # the driven test has no upper bound on s: over-strength is punished by
    # losing your balance, and there is no balance to lose. Free-root survival
    # is the missing half of the objective.
    fall = -1
    if not driven:
        up = 2.0 * (root[:, 2] * root[:, 3] + root[:, 0] * root[:, 1])
        ref_up = 2.0 * (ref[1:, 5] * ref[1:, 6] + ref[1:, 3] * ref[1:, 4])
        alive = (root[:, 2] > 0.5 * ref[1:, 2]) & (up > ref_up - 0.8)
        bad = np.flatnonzero(~alive)
        fall = int(bad[0]) if bad.size else -1

    return {"body": body, "s": s, "task": task,
            "pose_err_rad": float(np.abs(err).mean()),
            "sat_frac": float((ratio >= 0.999).mean()),
            "e_tau": float((ratio ** 2).mean()),
            "jerk_ratio": jerk(q) / max(jerk(ref[1:, 7:]), 1e-12),
            "survived_frac": (T if fall < 0 else fall) / T,
            "blew_up": int(blew)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="assets/robots",
                    help="tree whose actuators are scaled. Default is the raw "
                         "originals, so s* is directly comparable to the k that "
                         "calibrate_actuators.py would have chosen")
    ap.add_argument("--bodies", nargs="*", default=None)
    ap.add_argument("--task-prefixes", nargs="*", default=["move", "jump"])
    ap.add_argument("--split", default="datasets/crossenbodiment-1-datasets/splits/train_tasks.txt",
                    help="search on these tasks; validate elsewhere")
    ap.add_argument("--clips-per-task", type=int, default=2)
    ap.add_argument("--s-min", type=float, default=0.125)
    ap.add_argument("--s-max", type=float, default=64.0)
    ap.add_argument("--s-steps", type=int, default=19)
    ap.add_argument("--max-jerk", type=float, default=1.0,
                    help="reject s whose realized motion is jerkier than the reference")
    ap.add_argument("--max-sat", type=float, default=0.001)
    ap.add_argument("--root", choices=["driven", "free"], default="driven",
                    help="driven pins the root to the reference (isolates the joints, "
                         "but then over-strength is unpunished -- there is no balance "
                         "to lose). free makes the body hold itself up, which is where "
                         "an upper bound on s comes from.")
    ap.add_argument("--timestep", type=float, default=1.0 / 450.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--write-tree", default=None,
                    help="also write the chosen actuators out as a new asset tree")
    ap.add_argument("--out", default="outputs/actuator_search")
    args = ap.parse_args()

    cfg = BilevelConfig()
    bodies = args.bodies or (list(cfg.train_bodies) + list(cfg.heldout_bodies))
    tasks = {l.strip() for l in (REPO_ROOT / args.split).read_text().splitlines() if l.strip()}
    tasks = {t for t in tasks if any(t.startswith(p) for p in args.task_prefixes)}

    clips = []
    for task in sorted(tasks):
        got = sorted((REPO_ROOT / "data" / "origin_motion" / task).glob("*.npz"))
        clips += [(task, str(p)) for p in got[:args.clips_per_task]]
    if not clips:
        raise SystemExit(f"no clips for {args.task_prefixes} in {args.split}")

    grid = np.unique(np.round(np.geomspace(args.s_min, args.s_max, args.s_steps), 4))
    src_h = float(mujoco.MjModel.from_xml_path(
        str(REPO_ROOT / args.base / cfg.source_body / "robot.xml")).qpos0[2])

    jobs = [(b, float(s), t, p) for b in bodies for s in grid for (t, p) in clips]
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(bodies)} bodies x {len(grid)} s-values x {len(clips)} clips = {len(jobs)} runs")
    print(f"base {args.base} | search tasks {sorted(tasks)[:3]}... ({len(tasks)})")
    print(f"rule: min pose_err  s.t. jerk_ratio <= {args.max_jerk}, "
          f"sat_frac <= {args.max_sat}\n")

    t0 = time.time()
    rows = []
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.base, args.timestep, cfg.action_repeat, src_h,
                            args.root)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=len(clips)), 1):
            rows.append(r)
            if i % 1000 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate to (body, s) and choose --------------------------------
    curve: Dict[Tuple[str, float], Dict] = {}
    for b in bodies:
        for s in grid:
            rs = [r for r in rows if r["body"] == b and r["s"] == float(s)]
            curve[(b, float(s))] = {
                k: float(np.mean([r[k] for r in rs]))
                for k in ("pose_err_rad", "sat_frac", "e_tau", "jerk_ratio",
                          "survived_frac")
            } | {"blew_up": sum(r["blew_up"] for r in rs)}

    chosen = {}
    for b in bodies:
        cand = [(s, curve[(b, float(s))]) for s in grid]
        ok = [(s, c) for s, c in cand
              if not c["blew_up"] and c["jerk_ratio"] <= args.max_jerk
              and c["sat_frac"] <= args.max_sat]
        pick = min(ok, key=lambda x: x[1]["pose_err_rad"]) if ok else None
        greedy = min(cand, key=lambda x: x[1]["pose_err_rad"])
        chosen[b] = {
            "s_star": None if pick is None else float(pick[0]),
            "metrics": None if pick is None else pick[1],
            "s_unconstrained": float(greedy[0]),
            "metrics_unconstrained": greedy[1],
            "n_feasible": len(ok),
        }

    with open(out_dir / "curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["body", "s", "pose_err_rad", "sat_frac", "e_tau", "jerk_ratio",
                    "survived_frac", "blew_up"])
        for b in bodies:
            for s in grid:
                c = curve[(b, float(s))]
                w.writerow([b, s, f"{c['pose_err_rad']:.6f}", f"{c['sat_frac']:.6f}",
                            f"{c['e_tau']:.6f}", f"{c['jerk_ratio']:.4f}",
                            f"{c['survived_frac']:.4f}", c["blew_up"]])
    (out_dir / "chosen.json").write_text(json.dumps(
        {"base": args.base, "rule": {"max_jerk": args.max_jerk, "max_sat": args.max_sat},
         "grid": grid.tolist(), "chosen": chosen}, indent=2))

    report(bodies, grid, curve, chosen, args)
    if args.write_tree:
        write_tree(args.base, args.write_tree, cfg, chosen)
    print(f"\nwrote {out_dir}/curve.csv, chosen.json   ({(time.time()-t0)/60:.1f} min)")


def report(bodies, grid, curve, chosen, args) -> None:
    print("\n" + "=" * 100)
    print("CHOSEN SCALE per body   (s* on top of the raw actuators)")
    print("=" * 100)
    print(f"{'body':<15}{'s*':>8}{'pose_err':>11}{'jerk':>8}{'sat':>10}{'e_tau':>9}"
          f"{'| greedy s':>12}{'its err':>10}{'its jerk':>10}")
    for b in bodies:
        c = chosen[b]
        g, gm = c["s_unconstrained"], c["metrics_unconstrained"]
        if c["s_star"] is None:
            print(f"{b:<15}{'NONE':>8}   (no s satisfied the constraints)"
                  f"{'':>30}{g:>12.3g}{gm['pose_err_rad']:>10.4f}{gm['jerk_ratio']:>10.2f}")
            continue
        m = c["metrics"]
        print(f"{b:<15}{c['s_star']:>8.3g}{m['pose_err_rad']:>11.4f}{m['jerk_ratio']:>8.2f}"
              f"{m['sat_frac']:>10.5f}{m['e_tau']:>9.4f}"
              f"{g:>12.3g}{gm['pose_err_rad']:>10.4f}{gm['jerk_ratio']:>10.2f}")
    print("\n  `greedy` = the unconstrained argmin of pose_err. It is shown to make the")
    print("  degeneracy visible: error falls monotonically in s, so the greedy pick runs")
    print("  to the top of the grid and buys its accuracy with jerk.")

    print("\n" + "=" * 100)
    print("TRADE-OFF CURVE  pose_err (rad) / jerk_ratio, per s")
    print("=" * 100)
    show = [s for s in grid][::2]
    print(f"{'body':<13}" + "".join(f"{s:>9.3g}" for s in show))
    for b in bodies:
        print(f"{b:<13}" + "".join(f"{curve[(b,float(s))]['pose_err_rad']:>9.3f}" for s in show))
        print(f"{'  jerk':<13}" + "".join(f"{curve[(b,float(s))]['jerk_ratio']:>9.2f}" for s in show))


def write_tree(base: str, dest: str, cfg, chosen) -> None:
    """Bake s* into a new asset tree, via calibrate_actuators' own rewriters so
    the q*-preserving and armature-scaling rules are applied identically."""
    from calibrate_actuators import rewrite_actuators, rewrite_joints

    out = REPO_ROOT / dest
    out.mkdir(parents=True, exist_ok=True)
    for src in sorted((REPO_ROOT / base).iterdir()):
        if not (src / "robot.xml").exists():
            continue
        d = out / src.name
        d.mkdir(exist_ok=True)
        for extra in ("parameter.json", "skeleton.json"):
            if (src / extra).exists():
                shutil.copy2(src / extra, d / extra)
        s = chosen.get(src.name, {}).get("s_star")
        if src.name == cfg.source_body or s is None:
            shutil.copy2(src / "robot.xml", d / "robot.xml")
            continue
        m = mujoco.MjModel.from_xml_path(str(src / "robot.xml"))
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        k = np.full(m.nu, s)
        text = rewrite_joints(rewrite_actuators((src / "robot.xml").read_text(), k, names),
                              k, names, m)
        (d / "robot.xml").write_text(text)
        par = json.loads((d / "parameter.json").read_text())
        par["scale_actuators"] = "searched"
        par["actuator_scale_median"] = float(s)
        (d / "parameter.json").write_text(json.dumps(par, indent=2) + "\n")
    print(f"\nwrote asset tree -> {out}")


if __name__ == "__main__":
    main()
