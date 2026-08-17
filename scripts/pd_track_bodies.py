"""Does this body's actuator actually reach the angle it is commanded?

The test that open-loop action replay CANNOT be: replaying a fixed ctrl stream
has no feedback, so a stronger actuator can only add momentum and never exploit
its extra authority -- it is structurally incapable of winning. Here the
actuator is asked to do its own job instead, and the position servo IS the
feedback loop, so under-sizing and over-sizing both show up.

The command is exact, not fitted. The actuators are affine position servos, so

    a_ref[t] = clip( 2*(q_hat[t+1] - lo)/(hi - lo) - 1 , -1, 1 )

is the ctrl that commands the reference's next joint angle, in closed form
(the ctrl <-> jnt_range identity, measured to 4.4e-16 -- proposal.md 1.3). It is
the same target ppo.py uses for behaviour cloning.

Balance is removed on purpose
-----------------------------
`--root driven` (default) overwrites the root's qpos/qvel with the reference's
own before each step, i.e. it assumes an ideal controller already solved
balance, and asks only whether the JOINTS can be driven. Without this the body
falls over in ~20 steps (measured: every non-adult body, every move/jump clip)
and the error is then dominated by lying on the floor rather than by actuator
sizing -- exactly the confound that makes the replay test unusable here.
Contacts stay live, so stance load is real. `--root free` reproduces the
confounded version for comparison.

What each metric catches
------------------------
    pose_err_rad   under-sized: the servo cannot hold against gravity/inertia
    sat_frac       under-sized: force pinned at forcerange, MuJoCo clamps silently
    e_tau          rewards.py's own effort measure, mean (force/forcerange)^2
    jerk_ratio     OVER-sized: realized 2nd-difference over the reference's.
                   ~1 is faithful; >>1 is a servo stiff enough to chatter
    stab_margin    dt*sqrt(Kp/armature), the explicit integrator's hard limit.
                   >= 2 is unstable -- calibrate_actuators.py measured
                   pose_err 0.05 -> 8126 when this was violated

Evaluate on tasks the calibration never saw, or the comparison is circular:
calibrate_actuators.py fits its percentile to the pose sample, so scoring it on
those same tasks scores the fit, not the design.

Usage (from project root):
    uv run scripts/pd_track_bodies.py \
        --trees assets/robots assets/robots_cal_alltrain assets/robots_cal_movetrain \
        --tasks-file datasets/crossenbodiment-1-datasets/splits/test_tasks.txt

Writes under --out (default outputs/pd_track_bodies/):
    per_clip.csv   one row per (tree, body, clip)
    summary.json   per-tree and per-tree-x-body aggregates
"""

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.bilevel.config import BilevelConfig

_G: Dict = {}


def _init_worker(timestep: float, action_repeat: int, root_mode: str, src_rest_h: float):
    _G.update(models={}, timestep=timestep, action_repeat=action_repeat,
              root_mode=root_mode, src_rest_h=src_rest_h)


def _model(tree: str, body: str) -> mujoco.MjModel:
    key = (tree, body)
    if key not in _G["models"]:
        m = mujoco.MjModel.from_xml_path(str(REPO_ROOT / tree / body / "robot.xml"))
        m.opt.timestep = _G["timestep"]     # assets say 0.002; humenv ran 1/450
        _G["models"][key] = m
    return _G["models"][key]


def phi0_retarget(src_qpos: np.ndarray, m: mujoco.MjModel, src_rest_h: float) -> np.ndarray:
    """Root xyz by the rest-height ratio, hinges copied then clamped.

    Identical to model/bilevel/retarget.apply_retarget at u=0 and to
    calibrate_actuators.retarget_naive -- the project's naive retarget.
    """
    q = src_qpos.copy()
    q[:, 0:3] *= float(m.qpos0[2]) / src_rest_h
    np.clip(q[:, 7:], m.jnt_range[1:, 0], m.jnt_range[1:, 1], out=q[:, 7:])
    return q


def bc_target(ref: np.ndarray, m: mujoco.MjModel) -> np.ndarray:
    """ref (T, 76) -> ctrl (T-1, 69) commanding ref[t+1]'s hinge angles."""
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    return np.clip(2.0 * (ref[1:, 7:] - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def run_one(job: Tuple[str, str, str, int, str]) -> Dict:
    tree, body, task, trial, path = job
    m = _model(tree, body)
    d = mujoco.MjData(m)
    src = np.load(path)["qpos"]
    ref = phi0_retarget(src, m, _G["src_rest_h"])
    a = bc_target(ref, m)
    T = a.shape[0]
    rep = _G["action_repeat"]
    dt = _G["timestep"] * rep
    frange = np.abs(m.actuator_forcerange[:, 1])

    # Reference root velocities, in MuJoCo's own qvel coordinates. Must go
    # through mj_differentiatePos: the free joint's qvel[3:6] is a BODY-LOCAL
    # angular velocity a quaternion difference cannot produce.
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

    q_out = np.empty((T, 69))
    f_out = np.empty((T, 69))
    for t in range(T):
        if _G["root_mode"] == "driven":
            d.qpos[:7] = ref[t, :7]
            d.qvel[:6] = rvel[t]
        d.ctrl[:] = a[t]
        mujoco.mj_step(m, d, nstep=rep)
        q_out[t] = d.qpos[7:]
        f_out[t] = d.actuator_force

    blew = bool(~np.isfinite(q_out).all() or np.abs(q_out).max() > 1e3)
    q_out = np.nan_to_num(q_out, nan=0.0, posinf=0.0, neginf=0.0)

    err = q_out - ref[1:, 7:]
    ratio = np.abs(f_out) / np.maximum(frange, 1e-9)

    def jerk(x):
        return float((np.diff(x, n=2, axis=0) ** 2).mean())

    j_ref = jerk(ref[1:, 7:])
    return {
        "tree": tree, "body": body, "task": task, "trial": trial, "frames": T,
        "pose_err_rad": float(np.abs(err).mean()),
        "pose_mse": float((err ** 2).mean()),
        "pose_err_p95": float(np.quantile(np.abs(err), 0.95)),
        "sat_frac": float((ratio >= 0.999).mean()),
        "e_tau": float((ratio ** 2).mean()),
        "jerk_ratio": jerk(q_out) / max(j_ref, 1e-12),
        "blew_up": int(blew),
    }


def stability_margin(tree: str, body: str, timestep: float) -> float:
    """max over joints of dt*sqrt(Kp/armature); >= 2 is an unstable explicit step."""
    m = mujoco.MjModel.from_xml_path(str(REPO_ROOT / tree / body / "robot.xml"))
    kp = np.abs(m.actuator_biasprm[:, 1])
    dof = np.array([int(m.jnt_dofadr[int(m.actuator_trnid[i, 0])]) for i in range(m.nu)])
    arm = np.maximum(m.dof_armature[dof], 1e-12)
    return float((timestep * np.sqrt(kp / arm)).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trees", nargs="+", required=True, help="asset trees to compare")
    ap.add_argument("--bodies", nargs="*", default=None)
    ap.add_argument("--tasks-file", default=None,
                    help="evaluate only these tasks; use the HELD-OUT split")
    ap.add_argument("--root", choices=["driven", "free"], default="driven")
    ap.add_argument("--timestep", type=float, default=1.0 / 450.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="outputs/pd_track_bodies")
    args = ap.parse_args()

    cfg = BilevelConfig()
    bodies = args.bodies or ([cfg.source_body] + list(cfg.train_bodies)
                             + list(cfg.heldout_bodies))
    tasks = None
    if args.tasks_file:
        tasks = {l.strip() for l in Path(args.tasks_file).read_text().splitlines() if l.strip()}

    clips = []
    for p in sorted((REPO_ROOT / "data" / "origin_motion").rglob("*.npz")):
        if tasks is None or p.parent.name in tasks:
            clips.append((p.parent.name, int(p.stem.rsplit("_", 1)[1]), str(p)))
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        raise SystemExit("no clips matched")

    src_m = mujoco.MjModel.from_xml_path(str(REPO_ROOT / args.trees[0] / cfg.source_body / "robot.xml"))
    src_rest_h = float(src_m.qpos0[2])

    jobs = [(t, b, task, tr, p) for t in args.trees for b in bodies for (task, tr, p) in clips]
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(clips)} clips x {len(bodies)} bodies x {len(args.trees)} trees = {len(jobs)} runs")
    print(f"root: {args.root} | tasks: {args.tasks_file or 'ALL'}\n")

    t0 = time.time()
    rows: List[Dict] = []
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.timestep, cfg.action_repeat, args.root, src_rest_h)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=4), 1):
            rows.append(r)
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    stab = {(t, b): stability_margin(t, b, args.timestep) for t in args.trees for b in bodies}
    rows.sort(key=lambda r: (r["tree"], r["body"], r["task"], r["trial"]))
    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = build_summary(rows, args.trees, bodies, stab, args)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report(rows, args.trees, bodies, summary, stab, cfg)
    print(f"\nwrote {out_dir}/per_clip.csv, summary.json   ({(time.time()-t0)/60:.1f} min)")


METRICS = ["pose_err_rad", "pose_err_p95", "sat_frac", "e_tau", "jerk_ratio"]


def _agg(rs) -> Dict:
    out = {k: float(np.mean([r[k] for r in rs])) for k in METRICS}
    out["blew_up"] = int(sum(r["blew_up"] for r in rs))
    out["n"] = len(rs)
    return out


def build_summary(rows, trees, bodies, stab, args) -> Dict:
    tgt = [r for r in rows if r["body"] != "adult"]
    return {
        "root_mode": args.root, "tasks_file": args.tasks_file,
        "by_tree_targets_only": {t: _agg([r for r in tgt if r["tree"] == t]) for t in trees},
        "by_tree_body": {t: {b: _agg([r for r in rows if r["tree"] == t and r["body"] == b])
                             for b in bodies} for t in trees},
        "stability_margin": {t: {b: stab[(t, b)] for b in bodies} for t in trees},
    }


def report(rows, trees, bodies, summary, stab, cfg) -> None:
    short = {t: Path(t).name.replace("robots_", "") for t in trees}
    print("\n" + "=" * 96)
    print("PD TRACKING on held-out tasks  (target bodies only; lower is better except jerk~1)")
    print("=" * 96)
    print(f"{'tree':<16}{'pose_err (rad)':>16}{'p95 err':>11}{'sat_frac':>11}"
          f"{'e_tau':>10}{'jerk_ratio':>13}{'blowups':>10}")
    for t in trees:
        a = summary["by_tree_targets_only"][t]
        print(f"{short[t]:<16}{a['pose_err_rad']:>16.4f}{a['pose_err_p95']:>11.4f}"
              f"{a['sat_frac']:>11.4f}{a['e_tau']:>10.4f}{a['jerk_ratio']:>13.2f}"
              f"{a['blew_up']:>10}")

    print("\n" + "=" * 96)
    print("pose_err (rad) per body")
    print("=" * 96)
    print(f"{'body':<15}" + "".join(f"{short[t]:>18}" for t in trees) + f"{'best':>14}")
    for b in bodies:
        v = [summary["by_tree_body"][t][b]["pose_err_rad"] for t in trees]
        best = short[trees[int(np.argmin(v))]]
        tag = f"{b} *" if b == cfg.source_body else b
        print(f"{tag:<15}" + "".join(f"{x:>18.4f}" for x in v) + f"{best:>14}")
    print(f"{'':15}* adult is identical in every tree (calibration copies it verbatim)")

    print("\n" + "=" * 96)
    print("saturation fraction per body   (force pinned at forcerange -> silently clamped)")
    print("=" * 96)
    print(f"{'body':<15}" + "".join(f"{short[t]:>18}" for t in trees))
    for b in bodies:
        print(f"{b:<15}" + "".join(
            f"{summary['by_tree_body'][t][b]['sat_frac']:>18.4f}" for t in trees))

    print("\n" + "=" * 96)
    print("integrator stability margin  dt*sqrt(Kp/armature)   (>= 2.0 is UNSTABLE)")
    print("=" * 96)
    print(f"{'body':<15}" + "".join(f"{short[t]:>18}" for t in trees))
    for b in bodies:
        row = "".join(f"{stab[(t,b)]:>18.3f}" for t in trees)
        flag = "  <-- UNSTABLE" if max(stab[(t, b)] for t in trees) >= 2.0 else ""
        print(f"{b:<15}{row}{flag}")


if __name__ == "__main__":
    main()
