"""Replay the adult's recorded ctrl stream open-loop on other bodies, and ask
one question: does the body fall over?

No policy, no retargeted reference, no tracking reward. Just
`data.ctrl[:] = action[t]; mj_step(nstep=15)` on a different skeleton. The point
is that the joint-space command is already morphology-free:

    the actuators are affine position servos, so ctrl in [-1,1] maps EXACTLY
    onto q in jnt_range (measured 4.4e-16, proposal.md 1.3), and all 13 bodies
    share one identical jnt_range (scale_robot.py scales lengths and masses,
    never joint ranges)

so replaying the same ctrl on a child COMMANDS THE SAME JOINT ANGLES. Nothing in
joint space differs. Every difference in outcome is dynamics: mass, inertia,
limb length, actuator authority, and the six unactuated root DoF (nv=75, nu=69)
that no ctrl can touch.

Fall criterion
--------------
Two of the four terms in rewards.py's `alive` (:258), and deliberately not the
other two:

    KEPT    root_z  > term_root_height_frac * (h_tgt/h_src) * adult_root_z[t]
    KEPT    up_z    > adult_up_z[t] - term_up_margin
    DROPPED root_dist < ...      these two are TRACKING-failure tests. Open-loop
    DROPPED pose_err  < ...      replay is not tracking anything; a child that
                                 walks a shorter distance has not fallen, it has
                                 shorter legs. Including them would score the
                                 expected result as a failure.

Both survivors are measured against the ADULT'S OWN REPLAY at the same step, not
an absolute threshold -- the same argument as config.term_up_margin. jump-2
spends frames airborne and frames crouched, and an absolute root-height floor
calls the crouch a fall. The height test carries the rest-pose ratio h_tgt/h_src
because a shorter body is legitimately lower. up_z is dimensionless and is not
scaled. An absolute variant is computed alongside as a cross-check and reported
in its own column.

Two controls, and they measure different things
-----------------------------------------------
`--verify-harness` replays a few clips on humenv's OWN model and asserts the
result is bit-for-bit identical to the recording. That validates this file's
replay loop. Measured: exact.

The `adult` ROW is not that test. It replays on assets/robots_calib/adult like
every other body, and that asset is not bit-identical to humenv's: same
jnt_range, same masses, same gainprm/biasprm/forcerange (all max|diff| = 0.0),
but the derived inertia terms differ by ~1e-12 (dof_invweight0 1.9e-12,
actuator_acc0 2.6e-12). Free-joint semantics are fine -- the same qpos gives
identical world poses in both models (verified, max|diff| = 0.0) -- so this is
pure numerical noise, not a modelling difference.

Chaos then amplifies it, at roughly a decade per 20 steps:

    t=50   3.1e-14      t=150  6.2e-11      t=299  2.6e-05

So the adult row is the CHAOS FLOOR: the same physical body, perturbed only at
the 1e-12 level. Its fall rate is the criterion's false-positive rate, and any
other body's fall rate is only meaningful above it. (Measured on move/jump the
adult floor is 0% -- the fall signal survives the chaos entirely, which is what
makes the cross-body numbers trustworthy.)

    TIMESTEP. humenv's model runs at opt.timestep = 1/450 = 0.002222; every XML
    under assets/robots_calib/ declares 0.002. At action_repeat=15 that is
    0.0333 s vs 0.0300 s per control step -- an 11% faster clock. This script
    sets opt.timestep = --timestep (default 1/450) on every body so the replay
    matches the sim the actions were generated in and all bodies share one
    clock. (Note in passing: model/bilevel/longeval.py:101 steps the repo assets
    at their XML value while config.py:73 documents 1/450, so the training path
    runs 11% fast. Not touched here.)

Usage (from project root):
    uv run scripts/replay_actions_on_body.py
    uv run scripts/replay_actions_on_body.py --bodies child giant --workers 8
    uv run scripts/replay_actions_on_body.py --task-prefixes move jump crawl

Writes under --out (default outputs/replay_fall/):
    per_clip.csv    one row per (clip, body): fall step, reason, survival, drift
    summary.json    per-body and per-task aggregates + the thresholds used
    survival.csv    fraction still standing at each step, per body
"""

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.bilevel.config import BilevelConfig

_G: Dict = {}


def up_z(quat_wxyz: np.ndarray) -> np.ndarray:
    """up_z = 2*(qy*qz + qw*qx), vectorized over a (..., 4) array.

    NOT the textbook 1 - 2*(qx^2+qy^2): this asset's pelvis carries
    euler="90 0 0", so its local +Y is world up. Verbatim from
    model/bilevel/rewards.py:_up_z, and asserted against it in main().
    """
    q = np.asarray(quat_wxyz)
    return 2.0 * (q[..., 2] * q[..., 3] + q[..., 0] * q[..., 1])


def _init_worker(robots_dir: str, timestep: float, action_repeat: int) -> None:
    _G["models"] = {}
    _G["robots_dir"] = robots_dir
    _G["timestep"] = timestep
    _G["action_repeat"] = action_repeat


def _model(name: str) -> mujoco.MjModel:
    if name not in _G["models"]:
        m = mujoco.MjModel.from_xml_path(
            str(REPO_ROOT / _G["robots_dir"] / name / "robot.xml")
        )
        # See the module docstring: the assets declare 0.002, humenv ran 1/450.
        m.opt.timestep = _G["timestep"]
        _G["models"][name] = m
    return _G["models"][name]


def replay_one(job: Tuple[str, str, int, str]) -> Dict:
    body, task, trial, path = job
    m = _model(body)
    d = mujoco.MjData(m)
    f = np.load(path)
    action, src_q = f["action"], f["qpos"]
    T = action.shape[0]

    src_m = _model("adult")
    h_ratio = float(m.qpos0[2] / src_m.qpos0[2])

    # p = 0 initial state: root xyz scaled by the rest-pose pelvis height
    # ratio, every joint angle copied. That is retarget.apply_retarget at u=0
    # and scripts/qpos_retarget.py:91, i.e. the project's naive retarget. Writing
    # the adult's raw qpos onto a child would start it 0.37 m in the air.
    q0 = f["qpos_init"].copy()
    q0[0:3] *= h_ratio
    v0 = f["qvel_init"].copy()
    v0[0:3] *= h_ratio

    mujoco.mj_resetData(m, d)
    d.qpos[:] = q0
    d.qvel[:] = v0
    mujoco.mj_forward(m, d)

    qs = np.empty((T, m.nq))
    for t in range(T):
        d.ctrl[:] = action[t]
        mujoco.mj_step(m, d, nstep=_G["action_repeat"])
        qs[t] = d.qpos

    return score(body, task, trial, qs, src_q, h_ratio, m, src_m)


def score(body, task, trial, qs, src_q, h_ratio, m, src_m) -> Dict:
    cfg = _G["cfg"]
    T = qs.shape[0]
    root_z, src_root_z = qs[:, 2], src_q[:, 2]
    u, src_u = up_z(qs[:, 3:7]), up_z(src_q[:, 3:7])

    # The two postural tests, both relative to the adult's own replay.
    ok_h = root_z > cfg.term_root_height_frac * h_ratio * src_root_z
    ok_u = u > src_u - cfg.term_up_margin
    alive = ok_h & ok_u

    # Absolute cross-check: valid here only because move/jump are upright tasks.
    # It is wrong for headstand/crawl/lieonground, which is why it is secondary.
    alive_abs = (root_z > cfg.term_root_height_frac * m.qpos0[2]) & (u > 0.0)

    def first_false(a):
        bad = np.flatnonzero(~a)
        return int(bad[0]) if bad.size else -1

    fall = first_false(alive)
    reason = "none"
    if fall >= 0:
        reason = ("height" if ok_u[fall] else "upright" if ok_h[fall] else "both")

    # Did it stay down, or was it a transient (a jump landing dipping below the
    # floor fraction for a step or two)? Transients are not falls.
    stayed = bool(fall >= 0 and not alive[fall:].any())

    disp = float(np.linalg.norm(qs[-1, :2] - qs[0, :2]))
    src_disp = float(np.linalg.norm(src_q[-1, :2] - src_q[0, :2]))

    return {
        "body": body, "task": task, "trial": trial, "frames": T,
        "fall_step": fall,
        "fall_reason": reason,
        "fell": int(fall >= 0),
        "fell_and_stayed": int(stayed),
        "survived_frac": (T if fall < 0 else fall) / T,
        "alive_frac": float(alive.mean()),
        "fall_step_abs": first_false(alive_abs),
        "min_height_ratio": float(
            (root_z / np.maximum(h_ratio * src_root_z, 1e-9)).min()
        ),
        "min_up_z": float(u.min()),
        "final_up_z": float(u[-1]),
        # Distance travelled, in leg-lengths, against the adult's own. ~1.0 means
        # the body covered the proportionally same ground.
        "disp_m": disp,
        "disp_ratio": disp / max(src_disp, 1e-9),
        # Harness control: 0.0 for `adult` or the replay is not reproducing.
        "max_qpos_dev_from_source": float(np.abs(qs - src_q).max()),
        "alive_mask": alive,           # stripped before CSV, used for the curve
    }


def build_jobs(act_root: Path, prefixes: Optional[List[str]],
               bodies: List[str], limit: Optional[int]) -> List[Tuple]:
    clips = []
    for p in sorted(act_root.rglob("*.npz")):
        task = p.parent.name
        if prefixes and not any(task.startswith(x) for x in prefixes):
            continue
        clips.append((task, int(p.stem.rsplit("_", 1)[1]), str(p)))
    if limit:
        clips = clips[:limit]
    return [(b, t, tr, p) for b in bodies for (t, tr, p) in clips]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bodies", nargs="*", default=None,
                    help="default: adult (control) + all train + all held-out")
    ap.add_argument("--task-prefixes", nargs="*", default=["move", "jump"],
                    help="task-name prefixes to include; [] for all")
    ap.add_argument("--action-dir", default="data/origin_action")
    ap.add_argument("--limit", type=int, default=None, help="first N clips, for a smoke test")
    ap.add_argument("--timestep", type=float, default=1.0 / 450.0,
                    help="opt.timestep forced on every body. Default matches "
                         "humenv, which is what the actions were generated in; "
                         "the XMLs say 0.002 (see module docstring)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-verify-harness", dest="verify_harness", action="store_false",
                    help="skip the bit-exactness check against humenv's own model")
    ap.add_argument("--robots-dir", default=None,
                    help="which asset tree to simulate. Default BilevelConfig.robots_dir "
                         "(assets/robots_calib) -- actuators sized by "
                         "calibrate_actuators.py so every body has the adult's torque "
                         "margin, i.e. this isolates GEOMETRY. Pass assets/robots for "
                         "the uncalibrated originals, where only 8 of 13 bodies can "
                         "hold their own rest pose, i.e. geometry AND actuator "
                         "shortfall together. Use two --out dirs to compare.")
    ap.add_argument("--out", default="outputs/replay_fall")
    args = ap.parse_args()

    cfg = BilevelConfig()
    if args.robots_dir:
        cfg.robots_dir = args.robots_dir

    # The vectorized up_z here must agree with the scalar one the reward uses.
    from model.bilevel.rewards import _up_z as _ref_up_z
    probe = np.array([0.7071, 0.7071, 0.0, 0.0])
    assert abs(float(up_z(probe)) - _ref_up_z(probe)) < 1e-15, "up_z drifted from rewards.py"

    bodies = args.bodies or ([cfg.source_body] + list(cfg.train_bodies)
                             + list(cfg.heldout_bodies))
    act_root = REPO_ROOT / args.action_dir
    if not act_root.exists():
        raise SystemExit(f"{act_root} not found -- run metamotivo_motion_rollout.py first")
    jobs = build_jobs(act_root, args.task_prefixes or None, bodies, args.limit)
    if not jobs:
        raise SystemExit(f"no clips matched prefixes {args.task_prefixes}")
    n_clips = len(jobs) // len(bodies)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{n_clips} clips x {len(bodies)} bodies = {len(jobs)} replays | "
          f"timestep {args.timestep:.6f} x {cfg.action_repeat} = "
          f"{args.timestep*cfg.action_repeat*1000:.1f} ms/control step")
    print(f"fall: root_z > {cfg.term_root_height_frac} * (h_tgt/h_src) * adult_root_z "
          f"AND up_z > adult_up_z - {cfg.term_up_margin}")

    harness_dev = None
    if args.verify_harness:
        harness_dev = verify_harness(act_root, args.task_prefixes or None,
                                     args.timestep, cfg.action_repeat)
        if harness_dev is not None:
            ok = "OK, bit-exact" if harness_dev == 0.0 else f"FAILED, max|d|={harness_dev:.3e}"
            print(f"harness check (replay on humenv's own model): {ok}")
            if harness_dev != 0.0:
                raise SystemExit(
                    "the replay loop does not reproduce the recording on the model "
                    "the actions were generated in -- fix that before reading any "
                    "fall number below"
                )
    print()

    t0 = time.time()
    ctx = mp.get_context("fork")
    _G["cfg"] = cfg      # inherited by fork; also set in the initializer below
    rows: List[Dict] = []
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(cfg.robots_dir, args.timestep, cfg.action_repeat)) as pool:
        for i, r in enumerate(pool.imap_unordered(replay_one, jobs, chunksize=4), 1):
            rows.append(r)
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    curves, n_at = build_curves(rows, bodies)
    for r in rows:
        r.pop("alive_mask", None)
    rows.sort(key=lambda r: (r["body"], r["task"], r["trial"]))

    with open(out_dir / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "survival.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "n_clips"] + bodies)
        for s in range(len(n_at)):
            w.writerow([s, int(n_at[s])] + [f"{curves[b][s]:.4f}" for b in bodies])

    summary = build_summary(cfg, rows, bodies, args)
    summary["harness_max_dev_vs_humenv"] = harness_dev
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report(cfg, rows, bodies, summary, curves, n_at, harness_dev)
    print(f"\nwrote {out_dir}/per_clip.csv, survival.csv, summary.json")
    print(f"wall time {(time.time()-t0)/60:.1f} min")
    plot_survival_curve(out_dir / "survival.csv", out_dir)


def plot_survival_curve(survival_csv: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(survival_csv)
    plt.figure(figsize=(10, 6))
    for body in df.columns[2:]:
        plt.plot(df["step"], df[body], label=body)
    plt.xlabel("Step")
    plt.ylabel("Fraction Still Standing")
    plt.title("Survival Curve by Body")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir / "survival_curve.png")
    plt.close()


def build_curves(rows, bodies) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Fraction of clips still standing at step s (never-fallen-yet), per body.

    Clip lengths differ (move-* run 300 steps, jump-2 150), so the denominator
    shrinks past the shortest clip rather than the curve being truncated to it.
    `n_at` is that denominator, returned so the report can say where the
    population thins out instead of showing a rate over three clips as if it
    were over 240.
    """
    n = max(len(r["alive_mask"]) for r in rows)
    out, den_ref = {}, None
    for b in bodies:
        num, den = np.zeros(n), np.zeros(n)
        for r in rows:
            if r["body"] != b:
                continue
            m = r["alive_mask"]
            num[:len(m)] += np.logical_and.accumulate(m)
            den[:len(m)] += 1
        out[b] = num / np.maximum(den, 1)
        den_ref = den
    return out, den_ref


def verify_harness(act_root: Path, prefixes, timestep: float, action_repeat: int,
                   n_clips: int = 2) -> Optional[float]:
    """Replay on humenv's own model; must be bit-for-bit identical.

    This is the test that this file's replay loop is correct. It is separate
    from the `adult` row, which uses the repo asset and measures the chaos floor
    instead -- see the module docstring.
    """
    try:
        from humenv import make_humenv
    except Exception as e:                                    # pragma: no cover
        print(f"  harness verification skipped ({e})")
        return None

    env, _ = make_humenv(num_envs=1, task=None, state_init="Default")
    m = env.unwrapped.model
    m.opt.timestep = timestep
    worst = 0.0
    paths = [p for p in sorted(act_root.rglob("*.npz"))
             if not prefixes or any(p.parent.name.startswith(x) for x in prefixes)]
    for p in paths[:n_clips]:
        f = np.load(p)
        act, src = f["action"], f["qpos"]
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = f["qpos_init"], f["qvel_init"]
        mujoco.mj_forward(m, d)
        got = np.empty_like(src)
        for t in range(act.shape[0]):
            d.ctrl[:] = act[t]
            mujoco.mj_step(m, d, nstep=action_repeat)
            got[t] = d.qpos
        worst = max(worst, float(np.abs(got - src).max()))
    env.close()
    return worst


def build_summary(cfg, rows, bodies, args) -> Dict:
    def agg(rs):
        fell = np.array([r["fell"] for r in rs])
        fs = np.array([r["fall_step"] for r in rs], dtype=float)
        surv = np.array([r["survived_frac"] for r in rs])
        reasons = defaultdict(int)
        for r in rs:
            reasons[r["fall_reason"]] += 1
        return {
            "n": len(rs),
            "fall_rate": float(fell.mean()),
            "stayed_down_rate": float(np.mean([r["fell_and_stayed"] for r in rs])),
            "median_fall_step": float(np.median(fs[fs >= 0])) if (fs >= 0).any() else None,
            "mean_survived_frac": float(surv.mean()),
            "mean_disp_ratio": float(np.mean([r["disp_ratio"] for r in rs])),
            "max_qpos_dev_from_source": float(max(r["max_qpos_dev_from_source"] for r in rs)),
            "fall_reasons": dict(reasons),
        }

    tasks = sorted({r["task"] for r in rows})
    tgt = [r for r in rows if r["body"] != cfg.source_body]
    return {
        "criterion": {
            "term_root_height_frac": cfg.term_root_height_frac,
            "term_up_margin": cfg.term_up_margin,
            "reference": "adult open-loop replay, height scaled by h_tgt/h_src",
            "dropped_from_rewards_alive": ["root_dist", "pose_err"],
            "timestep": args.timestep,
            "action_repeat": cfg.action_repeat,
            "init": "p=0 retarget of qpos_init (root xyz * h_tgt/h_src)",
        },
        # Which asset tree was simulated is the single biggest confound between
        # two runs of this script, so it is recorded rather than inferred from
        # the output path.
        "robots_dir": cfg.robots_dir,
        "task_prefixes": args.task_prefixes,
        "by_body": {b: agg([r for r in rows if r["body"] == b]) for b in bodies},
        "by_task_targets_only": {t: agg([r for r in tgt if r["task"] == t]) for t in tasks},
    }


def report(cfg, rows, bodies, summary, curves, n_at, harness_dev) -> None:
    floor = summary["by_body"][cfg.source_body]["fall_rate"]
    print("\n" + "=" * 108)
    print("FALL RATE  (open-loop replay of the adult's ctrl on each body)")
    print("=" * 108)
    print(f"{'body':<14}{'fall rate':>11}{'stayed down':>13}{'median fall':>13}"
          f"{'mean surv':>11}{'disp ratio':>12}{'reason h/u/both':>18}{'chaos dev':>12}")
    for b in bodies:
        a = summary["by_body"][b]
        rr = a["fall_reasons"]
        mf = "-" if a["median_fall_step"] is None else f"{a['median_fall_step']:.0f}"
        tag = f"{b} *" if b == cfg.source_body else b
        print(f"{tag:<14}{a['fall_rate']*100:>10.1f}%{a['stayed_down_rate']*100:>12.1f}%"
              f"{mf:>13}{a['mean_survived_frac']:>11.3f}{a['mean_disp_ratio']:>12.2f}"
              f"{rr.get('height',0):>9}/{rr.get('upright',0)}/{rr.get('both',0)}"
              f"{a['max_qpos_dev_from_source']:>12.2e}")

    print(f"\n  * CHAOS FLOOR, not a harness check. `adult` is the same physical body as the")
    print(f"    source; {cfg.robots_dir}/adult differs from humenv's model only in derived")
    print(f"    inertia terms (~1e-12), which chaos amplifies. Its fall rate is this")
    print(f"    criterion's FALSE-POSITIVE RATE: {floor*100:.1f}%. Read every other row against it.")
    if harness_dev is not None:
        ok = "bit-exact" if harness_dev == 0.0 else f"max|d|={harness_dev:.3e}"
        print(f"    Harness check (replay on humenv's own model): {ok}.")

    print("\n" + "=" * 108)
    print("SURVIVAL CURVE  (fraction never yet fallen, at step ...)")
    print("=" * 108)
    n = len(n_at)
    marks = [s for s in (0, 15, 30, 60, 90, 120, 150, 200, 250, n - 1) if s < n]
    print(f"{'body':<14}" + "".join(f"{'t=' + str(s):>9}" for s in marks))
    for b in bodies:
        print(f"{b:<14}" + "".join(f"{curves[b][s]*100:>8.0f}%" for s in marks))
    print(f"{'(n clips)':<14}" + "".join(f"{int(n_at[s]):>9}" for s in marks)
          + "   <- denominator; shorter clips drop out")

    bt = summary["by_task_targets_only"]
    order = sorted(bt, key=lambda t: -bt[t]["fall_rate"])
    print("\n" + "=" * 104)
    print("BY TASK  (target bodies only, hardest first)")
    print("=" * 104)
    print(f"{'task':<34}{'fall rate':>11}{'median fall':>13}{'mean surv':>11}{'disp ratio':>12}")
    for t in order:
        a = bt[t]
        mf = "-" if a["median_fall_step"] is None else f"{a['median_fall_step']:.0f}"
        print(f"{t:<34}{a['fall_rate']*100:>10.1f}%{mf:>13}"
              f"{a['mean_survived_frac']:>11.3f}{a['mean_disp_ratio']:>12.2f}")


if __name__ == "__main__":
    main()
