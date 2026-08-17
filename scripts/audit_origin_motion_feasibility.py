"""Audit every clip in data/origin_motion/ with the UPPER LEVEL's own C term.

Not humenv's reward. C (model/bilevel/semantics.py:reference_feasibility) asks a
different and, for this project, more fundamental question:

    humenv reward:  "did the robot do the task well?"   (needs a simulation)
    upper-level C:  "is this reference physically ASKABLE of this body at all?"
                    (pure kinematics -- no simulation, no policy, no reward model)

C is evaluated at p = 0, i.e. the NAIVE retarget, which apply_retarget
reproduces exactly (retarget.py:26): root xyz scaled by the rest-pose pelvis
height ratio, every joint angle copied verbatim. So every number here is a
property of the source motion + the target skeleton, with nothing learned in
between. That makes this the baseline the upper level has to beat, and the
`adult` row is the control: at p = 0 the source body's retarget is the
identity, so whatever C reports there is the floor, not damage from retargeting.

The five C terms (semantics.py:183):


alongside interpretable companions C itself does not report: the fraction of
frames with at least one illegal joint (the headline 95.8%), the violation
magnitude in RADIANS rather than ctrl units, per-joint violation rates, and
penetration depth in mm.

S (semantic_fidelity) is reported as a secondary block. Two of its channels are
structurally zero at p = 0 -- s_pose is 0 exactly and s_heading ~1e-32,
because the hinge angles and root quaternion are copied verbatim -- but
s_reach / s_contact / s_froude are not, and they measure how much the body swap
ALONE distorts the motion before any correction is applied.

Usage (from project root):
    uv run scripts/audit_origin_motion_feasibility.py
    uv run scripts/audit_origin_motion_feasibility.py --bodies child teen giant
    uv run scripts/audit_origin_motion_feasibility.py --limit-clips 20 --device cpu

Writes under --out (default outputs/audit_origin_motion_feasibility/):
    per_clip.csv    one row per (clip, body): all C and S terms + diagnostics
    per_joint.csv   one row per (body, joint): violation rate, mean/max excess rad
    summary.json    per-body and per-task aggregates, plus the cfg values used
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Same convention as scripts/audit_bodies.py: runnable directly from the repo
# root without an editable install.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.bilevel.config import BilevelConfig
from model.bilevel.data import REPO_ROOT, load_body
from model.bilevel.retarget import U_DIM, RetargetParams, apply_retarget, joint_group_index
from model.bilevel.semantics import (
    UpperGeometry, contact_signal, reference_feasibility, semantic_fidelity,
)

C_TERMS = ["c_limit", "c_penetrate", "c_float", "c_slide", "c_smooth"]
S_TERMS = ["s_pose", "s_reach", "s_contact", "s_heading", "s_froude"]
DIAG = ["frac_illegal", "frac_pairs_illegal", "excess_rad_mean", "excess_rad_p99",
        "excess_rad_max", "n_joints_ever_illegal", "min_z_median", "min_z_p01",
        "frac_frames_penetrating", "src_contact_frac"]


# ------------------------------------------------------------------ per-clip C

def per_clip_c(cfg, kin, ref_raw: torch.Tensor, ref_geo: UpperGeometry,
               src_geo: UpperGeometry, leg_len: float) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """semantics.reference_feasibility, reduced per clip instead of per batch.

    Every term there is a plain .mean() over a tensor whose leading axis is the
    clip, so keeping that axis and reducing the rest gives values whose mean is
    the batch value exactly -- asserted in _check_matches_reference().

    Returns (terms, a_raw) -- a_raw is reused for the radian-space diagnostics.
    """
    dt = cfg.dt

    a_raw = kin.normalized_ctrl(ref_raw[..., 7:])                       # (B, T, 69)
    c_limit = F.relu(a_raw.abs() - 1.0).pow(2).mean(dim=(1, 2))

    minz = ref_geo.min_geom_z                                           # (B, T)
    c_pen = F.relu(-(minz - cfg.ground_tol)).pow(2).mean(dim=1)

    src_contact = (contact_signal(src_geo.foot_height(), src_geo.foot_rest_z,
                                  cfg.s_contact_k).max(dim=-1).values > 0.5)
    c_float = (F.relu(minz - cfg.float_tol).pow(2) * src_contact).mean(dim=1)

    c_src = contact_signal(src_geo.foot_height(), src_geo.foot_rest_z,
                           cfg.s_contact_k)[:, :-1]                     # (B, T-1, 2)
    v = ref_geo.foot_vel_xy(dt) / leg_len                                # (B, T-1, 2, 2)
    c_slide = (c_src * v.pow(2).sum(-1)).mean(dim=(1, 2))

    acc = ref_raw[:, 2:, 7:] - 2 * ref_raw[:, 1:-1, 7:] + ref_raw[:, :-2, 7:]
    c_smooth = acc.pow(2).mean(dim=(1, 2))

    return {"c_limit": c_limit, "c_penetrate": c_pen, "c_float": c_float,
            "c_slide": c_slide, "c_smooth": c_smooth}, a_raw


def per_clip_s(cfg, ref_geo: UpperGeometry, src_geo: UpperGeometry,
               ee_len: torch.Tensor, src_ee_len: torch.Tensor,
               leg_len: float, src_leg_len: float) -> Dict[str, torch.Tensor]:
    """semantics.semantic_fidelity, reduced per clip. Same argument as above.

    binary_cross_entropy is expanded by hand only because it has no per-sample
    reduction that keeps just the leading axis; the expression is its definition.
    """
    dt = cfg.dt

    d_pose = (ref_geo.hinge - src_geo.hinge).pow(2).mean(dim=(1, 2))

    e_ref = ref_geo.ee_relative() / ee_len.view(1, 1, -1, 1)
    e_src = src_geo.ee_relative() / src_ee_len.view(1, 1, -1, 1)
    d_reach = (e_ref - e_src).pow(2).sum(-1).mean(dim=(1, 2))

    p = contact_signal(ref_geo.foot_height(), ref_geo.foot_rest_z,
                       cfg.s_contact_k).clamp(1e-6, 1 - 1e-6)
    c = contact_signal(src_geo.foot_height(), src_geo.foot_rest_z, cfg.s_contact_k)
    d_contact = (-(c * p.log() + (1 - c) * (1 - p).log())).mean(dim=(1, 2))

    def wrap(a):
        return (a + torch.pi) % (2 * torch.pi) - torch.pi

    dpsi_ref = wrap(ref_geo.heading()[:, 1:] - ref_geo.heading()[:, :-1])
    dpsi_src = wrap(src_geo.heading()[:, 1:] - src_geo.heading()[:, :-1])
    d_heading = (dpsi_ref - dpsi_src).pow(2).mean(dim=1)

    d_froude = (ref_geo.froude(leg_len, dt) - src_geo.froude(src_leg_len, dt)).pow(2).mean(dim=1)

    return {"s_pose": d_pose, "s_reach": d_reach, "s_contact": d_contact,
            "s_heading": d_heading, "s_froude": d_froude}


def _check_matches_reference(name: str, per_clip: Dict[str, torch.Tensor],
                             batch: Dict[str, torch.Tensor], tol: float = 1e-9) -> None:
    """The per-clip reductions above must reproduce semantics.py's batch values.

    Within a length group every clip has the same number of frames, so the batch
    mean IS the mean of the per-clip means. If this ever fires, the per-clip
    copies have drifted from the functions the upper level actually optimizes
    and every number in the report is suspect.
    """
    for k, v in batch.items():
        got, want = float(per_clip[k].mean()), float(v)
        if abs(got - want) > tol * max(1.0, abs(want)):
            raise AssertionError(
                f"{name}.{k}: per-clip mean {got!r} != semantics.py's {want!r}. "
                f"per_clip_c/per_clip_s have drifted from semantics.py."
            )


# ------------------------------------------------------------------ loading

def load_clips(motion_root: Path, tasks: List[str] | None,
               limit: int | None) -> List[Tuple[str, int, np.ndarray]]:
    out = []
    for p in sorted(motion_root.rglob("*.npz")):
        task = p.parent.name
        if tasks is not None and task not in tasks:
            continue
        out.append((task, int(p.stem.rsplit("_", 1)[1]), np.load(p)["qpos"].astype(np.float64)))
    if limit is not None:
        out = out[:limit]
    if not out:
        raise SystemExit(f"no clips found under {motion_root}")
    return out


def group_by_length(clips) -> Dict[int, List[int]]:
    g = defaultdict(list)
    for i, (_, _, q) in enumerate(clips):
        g[q.shape[0]].append(i)
    return dict(sorted(g.items()))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bodies", nargs="*", default=None,
                    help="default: source + all train + all held-out bodies")
    ap.add_argument("--tasks", nargs="*", default=None, help="restrict to these task dirs")
    ap.add_argument("--limit-clips", type=int, default=None, help="smoke-test on the first N clips")
    ap.add_argument("--chunk", type=int, default=64, help="clips per FK batch")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/audit_origin_motion_feasibility")
    args = ap.parse_args()

    cfg = BilevelConfig()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.bodies or ([cfg.source_body] + list(cfg.train_bodies) + list(cfg.heldout_bodies))
    clips = load_clips(REPO_ROOT / cfg.origin_motion_dir, args.tasks, args.limit_clips)
    groups = group_by_length(clips)
    total_frames = sum(q.shape[0] for _, _, q in clips)

    print(f"device {device} | {len(clips)} clips, {total_frames} frames, "
          f"lengths {dict((k, len(v)) for k, v in groups.items())}")
    print(f"bodies: {', '.join(names)}   (p = 0, the naive retarget)\n")

    source = load_body(cfg, cfg.source_body, is_train=False)
    source.kin.to(device)
    src_ee_len = torch.as_tensor(source.ee_limb_lengths, dtype=torch.float64, device=device)
    src_foot_rest = torch.as_tensor(source.foot_rest_heights, dtype=torch.float64, device=device)

    rows: List[Dict] = []
    joint_rows: List[Dict] = []

    for name in names:
        body = source if name == cfg.source_body else load_body(cfg, name, is_train=False)
        kin = body.kin.to(device)
        params = RetargetParams(cfg, source.rest_h, kin.root_rest_height).to(device)
        group_idx = joint_group_index(kin.hinge_names).to(device)
        foot_rest = torch.as_tensor(body.foot_rest_heights, dtype=torch.float64, device=device)
        ee_len = torch.as_tensor(body.ee_limb_lengths, dtype=torch.float64, device=device)
        half_range = ((kin.hinge_hi - kin.hinge_lo) / 2.0).to(device)      # ctrl units -> rad

        # Per-joint accumulators, frame-weighted so length groups mix correctly.
        j_viol = torch.zeros(len(kin.hinge_names), dtype=torch.float64, device=device)
        j_exc_sum = torch.zeros_like(j_viol)
        j_exc_max = torch.zeros_like(j_viol)
        n_frames = 0
        checked = False

        for T, idxs in groups.items():
            for lo in range(0, len(idxs), args.chunk):
                sel = idxs[lo:lo + args.chunk]
                src = torch.as_tensor(
                    np.stack([clips[i][2] for i in sel]), dtype=torch.float64, device=device
                )
                B = src.shape[0]

                with torch.no_grad():
                    u = torch.zeros(B, U_DIM, dtype=torch.float64, device=device)
                    ref_raw, ref = apply_retarget(src, params(u), kin, group_idx, n_out=T)

                    ref_geo = UpperGeometry(kin, ref, foot_rest)
                    src_geo = UpperGeometry(source.kin, src, src_foot_rest)

                    c, a_raw = per_clip_c(cfg, kin, ref_raw, ref_geo, src_geo, body.leg_len)
                    s = per_clip_s(cfg, ref_geo, src_geo, ee_len, src_ee_len,
                                   body.leg_len, source.leg_len)

                    # Verify once per body that these reproduce semantics.py.
                    if not checked:
                        _check_matches_reference("C", c, reference_feasibility(
                            cfg, kin, ref_raw, ref_geo, src_geo, leg_len=body.leg_len))
                        _check_matches_reference("S", s, semantic_fidelity(
                            cfg, ref_geo, src_geo, ee_len, src_ee_len,
                            body.leg_len, source.leg_len))
                        checked = True

                    # ---- diagnostics C does not report ----------------------
                    bad = a_raw.abs() > 1.0                                   # (B, T, 69)
                    excess_rad = F.relu(a_raw.abs() - 1.0) * half_range        # radians
                    minz = ref_geo.min_geom_z
                    src_contact = contact_signal(
                        src_geo.foot_height(), src_geo.foot_rest_z, cfg.s_contact_k
                    ).max(dim=-1).values > 0.5

                    flat = excess_rad.reshape(B, -1)
                    diag = {
                        "frac_illegal": bad.any(-1).to(torch.float64).mean(1),
                        "frac_pairs_illegal": bad.to(torch.float64).mean((1, 2)),
                        "excess_rad_mean": excess_rad.sum((1, 2)) / bad.to(torch.float64).sum((1, 2)).clamp(min=1),
                        "excess_rad_p99": flat.quantile(0.99, dim=1),
                        "excess_rad_max": flat.max(dim=1).values,
                        "n_joints_ever_illegal": bad.any(1).to(torch.float64).sum(1),
                        "min_z_median": minz.median(dim=1).values,
                        "min_z_p01": minz.quantile(0.01, dim=1),
                        "frac_frames_penetrating": (minz < cfg.ground_tol).to(torch.float64).mean(1),
                        "src_contact_frac": src_contact.to(torch.float64).mean(1),
                    }

                    j_viol += bad.to(torch.float64).sum((0, 1))
                    j_exc_sum += excess_rad.sum((0, 1))
                    j_exc_max = torch.maximum(j_exc_max, excess_rad.amax((0, 1)))
                    n_frames += B * T

                for bi, ci in enumerate(sel):
                    task, trial, q = clips[ci]
                    row = {"body": name, "task": task, "trial": trial, "frames": q.shape[0]}
                    for k in C_TERMS:
                        row[k] = float(c[k][bi])
                    for k in S_TERMS:
                        row[k] = float(s[k][bi])
                    for k in DIAG:
                        row[k] = float(diag[k][bi])
                    rows.append(row)

        rate = (j_viol / max(n_frames, 1)).cpu().numpy()
        exc_mean = (j_exc_sum / j_viol.clamp(min=1)).cpu().numpy()
        exc_max = j_exc_max.cpu().numpy()
        lo_np, hi_np = kin.hinge_lo.cpu().numpy(), kin.hinge_hi.cpu().numpy()
        for ji, jn in enumerate(kin.hinge_names):
            joint_rows.append({
                "body": name, "joint": jn,
                "range_rad": float(hi_np[ji] - lo_np[ji]),
                "viol_rate": float(rate[ji]),
                "excess_rad_mean": float(exc_mean[ji]),
                "excess_rad_max": float(exc_max[ji]),
            })

        fi = np.mean([r["frac_illegal"] for r in rows if r["body"] == name])
        cl = np.mean([r["c_limit"] for r in rows if r["body"] == name])
        print(f"  {name:<14} frac_illegal {fi:6.3f}   c_limit {cl:.4g}")

    # ------------------------------------------------------------------ write
    per_clip_csv = out_dir / "per_clip.csv"
    fields = ["body", "task", "trial", "frames"] + C_TERMS + S_TERMS + DIAG
    with open(per_clip_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    per_joint_csv = out_dir / "per_joint.csv"
    with open(per_joint_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["body", "joint", "range_rad", "viol_rate",
                                          "excess_rad_mean", "excess_rad_max"])
        w.writeheader()
        w.writerows(joint_rows)

    summary = build_summary(cfg, rows, joint_rows, names)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    report(cfg, rows, joint_rows, names, summary)
    print(f"\nwrote {per_clip_csv}\n      {per_joint_csv}\n      {out_dir/'summary.json'}")


# ------------------------------------------------------------------ reporting

def _agg(rows: List[Dict], keys: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                  "p90": float(np.quantile(v, 0.90)), "max": float(v.max())}
    return out


def build_summary(cfg, rows, joint_rows, names) -> Dict:
    metrics = C_TERMS + S_TERMS + DIAG
    by_body = {n: _agg([r for r in rows if r["body"] == n], metrics) for n in names}

    targets = [r for r in rows if r["body"] != cfg.source_body]
    tasks = sorted({r["task"] for r in rows})
    by_task = {t: _agg([r for r in targets if r["task"] == t], metrics)
               for t in tasks} if targets else {}

    return {
        "config": {
            "p": 0.0,
            "source_body": cfg.source_body,
            "robots_dir": cfg.robots_dir,
            "origin_motion_dir": cfg.origin_motion_dir,
            "dt": cfg.dt,
            "ground_tol": cfg.ground_tol,
            "float_tol": cfg.float_tol,
            "s_contact_k": cfg.s_contact_k,
            "C_weights": {k: getattr(cfg, k) for k in C_TERMS},
            "S_weights": {k: getattr(cfg, k) for k in S_TERMS},
        },
        "n_clips": len({(r["task"], r["trial"]) for r in rows}),
        "n_bodies": len(names),
        "by_body": by_body,
        "by_task_targets_only": by_task,
    }


def report(cfg, rows, joint_rows, names, summary) -> None:
    w = f"{'body':<14}"
    print("\n" + "=" * 120)
    print("C -- reference feasibility at p = 0   (mean over clips; `adult` is the identity control)")
    print("=" * 120)
    print(w + "".join(f"{k:>13}" for k in C_TERMS)
          + f"{'frac_illegal':>14}{'frac_pairs':>12}{'excess_rad':>12}")
    for n in names:
        b = summary["by_body"][n]
        tag = f"{n} *" if n == cfg.source_body else n
        print(f"{tag:<14}" + "".join(f"{b[k]['mean']:>13.4g}" for k in C_TERMS)
              + f"{b['frac_illegal']['mean']:>14.3f}{b['frac_pairs_illegal']['mean']:>12.4f}"
              + f"{b['excess_rad_mean']['mean']:>12.4f}")
    print(f"{'':<14}* p=0 on the source body is the identity retarget -- this row is the floor.")

    # Which terms actually depend on the target body at all. At p=0 the hinge
    # angles and root quaternion are copied verbatim and all 13 bodies share one
    # jnt_range (verified: scale_robot.py scales lengths and masses, never joint
    # ranges), so every joint-space term is body-INVARIANT by construction and
    # only the world-space ones can carry a cross-embodiment signal. Reported as
    # a measured spread rather than asserted, so a future asset change that
    # breaks the assumption shows up here instead of silently.
    print("\n" + "=" * 120)
    print("Body dependence at p = 0   (spread = max-min of the per-body mean, across bodies)")
    print("=" * 120)
    print(f"{'term':<16}{'min':>14}{'max':>14}{'spread':>14}{'rel spread':>13}   verdict")
    for k in C_TERMS + S_TERMS:
        v = np.array([summary["by_body"][n][k]["mean"] for n in names])
        spread = float(v.max() - v.min())
        rel = spread / max(abs(float(v.mean())), 1e-30)
        verdict = "body-INVARIANT" if rel < 1e-9 else "varies with body"
        print(f"{k:<16}{v.min():>14.4g}{v.max():>14.4g}{spread:>14.4g}{rel:>13.2e}   {verdict}")

    print("\n" + "=" * 108)
    print("Ground clearance and contact  (min_geom_z in mm; negative = inside the floor)")
    print("=" * 108)
    print(f"{'body':<14}{'median':>11}{'p01':>11}{'%frames pen':>14}{'c_penetrate':>14}"
          f"{'c_float':>12}{'src contact%':>14}")
    for n in names:
        b = summary["by_body"][n]
        print(f"{n:<14}{b['min_z_median']['mean']*1000:>11.1f}{b['min_z_p01']['mean']*1000:>11.1f}"
              f"{b['frac_frames_penetrating']['mean']*100:>14.1f}{b['c_penetrate']['mean']:>14.4g}"
              f"{b['c_float']['mean']:>12.4g}{b['src_contact_frac']['mean']*100:>14.1f}")

    print("\n" + "=" * 108)
    print("S -- semantic distortion from the body swap ALONE at p = 0")
    print("=" * 108)
    print(f"{'body':<14}" + "".join(f"{k:>15}" for k in S_TERMS))
    for n in names:
        b = summary["by_body"][n]
        print(f"{n:<14}" + "".join(f"{b[k]['mean']:>15.4g}" for k in S_TERMS))
    print(f"{'':<14}s_pose and s_heading are 0 BY CONSTRUCTION at p=0 (angles/quat copied verbatim).")

    targets = [r for r in rows if r["body"] != cfg.source_body]
    if targets:
        print("\n" + "=" * 108)
        print("Hardest tasks for the naive retarget  (averaged over target bodies)")
        print("=" * 108)
        bt = summary["by_task_targets_only"]
        order = sorted(bt, key=lambda t: -bt[t]["c_limit"]["mean"])

        def line(t):
            a = bt[t]
            print(f"{t:<34}{a['c_limit']['mean']:>12.4g}{a['frac_illegal']['mean']:>14.3f}"
                  f"{a['excess_rad_mean']['mean']:>12.4f}{a['excess_rad_max']['max']:>12.4f}"
                  f"{a['c_penetrate']['mean']:>14.4g}{a['c_slide']['mean']:>12.4g}")

        print(f"{'task':<34}{'c_limit':>12}{'frac_illegal':>14}{'excess_rad':>12}"
              f"{'exc_max':>12}{'c_penetrate':>14}{'c_slide':>12}")
        head, tail = (order[:12], order[-3:]) if len(order) > 18 else (order, [])
        for t in head:
            line(t)
        if tail:
            print(f"{'... (' + str(len(order) - 15) + ' more)':<34}")
            for t in tail:
                line(t)

    print("\n" + "=" * 108)
    print("Worst joints  (violation rate over all frames x clips, target bodies only)")
    print("=" * 108)
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for r in joint_rows:
        if r["body"] == cfg.source_body:
            continue
        a = agg[r["joint"]]
        a[0] += r["viol_rate"]
        a[1] += r["excess_rad_mean"]
        a[2] = max(a[2], r["excess_rad_max"])
        a[3] += 1
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][0] / max(kv[1][3], 1))
    rng = {r["joint"]: r["range_rad"] for r in joint_rows if r["body"] != cfg.source_body}
    print(f"{'joint':<22}{'range (rad)':>13}{'viol rate':>12}{'excess mean':>14}{'excess max':>13}")
    for j, (vr, em, mx, n) in ranked[:15]:
        print(f"{j:<22}{rng.get(j, float('nan')):>13.3f}{vr/n:>12.3f}{em/n:>14.4f}{mx:>13.4f}")


if __name__ == "__main__":
    main()
