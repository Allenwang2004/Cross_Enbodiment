#!/usr/bin/env python3
"""aggregate_motion_k.py — one actuator scaling from all 54 motions, after the
motions that disagree with the rest are thrown out.

scripts/scale_child_actuators.py builds robot_torque.xml from ONE clip's fit, so
whatever that clip happens to load determines the whole body's actuator sizing.
scripts/torque_ratio_across_motions.py showed that is not safe: the k vectors of
the 54 motions correlate at median 0.93 but down to 0.53, and the disagreement
is concentrated in the torso (Chest_y varies 119% across motions). This script
takes that same k matrix, drops the motions that sit clearly outside the pack,
averages what remains, and writes the scaled MJCF.

Which motions get dropped, and why by this rule
------------------------------------------------
Each motion's mean correlation to the other 53 is a single number saying "does
this motion agree with the consensus". They form ONE continuous run from 0.79 to
0.95 -- there is no bimodal gap, so no clustering method would find a clean
"other group" either; the honest reading is a tail, not a second cluster. So the
cut is a robust outlier rule on that number: drop anything more than --z scaled
MADs below the median (median absolute deviation x 1.4826, so --z is in
normal-sigma units). At the default --z 3 that removes the arm-raising cluster
plus move-ego-0-0, which is exactly the tail the heatmap shows as green.

Per-joint, not just per-motion
------------------------------
A kept motion can still have one meaningless joint: a joint carrying almost no
gravity torque in that clip fits an arbitrary k at R^2~0. So a joint's average
uses only the kept motions where that joint ALSO cleared --r2-min, and a joint
with no such motion anywhere falls back to k_predicted_subtree (downstream
subtree mass x lever arm), the same positive-by-construction fallback
scale_child_actuators.py uses.

Usage:
  uv run scripts/aggregate_motion_k.py
  uv run scripts/aggregate_motion_k.py --z 2.5 --agg median
  uv run scripts/aggregate_motion_k.py --out assets/robots/child/robot_torque.xml
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.scale_child_actuators import write_scaled_xml  # noqa: E402
from scripts.torque_ratio_per_joint import predicted_ratios  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = ROOT / "outputs/torque_ratio_across_motions/gravity"
DEFAULT_SRC = ROOT / "assets/robots/child/robot.xml"
DEFAULT_OUT = ROOT / "assets/robot_torque/robot_torque.xml"


def read_matrix(path):
    rows = list(csv.reader(open(path)))
    return rows[0][1:], [r[0] for r in rows[1:]], np.array(
        [[float(v) for v in r[1:]] for r in rows[1:]])


def keep_mask(C, z):
    """Motions whose mean correlation to the others is not a low outlier.

    Robust (median / scaled MAD) rather than mean / std: the tail being cut is
    exactly what would inflate a standard deviation and hide itself.
    """
    mean_c = (C.sum(axis=1) - np.diag(C)) / (len(C) - 1)
    med = np.median(mean_c)
    mad = np.median(np.abs(mean_c - med)) * 1.4826
    if mad < 1e-12:
        return np.ones(len(C), dtype=bool), mean_c, med, mad
    return mean_c >= med - z * mad, mean_c, med, mad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default=str(DEFAULT_MATRIX),
                   help="directory written by torque_ratio_across_motions.py")
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--z", type=float, default=3.0,
                   help="drop motions this many scaled MADs below the median "
                        "mean-correlation; 0 keeps every motion")
    p.add_argument("--r2-min", type=float, default=0.9,
                   help="a motion contributes to a joint's average only if its "
                        "fit for that joint reached this R²")
    p.add_argument("--agg", choices=["mean", "median"], default="mean")
    args = p.parse_args()

    mdir = Path(args.matrix)
    joints, motions, K = read_matrix(mdir / "_k_matrix.csv")
    joints_r, motions_r, R2 = read_matrix(mdir / "_r2_matrix.csv")
    corr_motions, _, C = read_matrix(mdir / "_corr.csv")
    if joints != joints_r or motions != motions_r or corr_motions != motions:
        raise SystemExit(f"{mdir} matrices disagree on their rows/columns")

    keep, mean_c, med, mad = (keep_mask(C, args.z) if args.z > 0
                              else (np.ones(len(C), bool), None, None, None))
    dropped = [motions[i] for i in np.where(~keep)[0]]
    print(f"{len(motions)} motions, {len(joints)} joints")
    if args.z > 0:
        print(f"agreement cut: mean-corr < {med - args.z * mad:.4f} "
              f"(median {med:.4f} - {args.z:g} x scaled MAD {mad:.4f})")
    if dropped:
        print(f"dropped {len(dropped)}:")
        for i in np.argsort(mean_c):
            if not keep[i]:
                print(f"    {motions[i]:34s} mean-corr {mean_c[i]:.4f}")
    print(f"kept {int(keep.sum())} motions"
          + (f"  (weakest kept: {motions[int(np.argmin(np.where(keep, mean_c, np.inf)))]} "
             f"{np.where(keep, mean_c, np.inf).min():.4f})" if args.z > 0 else ""))

    m_child = mujoco.MjModel.from_xml_path(str(args.src))
    m_adult = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/adult/robot.xml"))
    k_pred = predicted_ratios(m_adult, m_child)
    model_names = [mujoco.mj_id2name(m_child, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                   for i in range(m_child.nu)]
    if model_names != joints:
        raise SystemExit("the k matrix's joints do not match the model's actuators")

    Kk, R2k = K[keep], R2[keep]
    kmap, spread, notes = {}, {}, []
    for j, name in enumerate(joints):
        ok = np.isfinite(Kk[:, j]) & (R2k[:, j] >= args.r2_min) & (Kk[:, j] > 0)
        if ok.sum() == 0:
            kp = k_pred[j]
            kmap[name] = float(kp) if np.isfinite(kp) and kp > 0 else 1.0
            spread[name] = (0, np.nan)
            notes.append(f"{name}: no kept motion reached R² {args.r2_min:g} "
                         f"(best {np.nanmax(R2k[:, j]):.3f}) -> predictor "
                         f"{kmap[name]:.4f}")
            continue
        vals = Kk[ok, j]
        kmap[name] = float(np.mean(vals) if args.agg == "mean" else np.median(vals))
        spread[name] = (int(ok.sum()), float(vals.std()))

    kv = np.array([kmap[n] for n in joints])
    cv = np.array([spread[n][1] / abs(kmap[n]) if spread[n][0] else np.nan
                   for n in joints])
    used = np.array([spread[n][0] for n in joints])
    print(f"\nk per actuator: {args.agg} over the kept motions that cleared "
          f"R² {args.r2_min:g} (median {int(np.median(used))} motions/joint, "
          f"min {used.min()})")
    print(f"  k median {np.median(kv):.4f}   range {kv.min():.4f}–{kv.max():.4f}")
    worst = np.argsort(np.where(np.isfinite(cv), cv, -1))[-6:][::-1]
    print("  least settled joints (std/mean of k across kept motions):")
    for i in worst:
        print(f"    {joints[i]:14s} k {kv[i]:7.4f}  ±{cv[i]:.1%}  "
              f"from {used[i]} motions")
    if notes:
        print(f"  {len(notes)} actuator(s) fell back to the predictor:")
        for n in notes:
            print(f"    {n}")

    agg_csv = mdir / "_ratios_aggregate.csv"
    with open(agg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actuator", "k_aggregate", "n_motions", "std_across_motions",
                    "k_predicted_subtree"])
        for j, name in enumerate(joints):
            w.writerow([name, f"{kv[j]:.6f}", used[j],
                        f"{spread[name][1]:.6f}" if used[j] else "",
                        f"{k_pred[j]:.6f}"])
    print(f"  per-actuator table -> {agg_csv}")

    print()
    write_scaled_xml(args.src, args.out, kmap)

    # How much the motion choice actually mattered: the single-clip k this
    # replaces vs the aggregate, per joint.
    k_all = np.array([np.mean(K[np.isfinite(K[:, j]) & (R2[:, j] >= args.r2_min)
                                & (K[:, j] > 0), j]) if (R2[:, j] >= args.r2_min).any()
                      else np.nan for j in range(len(joints))])
    d = np.abs(kv - k_all) / np.abs(k_all)
    print(f"\nvs averaging all {len(motions)} motions: median |Δk|/k "
          f"{np.nanmedian(d):.2%}, worst {joints[int(np.nanargmax(d))]} "
          f"{np.nanmax(d):.1%}")


if __name__ == "__main__":
    main()
