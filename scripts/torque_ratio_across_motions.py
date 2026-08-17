#!/usr/bin/env python3
"""torque_ratio_across_motions.py — is the per-joint child/adult torque ratio a
property of the BODY, or of the motion?

scripts/torque_ratio_per_joint.py answers "does a ratio k exist" for one clip.
This asks the follow-up: run that same fit on the first clip of every motion in
data/origin_motion, giving one 69-dim k vector per motion, and correlate those
vectors against each other. If k is a body property the 54 vectors are nearly
identical and the correlation matrix is uniformly ~1; wherever it is not, the
motion is changing which joints carry the load, and a single scaled MJCF cannot
serve every motion equally.

Same measurement as the per-joint script (it imports joint_torques/fit_ratio
straight from it), so the k column of any single motion here reproduces that
script's _ratios.csv for the same clip. No per-joint figures are drawn.

Outputs (default outputs/torque_ratio_across_motions/<mode>/):
  _k_matrix.csv   motions x 69 fitted k          (plus R^2 in _r2_matrix.csv)
  _corr.csv       motions x motions correlation of those k vectors
  _corr.png       that matrix as a heatmap

Usage:
  uv run scripts/torque_ratio_across_motions.py
  uv run scripts/torque_ratio_across_motions.py --metric spearman
  uv run scripts/torque_ratio_across_motions.py --r2-min 0.9
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.torque_ratio_per_joint import fit_ratio, joint_torques  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Two maps, chosen by the data (see draw_corr): diverging blue<->red pinned to
# [-1,1] when any correlation is negative -- polarity is then the story and the
# neutral must sit exactly at 0 -- and a one-hue blue ramp over the observed
# range when they are all positive, which is a magnitude question and where the
# diverging map would spend half its range on empty negatives and flatten
# everything into one blue.
DIVERGING = LinearSegmentedColormap.from_list("corr_div", [
    "#7a1513", "#c0332f", "#e34948", "#ef8a84", "#f7c4bf",
    "#f0efec",
    "#b7d3f6", "#86b6ef", "#3987e5", "#256abf", "#104281",
])
# Sequential: the classic YlGnBu ramp run dark->light, i.e. deep blue at
# correlation 1 through teal to pale green at the low end. Two hues instead of
# one, but they move monotonically in lightness, which is what makes a band of
# 0.5-1.0 readable at all; the pale-yellow tail is cut so the top end stays green.
SEQUENTIAL = LinearSegmentedColormap.from_list("corr_seq", [
    "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#253494", "#122160",
])


def first_clip(motion_dir):
    """The first .npz of a motion, i.e. the *_0 take when the names are padded."""
    clips = sorted(motion_dir.glob("*.npz"))
    return clips[0] if clips else None


def k_vector(a_path, c_path, adult_xml, child_xml, dt, mode):
    """One motion -> (k[69], r2[69], names, frames). None if the pair is unusable."""
    qa = np.load(a_path)["qpos"]
    qc = np.load(c_path)["qpos"]
    if len(qa) != len(qc):
        return None
    ta, _, names, _ = joint_torques(adult_xml, qa, dt, mode=mode)
    tc, _, names_c, _ = joint_torques(child_xml, qc, dt, mode=mode)
    if names != names_c:
        return None
    k = np.full(len(names), np.nan)
    r2 = np.full(len(names), np.nan)
    for i in range(len(names)):
        k[i], r2[i] = fit_ratio(ta[:, i], tc[:, i])
    return k, r2, names, len(qa)


def rankdata(x):
    """Average-tie ranks, so --metric spearman needs no scipy."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average the ranks inside each run of equal values
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def correlate(K, metric):
    """Motion x motion correlation of the k vectors.

    pearson  — on the k values themselves. A handful of joints (head/neck/chest)
               sit near k=1.6 while the 55 limb joints sit near 0.21, so this is
               dominated by whether a motion agrees about those outliers.
    spearman — on the per-joint RANKS, which asks the weaker but more robust
               question "do the motions order the joints the same way".
    """
    X = np.array([rankdata(row) for row in K]) if metric == "spearman" else K
    return np.corrcoef(X)


def draw_corr(labels, C, out_path, subtitle):
    n = len(labels)
    size = max(9.0, 0.20 * n + 3.2)
    fig, ax = plt.subplots(figsize=(size + 1.4, size), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    off = C[~np.eye(n, dtype=bool)]
    if off.min() < 0:
        cmap, vmin, vmax = DIVERGING, -1.0, 1.0
        ticks = [-1, -0.5, 0, 0.5, 1]
    else:
        # floor to the next 0.05 below the weakest pair, so the colorbar states
        # the range being spent and nobody reads dark blue as "correlation 1".
        vmin = float(np.floor(off.min() * 20) / 20)
        cmap, vmax = SEQUENTIAL, 1.0
        ticks = list(np.round(np.linspace(vmin, 1.0, 5), 3))
    im = ax.imshow(C, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5, color=INK_2)
    ax.set_yticklabels(labels, fontsize=6.5, color=INK_2)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.0)  # 1px surface gap between cells
    ax.tick_params(which="both", colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_color(AXIS)

    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015, ticks=ticks)
    cb.outline.set_color(AXIS)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.set_label("correlation of the two motions' k vectors", color=INK_2, fontsize=9)

    ax.set_title("Per-joint torque-ratio vector, motion vs motion",
                 loc="left", fontsize=13, color=INK, pad=30)
    ax.text(0, 1.006, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color=INK_2)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--origin", default=str(ROOT / "data/origin_motion"))
    p.add_argument("--retarget", default=str(ROOT / "data/retargeting_motion"))
    p.add_argument("--adult-xml", default=str(ROOT / "assets/robots/adult/robot.xml"))
    p.add_argument("--child-xml", default=str(ROOT / "assets/robots/child/robot.xml"))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--mode", choices=["gravity", "quasi", "full"], default="gravity")
    p.add_argument("--metric", choices=["pearson", "spearman"], default="pearson")
    p.add_argument("--r2-min", type=float, default=0.0,
                   help="drop any joint whose fit falls below this R^2 in ANY "
                        "motion, so the correlation is over joints where k is "
                        "actually meaningful everywhere")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    origin, retarget = Path(args.origin), Path(args.retarget)
    motions = sorted(d.name for d in origin.iterdir() if d.is_dir())
    outdir = Path(args.outdir or ROOT / "outputs/torque_ratio_across_motions" / args.mode)
    outdir.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / args.fps
    used, K, R2, names = [], [], [], None
    for m in motions:
        a = first_clip(origin / m)
        c_dir = retarget / m
        c = first_clip(c_dir) if c_dir.is_dir() else None
        if a is None or c is None or a.name != c.name:
            print(f"  skip {m}: no matching first clip")
            continue
        got = k_vector(a, c, args.adult_xml, args.child_xml, dt, args.mode)
        if got is None:
            print(f"  skip {m}: frame count or actuator names differ")
            continue
        k, r2, names, T = got
        used.append(m)
        K.append(k)
        R2.append(r2)
        print(f"  {m:34s} {a.name:34s} {T:4d} frames   "
              f"k median {np.nanmedian(k):.3f}  R² median {np.nanmedian(r2):.4f}")

    if len(used) < 2:
        raise SystemExit("need at least two usable motions")
    K = np.array(K)
    R2 = np.array(R2)

    # A joint enters the correlation only if it is finite in EVERY motion:
    # a per-motion mask would make each pair of vectors a different quantity.
    keep = np.isfinite(K).all(axis=0) & np.isfinite(R2).all(axis=0)
    if args.r2_min > 0:
        keep &= (R2 >= args.r2_min).all(axis=0)
    dropped = [names[i] for i in range(len(names)) if not keep[i]]
    print(f"\n{len(used)} motions x {int(keep.sum())}/{len(names)} joints"
          + (f"   dropped: {', '.join(dropped)}" if dropped else ""))

    C = correlate(K[:, keep], args.metric)
    off = C[~np.eye(len(C), dtype=bool)]
    subtitle = (f"{len(used)} motions (first clip of each) x {int(keep.sum())} joints "
                f"— {args.metric}, {args.mode} torque"
                + (f", R² ≥ {args.r2_min:g} in every motion" if args.r2_min > 0 else ""))
    draw_corr(used, C, outdir / "_corr.png", subtitle)

    with open(outdir / "_k_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["motion"] + names)
        for m, row in zip(used, K):
            w.writerow([m] + [f"{v:.6f}" for v in row])
    with open(outdir / "_r2_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["motion"] + names)
        for m, row in zip(used, R2):
            w.writerow([m] + [f"{v:.6f}" for v in row])
    with open(outdir / "_corr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["motion"] + used)
        for m, row in zip(used, C):
            w.writerow([m] + [f"{v:.6f}" for v in row])

    print(f"\n=== is k a body property or a motion property? ===")
    print(f"off-diagonal correlation : median {np.median(off):.4f}   "
          f"min {off.min():.4f}   p05 {np.percentile(off, 5):.4f}")
    worst_pair = np.unravel_index(np.argmin(np.where(np.eye(len(C), dtype=bool),
                                                     np.inf, C)), C.shape)
    print(f"least agreeing pair      : {used[worst_pair[0]]} vs "
          f"{used[worst_pair[1]]}  ({C[worst_pair]:.4f})")
    mean_c = (C.sum(axis=1) - 1) / (len(C) - 1)
    odd = np.argsort(mean_c)[:5]
    print("most atypical motions    : "
          + ", ".join(f"{used[i]} ({mean_c[i]:.3f})" for i in odd))
    spread = K[:, keep].std(axis=0) / np.abs(K[:, keep].mean(axis=0))
    ji = np.array([i for i in range(len(names)) if keep[i]])
    wj = np.argsort(spread)[-5:][::-1]
    print("joints whose k moves most across motions (CV): "
          + ", ".join(f"{names[ji[i]]} {spread[i]:.1%}" for i in wj))
    print(f"\n-> {outdir}/_corr.png, _corr.csv, _k_matrix.csv, _r2_matrix.csv")


if __name__ == "__main__":
    main()
