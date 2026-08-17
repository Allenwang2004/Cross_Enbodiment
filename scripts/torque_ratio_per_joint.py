#!/usr/bin/env python3
"""torque_ratio_per_joint.py — is the inverse-dynamics torque of the child a
constant multiple of the adult's, joint by joint?

Draws one figure per actuator (69 for this skeleton): the two torque traces over
frames, plus a child-vs-adult scatter with a least-squares ratio fitted through
the origin, so "is there a ratio relation" is answered by a number and an R^2
rather than by eyeballing two wiggly lines.

Why this comparison is meaningful at all
----------------------------------------
data/retargeting_motion is data/origin_motion with the root xyz scaled by the
rest-pelvis-height ratio (0.611) and every hinge angle copied unchanged
(scripts/qpos_retarget.py:91). So both bodies trace the SAME joint trajectory,
and any torque difference is purely the body scaling: adult 71.8 kg / 0.957 m
pelvis vs child 38.2 kg / 0.585 m. That is exactly the condition under which a
per-joint ratio could exist.

Which torque, and why the default is not mj_inverse
---------------------------------------------------
mj_inverse invents whatever constraint force explains the given qacc, and the
child's retargeted feet are in contact on EVERY frame of these clips (ground
correction plants them; child ncon median 6 vs adult 3). So there is no
contact-free frame to fall back on, and any mj_inverse torque is contaminated on
every frame — unequally on the two bodies, which is fatal for a ratio test.
--mode gravity therefore uses qfrc_bias at qvel=0, the pure gravity torque of the
pose, which contains no constraint force at all. Note it also excludes the joint
springs (qfrc_passive); see joint_torques() for why including them costs the
clean ratio. --mode quasi/full expose the mj_inverse versions for comparison;
read them knowing the above.

Usage:
  uv run scripts/torque_ratio_per_joint.py
  uv run scripts/torque_ratio_per_joint.py --clip move-ego-0-0/move-ego-0-0_0.npz
  uv run scripts/torque_ratio_per_joint.py --mode full
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
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIP = "jump-2/jump-2_0.npz"

# Palette: categorical slots 1-2 of the reference data-viz palette (pre-validated
# all-pairs, both modes). Line style doubles the encoding so identity never rests
# on hue alone; neutrals below are the same palette's axis/grid steps.
C_ADULT = "#2a78d6"
C_CHILD = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SHADE = "#f0efe9"


def joint_torques(xml, qpos, dt, *, mode="gravity"):
    """Per-joint torque demand for the 69 actuated hinges. Returns (T,69), ncon.

    mode:
      "gravity" — qfrc_bias at qvel=0: the PURE gravity torque of the pose
                  (verified: switch opt.gravity off and qfrc_bias is exactly 0).
                  It does NOT include the joint springs -- those live in
                  qfrc_passive, a separate field, and they are not small here
                  (|passive|/|gravity| median 1.12 on adult). Subtracting them
                  to get the true static demand costs the clean ratio: limb k
                  moves 0.213 -> 0.152 and joints at R^2>=0.99999 drop 50 -> 5,
                  because child jnt_stiffness was scaled by ~0.135 while its
                  gravity torque scales by ~0.213, so the two terms mix
                  pose-dependently. What this mode measures is therefore the
                  gravity term alone, which is the part that HAS a ratio.
                  This is the only one of the three that contains NO constraint
                  force, so it is the only clean ratio test on these clips: the
                  child's retargeted feet are in contact on every single frame
                  (ground-correction plants them), so anything derived from
                  mj_inverse is contaminated on every frame and contaminated
                  differently on the two bodies (child ncon median 6 vs adult 3).
      "quasi"   — mj_inverse with qvel=qacc=0. Adds the constraint force back.
      "full"    — mj_inverse with the differenced qvel/qacc. Closest to the real
                  demand in principle, but on a retargeted clip the differenced
                  qacc is unreliable and contacts amplify it (see
                  notebooks/actuator.py --verify).
    """
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             or f"act{i}" for i in range(model.nu)]
    dofs = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    # forcerange is in pre-gear actuator units; gear is 1 here, but divide anyway
    # so a future non-unit gear does not silently corrupt the comparison.
    gear = model.actuator_gear[:, 0]

    if mode == "gravity":
        T = len(qpos) 
        tau = np.zeros((T, model.nv))
        ncon = np.zeros(T, dtype=int)
        for t in range(T):
            data.qpos[:] = qpos[t]
            data.qvel[:] = 0.0          # qvel=0 -> qfrc_bias is gravity + springs
            mujoco.mj_forward(model, data)
            tau[t] = data.qfrc_bias
            ncon[t] = data.ncon         # recorded for the plot, not used in the fit
    # else:
    #     # imported lazily: notebooks/actuator.py is only needed by quasi/full,
    #     # and the default gravity mode must run even when it is absent.
    #     from notebooks.actuator import differentiate_trajectory, inverse_dynamics

    #     qvel, qacc = differentiate_trajectory(model, qpos, dt)
    #     tau, ncon = inverse_dynamics(model, data, qpos, qvel, qacc,
    #                                  quasi_static=(mode == "quasi"))

    return tau[:, dofs] / gear, ncon, names, model


def fit_ratio(x, y):
    """Least-squares k for y ~ k*x through the origin, plus R^2 about that fit.

    Through the origin, not a free intercept: the claim being tested is
    "child torque is a fixed multiple of adult torque", and a fitted offset
    would let a joint pass with a constant bias it has no physical reason to
    carry. R^2 is 1 - SSE/SS_total-about-zero, so it is comparable to that claim.
    """
    denom = float(np.dot(x, x))
    if denom < 1e-18:
        return np.nan, np.nan
    k = float(np.dot(x, y) / denom)
    sse = float(np.sum((y - k * x) ** 2))
    sst = float(np.sum(y ** 2))
    return k, (1.0 - sse / sst if sst > 1e-18 else np.nan)


def predicted_ratios(m_adult, m_child):
    """Per-joint child/adult gravity-torque ratio predicted from the two models.

    Gravity torque about a hinge is (downstream subtree mass) x g x (lever arm
    from the joint axis to that subtree's centre of mass), so the ratio needs no
    scaling exponent guessed a priori -- both factors are read straight out of a
    forward pass on each model. Same construction as
    utils/actuator.py:derive_actuator_scale in "inertia" mode, but with arm^1
    rather than arm^2, because this predicts a gravity torque and not an
    equivalent rotational inertia.
    """
    da, dc = mujoco.MjData(m_adult), mujoco.MjData(m_child)
    mujoco.mj_forward(m_adult, da)
    mujoco.mj_forward(m_child, dc)
    out = np.full(m_adult.nu, np.nan)
    for i in range(m_adult.nu):
        ja, jc = m_adult.actuator_trnid[i, 0], m_child.actuator_trnid[i, 0]
        ba, bc = m_adult.jnt_bodyid[ja], m_child.jnt_bodyid[jc]
        arm_a = np.linalg.norm(da.subtree_com[ba] - da.xanchor[ja])
        arm_c = np.linalg.norm(dc.subtree_com[bc] - dc.xanchor[jc])
        num = m_child.body_subtreemass[bc] * arm_c
        den = m_adult.body_subtreemass[ba] * arm_a
        out[i] = num / den if den > 1e-12 else np.nan
    return out


def draw_joint(name, ta, tc, contact, k, r2, out_path,
               ylabel, split_scatter, shade_contact):
    fig = plt.figure(figsize=(12, 3.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1], wspace=0.22,
                          left=0.062, right=0.985, top=0.815, bottom=0.15)
    ax = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    sc = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    frames = np.arange(len(ta))

    # --- trace: one axis, both series in N*m. never two y-scales.
    # Shaded only when contact actually enters the numbers. In gravity mode it
    # does not, and on these clips it is true on every frame, so shading would
    # just flood the panel with a meaningless wash.
    if shade_contact and contact.any():
        ax.fill_between(frames, 0, 1, where=contact, transform=ax.get_xaxis_transform(),
                        color=SHADE, lw=0, zorder=0, label="in contact")
    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    ax.plot(frames, ta, color=C_ADULT, lw=1.7, zorder=3, label="adult (origin_motion)")
    ax.plot(frames, tc, color=C_CHILD, lw=1.7, ls=(0, (4, 2)), zorder=4,
            label="child (retargeting_motion)")
    ax.set_xlabel("frame", color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xlim(0, max(len(ta) - 1, 1))
    ax.grid(color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.008), ncols=3, fontsize=8,
              frameon=False, labelcolor=INK_2, handlelength=2.4,
              columnspacing=1.8, borderpad=0)

    # --- scatter: the ratio claim itself. no series hues here; the axes carry
    # the identity, and grey vs ink separates untrusted from trusted samples.
    if not split_scatter:
        sc.scatter(ta, tc, s=9, color=INK, alpha=0.5, lw=0, zorder=3)
    else:
        if (~contact).any():
            sc.scatter(ta[~contact], tc[~contact], s=9, color=INK, alpha=0.55,
                       lw=0, zorder=3, label="contact-free (fitted)")
        if contact.any():
            sc.scatter(ta[contact], tc[contact], s=9, color=AXIS, alpha=0.7,
                       lw=0, zorder=2, label="in contact (excluded)")
    if np.isfinite(k):
        span = np.array([ta.min(), ta.max()])
        sc.plot(span, k * span, color=INK, lw=1.6, ls=(0, (5, 2)), zorder=4)
        sc.text(0.04, 0.94, f"k = {k:.4f}\nR² = {r2:.5f}", transform=sc.transAxes,
                va="top", ha="left", fontsize=8.5, color=INK,
                bbox=dict(fc=SURFACE, ec=GRID, lw=0.8, pad=3.5))
    sc.axhline(0, color=AXIS, lw=0.9, zorder=1)
    sc.axvline(0, color=AXIS, lw=0.9, zorder=1)
    sc.set_xlabel("adult torque (N·m)", color=INK_2, fontsize=9)
    sc.set_ylabel("child torque (N·m)", color=INK_2, fontsize=9)
    sc.grid(color=GRID, lw=0.8, zorder=0)
    sc.set_axisbelow(True)
    if split_scatter:
        sc.legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=INK_2)

    fig.suptitle(f"{name}    child/adult torque ratio k = "
                 f"{'n/a' if not np.isfinite(k) else f'{k:.4f}'}"
                 f"   (R² {'n/a' if not np.isfinite(r2) else f'{r2:.5f}'})",
                 x=0.008, ha="left", fontsize=11.5, color=INK, y=0.965)
    for a in (ax, sc):
        a.tick_params(colors=MUTED, labelsize=8)
        for s in a.spines.values():
            s.set_color(AXIS)
    fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def draw_overview(names, ks, r2s, out_path):
    """All 69 fitted ratios in one view — this method's result only.

    The candidate scaling laws (mass, length, length^3, ...) are deliberately
    NOT drawn here: five near-parallel lines inside 0.14-0.61 buried the thing
    the figure is about. They are still on every per-joint figure's footnote and
    in the printed summary if the comparison is wanted.
    """
    order = np.argsort(ks)
    fig, ax = plt.subplots(figsize=(11, 13), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(order))

    good = np.array(r2s)[order] >= 0.9
    ax.barh(y[good], np.array(ks)[order][good], height=0.72, color=C_ADULT,
            zorder=3, label="R² ≥ 0.9  (ratio holds)")
    ax.barh(y[~good], np.array(ks)[order][~good], height=0.72, color=AXIS,
            zorder=3, label="R² < 0.9  (no single ratio)")

    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=7.5, color=INK_2)
    ax.set_ylim(-1, len(order))
    ax.set_xlabel("fitted child/adult torque ratio  k", color=INK_2, fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, labelcolor=INK_2,
              facecolor=SURFACE, edgecolor=GRID, framealpha=1.0)
    ax.set_title(f"Per-joint child/adult torque ratio, all {len(order)} actuators",
                 loc="left", fontsize=12, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default=DEFAULT_CLIP,
                   help="path under data/origin_motion and data/retargeting_motion")
    p.add_argument("--adult-xml", default=str(ROOT / "assets/robots/adult/robot.xml"))
    p.add_argument("--child-xml", default=str(ROOT / "assets/robots/child/robot.xml"))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--mode", choices=["gravity", "quasi", "full"], default="gravity",
                   help="torque instrument; 'gravity' is the only contact-free one "
                        "and the only clean ratio test on these clips")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    a_path = ROOT / "data/origin_motion" / args.clip
    c_path = ROOT / "data/retargeting_motion" / args.clip
    for path in (a_path, c_path):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    qa = np.load(a_path)["qpos"]
    qc = np.load(c_path)["qpos"]
    if len(qa) != len(qc):
        raise SystemExit(f"frame count differs: {len(qa)} vs {len(qc)}")
    if not np.allclose(qa[:, 7:], qc[:, 7:]):
        print("  NOTE: hinge angles differ between the two clips, so a torque "
              "difference is not purely body scaling")

    dt = 1.0 / args.fps
    tag = args.mode
    outdir = Path(args.outdir or ROOT / "outputs/torque_ratio_per_joint" /
                  f"{Path(args.clip).stem}_{tag}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"clip {args.clip}  ({len(qa)} frames @ {args.fps:g} fps, {tag})")
    ta, ncon_a, names, m_adult = joint_torques(args.adult_xml, qa, dt, mode=args.mode)
    tc, ncon_c, names_c, m_child = joint_torques(args.child_xml, qc, dt, mode=args.mode)
    if names != names_c:
        raise SystemExit("actuator names differ between the two models")

    # A frame is trusted only if BOTH bodies are contact-free there: a torque
    # pair is only comparable when neither side has invented constraint forces.
    contact = (ncon_a > 0) | (ncon_c > 0)
    if args.mode == "gravity":
        # qfrc_bias carries no constraint force, so contact does not corrupt it.
        # The shading stays on the plot as context, but every frame is fitted.
        trusted = np.ones(len(qa), dtype=bool)
        ylabel = "torque  (N·m)"
    else:
        trusted = ~contact
        if not trusted.any():
            print("  NOTE: every frame has contact in at least one body; fits use "
                  "all frames and are correspondingly unreliable")
            trusted = np.ones(len(qa), dtype=bool)
        ylabel = "torque  (N·m)"

    mass_r = float(m_child.body_mass.sum() / m_adult.body_mass.sum())
    d = mujoco.MjData(m_adult); mujoco.mj_forward(m_adult, d); ha = float(d.qpos[2])
    d = mujoco.MjData(m_child); mujoco.mj_forward(m_child, d); hc = float(d.qpos[2])
    len_r = hc / ha
    # Printed only. The figures show the measured k and nothing else.
    refs = "  ".join(f"{n} {v:.3f}" for n, v in
                     [("mass", mass_r), ("length", len_r),
                      ("mass×length", mass_r * len_r),
                      ("length³", len_r ** 3), ("length⁴", len_r ** 4)])
    print(f"  mass ratio {mass_r:.4f}   length ratio {len_r:.4f}   "
          f"contact-free frames {int((~contact).sum())}/{len(qa)}")

    k_pred = predicted_ratios(m_adult, m_child)

    ks, r2s = [], []
    for i, name in enumerate(names):
        k, r2 = fit_ratio(ta[trusted, i], tc[trusted, i])
        ks.append(k); r2s.append(r2)
        draw_joint(name, ta[:, i], tc[:, i], contact, k, r2,
                   outdir / f"{i:02d}_{name}.png", ylabel,
                   split_scatter=(args.mode != "gravity"),
                   shade_contact=(args.mode != "gravity"))
    print(f"  wrote {len(names)} per-joint figures -> {outdir}")

    draw_overview(names, ks, r2s, outdir / "_overview.png")

    with open(outdir / "_ratios.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actuator", "k_fitted", "r2", "k_predicted_subtree",
                    "adult_peak_Nm", "child_peak_Nm", "mass_ratio", "length_ratio"])
        for i, name in enumerate(names):
            w.writerow([name, f"{ks[i]:.6f}", f"{r2s[i]:.6f}", f"{k_pred[i]:.6f}",
                        f"{np.abs(ta[trusted, i]).max():.4f}",
                        f"{np.abs(tc[trusted, i]).max():.4f}",
                        f"{mass_r:.6f}", f"{len_r:.6f}"])

    ka, r2a = np.array(ks), np.array(r2s)
    ok = np.isfinite(ka) & np.isfinite(r2a)
    tight = ok & (r2a >= 0.9)
    print(f"\n=== does a per-joint ratio hold? ===")
    print(f"joints with R² >= 0.9 : {int(tight.sum())}/{len(names)}")
    print(f"k over those joints   : median {np.median(ka[tight]):.3f}  "
          f"range {ka[tight].min():.3f}–{ka[tight].max():.3f}"
          if tight.any() else "k: no joint reaches R² 0.9")
    print(f"k over all joints     : median {np.median(ka[ok]):.3f}  "
          f"IQR {np.percentile(ka[ok],25):.3f}–{np.percentile(ka[ok],75):.3f}")
    print(f"candidate scalings    : {refs}")
    pm = ok & np.isfinite(k_pred) & (k_pred > 0)
    if pm.any():
        rel = np.abs(ka[pm] - k_pred[pm]) / k_pred[pm]
        print(f"vs per-joint predictor (downstream subtree mass x lever arm):")
        print(f"  median |k - k_pred|/k_pred : {np.median(rel):.1%}   "
              f"corr {np.corrcoef(ka[pm], k_pred[pm])[0, 1]:.3f}")
    print(f"  -> a single global ratio holds only if the k spread is narrow AND "
          f"lands on one of those")
    print(f"overview + csv -> {outdir}/_overview.png, _ratios.csv")


if __name__ == "__main__":
    main()
