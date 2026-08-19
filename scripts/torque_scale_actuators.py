#!/usr/bin/env python3
"""torque_scale_actuators.py — write a child MJCF whose actuators are scaled by the
per-joint torque ratio k measured against the adult.

The child's actuators are byte-identical to the adult's (every gainprm /
biasprm / forcerange ratio is exactly 1.0000), so the child carries adult-sized
motors on a 38 kg body. This applies k from
outputs/torque_ratio_per_joint/<clip>/_ratios.csv, so each actuator's torque authority
matches that joint's measured demand ratio.

All four affine terms scale, biasprm[0] included
------------------------------------------------
The actuator is an affine position servo:

    force = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*length + biasprm[2]*velocity

For the scaled actuator to deliver exactly k times the force in every state,
ALL FOUR coefficients must be multiplied by k. Then the equilibrium angle
    q* = -(gainprm[0]*ctrl + biasprm[0]) / biasprm[1]
is unchanged, so ctrl keeps its meaning and only the torque authority moves.
Leaving biasprm[0] alone (as utils/actuator.py:apply_actuator_scale does)
instead shifts the neutral angle by biasprm[0]*(1/k - 1) -- at k=0.21 that is a
~4.7x amplification of an offset that reaches 262 N*m on this model, so the body
would sag into a different rest pose. forcerange scales too: it is the torque
ceiling, and the point of the exercise is to move it.

Which k, and the two joints that cannot use the fitted one
----------------------------------------------------------
k_fitted is trustworthy only where the fit had signal. Two joints have almost no
gravity torque all clip, so their least-squares k is noise: Torso_y (R^2 0.878)
and Chest_y (R^2 0.003, k_fitted = -0.057). A NEGATIVE gain would inverse the
position servo into positive feedback and the model would diverge on contact, so
those fall back to k_predicted_subtree -- the physical prediction (downstream
subtree mass x lever arm) read from the two MJCFs, which is positive for every
joint by construction. --r2-min controls the threshold.

Left and right get the SAME k
-----------------------------
The fit is per actuator, so L_Shoulder_y and R_Shoulder_y come out at 0.2161 and
0.2122 on move-ego-90 -- a 1.8% difference that is entirely an artefact of the
clip turning one way. The body itself is symmetric: k_predicted_subtree, which
is read from the two MJCFs and knows nothing about any motion, agrees left/right
to 0.026% on every pair. Letting the clip's bias through would size the two legs'
motors differently for no physical reason, so --symmetry pair (the default) gives
each of the 27 mirrored pairs one shared k; the 15 midline actuators
(Torso/Spine/Chest/Neck/Head) have no partner and keep their own.

  both sides cleared --r2-min   ->  mean of the two fitted k
  one side cleared it           ->  that side's k (the other is noise)
  neither                       ->  mean of the two fallbacks

Scaling all four affine terms preserves the mirror exactly. The two MJCFs are
mirror images rather than copies -- gainprm, biasprm[1..2] and forcerange are
identical L/R while biasprm[0] flips sign on the y and z axes -- and k multiplies
+b and -b alike, so an equal k leaves the pair an exact mirror. check_mirror()
asserts that on the written file rather than trusting it.

Usage:
  uv run scripts/torque_scale_actuators.py
  uv run scripts/torque_scale_actuators.py --out assets/robots/child/robot_torque.xml
  uv run scripts/torque_scale_actuators.py --symmetry none   # per-actuator k
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "outputs/torque_ratio_per_joint/move-ego-90-2_0_gravity/_ratios.csv"
DEFAULT_SRC = ROOT / "assets/robots/child/robot.xml"
DEFAULT_OUT = ROOT / "assets/robots/child/robot_torque.xml"


def fmt(x: float) -> str:
    """Round-trippable float text, without numpy's array repr."""
    return repr(float(x))


def choose_k(rows, r2_min):
    """Per-actuator scale factor, with the fallback rule applied and reported.

    Returns (kmap, notes, trusted); trusted holds the actuators whose own fitted
    k survived, which is what symmetrize_k needs to know when the two sides of a
    pair disagree about whether their fit meant anything.
    """
    out, notes, trusted = {}, [], set()
    for r in rows:
        name = r["actuator"]
        kf, r2, kp = (float(r["k_fitted"]), float(r["r2"]),
                      float(r["k_predicted_subtree"]))
        if r2 >= r2_min and kf > 0:
            out[name] = kf
            trusted.add(name)
            continue
        if not (kp > 0) or not np.isfinite(kp):
            out[name] = 1.0
            notes.append(f"{name}: R²={r2:.3f}, k_fitted={kf:+.4f}, predictor "
                         f"unusable too -> left unscaled (k=1)")
            continue
        out[name] = kp
        why = "k_fitted<=0" if kf <= 0 else f"R²={r2:.3f}<{r2_min}"
        notes.append(f"{name}: {why}, k_fitted={kf:+.4f} -> using predictor {kp:.4f}")
    return out, notes, trusted


def mirror_pairs(names):
    """Split actuator names into (left, right) pairs and midline singles.

    Pairing is by the L_/R_ prefix the skeleton already uses. A half-pair means
    the naming convention broke, which would silently leave that joint
    asymmetric, so it aborts instead of skipping.
    """
    have = set(names)
    pairs, midline, orphans = [], [], []
    for n in names:
        if n.startswith("L_"):
            partner = "R_" + n[2:]
            (pairs.append((n, partner)) if partner in have else orphans.append(n))
        elif n.startswith("R_"):
            if "L_" + n[2:] not in have:
                orphans.append(n)
        else:
            midline.append(n)
    if orphans:
        raise SystemExit(f"{len(orphans)} actuator(s) have no mirror partner: "
                         f"{orphans[:5]}")
    return pairs, midline


def symmetrize_k(kmap, trusted):
    """Give both actuators of every mirrored pair one shared k, in place.

    The rule is in the module docstring. The one-sided case matters more than it
    looks: a joint that carries almost no gravity torque in a clip fits an
    arbitrary k, and averaging that into the good side would corrupt both
    actuators instead of neither.
    """
    pairs, midline = mirror_pairs(list(kmap))
    report = []
    for lname, rname in pairs:
        kl, kr = kmap[lname], kmap[rname]
        tl, tr = lname in trusted, rname in trusted
        if tl == tr:                      # both fitted, or both fell back
            k, why = 0.5 * (kl + kr), "mean" if tl else "mean of fallbacks"
        elif tl:
            k, why = kl, "L only (R untrusted)"
        else:
            k, why = kr, "R only (L untrusted)"
        kmap[lname] = kmap[rname] = k
        moved = max(abs(kl - k), abs(kr - k)) / max(abs(k), 1e-12)
        report.append((lname[2:], kl, kr, k, moved, why))
    return report, midline


def print_symmetry_report(report, midline, top=6):
    moved_any = [r for r in report if r[4] > 1e-12]
    print(f"symmetry: {len(report)} mirrored pairs share one k, "
          f"{len(midline)} midline actuators keep their own")
    if not moved_any:
        print("  every pair already agreed; no k changed")
        return
    worst = sorted(moved_any, key=lambda r: -r[4])[:top]
    print(f"  {len(moved_any)} pair(s) moved; largest {len(worst)}:")
    print(f"    {'joint':14s} {'k_L':>8s} {'k_R':>8s} {'k_pair':>8s} {'moved':>7s}  rule")
    for j, kl, kr, k, moved, why in worst:
        print(f"    {j:14s} {kl:8.4f} {kr:8.4f} {k:8.4f} {moved:6.2%}  {why}")
    special = [r for r in report if not r[5].startswith("mean")]
    if special:
        print(f"  {len(special)} pair(s) used one side only:")
        for j, kl, kr, k, _, why in special:
            print(f"    {j:14s} k_L {kl:.4f}  k_R {kr:.4f} -> {k:.4f}  ({why})")


def check_mirror(model, tol=1e-9):
    """Assert the written actuators are still exact mirror images.

    gainprm, biasprm[1..2] and forcerange are equal L/R in the source MJCF while
    biasprm[0] flips sign on the y and z axes, so the invariant is "equal, except
    biasprm[0] which is equal in magnitude". Scaling by an equal k preserves all
    four; an unequal k breaks every one of them, which is what this catches.
    """
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    idx = {n: i for i, n in enumerate(names)}
    pairs, _ = mirror_pairs(names)
    worst = 0.0
    for lname, rname in pairs:
        a, b = idx[lname], idx[rname]
        d = max(
            np.abs(model.actuator_gainprm[a] - model.actuator_gainprm[b]).max(),
            np.abs(model.actuator_biasprm[a, 1:3]
                   - model.actuator_biasprm[b, 1:3]).max(),
            np.abs(np.abs(model.actuator_forcerange[a])
                   - np.abs(model.actuator_forcerange[b])).max(),
            abs(abs(model.actuator_biasprm[a, 0])
                - abs(model.actuator_biasprm[b, 0])),
        )
        if d > tol:
            raise SystemExit(f"  !! {lname}/{rname} are not mirror images "
                             f"(max term difference {d:.3e})")
        worst = max(worst, float(d))
    return len(pairs), worst


def scale_line(line: str, k: float) -> str:
    """Multiply gainprm, all three biasprm and forcerange on one <general> line."""
    def one(m):
        return f'{m.group(1)}="{fmt(float(m.group(2)) * k)}"'

    def many(m):
        vals = [fmt(float(v) * k) for v in m.group(2).split()]
        return f'{m.group(1)}="{" ".join(vals)}"'

    line = re.sub(r'(gainprm)="([^"]+)"', one, line)
    line = re.sub(r'(biasprm)="([^"]+)"', many, line)
    line = re.sub(r'(forcerange)="([^"]+)"', many, line)
    return line


def verify(src_xml, out_xml, kmap, names, n_states=40, seed=0):
    """Confirm force_scaled == k * force_original in randomly sampled states.

    Checks the whole affine map at once -- random ctrl at random poses with
    random velocities -- because that is the only way to catch a term that was
    left unscaled. A pure-pose check would miss biasprm[2].
    """
    ma = mujoco.MjModel.from_xml_path(str(src_xml))
    mb = mujoco.MjModel.from_xml_path(str(out_xml))
    da, db = mujoco.MjData(ma), mujoco.MjData(mb)
    rng = np.random.default_rng(seed)
    k = np.array([kmap[n] for n in names])

    worst = 0.0
    for _ in range(n_states):
        q = np.zeros(ma.nq); q[3] = 1.0
        q[7:] = rng.uniform(ma.jnt_range[1:, 0], ma.jnt_range[1:, 1])
        v = rng.normal(0, 1.5, ma.nv)
        c = rng.uniform(-1, 1, ma.nu)
        for m, d in ((ma, da), (mb, db)):
            d.qpos[:] = q; d.qvel[:] = v; d.ctrl[:] = c
            mujoco.mj_forward(m, d)
        # forcerange clamps, so compare the pre-clamp affine output: read
        # actuator_force but only where neither model is saturated.
        fa, fb = da.actuator_force.copy(), db.actuator_force.copy()
        lim_a = np.abs(ma.actuator_forcerange).min(axis=1)
        lim_b = np.abs(mb.actuator_forcerange).min(axis=1)
        free = (np.abs(fa) < 0.999 * lim_a) & (np.abs(fb) < 0.999 * lim_b)
        if free.any():
            rel = np.abs(fb[free] - k[free] * fa[free]) / np.maximum(
                np.abs(k[free] * fa[free]), 1e-9)
            worst = max(worst, float(rel.max()))
    return worst, mb


def write_scaled_xml(src, out, kmap, check_symmetry=True):
    """Apply kmap to every <general> line of src, write out, and verify exactly.

    Shared with scripts/aggregate_motion_k.py, which supplies a k averaged over
    many motions instead of a single clip's fit; everything downstream of "here
    is one k per actuator" is identical, and the verification below is the part
    that must not be duplicated and drift.

    check_symmetry asserts the mirrored pairs came out identical, so it must be
    off when the caller deliberately used a per-actuator k (--symmetry none).
    """
    src, out = Path(src), Path(out)
    model = mujoco.MjModel.from_xml_path(str(src))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    missing = [n for n in names if n not in kmap]
    if missing:
        raise SystemExit(f"no k for {len(missing)} actuator(s): {missing[:5]}")

    text = src.read_text().splitlines(keepends=True)
    edited = 0
    for i, line in enumerate(text):
        m = re.search(r'<general\s+name="([^"]+)"', line)
        if m and m.group(1) in kmap:
            text[i] = scale_line(line, kmap[m.group(1)])
            edited += 1
    if edited != model.nu:
        raise SystemExit(f"edited {edited} actuator lines but the model has "
                         f"{model.nu}; aborting rather than writing a partial file")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        backup = out.with_suffix(out.suffix + ".bak")
        shutil.copy2(out, backup)
        print(f"existing {out.name} backed up to {backup.name}")
    out.write_text("".join(text))

    kv = np.array([kmap[n] for n in names])
    print(f"wrote {out}  ({edited} actuators scaled)")
    print(f"  k: median {np.median(kv):.4f}  range {kv.min():.4f}–{kv.max():.4f}")

    worst, mb = verify(src, out, kmap, names)
    print(f"\nverification (random poses/velocities/ctrl, unsaturated actuators):")
    print(f"  max relative error of force_scaled / (k * force_original): {worst:.2e}")
    if worst > 1e-9:
        raise SystemExit("  !! scaling is NOT exact -- an affine term was missed")
    print("  exact: gainprm, biasprm[0..2] and forcerange all scaled consistently")

    gain_r = mb.actuator_gainprm[:, 0] / model.actuator_gainprm[:, 0]
    print(f"  neutral-angle shift vs original: "
          f"{np.abs(mb.actuator_biasprm[:, 0] / mb.actuator_biasprm[:, 1] - model.actuator_biasprm[:, 0] / model.actuator_biasprm[:, 1]).max():.2e} rad "
          f"(0 = ctrl keeps its meaning)")
    print(f"  gainprm ratio matches k: {np.abs(gain_r - kv).max():.2e}")
    print(f"  new forcerange: mean ±{np.abs(mb.actuator_forcerange[:, 1]).mean():.1f} N·m "
          f"(was ±{np.abs(model.actuator_forcerange[:, 1]).mean():.1f})")
    if check_symmetry:
        n_pairs, worst_mirror = check_mirror(mb)
        print(f"  left/right mirror holds for all {n_pairs} pairs "
              f"(max term difference {worst_mirror:.2e})")
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--r2-min", type=float, default=0.9,
                   help="below this the fitted k is replaced by the predictor")
    p.add_argument("--symmetry", choices=["pair", "none"], default="pair",
                   help="'pair' gives each mirrored L/R actuator pair one shared "
                        "k; 'none' keeps the raw per-actuator fit")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    src, out = Path(args.src), Path(args.out)
    if not rows:
        raise SystemExit(f"{args.csv} is empty")

    model = mujoco.MjModel.from_xml_path(str(src))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    csv_names = [r["actuator"] for r in rows]
    if set(csv_names) != set(names):
        raise SystemExit(f"csv covers {len(csv_names)} actuators, model has "
                         f"{len(names)}; names do not match")

    kmap, notes, trusted = choose_k(rows, args.r2_min)
    if notes:
        print(f"{len(notes)} actuator(s) did not use the fitted k:")
        for n in notes:
            print(f"    {n}")

    if args.symmetry == "pair":
        report, midline = symmetrize_k(kmap, trusted)
        print()
        print_symmetry_report(report, midline)
    print()
    write_scaled_xml(src, out, kmap, check_symmetry=(args.symmetry == "pair"))


if __name__ == "__main__":
    main()
