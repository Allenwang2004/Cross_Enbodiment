#!/usr/bin/env python3
"""scale_child_actuators.py — write a child MJCF whose actuators are scaled by the
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

Usage:
  uv run scripts/scale_child_actuators.py
  uv run scripts/scale_child_actuators.py --out assets/robots/child/robot_torque.xml
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
    """Per-actuator scale factor, with the fallback rule applied and reported."""
    out, notes = {}, []
    for r in rows:
        name = r["actuator"]
        kf, r2, kp = (float(r["k_fitted"]), float(r["r2"]),
                      float(r["k_predicted_subtree"]))
        if r2 >= r2_min and kf > 0:
            out[name] = kf
            continue
        if not (kp > 0) or not np.isfinite(kp):
            out[name] = 1.0
            notes.append(f"{name}: R²={r2:.3f}, k_fitted={kf:+.4f}, predictor "
                         f"unusable too -> left unscaled (k=1)")
            continue
        out[name] = kp
        why = "k_fitted<=0" if kf <= 0 else f"R²={r2:.3f}<{r2_min}"
        notes.append(f"{name}: {why}, k_fitted={kf:+.4f} -> using predictor {kp:.4f}")
    return out, notes


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


def write_scaled_xml(src, out, kmap):
    """Apply kmap to every <general> line of src, write out, and verify exactly.

    Shared with scripts/aggregate_motion_k.py, which supplies a k averaged over
    many motions instead of a single clip's fit; everything downstream of "here
    is one k per actuator" is identical, and the verification below is the part
    that must not be duplicated and drift.
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
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--r2-min", type=float, default=0.9,
                   help="below this the fitted k is replaced by the predictor")
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

    kmap, notes = choose_k(rows, args.r2_min)
    if notes:
        print(f"{len(notes)} actuator(s) did not use the fitted k:")
        for n in notes:
            print(f"    {n}")

    write_scaled_xml(src, out, kmap)


if __name__ == "__main__":
    main()
