"""R3 hard gate: model/bilevel/torch_kin.py must agree with MuJoCo exactly.

From proposal.md 8.1 R3 -- if the torch FK is silently wrong, the upper level
optimizes a geometry the simulator does not have and every loss curve still
looks completely normal. Nothing else in model/bilevel/ may be trusted until
this passes on all 13 bodies.

Checks, over N random qpos drawn inside jnt_range on every assets/robots/*/robot.xml:
  1. xpos  vs data.xpos                              (mj_kinematics)
  2. xquat vs data.xquat                             (sign-canonicalized)
  3. com   vs data.subtree_com[0]                    (mj_comPos)
  4. min_geom_z vs scripts/qpos_retarget.py:99 _min_geom_z
  5. normalized_ctrl round-trip through the actuator's affine bias law
  6. rest pose: root_rest_height vs skeleton.json, where one exists

Run:
    uv run model/bilevel/tests/test_torch_kin.py
    uv run model/bilevel/tests/test_torch_kin.py --n 2000 --body child
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from model.bilevel.torch_kin import TorchKinematics

TOL = 1e-9  # float64 throughout; anything above ~1e-12 would already be a real disagreement

# normalized_ctrl gets its own, looser tolerance. On the 12 bodies that kept the
# adult's actuator parameters the identity is exact (4.4e-16). `elderly` is the
# one body whose gainprm/biasprm/forcerange were rescaled by
# scripts/scale_robot.py:279, and writing those scaled values back out as XML
# text rounds them -- gainprm[0] and biasprm[1] no longer share a bit-exact
# ratio, so q*(ctrl=+-1) misses jnt_range by ~7e-6 rad (4e-4 degrees).
# Physically irrelevant for both consumers (the BC target in ppo.py and the
# feasibility penalty in semantics.py), but worth failing on if it ever grows:
# a large error here would mean the affine servo assumption has actually broken.
CTRL_TOL = 1e-4


def _min_geom_z_reference(model, data):
    """Verbatim copy of scripts/qpos_retarget.py:99 _min_geom_z, so the test
    compares against the implementation that actually produced the shipped
    dataset rather than a paraphrase of it."""
    lo = np.inf
    for gid in range(model.ngeom):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) == "floor":
            continue
        gtype = model.geom_type[gid]
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            for signs in itertools.product([1, -1], repeat=3):
                lo = min(lo, (pos + mat @ (np.array(signs) * size))[2])
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            c1 = (pos + mat @ np.array([0, 0, size[1]]))[2] - size[0]
            c2 = (pos + mat @ np.array([0, 0, -size[1]]))[2] - size[0]
            lo = min(lo, c1, c2)
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            lo = min(lo, pos[2] - size[0])
        else:
            lo = min(lo, pos[2])
    return lo


def _random_qpos(model, n, rng):
    """n random poses: root anywhere sane, hinges uniform inside jnt_range,
    root quaternion uniform on SO(3)."""
    q = np.zeros((n, model.nq))
    q[:, 0:2] = rng.uniform(-3.0, 3.0, size=(n, 2))
    q[:, 2] = rng.uniform(0.1, 2.0, size=n)
    quat = rng.normal(size=(n, 4))
    q[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    lo = model.jnt_range[1:, 0]
    hi = model.jnt_range[1:, 1]
    q[:, 7:] = rng.uniform(lo, hi, size=(n, model.nq - 7))
    return q


def check_body(xml_path: Path, n: int, rng, verbose: bool = True) -> dict:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    tk = TorchKinematics(model, dtype=torch.float64)

    qpos = _random_qpos(model, n, rng)

    # ---- reference, one frame at a time through MuJoCo -------------------
    ref_xpos = np.zeros((n, model.nbody, 3))
    ref_xquat = np.zeros((n, model.nbody, 4))
    ref_com = np.zeros((n, 3))
    ref_minz = np.zeros(n)
    for t in range(n):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        ref_xpos[t] = data.xpos
        ref_xquat[t] = data.xquat
        ref_com[t] = data.subtree_com[0]
        ref_minz[t] = _min_geom_z_reference(model, data)

    # ---- torch, all frames at once --------------------------------------
    q_t = torch.as_tensor(qpos, dtype=torch.float64)
    xpos, xquat = tk(q_t)
    com = tk.com(xpos, xquat)
    minz = tk.min_geom_z(xpos, xquat)

    # quaternions are a double cover: canonicalize both sides to w >= 0
    def canon(a):
        return np.where(a[..., :1] < 0, -a, a)

    e_pos = np.abs(xpos.numpy() - ref_xpos).max()
    e_quat = np.abs(canon(xquat.numpy()) - canon(ref_xquat)).max()
    e_com = np.abs(com.numpy() - ref_com).max()
    e_minz = np.abs(minz.numpy() - ref_minz).max()

    # ---- normalized_ctrl vs the actuator's own affine law ----------------
    # force = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*q + biasprm[2]*qdot;
    # the equilibrium angle for a given ctrl solves force = 0 at qdot = 0.
    g = model.actuator_gainprm[:, 0]
    b0 = model.actuator_biasprm[:, 0]
    b1 = model.actuator_biasprm[:, 1]
    hinge_q = qpos[:, 7:]
    a = tk.normalized_ctrl(torch.as_tensor(hinge_q, dtype=torch.float64)).numpy()
    q_from_ctrl = -(g * a + b0) / b1
    e_ctrl = np.abs(q_from_ctrl - hinge_q).max()

    # ---- rest height vs skeleton.json, where present ---------------------
    e_rest = None
    skel = xml_path.parent / "skeleton.json"
    if skel.exists():
        d = json.loads(skel.read_text())
        root = next(b for b in d["bodies"] if b["parent"] == "world")
        e_rest = abs(tk.root_rest_height - root["world_pos"][2])

    res = {
        "body": xml_path.parent.name,
        "xpos": e_pos, "xquat": e_quat, "com": e_com,
        "min_geom_z": e_minz, "ctrl": e_ctrl, "rest_height": e_rest,
    }
    if verbose:
        rest = "  -" if e_rest is None else f"{e_rest:9.2e}"
        print(
            f"{res['body']:<14} xpos={e_pos:9.2e} xquat={e_quat:9.2e} com={e_com:9.2e} "
            f"minz={e_minz:9.2e} ctrl={e_ctrl:9.2e} rest={rest}"
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="random poses per body")
    ap.add_argument("--body", default=None, help="only check this label")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robots", type=Path, default=REPO_ROOT / "assets" / "robots",
                    help="asset directory to check; use this after regenerating or "
                         "recalibrating actuators (scripts/calibrate_actuators.py)")
    args = ap.parse_args()

    robots = sorted(p for p in args.robots.iterdir() if (p / "robot.xml").exists())
    if args.body:
        robots = [p for p in robots if p.name == args.body]
        if not robots:
            raise SystemExit(f"no such body: {args.body}")

    rng = np.random.default_rng(args.seed)
    print(f"torch FK vs mj_kinematics -- {args.n} random poses x {len(robots)} bodies, tol={args.tol:g}\n")
    results = [check_body(p / "robot.xml", args.n, rng) for p in robots]

    worst = {k: max(r[k] for r in results) for k in ("xpos", "xquat", "com", "min_geom_z", "ctrl")}
    print("\nworst over all bodies:", {k: f"{v:.2e}" for k, v in worst.items()})

    limits = {"xpos": args.tol, "xquat": args.tol, "com": args.tol,
              "min_geom_z": args.tol, "ctrl": CTRL_TOL}
    failed = [k for k, v in worst.items() if v > limits[k]]
    rest_fail = [r["body"] for r in results if r["rest_height"] is not None and r["rest_height"] > 1e-6]
    if rest_fail:
        failed.append(f"rest_height({','.join(rest_fail)})")

    if failed:
        print(f"\nFAIL: {failed}")
        raise SystemExit(1)
    print("\nPASS -- torch FK matches MuJoCo on every body. Upper level is unblocked.")


if __name__ == "__main__":
    main()
