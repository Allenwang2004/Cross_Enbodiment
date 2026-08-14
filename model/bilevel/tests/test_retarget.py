"""p = 0 must reproduce scripts/qpos_retarget.py:91 retarget_qpos exactly.

This is the anchor the whole anti-degeneracy argument rests on (proposal.md
3.2): the naive retarget is the ARCHITECTURAL ORIGIN of the parameterization,
not a soft target maintained by a penalty. If this drifts, `lambda_prox`, the
`upper_warmup_iters` freeze and every S/G baseline recorded in Stage 0 all
silently refer to the wrong reference.

Also checks:
  - the hard box actually bounds the outputs at extreme u
  - the 69 hinges partition cleanly into the 14 groups, left/right tied
  - RetargetNet is zero-initialized, so u == 0 at construction
  - gradients reach p through ref_raw (and survive where clamp would kill them)
  - the time-warp path is the identity at tau = 1

Run:
    uv run model/bilevel/tests/test_retarget.py
"""

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

from model.bilevel.config import BilevelConfig
from model.bilevel.retarget import (
    JOINT_GROUPS, N_GROUPS, U_DIM, RetargetNet, Retargeter, joint_group_index,
)
from model.bilevel.torch_kin import TorchKinematics

ROBOTS = REPO_ROOT / "assets" / "robots"


def reference_retarget_qpos(qpos, scale):
    """Verbatim from scripts/qpos_retarget.py:91 -- the function this must match."""
    out = qpos.copy()
    out[:, 0:3] *= scale
    return out


def main():
    cfg = BilevelConfig()
    src_model = mujoco.MjModel.from_xml_path(str(ROBOTS / cfg.source_body / "robot.xml"))
    src_h = float(src_model.qpos0[2])

    rng = np.random.default_rng(0)
    bodies = sorted(p.name for p in ROBOTS.iterdir() if (p / "robot.xml").exists())
    print(f"p=0 anchor vs scripts/qpos_retarget.py:91, source={cfg.source_body} (h={src_h:.6f})\n")

    net = RetargetNet(beta_dim=8, hidden_dims=cfg.retarget_hidden_dims).double()
    worst_anchor = 0.0

    for name in bodies:
        kin = TorchKinematics(mujoco.MjModel.from_xml_path(str(ROBOTS / name / "robot.xml")))
        rt = Retargeter(cfg, net, kin, src_h).double()

        # ---- 1. p = 0 anchor -----------------------------------------
        T = 40
        src = np.zeros((1, T, 76))
        src[0, :, 0:3] = rng.uniform(-2, 2, size=(T, 3))
        q = rng.normal(size=(T, 4))
        src[0, :, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
        src[0, :, 7:] = rng.uniform(-1.0, 1.0, size=(T, 69))

        beta = torch.zeros(1, 8, dtype=torch.float64)
        u0 = torch.zeros(1, U_DIM, dtype=torch.float64)
        ref_raw, ref, _ = rt(torch.as_tensor(src), beta, u_override=u0)

        expected = reference_retarget_qpos(src[0], kin.root_rest_height / src_h)
        e_anchor = np.abs(ref_raw[0].detach().numpy() - expected).max()
        worst_anchor = max(worst_anchor, e_anchor)

        # ---- 2. the clamp is the only difference between raw and ref ---
        lo, hi = kin.hinge_lo.numpy(), kin.hinge_hi.numpy()
        e_clamp = np.abs(ref[0, :, 7:].detach().numpy() - np.clip(expected[:, 7:], lo, hi)).max()
        e_root_untouched = np.abs(ref[0, :, :7].detach().numpy() - ref_raw[0, :, :7].detach().numpy()).max()

        # ---- 3. hard box holds at extreme u ----------------------------
        u_big = torch.full((1, U_DIM), 50.0, dtype=torch.float64)
        p = rt.params(u_big)
        box_ok = (
            float(p["joint_gain"].max()) <= 1.0 + cfg.box_joint_gain + 1e-9
            and float(p["joint_bias"].abs().max()) <= cfg.box_joint_bias + 1e-9
            and float(p["root_dz"].abs().max()) <= cfg.box_root_dz + 1e-9
            and float(p["root_rot"].abs().max()) <= cfg.box_root_rot + 1e-9
        )
        scale_ratio = float((p["root_scale"] / (kin.root_rest_height / src_h)).max())

        print(
            f"{name:<14} anchor={e_anchor:9.2e} clamp={e_clamp:9.2e} root_untouched={e_root_untouched:9.2e} "
            f"box_ok={box_ok} max_scale_dev={scale_ratio - 1:+.3f}"
        )
        assert box_ok, f"{name}: hard box violated"
        assert e_clamp < 1e-12, f"{name}: ref is not exactly clamp(ref_raw)"
        assert e_root_untouched < 1e-12, f"{name}: clamp touched the root"

    print(f"\nworst p=0 anchor error over all bodies: {worst_anchor:.2e}")
    assert worst_anchor < 1e-12, "p=0 does NOT reproduce retarget_qpos"

    # ---- 4. joint grouping ---------------------------------------------
    kin = TorchKinematics(mujoco.MjModel.from_xml_path(str(ROBOTS / "child" / "robot.xml")))
    gi = joint_group_index(kin.hinge_names)
    counts = torch.bincount(gi, minlength=N_GROUPS)
    print("\ngroup sizes:", {g: int(c) for g, c in zip(JOINT_GROUPS, counts)})
    assert int(counts.sum()) == 69
    assert (counts > 0).all(), "some group is empty"
    for k, name in enumerate(kin.hinge_names):
        mirror = name.replace("L_", "R_", 1) if name.startswith("L_") else name.replace("R_", "L_", 1)
        if mirror in kin.hinge_names:
            j = kin.hinge_names.index(mirror)
            assert gi[k] == gi[j], f"{name} and {mirror} are in different groups"
    print("left/right tied: OK")

    # ---- 5. zero init ---------------------------------------------------
    net_f = RetargetNet(beta_dim=8, hidden_dims=cfg.retarget_hidden_dims)
    u_init = net_f(torch.randn(16, 8)).detach()
    assert float(u_init.abs().max()) == 0.0, "RetargetNet is not zero-initialized"
    print("RetargetNet zero-init: OK (u == 0 at construction)")

    # ---- 6. gradient reaches p, including outside jnt_range ----------
    rt = Retargeter(cfg, RetargetNet(8, cfg.retarget_hidden_dims).double(), kin, src_h).double()
    # force every hinge far outside its range so a clamp-only path would be dead
    src_t = torch.as_tensor(src).clone()
    src_t[..., 7:] = 5.0
    beta = torch.randn(1, 8, dtype=torch.float64)
    ref_raw, ref, u = rt(src_t, beta)
    a_raw = kin.normalized_ctrl(ref_raw[..., 7:])
    loss_raw = torch.relu(a_raw.abs() - 1).pow(2).mean()
    loss_raw.backward()
    g_raw = sum(p.grad.abs().sum() for p in rt.net.parameters() if p.grad is not None)
    rt.zero_grad()
    ref_raw2, ref2, _ = rt(src_t, beta)
    ref2[..., 7:].pow(2).mean().backward()
    g_clamped = sum(
        (p.grad.abs().sum() if p.grad is not None else torch.tensor(0.0)) for p in rt.net.parameters()
    )
    print(f"grad via ref_raw (feasibility): {float(g_raw):.3e}   via clamped ref: {float(g_clamped):.3e}")
    assert float(g_raw) > 0, "no gradient reaches p through ref_raw"
    assert float(g_clamped) == 0.0, (
        "expected the clamped path to be gradient-dead here -- that is exactly why "
        "apply_retarget returns ref_raw separately"
    )

    # ---- 7. time warp is the identity at tau = 1 -----------------------
    cfg_tw = BilevelConfig(enable_time_warp=True)
    rt_tw = Retargeter(cfg_tw, RetargetNet(8, cfg.retarget_hidden_dims).double(), kin, src_h).double()
    ref_a, _, _ = rt_tw(torch.as_tensor(src), torch.zeros(1, 8, dtype=torch.float64),
                        u_override=torch.zeros(1, U_DIM, dtype=torch.float64), n_out=24)
    ref_b, _, _ = rt(torch.as_tensor(src), torch.zeros(1, 8, dtype=torch.float64),
                     u_override=torch.zeros(1, U_DIM, dtype=torch.float64), n_out=24)
    e_tw = float((ref_a - ref_b).abs().max())
    print(f"time-warp identity at tau=1: {e_tw:.2e}")
    assert e_tw < 1e-12

    print("\nPASS -- p=0 is the naive retarget, the box holds, and gradients flow.")


if __name__ == "__main__":
    main()
