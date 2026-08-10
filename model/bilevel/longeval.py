"""Periodic long-rollout diagnostic (proposal.md R6).

Training optimizes 24-step windows and re-initializes each one from the
reference (RSI), so tracking error never accumulates across windows. A long
rollout has no such reset, and a policy that is excellent at 0.8 s can still be
useless at 10 s -- measured on stage 1: 22.3/24 steps survived in training
against 22/299 in a single continuous rollout of the same clip and body.

R6 asked for exactly this and it was never implemented, which is why the gap sat
undetected until someone rendered a video. Nothing here feeds a gradient; it is
a metric, so that "is the long-horizon behaviour improving?" stops being a
question you can only answer after the run finishes.

Deliberate choices:

  f_max = 0        Reported without the wrench crutch even while training uses
                   it, because that is the condition the deliverable runs in
                   (proposal.md R5).
  fixed pairs      The same (clip, body) pairs every time it is called, chosen
                   once from a fixed seed. A curve over iterations is only
                   readable if the test set does not move under it.
  no termination   The four alive tests are evaluated but NOT enforced: the
                   point of a long rollout is to see what happens after the
                   first stumble, which a training-style termination hides.
  single process   Roughly 1.15 ms per env-step, so 8 clips x 299 steps is about
                   2.8 s. At the default every-100-iterations that is under
                   0.03 s/iter -- cheaper than reusing SimPool, whose shared
                   buffers are sized for H+1 = 25 frames and cannot hold a
                   299-frame reference without being rebuilt.
"""

from typing import Dict, List, Optional

import mujoco
import numpy as np
import torch

from model.bilevel import rewards as R
from model.bilevel.data import ref_qvel_from_qpos
from model.bilevel.policy import CTRL_DIM
from model.bilevel.quat import quat_rot
from model.bilevel.rollout import root_features
from model.bilevel.sim.worker import _make_obs_fn

# Prefixed `long/` in W&B by train_bilevel._wandb_metrics.
LONG_KEYS = ("long_survive", "long_steps", "long_pose_err", "long_root_dist",
             "long_root_dist_max", "long_diverged")


def rollout_one(cfg, model, leg_len, ref, policy, z0, beta, obs_fn,
                wrench_frac=0.0, deterministic=True, device=None):
    """One continuous rollout of the whole reference. -> (qpos (T+1, nq), stats).

    Shared with scripts/rollout_video.py so the number on the W&B curve and the
    number under the video are produced by the same code.
    """
    d = mujoco.MjData(model)
    ctx = R.RewardContext.build(model, cfg.dt, leg_len)
    dev = device or next(policy.parameters()).device
    T = ref.shape[0] - 1

    # RSI at the reference's first frame exactly -- no noise. This measures
    # tracking, not robustness to a perturbed start.
    mujoco.mj_resetData(model, d)
    d.qpos[:] = ref[0]
    d.qvel[:] = ref_qvel_from_qpos(model, ref[0], ref[1], cfg.dt)
    mujoco.mj_forward(model, d)

    rc = R.reference_cache(ctx, ref)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=dev).unsqueeze(0)
    z0_t = torch.as_tensor(z0, dtype=torch.float32, device=dev).unsqueeze(0)
    with torch.no_grad():
        z_beta = policy.latent(beta_t, z0_t)

    f_max = torch.full((1,), wrench_frac * cfg.wrench_f_frac * ctx.weight,
                       device=dev, dtype=torch.float32)
    m_max = f_max * cfg.wrench_m_frac * ctx.leg_len

    qpos_out = np.zeros((T + 1, model.nq))
    qpos_out[0] = d.qpos
    pose_err: List[float] = []
    root_dist: List[float] = []
    alive_steps = 0
    first_failure: Optional[int] = None
    diverged = False

    for t in range(T):
        qp = torch.as_tensor(d.qpos, dtype=torch.float32, device=dev).unsqueeze(0)
        qv = torch.as_tensor(d.qvel, dtype=torch.float32, device=dev).unsqueeze(0)
        ob = torch.as_tensor(obs_fn(model, d), dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            mean, _ = policy.act_mean(ob, beta_t, z_beta, root_features(qp, qv))
            action = mean if deterministic else policy.dist(mean).sample()
            ctrl, force, torque = policy.split_action(action, f_max, m_max)
            xfrc = torch.cat([quat_rot(qp[:, 3:7], force),
                              quat_rot(qp[:, 3:7], torque)], dim=-1)

        d.ctrl[:] = ctrl[0].cpu().numpy().astype(np.float64)
        d.xfrc_applied[ctx.root_bid, :] = xfrc[0].cpu().numpy().astype(np.float64)
        try:
            mujoco.mj_step(model, d, nstep=cfg.action_repeat)
            mujoco.mj_step1(model, d)
            if d.warning.number.any():
                raise RuntimeError("mujoco warning")
        except (RuntimeError, ValueError):
            diverged = True
            if first_failure is None:
                first_failure = t
            qpos_out[t + 1:] = qpos_out[t]
            break

        qpos_out[t + 1] = d.qpos
        pe = float(np.mean((d.qpos[7:] - rc["hinge"][t + 1]) ** 2))
        rd = float(np.linalg.norm(d.xpos[ctx.root_bid] - rc["root"][t + 1]))
        pose_err.append(pe)
        root_dist.append(rd)

        up = R._up_z(d.qpos[3:7])
        ref_up = R._up_z(ref[t + 1, 3:7])
        ok = (d.qpos[2] > cfg.term_root_height_frac * float(rc["root"][t + 1][2])
              and up > ref_up - cfg.term_up_margin
              and rd < cfg.term_root_dist * (ctx.leg_len / R._ADULT_LEG_LEN)
              and pe < cfg.term_pose_err)
        if ok and alive_steps == t:
            alive_steps = t + 1
        elif first_failure is None:
            first_failure = t

    return qpos_out, {
        "steps": T,
        "alive_steps": alive_steps,
        "first_failure": first_failure,
        "diverged": diverged,
        # Averages over the ALIVE prefix only. Once the robot is on the floor
        # the numbers describe a corpse sliding, not tracking, and averaging
        # those in makes a policy that falls late look worse than one that falls
        # immediately.
        "pose_err": float(np.mean(pose_err[:alive_steps])) if alive_steps else float("nan"),
        "root_dist": float(np.mean(root_dist[:alive_steps])) if alive_steps else float("nan"),
        "root_dist_max": float(np.max(root_dist[:alive_steps])) if alive_steps else float("nan"),
    }


class LongEvaluator:
    """A fixed panel of (clip, body) pairs, re-rolled on demand."""

    def __init__(self, cfg, ds, device, n_clips: int = 8, horizon: int = 299, seed: int = 0):
        self.cfg = cfg
        self.ds = ds
        self.device = device
        self.horizon = horizon
        self.obs_fn = _make_obs_fn()

        rng = np.random.default_rng(seed)
        n_bodies = len(ds.bodies)
        # Stratified over bodies, and only clips long enough to be a real test.
        usable = [i for i, c in enumerate(ds.clips) if c.qpos.shape[0] >= horizon + 1]
        if not usable:
            usable = list(range(len(ds.clips)))
        pick = rng.choice(usable, size=n_clips, replace=len(usable) < n_clips)
        self.pairs = [(int(ci), i % n_bodies) for i, ci in enumerate(pick)]

    @torch.no_grad()
    def run(self, policy, retargeters, wrench_frac: float = 0.0) -> Dict[str, float]:
        cfg, ds, dev = self.cfg, self.ds, self.device
        was_training = policy.training
        policy.eval()
        surv, steps, pe, rd, rdm, div = [], [], [], [], [], 0
        try:
            for clip_idx, b in self.pairs:
                clip = ds.clips[clip_idx]
                spec = ds.bodies[b]
                T = min(self.horizon, clip.qpos.shape[0] - 1)
                src = torch.as_tensor(clip.qpos[: T + 1], dtype=torch.float64,
                                      device=dev).unsqueeze(0)
                beta = torch.as_tensor(spec.beta, dtype=torch.float64, device=dev).unsqueeze(0)
                _, ref, _ = retargeters[b](src, beta, n_out=T + 1)
                _, st = rollout_one(
                    cfg, spec.model, spec.leg_len, ref[0].cpu().numpy(),
                    policy, clip.z0, spec.beta, self.obs_fn,
                    wrench_frac=wrench_frac, deterministic=True, device=dev,
                )
                surv.append(st["alive_steps"] / max(1, st["steps"]))
                steps.append(st["alive_steps"])
                div += int(st["diverged"])
                if st["alive_steps"]:
                    pe.append(st["pose_err"])
                    rd.append(st["root_dist"])
                    rdm.append(st["root_dist_max"])
        finally:
            policy.train(was_training)

        nan = float("nan")
        return {
            "long_survive": float(np.mean(surv)),
            "long_steps": float(np.mean(steps)),
            "long_pose_err": float(np.mean(pe)) if pe else nan,
            "long_root_dist": float(np.mean(rd)) if rd else nan,
            "long_root_dist_max": float(np.max(rdm)) if rdm else nan,
            "long_diverged": float(div) / len(self.pairs),
        }
