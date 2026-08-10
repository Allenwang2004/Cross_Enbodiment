"""One sim worker process: owns `sims_per_worker` raw MuJoCo sims.

MUST NOT IMPORT TORCH. 32 of these are spawned; torch's import cost and its
thread pools would be pure waste, and its intra-op threads would fight the
affinity pinning.

Replicates the parts of humenv.env.HumEnv that matter and drops the parts that
get in the way:
  kept    -- mj_step(nstep=action_repeat) then mj_step1, then
             compute_humanoid_self_obs_v2 off data.sensordata. The trailing
             mj_step1 is NOT optional: the obs reads the 48 framelinvel/
             frameangvel sensors, which are only valid after a sensor-stage
             update. humenv/env.py:121-126 does the same thing.
  added   -- reference-state initialization from an arbitrary qpos/qvel with
             Gaussian noise on the hinges, a root wrench via xfrc_applied, and
             per-step reward terms.
  dropped -- gymnasium wrapping, and HumEnv.is_terminated() (which always
             returns False, humenv/env.py:138, so survival has to be judged
             here or not at all).
"""

import os
import time
from typing import Dict, List, Optional

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from model.bilevel import rewards as R
from model.bilevel.sim.protocol import (
    META_MODE, META_T, MODE_RESET, MODE_SHUTDOWN, MODE_STEP, SharedBuffers, WorkerInit,
)


class _Slot:
    """One simulation: its model, its data, and the state that has to persist
    across control steps (previous ctrl for the smoothness term, previous foot
    positions for the slip term, the cached reference FK for the window)."""

    __slots__ = ("model", "data", "ctx", "obs_fn", "ref", "prev_ctrl", "prev_foot_xy",
                 "frozen", "hinge_lo", "hinge_hi")

    def __init__(self, xml_path: str, leg_len: float, cfg, obs_fn):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = cfg.physics_dt
        self.data = mujoco.MjData(self.model)
        self.ctx = R.RewardContext.build(self.model, cfg.dt, leg_len)
        self.obs_fn = obs_fn
        self.ref: Optional[Dict[str, np.ndarray]] = None
        self.prev_ctrl: Optional[np.ndarray] = None
        self.prev_foot_xy = None
        self.frozen = False
        self.hinge_lo = self.model.jnt_range[1:, 0].copy()
        self.hinge_hi = self.model.jnt_range[1:, 1].copy()

    def observe(self) -> np.ndarray:
        return self.obs_fn(self.model, self.data)


def _make_obs_fn():
    """humenv's own observation builder, reproducing HumEnv.get_obs() exactly.

    Three details that are easy to get wrong and impossible to notice later,
    all read straight off humenv/env.py:144-152:
      - it calls mj_kinematics AGAIN, after mj_step1, before building the obs
      - compute_humanoid_self_obs_v2 returns an OrderedDict, not an array; the
        358-dim proprio vector is its values concatenated IN INSERTION ORDER
      - the flags are (upright_start=False, root_height_obs=True,
        humanoid_type="smpl")

    Getting any of these wrong would feed the frozen actor a permuted or
    mis-scaled observation -- which would degrade silently rather than crash.
    """
    from humenv.env import compute_humanoid_self_obs_v2

    def obs_fn(model, data):
        mujoco.mj_kinematics(model, data)
        d = compute_humanoid_self_obs_v2(
            model, data, upright_start=False, root_height_obs=True, humanoid_type="smpl"
        )
        return np.concatenate([v.ravel() for v in d.values()], axis=0).astype(np.float32)

    return obs_fn


def _rsi(slot: _Slot, ref_qpos: np.ndarray, ref_qvel0: np.ndarray,
         noise: np.ndarray, cfg) -> None:
    """Reference-state initialization with Gaussian perturbation on the hinges.

    new.md: "rollout 的時候 除了 root 以外 關節要加入高斯擾動". The root's
    position and orientation are set EXACTLY from the reference; only the 69
    hinges are perturbed. `noise` already carries the per-window sigma, drawn in
    the main process so it is seedable and can be shared between the members of
    an antithetic ES pair (CRN, see upper.py).
    """
    m, d = slot.model, slot.data
    scale = 1.0
    for _ in range(cfg.rsi_max_retries + 1):
        mujoco.mj_resetData(m, d)   # NB: this also zeroes xfrc_applied
        d.qpos[:] = ref_qpos[0]
        d.qpos[7:] += noise * scale
        np.clip(d.qpos[7:], slot.hinge_lo, slot.hinge_hi, out=d.qpos[7:])
        d.qvel[:] = ref_qvel0
        mujoco.mj_forward(m, d)     # fills sensordata, so obs is valid immediately
        if d.ncon == 0 or float(d.contact.dist[:d.ncon].min()) >= cfg.rsi_penetration_tol:
            break
        # The noise pushed the body into itself or through the floor. Wide
        # bodies (heavy, short_stocky) hit this most; halve and retry.
        scale *= 0.5
    slot.ref = R.reference_cache(slot.ctx, ref_qpos)
    slot.prev_ctrl = None
    slot.prev_foot_xy = None
    slot.frozen = False


def worker_main(init: WorkerInit) -> None:
    cfg = init.cfg
    if init.cpu is not None:
        try:
            os.sched_setaffinity(0, {init.cpu})
        except OSError:
            pass  # not fatal; pinning is a ~14% optimization, not a requirement
    # A worker is single-threaded by construction; letting BLAS spin up its own
    # pool would oversubscribe every pinned core.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    buf = SharedBuffers(init.specs, names=init.shm_names)
    cmd, ack, meta = buf["cmd"], buf["ack"], buf["meta"]
    ctrl_b, xfrc_b, prior_b = buf["ctrl"], buf["xfrc"], buf["raw_prior"]
    ref_b, refv_b, noise_b = buf["ref_qpos"], buf["ref_qvel0"], buf["rsi_noise"]
    obs_b, term_b, done_b = buf["obs"], buf["rew_terms"], buf["done"]
    qpos_b, qvel_b = buf["qpos"], buf["qvel"]

    obs_fn = _make_obs_fn()
    lo, hi = init.slot_lo, init.slot_hi
    slots = [
        _Slot(init.xml_paths[i], init.leg_lens[i], cfg, obs_fn) for i in range(hi - lo)
    ]

    w = init.worker_id
    seen = 0
    ack[w] = 0
    spin = cfg.spin_iters
    scratch = np.zeros(R.N_TERMS, dtype=np.float64)

    while True:
        # ---- wait for work (spin, then yield) ---------------------------
        k = 0
        while cmd[w] == seen:
            k += 1
            if k > spin:
                time.sleep(0)   # yield the core without a syscall-heavy sleep
        seen = int(cmd[w])
        if seen < 0 or meta[META_MODE] == MODE_SHUTDOWN:
            break

        mode = int(meta[META_MODE])
        t_ref = int(meta[META_T])

        for i, slot in enumerate(slots):
            e = lo + i
            if mode == MODE_RESET:
                _rsi(slot, ref_b[e], refv_b[e], noise_b[e], cfg)
                term_b[e] = 0.0
                done_b[e] = 0
            elif slot.frozen:
                # A terminated slot costs nothing for the rest of the window.
                # Its rows are masked out of GAE and every loss in ppo.py.
                done_b[e] = 1
                term_b[e] = 0.0
                continue
            else:
                d, m = slot.data, slot.model
                ctrl = ctrl_b[e].astype(np.float64)
                d.ctrl[:] = ctrl
                d.xfrc_applied[slot.ctx.root_bid, :] = xfrc_b[e]
                try:
                    mujoco.mj_step(m, d, nstep=cfg.action_repeat)
                    mujoco.mj_step1(m, d)   # refresh sensordata for the obs
                    if d.warning.number.any():
                        raise RuntimeError("mujoco warning")
                except (RuntimeError, ValueError):
                    # Divergence. humenv raises here (humenv/env.py:122); a
                    # worker must NEVER die -- the main process spins on ack and
                    # would deadlock. Terminate the slot instead.
                    mujoco.mj_resetData(m, d)
                    d.warning.number[:] = 0
                    slot.frozen = True
                    done_b[e] = 1
                    term_b[e] = 0.0
                    obs_b[e] = 0.0
                    continue

                slot.prev_foot_xy = R.step_terms(
                    slot.ctx, cfg, d, slot.ref, t_ref,
                    ctrl=ctrl, prev_ctrl=slot.prev_ctrl,
                    raw_prior=prior_b[e].astype(np.float64),
                    wrench=xfrc_b[e].astype(np.float64),
                    prev_foot_xy=slot.prev_foot_xy, out=scratch,
                )
                slot.prev_ctrl = ctrl
                term_b[e] = scratch
                dead = scratch[R.TERM_IDX["alive"]] < 0.5
                done_b[e] = 1 if dead else 0
                if dead:
                    slot.frozen = True

            obs_b[e] = slot.observe()
            qpos_b[e] = slot.data.qpos
            qvel_b[e] = slot.data.qvel

        ack[w] = seen

    buf.close()


def worker_entry(init: WorkerInit) -> None:
    """Spawn target. Kept trivially thin so a traceback points at worker_main."""
    import sys
    from pathlib import Path

    repo = str(Path(__file__).resolve().parents[3])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    worker_main(init)
