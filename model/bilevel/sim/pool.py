"""SimPool: 32 worker processes x 8 sims = 256 envs, driven from the main process.

Replaces gymnasium's SyncVectorEnv (model/train.py:235, batch 16, one process).
The measured shape of the problem, from proposal.md 6.1:

    16 workers x 16 sims  707 ms / 24-step iteration
    30 x 9                632 ms      <- best measured
    48 x 6               1550 ms
    64 x 4               1671 ms
    mp.Barrier instead of spin-yield: 1030 ms for the same work

Past ~32 the workers contend for physical cores and spin-waiting makes it much
worse; 32 x 8 is within noise of the 30 x 9 optimum and hits 256 exactly.

Threads were considered and rejected for v1: mj_step does release the GIL (8
threads scale ~8x on physics), but compute_humanoid_self_obs_v2 is pure numpy
and GIL-bound (8 threads -> 3.3x). A thread pool for physics plus a batched
obs in the main process is worth roughly 2x and is the optimization to reach
for if throughput ever becomes the blocker -- it needs
compute_humanoid_self_obs_v2 rewritten in batched form and proven bit-identical.
"""

import multiprocessing as mp
import os
import time
from typing import List, Optional, Sequence

import numpy as np

from model.bilevel import rewards as R
from model.bilevel.sim.protocol import (
    META_MODE, META_T, MODE_RESET, MODE_SHUTDOWN, MODE_STEP,
    SharedBuffers, WorkerInit, block_specs,
)
from model.bilevel.sim.worker import worker_entry


def physical_cpus(n: int) -> List[int]:
    """The first `n` PHYSICAL cores, avoiding hyperthread siblings.

    /sys/devices/system/cpu/cpuN/topology/thread_siblings_list lists each
    logical CPU's siblings; keeping only the first of each group leaves one
    logical CPU per physical core. Pinning workers to HT siblings is measurably
    worse -- two spinning workers on one physical core is the pathological case.
    """
    seen, out = set(), []
    base = "/sys/devices/system/cpu"
    try:
        cpus = sorted(
            int(d[3:]) for d in os.listdir(base)
            if d.startswith("cpu") and d[3:].isdigit()
        )
    except OSError:
        return list(range(n))
    for c in cpus:
        try:
            with open(f"{base}/cpu{c}/topology/thread_siblings_list") as f:
                sibs = f.read().strip()
        except OSError:
            sibs = str(c)
        if sibs in seen:
            continue
        seen.add(sibs)
        out.append(c)
        if len(out) >= n:
            break
    return out or list(range(n))


class SimPool:
    """Lockstep vectorized MuJoCo over shared memory.

    Usage:
        pool = SimPool(cfg, xml_paths, leg_lens, obs_dim)
        pool.reset_windows(ref_qpos, ref_qvel0, rsi_noise)     # RSI, no physics
        for t in range(1, H + 1):
            pool.step(ctrl, xfrc, raw_prior, t_ref=t)
            obs, terms, done = pool.obs, pool.rew_terms, pool.done
        pool.close()
    """

    def __init__(self, cfg, xml_paths: Sequence[str], leg_lens: Sequence[float],
                 obs_dim: int, nq: int = 76, nv: int = 75, nu: int = 69,
                 start_method: str = "spawn"):
        self.cfg = cfg
        self.n_envs = cfg.n_envs
        self.n_workers = cfg.n_workers
        self.per = cfg.sims_per_worker
        if len(xml_paths) != self.n_envs:
            raise ValueError(f"need one xml path per env slot ({self.n_envs}), got {len(xml_paths)}")

        self.specs = block_specs(
            n_envs=self.n_envs, n_workers=self.n_workers, horizon=cfg.horizon,
            nq=nq, nv=nv, nu=nu, obs_dim=obs_dim, n_terms=R.N_TERMS,
        )
        self.buf = SharedBuffers(self.specs)
        self._token = 0

        cpus = physical_cpus(self.n_workers) if cfg.pin_workers else [None] * self.n_workers
        ctx = mp.get_context(start_method)
        self.procs: List[mp.Process] = []
        for w in range(self.n_workers):
            lo, hi = w * self.per, (w + 1) * self.per
            init = WorkerInit(
                worker_id=w, slot_lo=lo, slot_hi=hi,
                xml_paths=[str(p) for p in xml_paths[lo:hi]],
                leg_lens=[float(v) for v in leg_lens[lo:hi]],
                shm_names=self.buf.names, specs=self.specs, cfg=cfg,
                cpu=cpus[w] if w < len(cpus) else None,
            )
            p = ctx.Process(target=worker_entry, args=(init,), daemon=True)
            p.start()
            self.procs.append(p)
        self._closed = False

    # ------------------------------------------------------------------ views
    @property
    def obs(self) -> np.ndarray: return self.buf["obs"]
    @property
    def rew_terms(self) -> np.ndarray: return self.buf["rew_terms"]
    @property
    def done(self) -> np.ndarray: return self.buf["done"]
    @property
    def qpos(self) -> np.ndarray: return self.buf["qpos"]
    @property
    def qvel(self) -> np.ndarray: return self.buf["qvel"]

    # ------------------------------------------------------------------ driving

    def _dispatch(self, mode: int, t_ref: int) -> None:
        b = self.buf
        b["meta"][META_MODE] = mode
        b["meta"][META_T] = t_ref
        self._token += 1
        # meta and every input block are written before the token bump; workers
        # only read them after seeing the new token (x86-TSO, see protocol.py).
        b["cmd"][:] = self._token
        self._await()

    def _await(self) -> None:
        ack = self.buf["ack"]
        tok = self._token
        spin = self.cfg.spin_iters
        deadline = time.perf_counter() + self.cfg.worker_timeout_s
        k = 0
        while True:
            if bool((ack == tok).all()):
                return
            k += 1
            if k > spin:
                time.sleep(0)
                if time.perf_counter() > deadline:
                    self._diagnose_stall(tok)

    def _diagnose_stall(self, tok: int) -> None:
        """A worker died or hung. The spin loop would otherwise wait forever, so
        report which slots are affected and fail loudly."""
        ack = self.buf["ack"]
        stuck = [w for w in range(self.n_workers) if int(ack[w]) != tok]
        dead = [w for w in stuck if not self.procs[w].is_alive()]
        detail = ", ".join(
            f"w{w}[slots {w * self.per}:{(w + 1) * self.per}]"
            f"{' DEAD exit=' + str(self.procs[w].exitcode) if w in dead else ''}"
            for w in stuck
        )
        self.close()
        raise RuntimeError(
            f"SimPool stalled after {self.cfg.worker_timeout_s}s waiting on token {tok}: {detail}"
        )

    def reset_windows(self, ref_qpos: np.ndarray, ref_qvel0: np.ndarray,
                      rsi_noise: np.ndarray) -> None:
        """RSI every slot to the start of its new reference window.

        Issued as its own command with no physics -- so a window costs 25
        dispatches for 24 physics steps, about 4% overhead, in exchange for
        never having to interleave reset and step logic.
        """
        self.buf["ref_qpos"][...] = ref_qpos
        self.buf["ref_qvel0"][...] = ref_qvel0
        self.buf["rsi_noise"][...] = rsi_noise
        self._dispatch(MODE_RESET, 0)

    def step(self, ctrl: np.ndarray, xfrc: np.ndarray, raw_prior: np.ndarray,
             t_ref: int) -> None:
        """One 30 Hz control step (action_repeat=15 physics substeps).

        `xfrc` is the root wrench in the WORLD frame -- RootWrenchHead emits it
        in the root's own frame and rollout.py rotates it, so the head stays
        rotation-equivariant. `t_ref` is the reference index the RESULTING state
        is scored against, i.e. call with t+1 for the t-th step.
        """
        self.buf["ctrl"][...] = ctrl
        self.buf["xfrc"][...] = xfrc
        self.buf["raw_prior"][...] = raw_prior
        self._dispatch(MODE_STEP, t_ref)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.buf["meta"][META_MODE] = MODE_SHUTDOWN
            self.buf["cmd"][:] = -1
        except Exception:
            pass
        for p in self.procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        self.buf.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
