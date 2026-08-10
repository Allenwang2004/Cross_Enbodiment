"""Shared-memory layout between the main process and the sim workers.

SINGLE SOURCE OF TRUTH. Both sides build their numpy views from the table
below, so a shape or dtype can only be changed in one place.

Nothing is pickled per step. Everything crosses the process boundary as a
numpy view onto a multiprocessing.shared_memory block; a full iteration moves
under 6 MB in each direction. This is the whole reason gymnasium's VectorEnv is
bypassed: its per-step pickling, its inability to set an arbitrary initial
state on an async worker, and its lack of any hook for data.xfrc_applied all
make the plan in proposal.md 6 impossible to express through it.

Synchronization
---------------
A monotone token per worker, spin-then-yield on both sides -- NOT
multiprocessing.Barrier, which measured 1030 ms/iteration against 632 ms for
the identical work (with 33 waiters every release is a thundering herd, and
there are 25 of them per iteration).

Ordering assumption: the main process writes `meta` and all input blocks BEFORE
bumping `cmd`, and a worker reads them only after observing a new `cmd[w]`.
That is safe on x86-TSO (stores are not reordered with other stores, loads not
with other loads) which is the only platform this targets. On a weakly-ordered
machine this would need explicit fences.
"""

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple

import numpy as np

# Bump whenever a block's name, shape, dtype or MEANING changes -- including
# reordering rewards.TERM_NAMES, which is the payload of `rew_terms`.
PROTOCOL_VERSION = 1

# meta[] slots
META_MODE = 0        # 0 = step, 1 = reset (RSI), -1 = shutdown
META_T = 1           # reference index to score the resulting state against
META_LEN = 4

MODE_STEP = 0
MODE_RESET = 1
MODE_SHUTDOWN = -1


@dataclass(frozen=True)
class BlockSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    direction: str  # "in" = main -> worker, "out" = worker -> main, "ctl" = sync


def block_specs(n_envs: int, n_workers: int, horizon: int, nq: int, nv: int,
                nu: int, obs_dim: int, n_terms: int) -> List[BlockSpec]:
    H1 = horizon + 1
    return [
        # ---- sync -------------------------------------------------------
        BlockSpec("cmd", (n_workers,), np.int64, "ctl"),
        BlockSpec("ack", (n_workers,), np.int64, "ctl"),
        BlockSpec("meta", (META_LEN,), np.int64, "ctl"),
        # ---- main -> worker ---------------------------------------------
        BlockSpec("ctrl", (n_envs, nu), np.float32, "in"),
        BlockSpec("xfrc", (n_envs, 6), np.float32, "in"),          # WORLD frame; main rotates
        BlockSpec("raw_prior", (n_envs, nu), np.float32, "in"),    # frozen actor's mean, for e_res
        BlockSpec("ref_qpos", (n_envs, H1, nq), np.float64, "in"),  # phi-adjusted, clamped
        BlockSpec("ref_qvel0", (n_envs, nv), np.float64, "in"),     # via mj_differentiatePos
        BlockSpec("rsi_noise", (n_envs, nu), np.float32, "in"),     # sampled in MAIN, so it is
                                                                    # seedable and CRN-shareable
        # ---- worker -> main ---------------------------------------------
        BlockSpec("obs", (n_envs, obs_dim), np.float32, "out"),
        BlockSpec("rew_terms", (n_envs, n_terms), np.float32, "out"),  # UNWEIGHTED
        BlockSpec("done", (n_envs,), np.uint8, "out"),
        BlockSpec("qpos", (n_envs, nq), np.float64, "out"),
        BlockSpec("qvel", (n_envs, nv), np.float64, "out"),
    ]


class SharedBuffers:
    """Owns (or attaches to) one shared_memory block per BlockSpec."""

    def __init__(self, specs: List[BlockSpec], names: Optional[Dict[str, str]] = None):
        self.specs = {s.name: s for s in specs}
        self._shm: Dict[str, shared_memory.SharedMemory] = {}
        self.arr: Dict[str, np.ndarray] = {}
        self.owner = names is None
        for s in specs:
            nbytes = int(np.prod(s.shape)) * np.dtype(s.dtype).itemsize
            if self.owner:
                shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 8))
            else:
                shm = shared_memory.SharedMemory(name=names[s.name])
            self._shm[s.name] = shm
            self.arr[s.name] = np.ndarray(s.shape, dtype=s.dtype, buffer=shm.buf)
            if self.owner:
                self.arr[s.name][...] = 0
        self.names = {k: v.name for k, v in self._shm.items()}

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arr[key]

    def close(self) -> None:
        for name, shm in self._shm.items():
            # Drop the numpy view first: SharedMemory.close() fails while a
            # memoryview onto its buffer is still alive.
            self.arr.pop(name, None)
            try:
                shm.close()
                if self.owner:
                    shm.unlink()
            except (FileNotFoundError, BufferError):
                pass
        self._shm.clear()


@dataclass(frozen=True)
class WorkerInit:
    """Everything a worker needs at spawn time. Pickled exactly once."""
    worker_id: int
    slot_lo: int
    slot_hi: int
    xml_paths: List[str]        # per slot in [slot_lo, slot_hi)
    leg_lens: List[float]       # per slot
    shm_names: Dict[str, str]
    specs: List[BlockSpec]
    cfg: object                 # BilevelConfig (a plain dataclass, picklable)
    cpu: Optional[int]
    protocol_version: int = PROTOCOL_VERSION
