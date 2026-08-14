"""Collect one on-policy batch: 256 windows x 24 steps.

Owns everything that has to happen in the main process:
  - turning the current p into a concrete reference window per body group
  - RSI noise sampling (here, not in the workers, so it is seedable and can be
    shared byte-for-byte between the members of an antithetic ES pair)
  - the CRN action-noise tensor (same reason -- see upper.py)
  - the batched policy/value forward on GPU
  - rotating the root-local wrench into world frame for xfrc_applied
  - masking everything after a termination out of the batch

The reference used for SIMULATION is detached. upper.py recomputes it with a
live graph when it needs dF/dp; that recomputation is one torch FK on
(256, 25, 76) and is far cheaper than holding a 24-step autograd graph across
a process boundary that has no gradient anyway.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from model.bilevel import rewards as R
from model.bilevel.policy import ACTION_DIM, CTRL_DIM, ROOT_FEAT_DIM
from model.bilevel.quat import quat_rot


def root_features(qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
    """(B, 76), (B, 75) -> (B, 8), all in the ROOT'S OWN FRAME.

    [root_z, up_z, v_local(3), w_local(3)]. Expressing them locally is what makes
    RootWrenchHead rotation-equivariant: the same lean produces the same
    corrective wrench regardless of which way the character happens to face.

    qvel[0:3] is a WORLD linear velocity but qvel[3:6] is already BODY-LOCAL
    angular velocity (MuJoCo free-joint convention), so only the former is
    rotated.
    """
    q = qpos[:, 3:7]
    w, x, y, z = q.unbind(-1)
    up_z = 2.0 * (y * z + w * x)          # see quat.py:quat_up_z
    q_inv = torch.cat([q[:, :1], -q[:, 1:]], dim=-1)
    v_local = quat_rot(q_inv, qvel[:, 0:3])
    return torch.cat(
        [qpos[:, 2:3], up_z.unsqueeze(-1), v_local, qvel[:, 3:6]], dim=-1
    )


def body_slices(assign: np.ndarray) -> List[Tuple[int, slice]]:
    """Contiguous assignment -> [(body_idx, slice), ...]. See data.body_assignment."""
    out, start = [], 0
    for i in range(1, len(assign) + 1):
        if i == len(assign) or assign[i] != assign[start]:
            out.append((int(assign[start]), slice(start, i)))
            start = i
    return out


class Collector:
    def __init__(self, cfg, ds, pool, policy, value_net, retargeters, device):
        self.cfg = cfg
        self.ds = ds
        self.pool = pool
        self.policy = policy
        self.value_net = value_net
        self.retargeters = retargeters          # body_idx -> Retargeter (share one RetargetNet)
        self.device = device
        self.assign = ds.body_assignment(cfg.n_envs)
        self.slices = body_slices(self.assign)
        self.H = cfg.horizon

        # Per-body wrench magnitude limits. m_max = f_frac * f_max * L_leg keeps
        # the torque limit dimensionally consistent with the force limit.
        f = np.array([cfg.wrench_f_frac * ds.bodies[b].mass * 9.81 for b in self.assign])
        m = np.array([cfg.wrench_m_frac * f[i] * ds.bodies[self.assign[i]].leg_len
                      for i in range(cfg.n_envs)])
        self.f_max_base = torch.as_tensor(f, dtype=torch.float32, device=device)
        self.m_max_base = torch.as_tensor(m, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------ reference

    def build_reference(self, batch: Dict[str, np.ndarray], u_per_env: torch.Tensor):
        """p -> the concrete (n_envs, H+1, 76) reference window, detached.

        `u_per_env` is (n_envs, 36), NOT one u per body. Antithetic ES needs the
        plus and minus members of a pair to run inside the SAME iteration on
        different env slots -- that is what makes it cost zero extra simulation
        (proposal.md 4.4). Slots of the same body normally share a u; the ES
        plan is what makes some of them differ.

        One Retargeter call per body group (10 groups of ~26), not per env.
        """
        n, H1 = self.cfg.n_envs, self.H + 1
        ref = np.empty((n, H1, 76), dtype=np.float64)
        src = torch.as_tensor(batch["src_qpos"], dtype=torch.float64, device=self.device)
        beta = torch.as_tensor(batch["beta"], dtype=torch.float64, device=self.device)
        u = u_per_env.to(torch.float64)
        with torch.no_grad():
            for b, sl in self.slices:
                _, r, _ = self.retargeters[b](src[sl], beta[sl], u_override=u[sl], n_out=H1)
                ref[sl] = r.cpu().numpy()
        return ref

    def reference_qvel(self, ref: np.ndarray) -> np.ndarray:
        """Initial qvel per window, in MuJoCo's own qvel coordinates."""
        from model.bilevel.data import ref_qvel_from_qpos

        out = np.empty((self.cfg.n_envs, 75), dtype=np.float64)
        for b, sl in self.slices:
            m = self.ds.bodies[b].model
            for e in range(sl.start, sl.stop):
                out[e] = ref_qvel_from_qpos(m, ref[e, 0], ref[e, 1], self.cfg.dt)
        return out

    # ------------------------------------------------------------------ collection

    def collect(
        self, it: int, batch: Dict[str, np.ndarray], u_per_env: torch.Tensor,
        rsi_noise: np.ndarray, action_noise: torch.Tensor, deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run one 24-step window on all 256 envs.

        `rsi_noise` and `action_noise` are supplied by the caller rather than
        drawn here: an antithetic ES pair MUST share them bit-for-bit, or the
        ~10% finite-difference signal is buried under rollout noise (proposal.md
        R9). train_bilevel.py draws them once per iteration.
        """
        cfg, H, dev = self.cfg, self.H, self.device
        n = cfg.n_envs

        ref = self.build_reference(batch, u_per_env)
        self.pool.reset_windows(ref, self.reference_qvel(ref), rsi_noise)

        beta = torch.as_tensor(batch["beta"], dtype=torch.float32, device=dev)
        z0 = torch.as_tensor(batch["z0"], dtype=torch.float32, device=dev)
        # z_beta is fixed for the window (beta and z0 are), so it is computed
        # once here. It is NOT frozen for the update: ppo.py recomputes it from
        # (beta, z0) inside every minibatch forward, because the adapter is part
        # of the policy and the PPO ratio has to account for its change.
        with torch.no_grad():
            z_beta = self.policy.latent(beta, z0)

        wr = cfg.wrench_scale(it)
        f_max, m_max = self.f_max_base * wr, self.m_max_base * wr

        obs_b = torch.empty(H, n, self.policy.obs_dim, device=dev)
        act_b = torch.empty(H, n, ACTION_DIM, device=dev)
        logp_b = torch.empty(H, n, device=dev)
        val_b = torch.empty(H + 1, n, device=dev)
        prior_b = torch.empty(H, n, CTRL_DIM, device=dev)
        rootf_b = torch.empty(H, n, ROOT_FEAT_DIM, device=dev)
        terms_b = torch.empty(H, n, R.N_TERMS, device=dev)
        done_b = torch.zeros(H, n, device=dev)
        valid_b = torch.zeros(H, n, device=dev)
        qpos_b = torch.empty(H, n, 76, device=dev)
        # (H+1, n, 76): the RSI state followed by the state after each control
        # step. amp.py needs consecutive PAIRS, so the pre-first-step state has
        # to be kept -- qpos_b alone is missing t=0's left-hand side.
        qpos_all = torch.empty(H + 1, n, 76, device=dev)

        alive = torch.ones(n, device=dev)

        def read_state():
            qp = torch.as_tensor(self.pool.qpos.copy(), dtype=torch.float32, device=dev)
            qv = torch.as_tensor(self.pool.qvel.copy(), dtype=torch.float32, device=dev)
            ob = torch.as_tensor(self.pool.obs.copy(), dtype=torch.float32, device=dev)
            return qp, qv, ob

        qp, qv, ob = read_state()
        qpos_all[0] = qp

        for t in range(H):
            rf = root_features(qp, qv)
            with torch.no_grad():
                mean, raw_prior = self.policy.act_mean(ob, beta, z_beta, rf)
                dist = self.policy.dist(mean)
                if deterministic:
                    action = mean
                else:
                    action = mean + self.policy.std * action_noise[:, t]
                logp = dist.log_prob(action).sum(-1)
                phase = torch.stack(
                    [torch.full((n,), t / H, device=dev), torch.full((n,), (H - t) / H, device=dev)],
                    dim=-1,
                )
                val_b[t] = self.value_net(ob, z_beta, beta, phase)

            obs_b[t], act_b[t], logp_b[t] = ob, action, logp
            prior_b[t], rootf_b[t], valid_b[t] = raw_prior, rf, alive

            ctrl, force, torque = self.policy.split_action(action, f_max, m_max)
            # Root-local -> world, using the CURRENT root orientation.
            rq = qp[:, 3:7]
            xfrc = torch.cat([quat_rot(rq, force), quat_rot(rq, torque)], dim=-1)

            self.pool.step(
                ctrl.cpu().numpy().astype(np.float32),
                xfrc.cpu().numpy().astype(np.float32),
                raw_prior.cpu().numpy().astype(np.float32),
                t_ref=t + 1,
            )

            terms_b[t] = torch.as_tensor(self.pool.rew_terms.copy(), dtype=torch.float32, device=dev)
            step_done = torch.as_tensor(self.pool.done.copy(), dtype=torch.float32, device=dev)
            # Only count a termination on the step it actually happened; a slot
            # that was already frozen contributes nothing at all.
            done_b[t] = step_done * alive
            qp, qv, ob = read_state()
            qpos_b[t] = qp
            qpos_all[t + 1] = qp
            alive = alive * (1.0 - done_b[t])

        with torch.no_grad():
            phase = torch.stack(
                [torch.ones(n, device=dev), torch.zeros(n, device=dev)], dim=-1
            )
            val_b[H] = self.value_net(ob, z_beta, beta, phase)

        return {
            "obs": obs_b, "action": act_b, "logp": logp_b, "value": val_b,
            "raw_prior": prior_b, "root_feats": rootf_b, "terms": terms_b,
            "done": done_b, "valid": valid_b, "qpos": qpos_b, "qpos_all": qpos_all,
            "beta": beta, "z0": z0, "z_beta": z_beta,
            "ref": torch.as_tensor(ref, dtype=torch.float64, device=dev),
            "src_qpos": torch.as_tensor(batch["src_qpos"], dtype=torch.float64, device=dev),
            "pair_id": torch.as_tensor(batch["pair_id"], device=dev),
            "wrench_scale": wr,
        }


def sample_rsi_noise(cfg, rng: np.random.Generator) -> np.ndarray:
    """Per-window sigma ~ U(0, rsi_sigma_max), then N(0, sigma) on the 69 hinges.

    Resampling sigma per window (rather than fixing it) shows the value function
    a spread of initial tracking errors instead of one operating point, and acts
    as an implicit robustness curriculum. The root is never touched -- that is
    handled in the worker, which only adds this to qpos[7:].
    """
    sigma = rng.uniform(0.0, cfg.rsi_sigma_max, size=(cfg.n_envs, 1))
    return (rng.standard_normal((cfg.n_envs, CTRL_DIM)) * sigma).astype(np.float32)


def sample_action_noise(cfg, gen: torch.Generator, device) -> torch.Tensor:
    """(n_envs, H, 75) standard normal, drawn once per iteration.

    Held as an explicit tensor so an antithetic ES pair can reuse the identical
    stream (common random numbers).
    """
    return torch.randn(cfg.n_envs, cfg.horizon, ACTION_DIM, generator=gen, device=device)
