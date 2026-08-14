"""Lower level: PPO with GAE, plus two analytic side-channels.

Why PPO and not the existing REINFORCE
--------------------------------------
model/train.py runs episode-level REINFORCE: one scalar of learning signal for
300 x 69 = 20700 sampled dimensions, with an EMA scalar baseline. Shortening
the episode to a 24-step window makes that WORSE, not better -- the cost's scale
shrinks while the score function's magnitude does not. And a scalar baseline
cannot do within-window credit assignment at all.

Per-step rewards + GAE + a beta-conditioned value function fix both. PPO's clip
is close to free insurance on top: this policy is a residual on a strong frozen
prior, and a single bad update destroys the prior.

Two things are NOT estimated by the policy gradient, because they do not have to
go through the simulator:

  lambda_z anchor -- ||z_beta - z0||, kept from model/train.py:305 but in COSINE
      form. z_beta is now projected onto the sphere of radius sqrt(z_dim), so a
      Euclidean penalty would partly be penalizing a radius that is fixed by
      construction; 1 - cos is the correct distance on that manifold. Evaluated
      over the UNIQUE (clip, body) pairs in the minibatch, not per transition,
      so it is not implicitly re-weighted by how many windows each pair got.

  behaviour cloning -- ||mu_ctrl - a_ref||. Because ctrl in [-1,1] maps to
      qpos in jnt_range exactly (4.4e-16, measured), the ctrl that commands the
      reference's next joint angles is available in closed form. A dense,
      exact, zero-variance gradient into ActionHead, far better conditioned than
      anything PPO produces. Annealed to zero by iteration 2000 -- held on it
      fights the frozen prior and ignores dynamics (the position target that
      SETTLES at q_hat is not the one that DRIVES you there under load).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model.bilevel import rewards as R
from model.bilevel.policy import CTRL_DIM


class PairAdvantageNormalizer:
    """Per-(clip, body) EMA of ADVANTAGE mean/std.

    model/diagnose_single_task.py exists because task composition dominated the
    old signal: some (clip, body) pairs are simply much harder than others, and
    a global baseline reads that heterogeneity as advantage. Standardizing
    within a pair removes it. proposal.md R7.

    The statistics must be of the same quantity that gets normalized. An earlier
    version fed `update()` the EPISODE RETURN and then subtracted that from the
    PER-STEP advantage, which is wrong twice over and was measured to break the
    lower level outright:

      - scale: ep_ret ~ 5.5 (24 steps x ~0.55) against a per-step GAE advantage
        of ~0.1, so the subtraction was a large constant offset, not a baseline;
      - variance: on a pair's first sighting `fresh` sets sq = ret*ret exactly,
        so var = 0, std clamps to 1e-3, and the division multiplies by ~1000.

    The result was an "advantage" that encoded which (clip, body) pair a
    transition came from rather than whether its action was good.

    Note on what this did NOT cause: the global re-standardization on the next
    line put the advantage back on unit variance either way, so ||grad pg|| was
    measured at 8-23 both before and after this fix. The pg gradient is large
    for an unrelated and legitimate reason -- d logp / d mu = (a - mu) / sigma^2
    with sigma = 0.2 summed over 75 action dimensions -- and the fix for that is
    grad_clip_norm and lambda_bc, not this class. What this fix buys is that the
    advantage now carries action-quality information instead of pair identity.
    """

    def __init__(self, n_pairs: int, momentum: float = 0.99, device="cpu"):
        self.m = momentum
        self.mean = torch.zeros(n_pairs, device=device)
        self.sq = torch.ones(n_pairs, device=device)
        self.seen = torch.zeros(n_pairs, device=device, dtype=torch.bool)

    @torch.no_grad()
    def update(self, pair_id: torch.Tensor, adv: torch.Tensor, valid: torch.Tensor) -> None:
        """adv, valid: (H, N). pair_id: (N,)."""
        w = valid.sum(0).clamp(min=1.0)
        m1 = (adv * valid).sum(0) / w
        m2 = (adv * adv * valid).sum(0) / w
        fresh = ~self.seen[pair_id]
        self.mean[pair_id] = torch.where(
            fresh, m1, self.m * self.mean[pair_id] + (1 - self.m) * m1
        )
        self.sq[pair_id] = torch.where(
            fresh, m2, self.m * self.sq[pair_id] + (1 - self.m) * m2
        )
        self.seen[pair_id] = True

    @torch.no_grad()
    def normalize(self, pair_id: torch.Tensor, adv: torch.Tensor,
                  std_floor: torch.Tensor) -> torch.Tensor:
        """`std_floor` keeps a pair whose advantage happens to be near-constant
        this iteration from being amplified by 1/eps -- a pair the policy has
        already solved should contribute a small advantage, not a huge one."""
        mu = self.mean[pair_id]
        std = (self.sq[pair_id] - mu * mu).clamp(min=0.0).sqrt().clamp(min=std_floor)
        return (adv - mu.unsqueeze(0)) / std.unsqueeze(0)


class RunningScalar:
    """Running mean/std for value-target normalization (PopArt-lite).

    With H=24 roughly half the return at t=0 comes from the bootstrap, so a
    mis-scaled V early in training makes GAE meaningless.
    """

    def __init__(self, device="cpu"):
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = 1e-4

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        bm, bv, bc = x.mean(), x.var(unbiased=False), x.numel()
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self) -> torch.Tensor:
        return self.var.clamp(min=1e-6).sqrt()


def compute_gae(
    reward: torch.Tensor, value: torch.Tensor, done: torch.Tensor, valid: torch.Tensor,
    gamma: float, lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """reward/done/valid (H, N), value (H+1, N) -> (advantage, return), each (H, N).

    The window end at t=H is a TRUNCATION, not a termination: it bootstraps with
    V(s_H). A fall or a tracking failure IS a termination and bootstraps with 0.
    Conflating the two would teach the policy that surviving to the end of an
    arbitrary 0.8 s slice is worthless.
    """
    H = reward.shape[0]
    adv = torch.zeros_like(reward)
    last = torch.zeros_like(reward[0])
    for t in reversed(range(H)):
        nonterminal = 1.0 - done[t]
        delta = reward[t] + gamma * value[t + 1] * nonterminal - value[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv * valid, (adv + value[:H]) * valid


def reference_ctrl(kin, ref: torch.Tensor) -> torch.Tensor:
    """The exact ctrl commanding the reference's NEXT joint angles. (N, H, 69).

    a_ref[t] = clip(2*(q_hat[t+1] - lo)/(hi - lo) - 1, -1, 1)

    Exact, not approximate -- see torch_kin.normalized_ctrl. This is the BC
    target and costs one elementwise op.
    """
    return kin.normalized_ctrl(ref[:, 1:, 7:]).clamp(-1.0, 1.0)


def update_lower(
    cfg, it: int, policy, value_net, optimizer, ep: Dict[str, torch.Tensor],
    pair_norm: PairAdvantageNormalizer, ret_norm: RunningScalar,
    bc_targets: Optional[torch.Tensor] = None,
    amp_reward: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """One PPO update on the collected window batch. Returns scalar metrics."""
    dev = ep["obs"].device
    H, N = ep["terms"].shape[0], ep["terms"].shape[1]
    gamma, lam = cfg.gamma_at(it), cfg.gae_lambda

    terms_np = ep["terms"].detach().cpu().numpy()
    w_amp = cfg.amp_scale(it)
    amp_np = amp_reward.detach().cpu().numpy() if amp_reward is not None else None
    reward = torch.as_tensor(R.combine(terms_np, cfg, r_amp=amp_np, w_amp=w_amp),
                             dtype=torch.float32, device=dev)
    valid, done = ep["valid"], ep["done"]
    reward = reward * valid

    # ---- GAE on normalized value ----------------------------------------
    v_raw = ep["value"] * ret_norm.std + ret_norm.mean
    adv, ret = compute_gae(reward, v_raw, done, valid, gamma, lam)

    pair_id = ep["pair_id"]
    with torch.no_grad():
        m = valid > 0
        # Floor the per-pair std at a fraction of the batch-wide one, so the
        # per-pair division can only ever rescale by ~10x, never by 1/eps.
        g_std = adv[m].std() if m.any() else torch.ones((), device=dev)
        pair_norm.update(pair_id, adv, valid)
        adv = pair_norm.normalize(pair_id, adv, (cfg.adv_std_floor * g_std).clamp(min=1e-6))
        if m.any():
            adv = torch.where(m, (adv - adv[m].mean()) / (adv[m].std() + 1e-8), adv)
        ret_norm.update(ret[m] if m.any() else ret)
        ret_n = (ret - ret_norm.mean) / ret_norm.std

    # ---- flatten (H, N) -> (H*N,) ---------------------------------------
    def flat(x):
        return x.reshape(H * N, *x.shape[2:])

    b_obs, b_act, b_logp = flat(ep["obs"]), flat(ep["action"]), flat(ep["logp"])
    b_prior, b_rootf = flat(ep["raw_prior"]), flat(ep["root_feats"])
    b_adv, b_ret, b_valid = flat(adv), flat(ret_n), flat(valid)
    b_vold = flat(ep["value"][:H])
    b_beta = ep["beta"].unsqueeze(0).expand(H, N, -1).reshape(H * N, -1)
    b_z0 = ep["z0"].unsqueeze(0).expand(H, N, -1).reshape(H * N, -1)
    b_phase = torch.stack([
        (torch.arange(H, device=dev) / H).unsqueeze(1).expand(H, N).reshape(-1),
        ((H - torch.arange(H, device=dev)) / H).unsqueeze(1).expand(H, N).reshape(-1),
    ], dim=-1)
    b_bc = flat(bc_targets.transpose(0, 1)) if bc_targets is not None else None

    lam_bc = cfg.bc_scale(it)
    for g in optimizer.param_groups:
        g["lr"] = cfg.lr_lower_at(it)

    n_valid = float(b_valid.sum().clamp(min=1.0))
    idx_all = torch.randperm(H * N, device=dev)
    mb = (H * N) // cfg.ppo_minibatches
    stats = {"pg": 0.0, "v": 0.0, "z": 0.0, "bc": 0.0, "kl": 0.0, "clipfrac": 0.0}
    n_updates = 0

    for _ in range(cfg.ppo_epochs):
        idx_all = idx_all[torch.randperm(H * N, device=dev)]
        for k in range(cfg.ppo_minibatches):
            idx = idx_all[k * mb:(k + 1) * mb]
            w = b_valid[idx]
            if float(w.sum()) < 1.0:
                continue

            # z_beta is RECOMPUTED here, not reused from collection: the adapter
            # is part of the policy, so the PPO ratio must reflect its update.
            z_beta = policy.latent(b_beta[idx], b_z0[idx])
            mean, raw = policy.act_mean(b_obs[idx], b_beta[idx], z_beta, b_rootf[idx])
            dist = policy.dist(mean)
            logp = dist.log_prob(b_act[idx]).sum(-1)

            ratio = (logp - b_logp[idx]).exp()
            a = b_adv[idx]
            pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * a)
            pg_loss = (pg * w).sum() / w.sum()

            v = value_net(b_obs[idx], z_beta.detach(), b_beta[idx], b_phase[idx])
            v_clipped = b_vold[idx] + (v - b_vold[idx]).clamp(-cfg.value_clip, cfg.value_clip)
            v_loss = torch.max((v - b_ret[idx]) ** 2, (v_clipped - b_ret[idx]) ** 2)
            v_loss = 0.5 * (v_loss * w).sum() / w.sum()

            # cosine anchor on the FB sphere, over unique pairs only
            z0n = F.normalize(b_z0[idx], dim=-1)
            z_loss = (1.0 - (F.normalize(z_beta, dim=-1) * z0n).sum(-1))
            z_loss = (z_loss * w).sum() / w.sum()

            bc_loss = torch.zeros((), device=dev)
            if b_bc is not None and lam_bc > 0:
                bc_loss = (((mean[:, :CTRL_DIM] - b_bc[idx]) ** 2).mean(-1) * w).sum() / w.sum()

            loss = (pg_loss + cfg.value_coef * v_loss
                    + cfg.lambda_z * z_loss + lam_bc * bc_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.trainable_parameters() + list(value_net.parameters()), cfg.grad_clip_norm
            )
            optimizer.step()

            with torch.no_grad():
                stats["pg"] += float(pg_loss)
                stats["v"] += float(v_loss)
                stats["z"] += float(z_loss)
                stats["bc"] += float(bc_loss)
                stats["kl"] += float((((b_logp[idx] - logp) * w).sum() / w.sum()))
                stats["clipfrac"] += float(
                    (((ratio - 1).abs() > cfg.ppo_clip).float() * w).sum() / w.sum()
                )
            n_updates += 1

    for k in stats:
        stats[k] /= max(1, n_updates)

    i = R.TERM_IDX
    tw = ep["terms"] * valid.unsqueeze(-1)
    denom = valid.sum().clamp(min=1.0)
    stats.update({
        "reward": float((reward * valid).sum() / denom),
        "r_track": float(sum(
            getattr(cfg, f"w_{k}") * tw[..., i[f"r_{k}"]].sum() / denom
            for k in ("pose", "vel", "ee", "root", "com")
        )),
        "r_pose": float(tw[..., i["r_pose"]].sum() / denom),
        "r_ee": float(tw[..., i["r_ee"]].sum() / denom),
        "e_ext": float(tw[..., i["e_ext"]].sum() / denom),
        "pose_err": float(tw[..., i["pose_err"]].sum() / denom),
        "term_rate": float(done.sum() / N),
        "mean_steps": float(valid.sum() / N),
        "lambda_bc": lam_bc,
        "w_amp": w_amp,
        "wrench_scale": ep["wrench_scale"],
    })
    return stats
