"""Upper level: F(phi), the truncated hypergradient, and the collapse diagnostics.

The hypergradient decomposes as

    dF/dphi = dF/dref . dref/dphi                   [T1] KEPT, exactly
            + dF/dtau . dtau/dref . dref/dphi       [T2] estimated by antithetic ES
            + dF/dtau . dtau/dpsi . dpsi*/dphi      [T3] DROPPED

T1 is plain autograd through RetargetNet and apply_retarget, and is exact and
zero-variance for S, C and the proximal term (they depend on nothing else) and
exact-given-tau for G. It carries the bulk of the signal, and it is the entire
reason the retargeting was reimplemented in torch.

    Note carefully: T1 of lambda_ext*E_ext and lambda_phys*P is IDENTICALLY
    ZERO, because those depend on phi only through the simulator. So the exact
    path alone cannot see the lower level cheating with the external wrench.
    That hole is exactly what ES fills.

T2 is not negligible. dtau/dref includes the RSI channel, where the initial
state literally IS ref_phi[t0] -- an upper level that ignores T2 does not know
that moving the reference also moves where the robot starts every window. And
dtau/dref is precisely what MuJoCo cannot give us.

T3 is dropped because dpsi*/dphi = -(H_psipsi J)^-1 H_psiphi J needs a
Hessian-inverse-vector product of an RL objective over a contact-discontinuous
MDP, where even the first-order term is only available as a score-function
estimate; the second-order variance scales as 1/sigma^4 and is unusable at 6144
samples per iteration. Without MJX there is no differentiable-simulator
alternative. This is the standard first-order / truncated-hypergradient
approximation (first-order MAML, first-order DARTS, one-step-unrolled bilevel),
and it is known to behave when the lower level's response varies smoothly in
the upper variable -- which holds here BY CONSTRUCTION, since the hard box
bounds how far the reward's target can move.

    HONEST CAVEAT: with T3 dropped we are not solving the bilevel problem. We
    are running Gauss-Seidel alternation on min_phi F(phi, psi_k) with
    psi_k <- RL(phi_k), whose stationary points differ. Concretely: phi has no
    incentive to choose a reference that is hard now but produces a better
    policy later. For this application that is a feature -- we want a faithful,
    feasible reference, not a curriculum -- and faithful+feasible is exactly
    what S and C encode. Do not claim convergence guarantees in this file.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from model.bilevel import rewards as R
from model.bilevel.retarget import U_DIM, U_ROOT_DZ, RetargetParams
from model.bilevel.rollout import body_slices
from model.bilevel.semantics import (
    UpperGeometry, frac_illegal_frames, gap, reference_feasibility, semantic_fidelity,
    weighted_sum,
)


# ---------------------------------------------------------------- ES plan

@dataclass
class EsGroup:
    body: int
    plus: np.ndarray       # slot indices carrying u + sigma*eps
    minus: np.ndarray      # slot indices carrying u - sigma*eps (CRN-matched to `plus`)


@dataclass
class EsPlan:
    """Which env slots carry which perturbation, fixed for the whole run.

    Each body owns a contiguous slot range (data.body_assignment), which is
    split into 2*es_pairs_per_body equal groups; pair p uses group 2p for +eps
    and group 2p+1 for -eps. Whatever is left over is `control` -- unperturbed
    slots used for clean logging and for the T1 evaluation.
    """
    groups: List[EsGroup] = field(default_factory=list)
    control: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    @classmethod
    def build(cls, cfg, assign: np.ndarray) -> "EsPlan":
        plan = cls()
        ctrl: List[int] = []
        for b, sl in body_slices(assign):
            slots = np.arange(sl.start, sl.stop)
            n_groups = 2 * cfg.es_pairs_per_body
            per = min(cfg.es_windows_per_group, len(slots) // n_groups)
            if per < 1:
                ctrl.extend(slots.tolist())
                continue
            used = per * n_groups
            for p in range(cfg.es_pairs_per_body):
                lo = p * 2 * per
                plan.groups.append(EsGroup(
                    body=b, plus=slots[lo:lo + per], minus=slots[lo + per:lo + 2 * per]
                ))
            ctrl.extend(slots[used:].tolist())
        plan.control = np.array(sorted(ctrl), dtype=np.int64)
        return plan

    def apply_crn(self, batch: Dict[str, np.ndarray], rsi_noise: np.ndarray,
                  action_noise: torch.Tensor) -> None:
        """Force each pair's minus slots to replay the plus slots' randomness.

        COMMON RANDOM NUMBERS ARE MANDATORY (proposal.md R9). The ES signal is a
        ~10% relative change in G; without CRN it is buried under rollout noise
        and the estimator is worse than useless. Everything stochastic that
        differs between the two members must be made identical: the clip, the
        window start, the RSI noise, and the entire (24, 75) action-noise stream.
        Only +-sigma*eps may differ.
        """
        for g in self.groups:
            for key in ("src_qpos", "z0", "pair_id", "clip_idx", "t0"):
                batch[key][g.minus] = batch[key][g.plus]
            rsi_noise[g.minus] = rsi_noise[g.plus]
            action_noise[g.minus] = action_noise[g.plus]

    def build_u(self, u_by_body: torch.Tensor, assign: np.ndarray, sigma: float,
                eps: Dict[int, torch.Tensor], n_envs: int) -> torch.Tensor:
        """(n_bodies, 36) -> (n_envs, 36) with the perturbations written in."""
        u = u_by_body[torch.as_tensor(assign, device=u_by_body.device)].clone()
        for gi, g in enumerate(self.groups):
            e = eps[gi]
            u[g.plus] = u_by_body[g.body] + sigma * e
            u[g.minus] = u_by_body[g.body] - sigma * e
        return u


def _rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Map values to evenly spaced ranks in [-0.5, 0.5].

    Standard ES practice: makes the estimator invariant to F_sim's drifting
    scale, which matters a lot here because G's magnitude changes by an order of
    magnitude over training.
    """
    n = x.numel()
    if n < 2:
        return torch.zeros_like(x)
    order = x.argsort()
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(n, dtype=x.dtype, device=x.device)
    return ranks / (n - 1) - 0.5


class UpperLevel:
    def __init__(self, cfg, ds, retargeters, retarget_net, device):
        self.cfg = cfg
        self.ds = ds
        self.retargeters = retargeters
        self.net = retarget_net
        self.device = device
        self.assign = ds.body_assignment(cfg.n_envs)
        self.slices = body_slices(self.assign)
        self.plan = EsPlan.build(cfg, self.assign) if cfg.es_enabled else None
        self.optimizer = torch.optim.Adam(retarget_net.parameters(), lr=cfg.lr_upper)
        self.u_prev: Optional[torch.Tensor] = None
        self._grad_accum: Optional[torch.Tensor] = None
        self._accum_n = 0
        # Per-term units, filled by calibrate(). Until then the raw magnitudes
        # are used, which span six orders of magnitude and would make the
        # configured weights mean nothing (see semantics.weighted_sum).
        self.scales: Optional[Dict[str, float]] = None
        self.baseline: Dict[str, float] = {}   # each term at phi = 0, for the Stage 2 gate

        # Source-body geometry constants, needed by S.
        self.src = ds.source
        self.src_ee_len = torch.as_tensor(
            self.src.ee_limb_lengths, dtype=torch.float64, device=device
        )
        self.src_foot_rest = torch.as_tensor(
            self.src.foot_rest_heights, dtype=torch.float64, device=device
        )

    # ------------------------------------------------------------------ u

    def u_by_body(self) -> torch.Tensor:
        """(n_bodies, 36) with a live graph back to RetargetNet.

        The upper level runs in float64 throughout: T1 is the only exact
        gradient in the system and the terms it sums (a 1e-3 rad joint offset
        against a 1e2 (rad/s^2) smoothness term) span enough orders of magnitude
        that float32 accumulation would lose real signal.
        """
        dtype = next(self.net.parameters()).dtype
        beta = torch.as_tensor(
            np.stack([b.beta for b in self.ds.bodies]), dtype=dtype, device=self.device
        )
        return self.net(beta)

    def u_per_env(self, u_bb: torch.Tensor, sigma_eps: Optional[Dict[int, torch.Tensor]] = None,
                  sigma: float = 0.0) -> torch.Tensor:
        if self.plan is None or sigma_eps is None:
            return u_bb[torch.as_tensor(self.assign, device=u_bb.device)]
        return self.plan.build_u(u_bb, self.assign, sigma, sigma_eps, self.cfg.n_envs)

    def draw_eps(self, gen: torch.Generator) -> Dict[int, torch.Tensor]:
        if self.plan is None:
            return {}
        return {
            gi: torch.randn(U_DIM, generator=gen, device=self.device)
            for gi in range(len(self.plan.groups))
        }

    # ------------------------------------------------------------------ warm start

    @torch.no_grad()
    def warm_start_ground(self, rng, n_windows: Optional[int] = None,
                          verbose: bool = True) -> float:
        """Install the ground offset dz_root analytically instead of learning it.

        The phi=0 reference sits INSIDE the floor on ~88% of frames (measured:
        child median -0.0120 m, teen -0.0155 m, worst -0.204 m), because
        replacing scripts/qpos_retarget.py:127 ground_correct_qpos's per-frame
        lift with a learned dz_root left nothing to do the lifting at
        initialization. RSI then starts every window with the body in the
        ground, physics ejects it, and it falls.

        Learning the offset is the wrong tool: it is exactly computable (it is
        the reference's own median penetration), while lr_upper = 1e-5 -- sized
        for the delicate anti-degeneracy work of stages 2 and 3 -- moved it
        0.00005 m of the required 0.014 m in 50 iterations.

        So it is measured here and written into RetargetNet's output bias;
        gradient descent only refines it from there. One shared bias across
        bodies (the per-body spread is 12-16 mm) which the net's weights then
        specialize per beta.
        """
        cfg = self.cfg
        n = n_windows or min(256, cfg.n_envs)
        H1 = cfg.horizon + 1
        needed = []
        for b, spec in enumerate(self.ds.bodies):
            ci, _, t0 = self.ds.sample(n, rng, body_idx=np.full(n, b))
            batch = self.ds.build_batch(ci, np.full(n, b), t0)
            src = torch.as_tensor(batch["src_qpos"], dtype=torch.float64, device=self.device)
            beta = torch.as_tensor(batch["beta"], dtype=torch.float64, device=self.device)
            u0 = torch.zeros(n, U_DIM, dtype=torch.float64, device=self.device)
            _, ref, _ = self.retargeters[b](src, beta, u_override=u0, n_out=H1)
            xp, xq = spec.kin(ref)
            minz = spec.kin.min_geom_z(xp, xq)
            # Lift so the median frame clears the same tolerance the feasibility
            # penalty uses, rather than chasing the deepest outlier.
            needed.append(float(cfg.ground_tol - minz.median()))

        lift = float(np.mean(needed))
        u0 = self.net.warm_start(U_ROOT_DZ, lift, cfg.box_root_dz)
        if verbose:
            per = ", ".join(f"{s.name} {v * 1000:+.1f}" for s, v in zip(self.ds.bodies, needed))
            print(f"ground warm start: reference needs lifting by {lift * 1000:+.1f} mm "
                  f"(per body, mm: {per})")
            print(f"    -> dz_root bias u = {u0:+.4f}  (box +-{cfg.box_root_dz} m)")
        return lift

    # ------------------------------------------------------------------ calibration

    @torch.no_grad()
    def calibrate(self, rng, n_windows: Optional[int] = None, n_corners: int = 4,
                  verbose: bool = True) -> Dict[str, float]:
        """Set each S/C term's unit to the magnitude it reaches at the BOX CORNER.

        Not to its phi=0 value. Several terms are structurally zero at phi=0 --
        measured: s_heading = 2e-32 (the root quaternion is copied verbatim, so
        the turning rate is identical by construction) and c_float = 0 exactly.
        Dividing by those gives weights of order 1e11 and the objective is
        destroyed by whichever degenerate term moves first.

        The reachable range is the meaningful scale instead. phi's outputs are
        bounded by the hard tanh box (retarget.py), so evaluating the terms at
        random box corners gives, for each term, the largest value phi can
        produce. Every normalized term then lands in roughly [0, 1] over phi's
        ENTIRE reachable set, which is exactly the condition under which the
        weights in BilevelConfig are comparable priorities.

        Also records the phi=0 values as `self.baseline` -- proposal.md 8.2
        Stage 0 item 6 wants those on record as the Stage 2 target line.

        Costs one torch FK per body per corner and no simulation at all.
        """
        cfg = self.cfg
        n = n_windows or cfg.n_envs
        H1 = cfg.horizon + 1
        at_zero: Dict[str, List[float]] = {}
        at_corner: Dict[str, List[float]] = {}

        gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        for b, spec in enumerate(self.ds.bodies):
            ci, _, t0 = self.ds.sample(n, rng, body_idx=np.full(n, b))
            batch = self.ds.build_batch(ci, np.full(n, b), t0)
            src = torch.as_tensor(batch["src_qpos"], dtype=torch.float64, device=self.device)
            beta = torch.as_tensor(batch["beta"], dtype=torch.float64, device=self.device)

            u0 = torch.zeros(n, U_DIM, dtype=torch.float64, device=self.device)
            rr, r, _ = self.retargeters[b](src, beta, u_override=u0, n_out=H1)
            for k, v in self._phi_terms(b, spec, rr, r, src, H1).items():
                at_zero.setdefault(k, []).append(float(v))

            for _ in range(n_corners):
                # +-6 pre-tanh saturates tanh to ~+-1, i.e. a corner of the box.
                sign = torch.randint(0, 2, (U_DIM,), generator=gen, device=self.device,
                                     dtype=torch.float64) * 2 - 1
                u = (6.0 * sign).expand(n, -1)
                rr, r, _ = self.retargeters[b](src, beta, u_override=u, n_out=H1)
                for k, v in self._phi_terms(b, spec, rr, r, src, H1).items():
                    at_corner.setdefault(k, []).append(float(v))

        self.baseline = {k: float(np.mean(v)) for k, v in at_zero.items()}
        # Floor at the phi=0 value: a term can only be normalized down to the
        # range it actually spans, never below where it already sits.
        self.scales = {
            k: max(float(np.mean(v)), self.baseline.get(k, 0.0), 1e-9)
            for k, v in at_corner.items()
        }
        if verbose:
            print("upper-level term calibration  (unit = value at the hard-box corner):")
            for k in sorted(self.scales):
                b0 = self.baseline.get(k, 0.0)
                print(f"    {k:<14} phi=0 {b0:>11.4g}   corner {self.scales[k]:>11.4g}"
                      f"   -> normalized at phi=0: {b0 / self.scales[k]:.3f}")
        return self.scales

    def _phi_terms(self, b: int, spec, ref_raw, ref, src, H1) -> Dict[str, torch.Tensor]:
        """The S and C raw terms for one body group."""
        dev = self.device
        foot_rest = torch.as_tensor(spec.foot_rest_heights, dtype=torch.float64, device=dev)
        ref_geo = UpperGeometry(spec.kin, ref, foot_rest)
        src_geo = UpperGeometry(self.src.kin, src[:, :H1], self.src_foot_rest)
        ee_len = torch.as_tensor(spec.ee_limb_lengths, dtype=torch.float64, device=dev)
        out = semantic_fidelity(self.cfg, ref_geo, src_geo, ee_len, self.src_ee_len,
                                spec.leg_len, self.src.leg_len)
        out.update(reference_feasibility(self.cfg, spec.kin, ref_raw, ref_geo, src_geo,
                                         leg_len=spec.leg_len))
        return out

    # ------------------------------------------------------------------ F(phi)

    def evaluate(self, ep: Dict[str, torch.Tensor], u_bb: torch.Tensor,
                 with_grad: bool = True) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        """F(phi) on the control slots, with the exact T1 graph attached.

        Returns (F, metrics, per_env_F_sim). The reference is RECOMPUTED here
        with a live graph -- rollout.py only ever built a detached one, because
        nothing downstream of env.step() carries a gradient anyway.
        """
        cfg = self.cfg
        dev = self.device
        total = torch.zeros((), dtype=torch.float64, device=dev)
        f_sim_env = torch.zeros(cfg.n_envs, dtype=torch.float64, device=dev)
        acc: Dict[str, float] = {}
        n_groups = 0

        src_all = ep["src_qpos"]
        beta_all = ep["beta"].to(torch.float64)
        H1 = cfg.horizon + 1

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            for b, sl in self.slices:
                spec = self.ds.bodies[b]
                rt = self.retargeters[b]
                kin = spec.kin
                n_b = sl.stop - sl.start

                ref_raw, ref, _ = rt(src_all[sl], beta_all[sl],
                                     u_override=u_bb[b].to(torch.float64).expand(n_b, -1),
                                     n_out=H1)

                foot_rest = torch.as_tensor(spec.foot_rest_heights, dtype=torch.float64, device=dev)
                ref_geo = UpperGeometry(kin, ref, foot_rest)
                src_geo = UpperGeometry(self.src.kin, src_all[sl][:, :H1], self.src_foot_rest)

                ee_len = torch.as_tensor(spec.ee_limb_lengths, dtype=torch.float64, device=dev)
                s_terms = semantic_fidelity(
                    cfg, ref_geo, src_geo, ee_len, self.src_ee_len,
                    spec.leg_len, self.src.leg_len,
                )
                c_terms = reference_feasibility(cfg, kin, ref_raw, ref_geo, src_geo,
                                                leg_len=spec.leg_len)
                # Normalized by each term's own phi=0 value (calibrate()), so the
                # weights in BilevelConfig are relative priorities and S == 1.0
                # at phi = 0 by construction.
                S = weighted_sum(cfg, s_terms, self.scales)
                C = weighted_sum(cfg, c_terms, self.scales)

                # tau is detached inside gap(): MuJoCo gave us no gradient, so
                # the only thing this contributes to dF/dphi is the pull on ref.
                tau = ep["qpos"][:, sl].transpose(0, 1).to(torch.float64)
                valid = ep["valid"][:, sl].transpose(0, 1).to(torch.float64)
                g_terms = gap(cfg, kin, tau, _shift_ref(ref_geo, ref))
                G = weighted_sum(cfg, g_terms, None)  # G's channels are already commensurate

                E_ext, P = self._sim_terms(ep, sl)

                total = total + (cfg.lambda_gap * G + cfg.lambda_fid * S + cfg.lambda_feas * C
                                 + cfg.lambda_ext * E_ext.mean() + cfg.lambda_phys_upper * P.mean())

                with torch.no_grad():
                    g_env = _gap_per_env(cfg, kin, tau, ref, valid)
                    f_sim_env[sl] = (cfg.lambda_gap * g_env
                                     + cfg.lambda_ext * E_ext
                                     + cfg.lambda_phys_upper * P)
                    for d in (s_terms, c_terms, g_terms):
                        for k, v in d.items():
                            acc[k] = acc.get(k, 0.0) + float(v)
                    acc["S"] = acc.get("S", 0.0) + float(S)
                    acc["C"] = acc.get("C", 0.0) + float(C)
                    acc["G"] = acc.get("G", 0.0) + float(G)
                    acc["frac_illegal"] = acc.get("frac_illegal", 0.0) + float(
                        frac_illegal_frames(kin, ref_raw)
                    )
                    # The reference's own ground clearance, and the dz_root that
                    # sets it. Stage 1 exists to drive these: at phi=0 the
                    # reference sits ~12-16 mm INSIDE the floor on 88% of
                    # frames, which is what makes RSI start every window in the
                    # ground. ref_min_z should come up to roughly +0.002 m.
                    p = rt.params(u_bb[b].to(torch.float64).unsqueeze(0))
                    acc["dz_root"] = acc.get("dz_root", 0.0) + float(p["root_dz"])
                    acc["ref_min_z"] = acc.get("ref_min_z", 0.0) + float(ref_geo.min_geom_z.median())
                n_groups += 1

            total = total / max(1, n_groups)
            prox = torch.zeros((), dtype=torch.float64, device=dev)
            if self.u_prev is not None:
                prox = (u_bb.to(torch.float64) - self.u_prev).pow(2).mean()
                total = total + cfg.lambda_prox * prox

        metrics = {k: v / max(1, n_groups) for k, v in acc.items()}
        metrics["F"] = float(total)
        metrics["prox"] = float(prox)
        # any body's RetargetParams will do -- they share cfg, hence free_mask
        metrics["u_saturation"] = float(self.retargeters[0].params.saturation(u_bb))
        metrics["u_absmax"] = float(u_bb.abs().max())
        return total, metrics, f_sim_env

    def _sim_terms(self, ep: Dict[str, torch.Tensor], sl: slice) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-env E_ext and P, straight out of the reward terms the workers shipped.

        P is a per-step physical-plausibility measure assembled from what is
        already on the wire (actuator saturation, foot slide, action jerk, and
        the fraction of the window survived). It is deliberately NOT
        losses.physics_penalty -- that function stays untouched as the
        whole-trajectory eval metric in eval_bilevel.py, so its numbers remain
        comparable with outputs/{baseline,eval}/report.json.
        """
        i = R.TERM_IDX
        terms = ep["terms"][:, sl].to(torch.float64)         # (H, n_b, 14)
        valid = ep["valid"][:, sl].to(torch.float64)         # (H, n_b)
        denom = valid.sum(0).clamp(min=1.0)
        e_ext = (terms[..., i["e_ext"]] * valid).sum(0) / denom
        phys = ((terms[..., i["e_tau"]] + terms[..., i["e_slip"]]
                 + terms[..., i["e_smooth"]]) * valid).sum(0) / denom
        fall = 1.0 - valid.sum(0) / valid.shape[0]           # fraction of the window lost
        return e_ext, phys + fall

    # ------------------------------------------------------------------ ES

    def accumulate_es(self, f_sim_env: torch.Tensor, eps: Dict[int, torch.Tensor],
                      u_bb: torch.Tensor) -> Optional[float]:
        """Antithetic finite differences on u, accumulated into a buffer.

            g_u = 1/(2*sigma*G_b) * sum_g [F_sim(u + sigma*eps_g) - F_sim(u - sigma*eps_g)] * eps_g

        Accumulated over K iterations before the upper step, so each body gets
        2*K*es_pairs_per_body antithetic samples for its 36 dims. The ES window
        and the TTSA cadence are deliberately the same K.
        """
        if self.plan is None or not self.plan.groups:
            return None
        n_bodies, dim = u_bb.shape
        if self._grad_accum is None:
            self._grad_accum = torch.zeros(n_bodies, dim, dtype=torch.float64, device=self.device)

        deltas = torch.stack([
            f_sim_env[g.plus].mean() - f_sim_env[g.minus].mean() for g in self.plan.groups
        ])
        shaped = _rank_normalize(deltas)
        for gi, g in enumerate(self.plan.groups):
            self._grad_accum[g.body] += shaped[gi] * eps[gi].to(torch.float64)
        self._accum_n += 1
        return float(deltas.abs().mean())

    # ------------------------------------------------------------------ step

    def step(self, ep: Dict[str, torch.Tensor], u_bb: torch.Tensor) -> Dict[str, float]:
        """One upper update: exact T1 backward + the accumulated ES gradient,
        then a hard L-infinity trust region on the resulting change in u."""
        cfg = self.cfg
        use_es = self._grad_accum is not None and self._accum_n > 0
        F, metrics, _ = self.evaluate(ep, u_bb, with_grad=True)

        self.optimizer.zero_grad(set_to_none=True)
        F.backward(retain_graph=use_es)

        if use_es:
            # Route the ES estimate into RetargetNet's weights through the same
            # graph, by handing u_bb the estimated dF_sim/du as an upstream grad.
            u_bb.backward(gradient=(self._grad_accum / self._accum_n).to(u_bb.dtype))
            metrics["es_grad_norm"] = float(self._grad_accum.norm() / self._accum_n)
            self._grad_accum = None
            self._accum_n = 0

        params_before = [p.detach().clone() for p in self.net.parameters()]
        u_before = u_bb.detach().clone().to(torch.float64)
        self.optimizer.step()

        # Trust region on u, not on phi. u = f(phi) is nonlinear, so the step is
        # projected back by backtracking rather than solved for: interpolate the
        # parameters toward their pre-step values until ||delta u||_inf fits.
        # This bounds how much the lower level's reward target can move in one
        # upper step, which is what keeps PPO's value targets near-stationary
        # (proposal.md R4).
        clipped = 0.0
        with torch.no_grad():
            for _ in range(8):
                over = float((self.u_by_body().to(torch.float64) - u_before).abs().max())
                if over <= cfg.upper_clip_linf:
                    break
                clipped = 1.0
                scale = 0.9 * cfg.upper_clip_linf / over
                for p, p0 in zip(self.net.parameters(), params_before):
                    p.data.copy_(p0 + scale * (p.data - p0))
            metrics["u_step_linf"] = over
            metrics["u_step_clipped"] = clipped
            self.u_prev = self.u_by_body().detach().to(torch.float64)
        return metrics


def _shift_ref(geo: UpperGeometry, ref: torch.Tensor) -> UpperGeometry:
    """G compares the state AFTER step t against ref[t+1], so the reference side
    is used from index 1 onward. Reuses the already-computed FK by slicing."""
    g = object.__new__(UpperGeometry)
    g.kin = geo.kin
    g.qpos = geo.qpos[:, 1:]
    g.xpos, g.xquat = geo.xpos[:, 1:], geo.xquat[:, 1:]
    g.root, g.root_quat = geo.root[:, 1:], geo.root_quat[:, 1:]
    g.ee, g.feet = geo.ee[:, 1:], geo.feet[:, 1:]
    g.foot_rest_z = geo.foot_rest_z
    g._com = None if geo._com is None else geo._com[:, 1:]
    g._minz = None if geo._minz is None else geo._minz[:, 1:]
    return g


@torch.no_grad()
def _gap_per_env(cfg, kin, tau: torch.Tensor, ref: torch.Tensor,
                 valid: torch.Tensor) -> torch.Tensor:
    """Per-env G, for the ES finite difference. (B, H, 76) -> (B,).

    Only the pose and root-position channels: they dominate G and this runs
    inside the ES inner loop, where a full FK per env per iteration would eat
    the upper level's entire time budget.
    """
    q_hat = ref[:, 1:, 7:]
    d_pose = ((tau[..., 7:] - q_hat) ** 2).mean(-1)
    d_root = ((tau[..., :3] - ref[:, 1:, :3]) ** 2).sum(-1)
    w = valid
    denom = w.sum(1).clamp(min=1.0)
    return (cfg.g_pose * (d_pose * w).sum(1) + cfg.g_root_pos * (d_root * w).sum(1)) / denom
