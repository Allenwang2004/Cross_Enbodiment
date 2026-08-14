"""Bilevel training entry point: the TTSA loop.

    for k = 1 .. num_iters:
        collect 256 windows x 24 steps with (phi_k, p_k)
        phi_{k+1} = PPO(phi_k)                                  every iteration
        accumulate the antithetic-ES estimate of dF_sim/du
        if k > upper_warmup_iters and k % upper_every == 0:
            p_{k+1} = Adam(p_k, T1 + ES/K + prox)            every K = 10

Timescale separation: eta_phi/eta_p = 30, upper cadence K = 10, so the
effective ratio is 300:1. Borkar's TTSA conditions (sum eta = inf, sum eta^2 <
inf, eta_p/eta_phi -> 0) are NOT literally satisfied by Adam -- the theory is
a guide for the ratio, and what actually delivers the stability it describes is
the cadence plus the proximal term plus the hard L-inf trust region on u. No
convergence guarantee is claimed.

p is frozen at 0 for the first upper_warmup_iters. F(p, phi) is only
meaningful once phi is worth measuring, and freezing is the practical stand-in
for the "phi near phi*(p)" that the analysis assumes.

Three stages, run in order, each carrying the previous one's weights forward
with --init-from (NOT --resume, which is for continuing one interrupted run):

    uv run model/bilevel/train_bilevel.py --stage 1 --iters 2000
    uv run model/bilevel/train_bilevel.py --stage 2 --iters 4000 \
        --init-from model/bilevel/checkpoints/stage1_002000.pt
    uv run model/bilevel/train_bilevel.py --stage 3 --iters 10000 \
        --init-from model/bilevel/checkpoints/stage2_004000.pt

The last one is the full-scale run; there is no separate "stage 4".
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from model.bilevel import rewards as R
from model.bilevel.amp import AmpTrainer
from model.bilevel.config import BilevelConfig
from model.bilevel.data import WindowDataset, load_splits
from model.bilevel.longeval import LONG_KEYS, LongEvaluator
from model.bilevel.policy import LowerPolicy, build_value_net, load_frozen_model
from model.bilevel.ppo import (
    PairAdvantageNormalizer, RunningScalar, reference_ctrl, update_lower,
)
from model.bilevel.retarget import U_ROOT_DZ, RetargetNet, Retargeter
from model.bilevel.rollout import Collector, sample_action_noise, sample_rsi_noise
from model.bilevel.sim.pool import SimPool
from model.bilevel.upper import UpperLevel


def build(cfg, verbose=True):
    """Everything, wired. Returns a dict of components."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device(cfg.device)

    train_tasks, test_tasks = load_splits(cfg)
    ds = WindowDataset(cfg, tasks=train_tasks, verbose=verbose)
    R.set_adult_leg_len(ds.source.leg_len)
    # Every TorchKinematics is a real nn.Module full of buffers; they are built
    # on CPU by load_body and have to be moved explicitly. The target bodies'
    # would get moved as a side effect of Retargeter.to(dev), but the SOURCE
    # body's is only ever used directly (by S in semantics.py), so it would
    # otherwise stay behind and blow up mid-iteration.
    for spec in [*ds.bodies, ds.source]:
        spec.kin.to(dev)

    frozen = load_frozen_model(cfg)
    beta_dim = ds.beta_dim
    policy = LowerPolicy(cfg, frozen, beta_dim).to(dev)
    value_net = build_value_net(cfg, frozen, beta_dim).to(dev)

    # ONE RetargetNet shared by every body. That sharing is what forces p to
    # be a genuine function of beta instead of 10 independent corrections.
    net = RetargetNet(beta_dim, cfg.retarget_hidden_dims).to(dev).double()
    retargeters = {
        b: Retargeter(cfg, net, ds.bodies[b].kin, ds.source.rest_h).to(dev).double()
        for b in range(len(ds.bodies))
    }

    assign = ds.body_assignment(cfg.n_envs)
    pool = SimPool(
        cfg,
        [str(ds.bodies[b].xml_path) for b in assign],
        [ds.bodies[b].leg_len for b in assign],
        obs_dim=frozen.cfg.obs_dim,
    )
    collector = Collector(cfg, ds, pool, policy, value_net, retargeters, dev)
    upper = UpperLevel(cfg, ds, retargeters, net, dev)
    amp = AmpTrainer(cfg, ds, dev) if cfg.amp_enabled else None
    long_eval = (
        LongEvaluator(cfg, ds, dev, n_clips=cfg.long_eval_clips,
                      horizon=cfg.long_eval_horizon, seed=cfg.seed)
        if cfg.long_eval_every else None
    )

    opt = torch.optim.Adam(
        policy.trainable_parameters() + list(value_net.parameters()), lr=cfg.lr_lower
    )
    return dict(
        cfg=cfg, ds=ds, dev=dev, frozen=frozen, policy=policy, value_net=value_net,
        net=net, retargeters=retargeters, pool=pool, collector=collector, upper=upper,
        opt=opt, assign=assign, test_tasks=test_tasks, long_eval=long_eval, amp=amp,
    )


class WandbLogger:
    """W&B logging that can never take the training run down with it.

    Every call is wrapped: a network blip, an expired token or a missing
    credential disables logging and prints why, but the loop keeps going. The
    .jsonl in cfg.log_dir is written unconditionally and stays the source of
    truth -- W&B is a convenience layer over it, not a replacement.

    Stage 1/2/3 are separate runs by design (the objective changes at each
    boundary, so one continuous curve would be misleading), joined by `group`
    so they read as a single experiment. Interrupting and `--resume`-ing
    reattaches to the SAME run via the id stored in the checkpoint, rather than
    scattering one training run across several.
    """

    def __init__(self, cfg, stage: int, resumed_id=None, init_from=None):
        self.enabled = bool(cfg.use_wandb)
        self.run = None
        self.id = resumed_id
        if not self.enabled:
            return
        try:
            import dataclasses
            import wandb

            self.wandb = wandb
            self.run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name or f"stage{stage}",
                group=cfg.wandb_group,
                job_type=f"stage{stage}",
                mode=cfg.wandb_mode,
                id=resumed_id,
                resume="allow" if resumed_id else None,
                config={
                    **dataclasses.asdict(cfg),
                    "stage": stage,
                    "init_from": str(init_from) if init_from else None,
                },
            )
            self.id = self.run.id
            print(f"W&B: {self.run.url if cfg.wandb_mode == 'online' else '(offline)'}")
        except Exception as e:
            self.enabled = False
            print(f"W&B disabled ({type(e).__name__}: {e}). Training continues; "
                  f"metrics still go to the .jsonl.")

    def log(self, metrics: dict, step: int):
        if not self.enabled:
            return
        try:
            self.wandb.log({k: v for k, v in metrics.items() if k != "iter"}, step=step)
        except Exception as e:
            self.enabled = False
            print(f"W&B logging failed ({type(e).__name__}: {e}); disabled for the rest of the run.")

    def log_calibration(self, scales, baseline):
        """The p=0 / box-corner table from UpperLevel.calibrate(), once.

        Worth having on the run page: it is what makes the S and C weights
        interpretable, and it differs whenever the assets or the body list
        change (see semantics.weighted_sum).
        """
        if not self.enabled or not scales:
            return
        try:
            tbl = self.wandb.Table(columns=["term", "p0", "box_corner", "normalized_at_p0"])
            for k in sorted(scales):
                b0 = float(baseline.get(k, 0.0))
                tbl.add_data(k, b0, float(scales[k]), b0 / max(scales[k], 1e-12))
            self.wandb.log({"upper/calibration": tbl}, step=0)
            self.wandb.run.summary.update({f"calib/{k}": v for k, v in scales.items()})
        except Exception:
            pass

    def finish(self):
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass


def _wandb_metrics(m: dict, stage: int) -> dict:
    """Group the flat metric dict into W&B panels.

    The names come straight from ppo.update_lower and UpperLevel.evaluate; the
    prefixes just decide which chart they land on.
    """
    lower = {"reward", "r_track", "r_pose", "r_ee", "e_ext", "pose_err", "term_rate",
             "mean_steps", "pg", "v", "z", "bc", "kl", "clipfrac"}
    upper = {"F", "G", "G_ref0", "S", "C", "prox", "frac_illegal", "u_saturation", "u_absmax",
             "dz_root", "ref_min_z",
             "u_step_linf", "u_step_clipped", "es_grad_norm", "es_delta"}
    sched = {"lambda_bc", "wrench_scale", "w_amp"}
    amp = {"amp_loss", "amp_d_real", "amp_d_fake", "amp_grad_pen", "amp_reward"}
    long = set(LONG_KEYS)
    out = {"stage": stage}
    for k, v in m.items():
        if k in ("iter", "elapsed"):
            out[k] = v
        elif k in lower:
            out[f"lower/{k}"] = v
        elif k in upper:
            out[f"upper/{k}"] = v
        elif k in sched:
            out[f"sched/{k}"] = v
        elif k in long:
            out[f"long/{k[len('long_'):]}"] = v
        elif k in amp:
            out[f"amp/{k[len('amp_'):]}"] = v
        elif k.startswith(("s_", "c_", "g_")):
            out[f"upper_terms/{k}"] = v          # the per-channel S/C/G breakdown
        else:
            out[k] = v
    return out


def save_checkpoint(path, it, C, rng, gen, pair_norm, ret_norm, stage):
    """Everything needed to resume bit-comparably.

    Saving only the network weights is not enough: the Adam moments, the RNG
    streams, the per-(clip, body) advantage EMA and the value-target normalizer
    are all training state, and dropping them makes a resumed run behave like a
    fresh one with a warm start rather than a continuation. The RNG states in
    particular decide which windows get sampled and -- if ES is on -- the
    common random numbers a perturbation pair shares.
    """
    torch.save({
        "iter": it,
        "stage": stage,
        "cfg": C["cfg"],
        "policy": C["policy"].state_dict(),
        "value": C["value_net"].state_dict(),
        "retarget_net": C["net"].state_dict(),
        "amp": C["amp"].d.state_dict() if C["amp"] is not None else None,
        "amp_opt": C["amp"].opt.state_dict() if C["amp"] is not None else None,
        "opt_lower": C["opt"].state_dict(),
        "opt_upper": C["upper"].optimizer.state_dict(),
        "upper_scales": C["upper"].scales,
        "upper_baseline": C["upper"].baseline,
        "upper_u_prev": C["upper"].u_prev,
        "es_accum": C["upper"]._grad_accum,
        "es_accum_n": C["upper"]._accum_n,
        # so --resume reattaches to the same W&B run instead of starting a new
        # one every time a long run is interrupted
        "wandb_id": C.get("wandb_id"),
        "rng": rng.bit_generator.state,
        "torch_gen": gen.get_state(),
        "pair_mean": pair_norm.mean, "pair_sq": pair_norm.sq, "pair_seen": pair_norm.seen,
        "ret_mean": ret_norm.mean, "ret_var": ret_norm.var, "ret_count": ret_norm.count,
    }, path)


def load_checkpoint(path, C, rng, gen, pair_norm, ret_norm):
    """Restore in place. Returns the iteration to resume FROM."""
    ck = torch.load(path, map_location=C["dev"], weights_only=False)
    C["policy"].load_state_dict(ck["policy"])
    C["value_net"].load_state_dict(ck["value"])
    C["net"].load_state_dict(ck["retarget_net"])
    C["opt"].load_state_dict(ck["opt_lower"])
    C["upper"].optimizer.load_state_dict(ck["opt_upper"])
    C["upper"].scales = ck["upper_scales"]
    C["upper"].baseline = ck.get("upper_baseline", {})
    C["upper"].u_prev = ck["upper_u_prev"]
    C["upper"]._grad_accum = ck.get("es_accum")
    C["upper"]._accum_n = ck.get("es_accum_n", 0)
    rng.bit_generator.state = ck["rng"]
    # A Generator's state is a ByteTensor that must live on the CPU even for a
    # CUDA generator; torch.load's map_location moved it to the device.
    gen.set_state(ck["torch_gen"].cpu())
    pair_norm.mean, pair_norm.sq, pair_norm.seen = ck["pair_mean"], ck["pair_sq"], ck["pair_seen"]
    ret_norm.mean, ret_norm.var, ret_norm.count = ck["ret_mean"], ck["ret_var"], ck["ret_count"]
    C["wandb_id"] = ck.get("wandb_id")
    return int(ck["iter"])


def init_from_checkpoint(path, C, verbose=True):
    """Carry the trained NETWORKS across a stage boundary -- nothing else.

    Distinct from load_checkpoint on purpose. Resuming continues one run, so it
    restores optimizer moments, RNG streams and the iteration counter. Moving
    Stage 1 -> Stage 2 is a different thing: the objective itself changes (p
    unfreezes, the wrench starts annealing), so the Adam moments are stale, the
    LR schedule must restart from 0, and carrying the iteration counter over
    would skip the new stage's warmup entirely.

    Weights are all that should survive; everything else starts clean.
    """
    ck = torch.load(path, map_location=C["dev"], weights_only=False)
    C["policy"].load_state_dict(ck["policy"])
    C["value_net"].load_state_dict(ck["value"])
    C["net"].load_state_dict(ck["retarget_net"])
    if verbose:
        print(f"initialized networks from {path} "
              f"(stage {ck.get('stage', '?')}, iter {ck.get('iter', '?')}); "
              f"optimizers, schedules and RNG start fresh")
    return ck


def latest_checkpoint(ckpt_dir: Path, stage: int):
    found = sorted(Path(ckpt_dir).glob(f"stage{stage}_*.pt"))
    return found[-1] if found else None


def apply_stage(cfg, stage: int) -> BilevelConfig:
    """Stage presets (proposal.md 0.9 and 8.2).

    Each stage introduces exactly ONE new source of difficulty, so that when
    something breaks you know what broke it. Each is gated on the previous
    one's success criteria -- do not skip ahead.

    There are three, not four. An earlier draft had a "Stage 4 -- full scale",
    but scale is `--iters`, not a stage: Stage 2 and 3 already train on every
    body and every clip, so a fourth preset was byte-identical to Stage 3 (0
    config fields differed) and bought nothing but a second checkpoint prefix.
    The full run IS Stage 3 with `--iters 10000`.
    """
    if stage == 1:
        # The lower level, on two similar bodies, with the crutch on and the
        # reference held still -- so this answers exactly one question, as
        # cheaply as possible: can the policy track at all?
        #
        # ONE exception to "p frozen": dz_root is free. Freezing all of p
        # was measured to make this stage unwinnable. The p=0 reference sits
        # ~12-16 mm INSIDE the floor on 88% of frames (child median -0.0120 m,
        # teen -0.0155 m, worst -0.204 m), because replacing
        # scripts/qpos_retarget.py:127 ground_correct_qpos's per-frame lift with
        # a learned dz_root means freezing p also freezes the only thing that
        # can lift it. RSI then starts every window with the body in the ground,
        # physics ejects it, and it falls: a 2130-iteration run plateaued at
        # r_track 0.537 (gate 0.6) with term_rate 0.87 (gate 0.05).
        #
        # This is still one variable at a time -- stage 1 gets the 1 dimension
        # that repairs a broken premise, stage 2 gets the other 35.
        cfg.train_bodies = ["child", "teen"]
        cfg.upper_free_dims = [U_ROOT_DZ]
        cfg.upper_warmup_iters = 100     # let the value net settle first, then lift
        cfg.es_enabled = False
        cfg.wrench_hold_iters = 10 ** 9
    elif stage == 2:
        # Adds morphology diversity AND the remaining 35 p dimensions. The
        # wrench stays constant deliberately: that would be a third new thing.
        cfg.upper_free_dims = None       # all 36
        cfg.es_enabled = False
        cfg.wrench_hold_iters = 10 ** 9
    elif stage == 3:
        # Removes the crutch and, necessarily at the same time, adds ES. The
        # exact T1 gradient is blind to wrench-cheating (E_ext depends on p
        # only through the non-differentiable simulator, so its T1 is
        # identically zero), so annealing the wrench without ES would leave the
        # upper level free to make the reference harder and let the lower level
        # prop itself up. See proposal.md 4.3.
        cfg.es_enabled = True
    else:
        raise ValueError(f"stage must be 1, 2 or 3 (got {stage}); "
                         f"the full-scale run is stage 3 with --iters 10000")

    if cfg.amp_enabled:
        # amp.py is off by default and no stage turns it on -- the experiment is
        # recorded there and in proposal.md. Kept reachable only so the negative
        # result can be re-measured without rebuilding the plumbing.
        cfg.term_pose_err = cfg.amp_term_pose_err
    return cfg


def train(cfg, stage: int, verbose=True, resume=None, init_from=None):
    C = build(cfg, verbose=verbose)
    cfg, ds, dev = C["cfg"], C["ds"], C["dev"]
    collector, upper, policy, value_net, opt = (
        C["collector"], C["upper"], C["policy"], C["value_net"], C["opt"]
    )
    long_eval = C["long_eval"]
    amp = C["amp"]

    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)

    # Measure every S/C term at p=0 and adopt it as that term's unit. Without
    # this the raw magnitudes span six orders of magnitude and the configured
    # weights are meaningless -- c_smooth measured 1.2e3 against c_limit's
    # 9.7e-4, so a nominal 4:1 priority for the joint-limit term was really
    # 1:140 against it. See semantics.weighted_sum.
    # Lift the reference out of the floor before anything else looks at it --
    # both the calibration below and every RSI depend on it (see the method).
    if not init_from and not resume:
        upper.warm_start_ground(np.random.default_rng(cfg.seed + 2), verbose=verbose)
    upper.calibrate(np.random.default_rng(cfg.seed + 1), verbose=verbose)
    pair_norm = PairAdvantageNormalizer(
        len(ds.clips) * len(ds.bodies), cfg.adv_norm_momentum, device=dev
    )
    ret_norm = RunningScalar(device=dev)

    log_dir = REPO_ROOT / cfg.log_dir
    ckpt_dir = REPO_ROOT / cfg.ckpt_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"stage{stage}_metrics.jsonl"
    log_f = open(log_path, "a")

    start_iter = 0
    if init_from and resume:
        raise SystemExit("--init-from and --resume are mutually exclusive: one starts a new "
                         "stage from trained weights, the other continues an existing run")
    if init_from:
        init_from_checkpoint(Path(init_from), C, verbose=verbose)
    if resume:
        path = latest_checkpoint(ckpt_dir, stage) if resume == "auto" else Path(resume)
        if path is None:
            print(f"--resume auto: no stage{stage} checkpoint in {ckpt_dir}, starting fresh")
        else:
            start_iter = load_checkpoint(path, C, rng, gen, pair_norm, ret_norm)
            print(f"resumed from {path} at iteration {start_iter}")
            # A stale checkpoint from an older, longer run of the same stage
            # sorts LAST by filename, so `--resume auto` picks it, resumes past
            # the requested horizon and exits after "done in 0.0 min" having
            # trained nothing. That looked exactly like a successful run for two
            # hours. Refuse instead.
            if start_iter >= cfg.num_iters:
                raise SystemExit(
                    f"{path} is already at iteration {start_iter}, which is not before "
                    f"--iters {cfg.num_iters}: there is nothing to run. Either raise "
                    f"--iters, or pass --resume <path> for the checkpoint you meant "
                    f"(--resume auto takes the highest-numbered stage{stage} file, which "
                    f"may belong to an earlier run)."
                )

    print(f"\nstage {stage} | {len(ds.bodies)} bodies | {len(ds.clips)} clips | "
          f"{cfg.n_envs} envs x {cfg.horizon} steps | "
          f"iterations {start_iter}..{cfg.num_iters}")
    print(f"logging -> {log_path}")

    wb = WandbLogger(cfg, stage, resumed_id=C.get("wandb_id"), init_from=init_from)
    C["wandb_id"] = wb.id
    wb.log_calibration(upper.scales, upper.baseline)
    print()

    t_start = time.perf_counter()
    try:
        for it in range(start_iter, cfg.num_iters):
            # ---- sample, with CRN applied for any ES pairs ---------------
            ci, bi, t0 = ds.sample(cfg.n_envs, rng, body_idx=C["assign"])
            batch = ds.build_batch(ci, bi, t0)
            rsi_noise = sample_rsi_noise(cfg, rng)
            act_noise = sample_action_noise(cfg, gen, dev)

            u_bb = upper.u_by_body()
            eps = {}
            if cfg.es_enabled and upper.plan is not None:
                eps = upper.draw_eps(gen)
                upper.plan.apply_crn(batch, rsi_noise, act_noise)
            u_env = upper.u_per_env(u_bb.detach(), eps if eps else None, cfg.es_sigma)

            # ---- rollout -------------------------------------------------
            ep = collector.collect(it, batch, u_env, rsi_noise, act_noise)

            # ---- BC target: the exact ctrl that commands ref[t+1] --------
            bc = None
            if cfg.bc_scale(it) > 0:
                bc = torch.empty(cfg.n_envs, cfg.horizon, 69, device=dev)
                for b, sl in collector.slices:
                    bc[sl] = reference_ctrl(ds.bodies[b].kin, ep["ref"][sl]).to(torch.float32)

            # ---- AMP: stationary style reward (model/bilevel/amp.py) ------
            # Scores the policy's own transitions against the reference's, with
            # no time index anywhere -- which is the whole point. Blended into
            # the tracking channel by ppo.update_lower via cfg.amp_scale(it).
            amp_r, amp_m = None, {}
            if amp is not None and cfg.amp_scale(it) > 0:
                kin_by_body = {b: ds.bodies[b].kin for b in range(len(ds.bodies))}
                amp_r, amp_m = amp.step(ep, kin_by_body, collector.slices)

            # ---- lower level (every iteration) ---------------------------
            if it < cfg.value_warmup_iters:
                # Value-only warm-up: with H=24 about half the return at t=0 is
                # bootstrap, and a mis-scaled V makes GAE meaningless.
                m = _value_only_update(cfg, it, value_net, policy, opt, ep, ret_norm)
            else:
                m = update_lower(cfg, it, policy, value_net, opt, ep,
                                 pair_norm, ret_norm, bc_targets=bc, amp_reward=amp_r)
            m.update(amp_m)

            # ---- upper level (every K, after warmup) ---------------------
            if cfg.es_enabled and eps:
                with torch.no_grad():
                    _, _, f_sim = upper.evaluate(ep, u_bb.detach(), with_grad=False)
                d = upper.accumulate_es(f_sim, eps, u_bb.detach())
                if d is not None:
                    m["es_delta"] = d

            if it >= cfg.upper_warmup_iters and (it + 1) % cfg.upper_every == 0:
                m.update(upper.step(ep, upper.u_by_body()))
            elif it % cfg.log_every == 0:
                with torch.no_grad():
                    _, um, _ = upper.evaluate(ep, u_bb.detach(), with_grad=False)
                m.update(um)

            # ---- long-rollout diagnostic (R6) -----------------------------
            # Deliberately NOT a gradient: the 24-step window cannot see whether
            # locally-good motion is globally coherent, and without this the
            # answer only arrives when the whole run is over.
            if long_eval is not None and it % cfg.long_eval_every == 0:
                m.update(long_eval.run(policy, C["retargeters"],
                                       wrench_frac=cfg.long_eval_wrench))

            # ---- log ------------------------------------------------------
            if it % cfg.log_every == 0:
                m["iter"] = it
                m["elapsed"] = time.perf_counter() - t_start
                log_f.write(json.dumps({k: float(v) for k, v in m.items()}) + "\n")
                log_f.flush()
                wb.log(_wandb_metrics(m, stage), step=it)
                print(
                    f"[{it:05d}] r={m.get('reward', 0):.3f} r_track={m.get('r_track', 0):.3f} "
                    f"pose_err={m.get('pose_err', 0):.3f} term={m.get('term_rate', 0):.2f} "
                    f"| G={m.get('G', 0):.3f} S={m.get('S', 0):.3f} C={m.get('C', 0):.3f} "
                    f"illegal={m.get('frac_illegal', 0):.3f} sat={m.get('u_saturation', 0):.2f} "
                    + (f"| long={m['long_survive']:.2f} " if "long_survive" in m else "")
                    + f"| {m['elapsed'] / max(1, it + 1 - start_iter):.2f}s/it"
                )

            if (it + 1) % cfg.ckpt_every == 0:
                p = ckpt_dir / f"stage{stage}_{it + 1:06d}.pt"
                save_checkpoint(p, it + 1, C, rng, gen, pair_norm, ret_norm, stage)
                print(f"  saved -> {p}")
    except KeyboardInterrupt:
        # Ctrl-C is a normal way to stop a multi-hour run; land on a resumable
        # checkpoint instead of throwing the work away.
        p = ckpt_dir / f"stage{stage}_{it + 1:06d}.pt"
        save_checkpoint(p, it + 1, C, rng, gen, pair_norm, ret_norm, stage)
        print(f"\ninterrupted at iteration {it + 1}, saved -> {p}")
        print(f"resume with: uv run model/bilevel/train_bilevel.py --stage {stage} --resume auto")
    finally:
        log_f.close()
        wb.finish()
        C["pool"].close()

    print(f"\ndone in {(time.perf_counter() - t_start) / 60:.1f} min")
    return C


def _value_only_update(cfg, it, value_net, policy, opt, ep, ret_norm):
    """Fit V before letting the policy loss loose on a garbage advantage."""
    from model.bilevel.ppo import compute_gae

    dev = ep["obs"].device
    H, N = ep["terms"].shape[:2]
    reward = torch.as_tensor(
        R.combine(ep["terms"].detach().cpu().numpy(), cfg), dtype=torch.float32, device=dev
    ) * ep["valid"]
    with torch.no_grad():
        v_raw = ep["value"] * ret_norm.std + ret_norm.mean
        _, ret = compute_gae(reward, v_raw, ep["done"], ep["valid"],
                             cfg.gamma_at(it), cfg.gae_lambda)
        ret_norm.update(ret[ep["valid"] > 0])
        ret_n = (ret - ret_norm.mean) / ret_norm.std

    phase = torch.stack([
        (torch.arange(H, device=dev) / H).unsqueeze(1).expand(H, N),
        ((H - torch.arange(H, device=dev)) / H).unsqueeze(1).expand(H, N),
    ], dim=-1)
    beta = ep["beta"].unsqueeze(0).expand(H, N, -1)
    v = value_net(ep["obs"], ep["z_beta"].unsqueeze(0).expand(H, N, -1), beta, phase)
    loss = 0.5 * (((v - ret_n) ** 2) * ep["valid"]).sum() / ep["valid"].sum().clamp(min=1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), cfg.grad_clip_norm)
    opt.step()

    i = R.TERM_IDX
    d = ep["valid"].sum().clamp(min=1)
    return {
        "v": float(loss.detach()), "pg": 0.0, "z": 0.0, "bc": 0.0, "kl": 0.0, "clipfrac": 0.0,
        "reward": float((reward * ep["valid"]).sum() / d),
        "r_track": float(sum(getattr(cfg, f"w_{k}") * (ep["terms"][..., i[f"r_{k}"]] * ep["valid"]).sum() / d
                             for k in ("pose", "vel", "ee", "root", "com"))),
        "pose_err": float((ep["terms"][..., i["pose_err"]] * ep["valid"]).sum() / d),
        "term_rate": float(ep["done"].sum() / N),
        "wrench_scale": ep["wrench_scale"], "lambda_bc": 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                    help="1: lower level only  2: + upper level  3: + wrench anneal and ES. "
                         "The full-scale run is stage 3 with --iters 10000.")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--envs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    # Overrides for short validation runs; the stage presets set these to their
    # real values and should be left alone for an actual training run.
    ap.add_argument("--upper-warmup", type=int, default=None)
    ap.add_argument("--amp-warmup", type=int, default=None,
                    help="iterations before the AMP style reward starts ramping in")
    ap.add_argument("--no-wrench", action="store_true",
                    help="train with NO external root wrench at all (f_max = 0 from "
                         "iteration 0). The wrench is a crutch the deliverable does not "
                         "have; this asks whether the lower level can stand without it "
                         "instead of being weaned off it.")
    ap.add_argument("--value-warmup", type=int, default=None)
    ap.add_argument("--upper-every", type=int, default=None)
    ap.add_argument("--bodies", default=None, help="comma-separated body labels")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    metavar="PATH",
                    help="continue an interrupted run of THIS stage; bare --resume picks "
                         "the latest checkpoint for the stage, or pass an explicit .pt path")
    ap.add_argument("--init-from", default=None, metavar="PATH",
                    help="start this stage from another stage's trained weights "
                         "(networks only; optimizers, schedules and RNG start fresh). "
                         "Use this to go Stage 1 -> Stage 2 -> Stage 3.")
    ap.add_argument("--wandb", action="store_true", help="enable W&B logging")
    ap.add_argument("--wandb-offline", action="store_true",
                    help="log to a local W&B dir for later `wandb sync` (implies --wandb); "
                         "use this when the machine has no credentials")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-name", default=None, help="run name (default: stage<N>)")
    ap.add_argument("--wandb-group", default=None,
                    help="ties this pipeline's stages together; give all three the same value")
    args = ap.parse_args()

    cfg = BilevelConfig()
    cfg = apply_stage(cfg, args.stage)
    if args.iters is not None:
        cfg.num_iters = args.iters
    if args.upper_warmup is not None:
        cfg.upper_warmup_iters = args.upper_warmup
    if args.amp_warmup is not None:
        cfg.amp_warmup_iters = args.amp_warmup
        cfg.amp_full_iters = max(cfg.amp_full_iters, args.amp_warmup + 1)
    if args.no_wrench:
        # Note this leaves the 75-dim action distribution untouched: policy.py
        # multiplies by f_max, so zeroing it removes the FORCE without changing
        # the log-probs (see LowerPolicy.split_action). The 6 wrench dimensions
        # are still sampled and still scored, they just do nothing -- which
        # keeps PPO's ratio well-defined and the checkpoint interchangeable
        # with a wrench-enabled run.
        cfg.wrench_enabled = False
    if args.value_warmup is not None:
        cfg.value_warmup_iters = args.value_warmup
    if args.upper_every is not None:
        cfg.upper_every = args.upper_every
    if args.bodies:
        cfg.train_bodies = args.bodies.split(",")
    if args.wandb or args.wandb_offline:
        cfg.use_wandb = True
    if args.wandb_offline:
        cfg.wandb_mode = "offline"
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project
    if args.wandb_name:
        cfg.wandb_run_name = args.wandb_name
    if args.wandb_group:
        cfg.wandb_group = args.wandb_group
    if args.device:
        cfg.device = args.device
    if args.envs:
        cfg.n_envs = args.envs
    if args.workers:
        cfg.n_workers = args.workers
    if args.seed is not None:
        cfg.seed = args.seed
    train(cfg, args.stage, resume=args.resume, init_from=args.init_from)


if __name__ == "__main__":
    main()
