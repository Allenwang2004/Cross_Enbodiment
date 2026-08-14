"""Every hyperparameter for the bilevel system, in one dataclass.

Values and their justifications come from proposal.md; the section number is
cited next to anything that is not an arbitrary default. Where a value
deliberately overturns one inherited from model/config.py TrainConfig, the old
value and the reason are both recorded -- those old numbers encode findings
that cost real time (see model/train.py's module docstring).
"""

import dataclasses
from typing import List, Optional, Tuple


@dataclasses.dataclass
class BilevelConfig:
    # ------------------------------------------------------------- paths / data
    metamotivo_repo: str = "facebook/metamotivo-M-1"
    origin_motion_dir: str = "data/origin_motion"      # <task>/<task>_<trial>.npz, key "qpos"
    z_dir: str = "data/z"                              # <task>/<task>_<trial>.npy, (1, 256)
    # Actuator-calibrated assets, produced by scripts/calibrate_actuators.py.
    # assets/robots/ (the shipped ones) only yields 8 usable bodies -- see
    # train_bodies below. The originals are left untouched so the legacy
    # model/train.py baseline and outputs/{baseline,eval}/report.json stay
    # reproducible.
    robots_dir: str = "assets/robots_calib_move"
    source_body: str = "adult"                         # the body the frozen policy was trained on
    # 10 training bodies + 2 held out for testing.
    #
    # This split only became possible after scripts/calibrate_actuators.py. With
    # the SHIPPED assets (assets/robots/) only 8 of 13 bodies can generate enough
    # torque to hold their own rest pose -- measured headroom
    # (forcerange / static torque at qpos0):
    #
    #     adult 10.35  child 7.54  elderly 1.76  long_limbed 1.72  teen 1.60
    #     petite 1.41  tall_slim 1.23  athletic 1.17
    #     ---- above can stand; below cannot ----
    #     pear_shaped 0.96  giant 0.48  short_limbed 0.09  short_stocky 0.04  heavy 0.03
    #
    # They ship with the ADULT's gainprm/biasprm/forcerange despite masses of
    # 38-210 kg (only `elderly` was ever rescaled; the 11 bodies from commit
    # 9495329 all record "scale_actuators": false). scripts/scale_robot.py's
    # predicted load model does not fix it and makes things worse on balance
    # (see scripts/regen_bodies.py); calibrating against MEASURED torque demand
    # does, and brings all 13 to >= 1.5 with reference-pose headroom of 0.4-0.7,
    # i.e. every body is now about as controllable as the adult the frozen
    # policy was trained on (0.745).
    #
    # Held-out pair is chosen for EXTRAPOLATION, not interpolation. Training
    # spans 38-101 kg and 0.59-1.07 m rest height; both test bodies sit outside
    # it on the axis they probe -- `giant` at 110.7 kg / 1.17 m (taller than any
    # training body) and `short_stocky` at 130.5 kg (heavier than any). Held-out
    # numbers therefore measure generalization rather than interpolation.
    train_bodies: List[str] = dataclasses.field(default_factory=lambda: [
        "child", "teen", "petite", "tall_slim",
        "long_limbed", "short_limbed", "athletic", "elderly", "pear_shaped",
    ])
    heldout_bodies: List[str] = dataclasses.field(default_factory=lambda: [
        "giant", "short_stocky",
    ])
    # `heavy` (210 kg, 2.9x the adult) used to sit here as an "unused" body and
    # has now been DELETED from assets/ entirely. Even after calibration its
    # rest-pose headroom was 0.975, and reaching a usable margin needed roughly
    # a 250x actuator (measured 300x, clamp-limited, on the move/jump-only
    # calibration) -- at which point it is no longer a plausible humanoid, just
    # a very strong machine. The audit numbers quoting it in docs/ and in
    # scripts/calibrate_actuators.py are kept as the record of that finding.
    # Task-level split, so correlated trials of one task cannot straddle it
    # (same rationale as scripts/split_tasks.py:4-7).
    splits_dir: str = "datasets/crossenbodiment-1-datasets/splits"

    device: str = "cuda:0"
    seed: int = 0

    # --------------------------------------------------------------- simulation
    control_hz: float = 30.0            # humenv: simulation_dt=1/450, action_repeat=15
    physics_dt: float = 1.0 / 450.0
    action_repeat: int = 15
    n_envs: int = 256                   # proposal.md 9: deliberately 256, not new.md's 1024
    horizon: int = 60                   # steps per window
    n_workers: int = 32                 # = physical cores; sims_per_worker = n_envs // n_workers
    pin_workers: bool = True            # os.sched_setaffinity, measured worth ~14% (6.1)
    spin_iters: int = 2000              # spin-then-yield; mp.Barrier measured 1030ms vs 632ms (6.1)
    worker_timeout_s: float = 5.0       # watchdog: a dead worker deadlocks the spin loop (6.6)

    # RSI (proposal.md 6.4). Gaussian on the 69 hinges only; root pos/quat exact.
    rsi_sigma_max: float = 0.08         # rad; sigma ~ U(0, this) resampled PER WINDOW
    rsi_penetration_tol: float = -0.01  # retry with halved noise if contact.dist goes below
    rsi_max_retries: int = 3

    # ------------------------------------------------------------ upper level (p)
    retarget_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [64, 64])
    # Hard tanh box half-widths (proposal.md 3.2). These are the primary
    # anti-degeneracy defence: they make collapse structurally impossible rather
    # than merely penalized, so they do not depend on getting a weight right.
    box_root_scale: float = 0.15        # log-space, ~ +-16% on root xyz scale
    box_root_dz: float = 0.08           # m
    box_root_rot: float = 0.15          # exp-map rad, ~ +-8.6 deg
    box_joint_gain: float = 0.20        # +-20% amplitude, per joint group
    box_joint_bias: float = 0.15        # rad, per joint group
    box_log_tau: float = 0.223144       # log(1.25); time-warp, frozen until Stage 3
    enable_time_warp: bool = False
    # Which of u's 36 dimensions p is allowed to move. None = all of them.
    # Stage 1 frees ONLY dz_root (see train_bilevel.apply_stage): the p=0
    # reference sits ~12-16 mm inside the floor on 88% of frames, because
    # replacing scripts/qpos_retarget.py's per-frame ground_correct_qpos with a
    # LEARNED dz_root means freezing p also freezes the only mechanism that
    # can lift it out. Names are in retarget.py (U_ROOT_DZ etc.).
    upper_free_dims: Optional[List[int]] = None

    # F(p) weights (proposal.md 3.4). lambda_fid/lambda_gap = 3 fixes the
    # maximum drift of the reference toward the robot at lambda_gap/(sum) = 25%.
    lambda_gap: float = 1.0
    lambda_fid: float = 3.0
    lambda_feas: float = 2.0
    lambda_ext: float = 0.5
    lambda_phys_upper: float = 0.5
    lambda_prox: float = 1.0

    # G channel weights
    g_pose: float = 1.0
    g_ee: float = 2.0
    g_root_pos: float = 1.0
    g_root_rot: float = 0.5
    # S channel weights
    s_pose: float = 1.0
    s_reach: float = 1.0
    s_contact: float = 2.0     # the workhorse: collapse breaks foot-off timing first
    s_heading: float = 1.0
    s_froude: float = 1.0
    s_contact_k: float = 200.0
    # C channel weights
    c_limit: float = 4.0       # encodes the 95.8%-illegal-frames finding (proposal.md 1.2)
    c_penetrate: float = 2.0
    c_float: float = 0.5
    c_slide: float = 1.0
    c_smooth: float = 0.5
    ground_tol: float = -0.005  # matches scripts/qpos_retarget.py:161 --tolerance
    float_tol: float = 0.03     # "airborne" threshold for the floating penalty (m)

    # TTSA (proposal.md 4.2)
    lr_upper: float = 1e-5              # vs lr_lower 3e-4 -> 30:1
    upper_every: int = 10               # K; effective timescale ratio 300:1
    upper_warmup_iters: int = 500       # p frozen at 0 until phi is worth measuring
    upper_clip_linf: float = 0.02       # hard per-step cap on ||delta u||_inf
    # Antithetic ES for the T2 / simulator-only terms (proposal.md 4.4).
    es_enabled: bool = False            # Stage 3
    es_sigma: float = 0.02              # pre-tanh
    es_pairs_per_body: int = 2
    es_windows_per_group: int = 6

    # ------------------------------------------------------------ lower level (phi)
    adapter_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 512, 512, 256])
    adapter_alpha: float = 0.05         # was 0.1 in model/config.py:15; halved because z_beta is
                                        # now projected back onto the sphere, so the delta acts
                                        # purely tangentially and a smaller step goes further
    adapter_alpha_learnable: bool = False
    project_z: bool = True              # model/networks.py:36 did NOT project; every stored z has
                                        # norm exactly sqrt(256)=16 and FBModel.project_z enforces
                                        # it, so the old adapter fed the actor off-manifold z
    action_head_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 128])
    wrench_head_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 128])
    value_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [512, 512])

    exploration_std: float = 0.05       # inherited from model/config.py:35 -- that value was
                                        # measured, not guessed; noise compounds over the rollout
    wrench_std: float = 0.05

    # External root wrench (new.md: 30 Hz force on the root so it does not fall).
    # A training crutch, NOT part of the deliverable: annealed to zero, and every
    # reported eval number is taken at f_max = 0 from iteration 1 (proposal.md R5).
    wrench_enabled: bool = True
    wrench_f_frac: float = 0.5          # f_max = this * M * g
    wrench_m_frac: float = 0.5          # m_max = this * f_max * L_leg
    # 5000, not 2000. The BC term finishes annealing at bc_anneal_end and the
    # wrench STARTS annealing at wrench_hold_iters; setting both to 2000 removed
    # the analytic crutch and began removing the physical one on the SAME
    # iteration. Measured on the 2026-08-10 stage-3 run: r_track fell 0.71 ->
    # 0.50 across iterations 1980-2100 and never recovered over the remaining
    # 8000 iterations. Staggering them keeps this stage's own principle -- one
    # new source of difficulty at a time -- inside the stage as well as between
    # stages.
    wrench_hold_iters: int = 5000       # constant until here
    wrench_anneal_end: int = 8000       # cosine to zero by here

    # PPO (proposal.md 5.5)
    lr_lower: float = 3e-4
    lr_lower_final: float = 1e-4        # cosine
    ppo_epochs: int = 4
    ppo_minibatches: int = 4
    ppo_clip: float = 0.2
    value_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0           # the frozen prior already supplies the behaviour prior
    # Measured ||grad|| of the PPO term alone is 8-23 (model/bilevel/ppo.py, 125
    # iterations of stage 1). That is not a pathology -- d logp/d mu is
    # (a - mu)/sigma^2 with sigma = 0.2, summed over 75 action dims -- but a clip
    # of 1.0 against it rescaled EVERY update by ~15x, so the real learning rate
    # was 2e-5 rather than 3e-4 and the clip, not lr_lower, was setting the step
    # size. 10.0 leaves the clip doing what it is for (catching the occasional
    # outlier update that would wipe the frozen prior) instead of acting as a
    # permanent throttle. Was 1.0; 5.0 in model/config.py:45.
    grad_clip_norm: float = 10.0
    gamma: float = 0.97                 # effective horizon 33 steps ~ 1.1s > the 0.8s window
    gamma_warmup: float = 0.95
    gamma_warmup_iters: int = 1000
    gae_lambda: float = 0.95
    value_warmup_iters: int = 100       # value-only; with H=24 half the return is bootstrap
    # Periodic long-rollout diagnostic (proposal.md R6, model/bilevel/longeval.py).
    # Training re-initializes every 24-step window from the reference, so error
    # never accumulates; a long continuous rollout is the only thing that shows
    # whether locally-good motion is globally coherent. Measured on stage 1:
    # 22.3/24 steps survived in training against 22/299 in one continuous
    # rollout. Costs ~2.8 s per call, so under 0.03 s/iter at every=100.
    # Set long_eval_every = 0 to disable.
    long_eval_every: int = 100
    long_eval_clips: int = 8
    long_eval_horizon: int = 299        # 10.0 s @ 30 Hz, same as eval_bilevel.py
    long_eval_wrench: float = 0.0       # always report without the crutch (R5)

    # ------------------------------------------------------- AMP (model/bilevel/amp.py)
    # A STATIONARY replacement for the time-indexed r_track. See amp.py's module
    # docstring for why the time index is the root cause of the compounding
    # error; in short, under a reward that depends on t the optimal policy is
    # pi*(s, t) while this architecture can only represent pi(s), and RSI was
    # silently supplying the missing clock every 24 steps.
    amp_enabled: bool = False
    amp_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [512, 256])
    amp_lr: float = 1e-4
    amp_minibatches: int = 4
    amp_grad_penalty: float = 5.0       # on REAL samples (AMP eq. 9). Drop it and D
                                        # wins in a few hundred iterations, the reward
                                        # pins at 0 and the policy gets nothing.
    amp_norm_momentum: float = 0.99
    # How much of the 0.65 tracking budget r_amp takes over, cosine-ramped from
    # amp_warmup_iters to amp_full_iters. Ramped rather than switched because
    # the time-indexed reward is a far better early teacher -- it just cannot be
    # what we deploy on. Same role as lambda_bc and the wrench: scaffolding.
    amp_weight_final: float = 1.0
    amp_warmup_iters: int = 200
    amp_full_iters: int = 1000
    # With r_amp in charge, the tracking-failure termination is itself a clock:
    # it kills a window for being out of phase. Relaxed rather than removed, so
    # a genuinely diverged rollout still ends.
    amp_term_pose_err: float = 3.0

    adv_norm_momentum: float = 0.99     # per-(clip, body) advantage EMA
    adv_std_floor: float = 0.1          # per-pair std floored at this x the batch std, so the
                                        # per-pair division can rescale by at most ~10x. Without
                                        # a floor a pair whose advantage is momentarily flat gets
                                        # divided by ~1e-3; see ppo.PairAdvantageNormalizer.

    lambda_z: float = 0.1               # anchor on ||z_beta - z0||, now in COSINE form (5.5)
    # Behaviour cloning onto the exact ctrl that commands the reference's next
    # joint angles. Free and zero-variance because ctrl in [-1,1] <=> qpos in
    # jnt_range holds to 4.4e-16 (proposal.md 1.3). MUST anneal to zero: held on
    # it fights the frozen prior and ignores dynamics.
    #
    # 100.0, not 1.0, and the factor is measured rather than tuned. At 1.0 the BC
    # gradient is 0.15 against the PPO term's 8-23, i.e. ~1% of the update
    # direction, and the BC error stayed at 0.75 for 3200 optimizer steps -- worse
    # than emitting zeros (0.31) or the frozen prior unchanged (0.47). The head is
    # not the limit: with nothing competing it fits a_ref to 0.06 in 250 steps and
    # 0.002 in 2000 (scratch probe, 256 windows of `child`). 100 puts the two
    # gradients on the same order so BC leads the first ~500 iterations, which is
    # what stage 1 is for.
    lambda_bc: float = 100.0
    # 4000, not 2000. See wrench_hold_iters: BC was leaving exactly as the
    # wrench began to. This also gives PPO 1000 iterations to hold r_track on
    # its own, with the wrench still fully on, before anything else changes.
    bc_anneal_end: int = 4000

    # --------------------------------------------------------- reward (proposal.md 5.6)
    r_track_weight: float = 0.65
    r_reg_weight: float = 0.15
    r_surv_weight: float = 0.20

    k_pose: float = 2.0
    k_vel: float = 0.005
    k_ee: float = 40.0
    k_root: float = 10.0
    k_com: float = 10.0
    w_pose: float = 0.35
    w_vel: float = 0.15
    w_ee: float = 0.25
    w_root: float = 0.15
    w_com: float = 0.10

    e_act: float = 0.1
    e_smooth: float = 1.0
    e_res: float = 0.5
    e_tau: float = 0.5
    e_ext: float = 8.0          # largest by far: the wrench is the cheapest way to satisfy
                                # every other term, so it has to be the most expensive one
    e_slip: float = 2.0

    # Termination (proposal.md 5.6). humenv's own is_terminated() always returns
    # False, so the worker evaluates this itself -- without it r_surv is a
    # constant and carries no signal at all.
    term_root_height_frac: float = 0.5   # root_z < this * reference root_z
    # Uprightness is judged RELATIVE to the reference's own up-vector, not
    # against an absolute threshold: `up_z > ref_up_z - term_up_margin`. The
    # dataset deliberately contains non-upright tasks -- measured over all 540
    # clips, 21.0% of frames have up_z < 0.2 and 11 of 54 tasks have more than
    # half their frames below it (headstand median -0.970, crawl-0.5-2-u -0.091,
    # lieonground-up -0.020). An absolute threshold terminated the robot for
    # performing a headstand CORRECTLY. Measured cost: driving with the exact
    # a_ref BC target still gave 33/48 terminations, 82% of them from this test,
    # while pose_err never fired at all.
    #
    # 0.8 is not a free parameter: with a perfectly upright reference
    # (ref_up_z = 1.0) it reproduces the old absolute 0.2 exactly, so upright
    # tasks are held to the identical standard and only the inverted ones are
    # re-based. One-sided on purpose -- being more upright than the reference is
    # a tracking error, not a fall, and r_pose already prices it.
    term_up_margin: float = 0.8          # up_z = 2*(qy*qz + qw*qx), see rewards._up_z
    # Root drift, in metres, scaled by L_leg(beta)/L_leg(adult).
    #
    # 0.75, not 0.5, because the root is UNACTUATED (nu=69 vs nv=75): the
    # reference's root path comes from the source body's footfalls rescaled by
    # height, so even joint tracking that is exact leaves the root free to drift.
    # Measured open-loop under the exact a_ref over 150 windows per body, peak
    # drift inside one 0.8 s window is p50 0.192 / p90 0.375 m (child) and
    # p50 0.144 / p90 0.464 m (teen). At 0.5 the threshold for `child` is
    # 0.5 * 0.618 = 0.31 m and this test alone fired on 19% of windows in which
    # the policy had done nothing wrong; at 0.75 (0.46 m for child) that drops to
    # 4% while still being a meaningful failure criterion over 0.8 s.
    #
    # Honest note on what this bought: measured CLOSED loop with the trained
    # stage-1 policy (checkpoint at iter 750, 64 windows), this test fires on 0%
    # of terminations -- a policy with feedback corrects the drift that an
    # open-loop replay cannot, so the open-loop 19% badly over-stated it.
    # term_rate only moved 0.535 -> ~0.46. The change is still right (it stops
    # punishing correct behaviour) but it was not the bottleneck. What actually
    # terminates the trained policy is root_h (30%) and up_z (14%).
    term_root_dist: float = 0.75
    term_pose_err: float = 0.6           # mean squared hinge error; the highest-leverage trick

    # ------------------------------------------------------------------- schedule
    num_iters: int = 10000
    log_every: int = 10
    ckpt_every: int = 250
    eval_every: int = 500
    long_rollout_every: int = 100        # 300-step diagnostic (proposal.md R6)
    ckpt_dir: str = "model/bilevel/checkpoints"
    log_dir: str = "outputs/bilevel_logs"

    # ------------------------------------------------------------------- W&B
    # Off by default and opt-in via --wandb. Two reasons: this machine has no
    # credentials (no ~/.netrc, no WANDB_*), and a logging backend must never be
    # able to kill a 6-hour training run -- see WandbLogger in train_bilevel.py,
    # which degrades to disabled on any failure rather than raising.
    # The .jsonl in log_dir is written regardless and remains the source of truth.
    use_wandb: bool = False
    wandb_project: str = "crossenbodiment-bilevel"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    # Stage 1/2/3 of one pipeline are separate runs (the objective changes at
    # each boundary), so they are grouped to be read as one experiment.
    wandb_group: Optional[str] = None
    wandb_mode: str = "online"          # "offline" queues locally for `wandb sync`

    # ---------------------------------------------------------------- derived
    @property
    def sims_per_worker(self) -> int:
        if self.n_envs % self.n_workers:
            raise ValueError(f"n_envs={self.n_envs} not divisible by n_workers={self.n_workers}")
        return self.n_envs // self.n_workers

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    def wrench_scale(self, it: int) -> float:
        """Cosine anneal of the external-wrench magnitude, 1.0 -> 0.0."""
        import math
        if not self.wrench_enabled:
            return 0.0
        if it <= self.wrench_hold_iters:
            return 1.0
        if it >= self.wrench_anneal_end:
            return 0.0
        frac = (it - self.wrench_hold_iters) / (self.wrench_anneal_end - self.wrench_hold_iters)
        return 0.5 * (1.0 + math.cos(math.pi * frac))

    def bc_scale(self, it: int) -> float:
        import math
        if it >= self.bc_anneal_end:
            return 0.0
        return self.lambda_bc * 0.5 * (1.0 + math.cos(math.pi * it / self.bc_anneal_end))

    def amp_scale(self, it: int) -> float:
        """0 -> amp_weight_final, cosine over [amp_warmup_iters, amp_full_iters]."""
        import math
        if not self.amp_enabled:
            return 0.0
        if it <= self.amp_warmup_iters:
            return 0.0
        if it >= self.amp_full_iters:
            return self.amp_weight_final
        frac = (it - self.amp_warmup_iters) / max(1, self.amp_full_iters - self.amp_warmup_iters)
        return self.amp_weight_final * 0.5 * (1.0 - math.cos(math.pi * frac))

    def gamma_at(self, it: int) -> float:
        return self.gamma_warmup if it < self.gamma_warmup_iters else self.gamma

    def lr_lower_at(self, it: int) -> float:
        import math
        frac = min(1.0, it / max(1, self.num_iters))
        return self.lr_lower_final + 0.5 * (self.lr_lower - self.lr_lower_final) * (
            1.0 + math.cos(math.pi * frac)
        )
