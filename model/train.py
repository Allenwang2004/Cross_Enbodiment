"""Single-body cross-embodiment adapter training (model.md's "單一身材訓練流程").

Pipeline per update (batched across `cfg.batch_size` parallel episodes on one
vectorized HumEnv, see "Batching" below):
    z_beta = G_theta(beta, z0)                      # LatentAdapter
    a_t    = ActionHead(actor(obs_t, z_beta), beta)  # residual action correction
    tau_beta = env.step(a_t) for t in 0..T           # rollout on the target body

    L = lambda_rtg * D(tau_beta, tau_beta_ref)
        + lambda_z * ||z_beta - z0||^2 + lambda_phys * L_phys(tau_beta)

No R_task term
---------------
Earlier versions included -R_task (the humenv reward for the sampled task).
Removed: humenv's reward functions are written against the ORIGINAL body's
kinematics/actuator strength (e.g. move-ego's reward integrates an ego-frame
velocity computed for the source skeleton's proportions) -- exactly what the
LatentAdapter is changing. Optimizing directly against R_task on the target
body either rewards a number that no longer means "did the task" once the
body has changed, or implicitly pressures the adapter to fight the
morphology change to look more like the source body again, defeating the
point of adapting at all. D (functional equivalence vs the retargeted
reference motion) and L_phys (feasibility) don't have this problem -- both
are computed straight from the target skeleton's own forward kinematics, not
from a reward tuned for a different body. (R_task is still reported as a
diagnostic in model/evaluate.py and model/baseline.py -- just not optimized.)

Why REINFORCE, not backprop-through-the-rollout
-------------------------------------------------
model._actor() itself IS differentiable w.r.t. z (verified: model.act()/
model.actor() are wrapped in @torch.no_grad(), but calling model._actor()
and model._normalize() directly is not -- gradients flow through the frozen
actor's activations into z just fine, its *weights* are just frozen).

The blocker is MuJoCo: env.step() runs physics (contacts, integration) with
no autodiff support, so D/L_phys -- both computed from the resulting qpos
trajectory -- cannot be backpropagated through. We use a score-function
(REINFORCE) estimator instead: sample actions from a Gaussian around the
(differentiable) mean, accumulate log-probabilities, and after the episode
weight them by the realized cost. This only needs episode-level returns, not
a differentiable simulator.

The lambda_z * ||z_beta - z0||^2 term is the one exception: it never touches
the environment, so it's added as a normal backprop term for a much
lower-variance gradient on that part of the loss. Both paths write into the
same z_beta leaf tensor and their gradients simply add.

Exploration noise
------------------
Actions are sampled from Normal(action_head_output, cfg.exploration_std),
NOT the frozen model's own actor_std (0.2) -- that std is tuned for
single-step inference-time behavior, but here it's injected every one of
300 steps into a MuJoCo rollout, where per-step noise compounds nonlinearly
through contacts/integration. Empirically (50 updates, batch_size=16) this
made D/L_phys swing by an order of magnitude update to update with no
visible trend, i.e. the exploration noise's effect on the trajectory was
larger than the effect of the adapter actually changing -- the learning
signal was buried under it. cfg.exploration_std defaults much smaller (0.05).

Batching
--------
All cfg.batch_size episodes run in ONE vectorized HumEnv (gymnasium
VectorEnv under the hood, `cfg.vectorization_mode` picks sync vs async)
stepped in lockstep: every timestep is a single batched forward pass through
the frozen actor + adapter + action_head (batch dim = cfg.batch_size) and a
single env.step() call, not a Python loop over individual episodes. D/
L_phys still need per-trajectory forward kinematics (functional_equivalence/
physics_penalty operate on one qpos array at a time, not vectorized), so
those stay a short per-item Python loop over the collected (B, T, nq)
qpos_beta after the rollout finishes -- cheap relative to simulation, which
is now batched. Relies on the vector env's default auto-reset (a sub-episode
ending early mid-rollout just resets that slot and keeps stepping; we don't
special-case it since cfg.steps_per_episode is short enough that this is rare
and losing part of one sub-episode to a reset boundary doesn't bias the batch).

qpos_ref (retargeted_motion) is produced by scripts/qpos_retarget.py, which
retargets each origin_motion trajectory directly onto the same
robot_<label>.xml skeleton used for the live rollout (target_xml in
config.py) -- so D() is comparing two trajectories on the same bone lengths,
as intended.

Usage (from project root, once datasets/crossenbodiment-1-datasets exists):
    uv run model/train.py
    uv run model/run_train.py --gpu 1
"""

import os
import random
from pathlib import Path
import sys

PARENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from torch.distributions import Normal
from tqdm import tqdm

from humenv import make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel

from model import losses
from model.config import TrainConfig
from model.dataset import CrossEmbodimentDataset, load_beta, load_task_list
from model.networks import ActionHead, LatentAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent


def rollout_batch(model, adapter, action_head, env, z0_t, beta_t, cfg):
    """z0_t/beta_t: (B, z_dim)/(B, beta_dim). Steps the vectorized env
    cfg.steps_per_episode times, all B sub-envs in lockstep. Returns the
    batched qpos trajectory (B, T, nq) and each sub-episode's MEAN log-prob
    (B,) for REINFORCE -- averaged over steps*action_dim (~20k terms for the
    defaults) rather than summed, so its magnitude stays O(1) instead of
    O(1e4). This is just a constant rescale of the score-function gradient
    (same direction, smaller step), which matters a lot in practice: an
    unscaled sum this large completely swamps the advantage's actual signal
    with the score function's own magnitude, making updates dominated by
    noise rather than the cost."""
    B = z0_t.shape[0]
    z_beta = adapter(beta_t, z0_t)  # (B, z_dim), differentiable

    obs, _ = env.reset()
    qpos_hist = []
    log_prob_sum = torch.zeros(B, device=cfg.device)

    for t in range(cfg.steps_per_episode):
        obs_t = torch.tensor(obs["proprio"], dtype=torch.float32, device=cfg.device)

        # Bypass model.act()/model.actor() (both @torch.no_grad()-wrapped) so
        # gradients can reach z_beta through the frozen actor's activations.
        obs_norm = model._normalize(obs_t)
        dist = model._actor(obs_norm, z_beta, model.cfg.actor_std)
        raw_mean = dist.mean  # (B, action_dim), differentiable via z_beta

        action_mean = action_head(raw_mean, beta_t)
        # cfg.exploration_std, not model.cfg.actor_std -- see config.py's comment:
        # this noise physically compounds over the rollout, so it's kept much
        # smaller than the frozen model's own (inference-time) actor_std.
        action_dist = Normal(action_mean, cfg.exploration_std)
        action = action_dist.sample()
        log_prob_sum = log_prob_sum + action_dist.log_prob(action).sum(dim=-1)

        action_np = action.detach().cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action_np)
        qpos_hist.append(info["qpos"].copy())  # (B, nq)

    action_dim = action_mean.shape[-1]
    log_prob_mean = log_prob_sum / (cfg.steps_per_episode * action_dim)

    qpos_beta = np.stack(qpos_hist, axis=1)  # (B, T, nq)
    return {"z_beta": z_beta, "qpos_beta": qpos_beta, "log_prob_mean": log_prob_mean}


def compute_batch_cost(fk_model, cfg, qpos_beta, qpos_refs):
    """qpos_beta: (B, T, nq) numpy. qpos_refs: length-B list, entries may be
    None (see losses.functional_equivalence). D/L_phys use forward kinematics
    per-trajectory (not batched), looped here since it's cheap vs simulation.
    Returns per-item cost/D/L_phys arrays, shape (B,)."""
    B = qpos_beta.shape[0]
    costs = np.empty(B, dtype=np.float32)
    d_totals = np.empty(B, dtype=np.float32)
    l_physes = np.empty(B, dtype=np.float32)

    d_weights = {
        "root": cfg.d_root_weight,
        "ee": cfg.d_ee_weight,
        "contact": cfg.d_contact_weight,
        "pose": cfg.d_pose_weight,
        "velocity": cfg.d_velocity_weight,
    }
    for i in range(B):
        d_total, _ = losses.functional_equivalence(fk_model, qpos_beta[i], qpos_refs[i], d_weights)
        l_phys = losses.physics_penalty(fk_model, qpos_beta[i])
        costs[i] = cfg.lambda_rtg * d_total + cfg.lambda_phys * l_phys
        d_totals[i] = d_total
        l_physes[i] = l_phys
    return costs, d_totals, l_physes


def plot_loss_curve(loss_history, d_history, l_phys_history, out_path):
    """total_loss (pg_loss + z_reg_loss) is NOT a progress signal for
    REINFORCE -- it's advantage-weighted log-prob, and advantage is centered
    by construction (cost minus a running mean), so its expected value
    oscillates around 0 whether or not the policy is improving. The actual
    thing to watch is D / L_phys (the real cost being optimized), so those
    get their own panels here rather than only total_loss."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(loss_history)
    axes[0].set_ylabel("total_loss")
    axes[0].set_title("pg_loss + z_reg_loss (surrogate, NOT expected to trend down -- see D/L_phys below)")
    axes[1].plot(d_history, color="tab:orange")
    axes[1].set_ylabel("D (mean)")
    axes[1].set_title("functional-equivalence cost vs retargeted reference -- this SHOULD trend down")
    axes[2].plot(l_phys_history, color="tab:green")
    axes[2].set_ylabel("L_phys (mean)")
    axes[2].set_xlabel("update")
    axes[2].set_title("physics-feasibility penalty -- this SHOULD trend down")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"wrote training curves -> {out_path}")


def train(cfg: TrainConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_tasks = None
    train_tasks_path = REPO_ROOT / cfg.dataset_dir / "splits" / "train_tasks.txt"
    if train_tasks_path.exists():
        train_tasks = load_task_list(train_tasks_path)
    dataset = CrossEmbodimentDataset(REPO_ROOT / cfg.dataset_dir, task_filter=train_tasks)
    beta_dim = len(load_beta(REPO_ROOT / cfg.target_morphology_json))

    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(cfg.device)
    model.eval()  # frozen: FBModel.__init__ already sets requires_grad_(False)

    env, _ = make_humenv(
        num_envs=cfg.batch_size,
        vectorization_mode=cfg.vectorization_mode,
        task=None,
        xml=str(REPO_ROOT / cfg.target_xml),
        state_init="Default",
    )
    # Separate MjModel just for D/L_phys forward-kinematics (numpy, no grad) --
    # same skeleton as the env's own model, loaded standalone so it works
    # whether env is a sync or async VectorEnv.
    fk_model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / cfg.target_xml))

    adapter = LatentAdapter(
        beta_dim=beta_dim,
        z_dim=model.cfg.archi.z_dim,
        hidden_dims=cfg.adapter_hidden_dims,
        alpha=cfg.adapter_alpha,
        alpha_learnable=cfg.adapter_alpha_learnable,
    ).to(cfg.device)
    action_head = ActionHead(
        action_dim=model.cfg.action_dim,
        beta_dim=beta_dim,
        hidden_dims=cfg.action_head_hidden_dims,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(
        list(adapter.parameters()) + list(action_head.parameters()), lr=cfg.lr
    )

    # EMA-tracked mean/std of the cost, used to standardize the REINFORCE
    # advantage each update: (cost - mean) / std. Centering alone (a plain
    # scalar baseline) isn't enough here -- cost's own scale drifts a lot
    # update to update (batch_size=4 is a small, noisy sample), so without
    # also normalizing by std the gradient magnitude swings wildly and
    # training doesn't converge. Both are tracked as slow EMAs (not raw
    # per-batch stats) since batch_size=4 alone is too small a sample to
    # trust for either estimate.
    baseline_mean = None
    baseline_sqmean = None

    ckpt_dir = REPO_ROOT / cfg.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loss_history = []
    d_history = []
    l_phys_history = []

    pbar = tqdm(range(cfg.num_updates), desc="train")
    for update in pbar:
        optimizer.zero_grad()

        idx = [random.randrange(len(dataset)) for _ in range(cfg.batch_size)]
        samples = [dataset[i] for i in idx]
        z0_t = torch.tensor(np.stack([s["z0"] for s in samples]), dtype=torch.float32, device=cfg.device)
        beta_t = torch.tensor(np.stack([s["beta"] for s in samples]), dtype=torch.float32, device=cfg.device)
        qpos_refs = [s["qpos_ref"] for s in samples]

        episode = rollout_batch(model, adapter, action_head, env, z0_t, beta_t, cfg)
        costs_np, d_totals, l_physes = compute_batch_cost(fk_model, cfg, episode["qpos_beta"], qpos_refs)
        costs_t = torch.tensor(costs_np, dtype=torch.float32, device=cfg.device)

        if baseline_mean is None:
            baseline_mean = costs_t.mean()
            baseline_sqmean = (costs_t ** 2).mean()
        baseline_std = (baseline_sqmean - baseline_mean ** 2).clamp(min=1e-6).sqrt()
        advantage = (costs_t - baseline_mean) / (baseline_std + 1e-6)
        baseline_mean = cfg.baseline_momentum * baseline_mean + (1 - cfg.baseline_momentum) * costs_t.mean()
        baseline_sqmean = cfg.baseline_momentum * baseline_sqmean + (1 - cfg.baseline_momentum) * (costs_t ** 2).mean()

        pg_loss = (episode["log_prob_mean"] * advantage.detach()).mean()
        z_reg_loss = cfg.lambda_z * ((episode["z_beta"] - z0_t) ** 2).mean()
        total_loss = pg_loss + z_reg_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(adapter.parameters()) + list(action_head.parameters()), cfg.grad_clip_norm
        )
        optimizer.step()

        loss_history.append(total_loss.item())
        d_history.append(float(d_totals.mean()))
        l_phys_history.append(float(l_physes.mean()))
        pbar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            D=f"{d_totals.mean():.4f}",
            L_phys=f"{l_physes.mean():.4f}",
        )

        if update % cfg.log_every == 0:
            tqdm.write(
                f"[{update:04d}/{cfg.num_updates}] total_loss={total_loss.item():.4f} "
                f"pg_loss={pg_loss.item():.4f} z_reg={z_reg_loss.item():.4f} "
                f"D={d_totals.mean():.4f} L_phys={l_physes.mean():.4f}"
            )

        if (update + 1) % cfg.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"update_{update + 1:05d}.pt"
            torch.save(
                {"adapter": adapter.state_dict(), "action_head": action_head.state_dict(),
                 "update": update + 1, "cfg": cfg},
                ckpt_path,
            )
            tqdm.write(f"saved checkpoint -> {ckpt_path}")

    env.close()

    plot_loss_curve(loss_history, d_history, l_phys_history, REPO_ROOT / cfg.loss_curve_path)


if __name__ == "__main__":
    train(TrainConfig())
