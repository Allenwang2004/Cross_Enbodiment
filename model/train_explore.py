"""Kinematics-only action-network exploration with a fixed, unmodified
standard-body z (no LatentAdapter, no beta conditioning, no retargeted-motion
reference). A stripped-down control experiment against the full adapter
pipeline in model/train.py -- see that file's docstring for the shared
REINFORCE/batching mechanics this reuses unchanged.

Pipeline per update (batched across cfg.batch_size parallel episodes on one
vectorized HumEnv, exactly as in train.py):
    z0 sampled straight from the dataset's origin_z, used UNMODIFIED
      (no adapter(beta, z0) call -- z0 is fed to the frozen actor as-is)
    a_t = ActionResidual(actor(obs_t, z0))    # residual action correction, no beta
    tau = env.step(a_t) for t in 0..T          # rollout on target_xml
      (assets/robots/child/robot_origact.xml: child body geometry/inertia,
      robot.xml's ORIGINAL unscaled actuator gains -- see
      scripts/make_child_origact_xml.py)

    L = lambda_phys * L_phys(tau)

L_phys (model/losses.physics_penalty) is computed purely from the live
rollout's own qpos via forward kinematics -- no retargeted reference, no
qpos_ref, no D term at all. This isolates: given the child's own scaled-down
morphology but full adult actuator strength, and z left completely alone, can
a small residual action network alone learn to keep the rollout physically
plausible (joint limits respected, not collapsing)?

Same REINFORCE-not-backprop reasoning as train.py: MuJoCo's env.step() has no
autodiff support, so L_phys (computed from the resulting qpos trajectory)
can't be backpropagated through -- actions are sampled from a Gaussian around
ActionResidual's (differentiable) mean, log-probabilities accumulated, and
weighted by the realized cost after the episode. Unlike train.py there is no
lambda_z regularizer term here: z is never touched, so there is nothing to
regularize back toward.

Usage (from project root, once datasets/crossenbodiment-1-datasets exists and
scripts/make_child_origact_xml.py has been run):
    uv run model/train_explore.py
"""

import argparse
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

import dataclasses

import mujoco
import numpy as np
import torch
import wandb
from torch.distributions import Normal
from tqdm import tqdm

from humenv import make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel

from model import losses
from model.config_explore import ExploreConfig
from model.dataset import CrossEmbodimentDataset, load_task_list
from model.networks import ActionResidual

REPO_ROOT = Path(__file__).resolve().parent.parent


def rollout_batch_explore(model, action_net, env, z0_t, cfg):
    """z0_t: (B, z_dim), fed to the frozen actor unmodified -- no adapter.
    Steps the vectorized env cfg.steps_per_episode times, all B sub-envs in
    lockstep. Returns the batched qpos trajectory (B, T, nq) and each
    sub-episode's MEAN log-prob (B,) for REINFORCE (see train.py's
    rollout_batch for why mean, not sum)."""
    B = z0_t.shape[0]

    obs, _ = env.reset()
    qpos_hist = []
    log_prob_sum = torch.zeros(B, device=cfg.device)

    for t in range(cfg.steps_per_episode):
        obs_t = torch.tensor(obs["proprio"], dtype=torch.float32, device=cfg.device)

        # Bypass model.act()/model.actor() (both @torch.no_grad()-wrapped) so
        # gradients can reach action_net's params through the frozen actor's
        # activations.
        obs_norm = model._normalize(obs_t)
        dist = model._actor(obs_norm, z0_t, model.cfg.actor_std)
        raw_mean = dist.mean  # (B, action_dim)

        action_mean = action_net(raw_mean)
        action_dist = Normal(action_mean, cfg.exploration_std)
        action = action_dist.sample()
        log_prob_sum = log_prob_sum + action_dist.log_prob(action).sum(dim=-1)

        action_np = action.detach().cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action_np)
        qpos_hist.append(info["qpos"].copy())  # (B, nq)

    action_dim = action_mean.shape[-1]
    log_prob_mean = log_prob_sum / (cfg.steps_per_episode * action_dim)

    qpos_beta = np.stack(qpos_hist, axis=1)  # (B, T, nq)
    return {"qpos_beta": qpos_beta, "log_prob_mean": log_prob_mean}


def compute_batch_cost_explore(fk_model, cfg, qpos_beta):
    """qpos_beta: (B, T, nq) numpy. L_phys is computed per-trajectory via
    forward kinematics (not batched), looped here since it's cheap vs
    simulation. No D/qpos_ref -- kinematics-only. Returns per-item cost/L_phys
    arrays (shape (B,)) and a list of each item's unweighted term breakdown
    (limit/fall/com_support/foot_slide/penetrate/smooth -- see
    losses.physics_penalty) for logging."""
    B = qpos_beta.shape[0]
    costs = np.empty(B, dtype=np.float32)
    l_physes = np.empty(B, dtype=np.float32)
    terms_list = []

    for i in range(B):
        l_phys, terms = losses.physics_penalty(fk_model, qpos_beta[i])
        costs[i] = cfg.lambda_phys * l_phys
        l_physes[i] = l_phys
        terms_list.append(terms)
    return costs, l_physes, terms_list


def plot_loss_curve_explore(loss_history, l_phys_history, out_path):
    """total_loss is advantage-weighted log-prob, not a progress signal for
    REINFORCE (same caveat as train.py) -- watch L_phys instead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(loss_history)
    axes[0].set_ylabel("total_loss")
    axes[0].set_title("pg_loss (surrogate, NOT expected to trend down -- see L_phys below)")
    axes[1].plot(l_phys_history, color="tab:green")
    axes[1].set_ylabel("L_phys (mean)")
    axes[1].set_xlabel("update")
    axes[1].set_title("physics-feasibility penalty -- this SHOULD trend down")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"wrote training curves -> {out_path}")


def train_explore(cfg: ExploreConfig, task_filter=None):
    """task_filter: optional iterable of reward_name strings. When given
    explicitly (e.g. a single-task debug run, see main()'s --task-name), used
    as-is and the train_tasks.txt split is skipped. When None (default),
    falls back to the existing train_tasks.txt-derived filter (or no filter
    if that file doesn't exist), unchanged from prior behavior."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if task_filter is None:
        train_tasks_path = REPO_ROOT / cfg.dataset_dir / "splits" / "train_tasks.txt"
        if train_tasks_path.exists():
            task_filter = load_task_list(train_tasks_path)
    # beta/qpos_ref fields on each row are ignored -- only z0 is used.
    dataset = CrossEmbodimentDataset(REPO_ROOT / cfg.dataset_dir, task_filter=task_filter)
    print(f"dataset: {len(dataset)} rows"
          + (f" (task_filter={list(task_filter)})" if task_filter is not None else " (no task_filter)"))

    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(cfg.device)
    model.eval()  # frozen: FBModel.__init__ already sets requires_grad_(False)

    env, _ = make_humenv(
        num_envs=cfg.batch_size,
        vectorization_mode=cfg.vectorization_mode,
        task=None,
        xml=str(REPO_ROOT / cfg.target_xml),
        state_init="Default",
    )
    # Separate MjModel just for L_phys forward-kinematics (numpy, no grad) --
    # same skeleton as the env's own model, loaded standalone so it works
    # whether env is a sync or async VectorEnv.
    fk_model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / cfg.target_xml))

    action_net = ActionResidual(
        action_dim=model.cfg.action_dim,
        hidden_dims=cfg.action_hidden_dims,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(action_net.parameters(), lr=cfg.lr)

    # EMA-tracked mean/std of the cost, used to standardize the REINFORCE
    # advantage each update (see train.py's config for why both mean and std).
    baseline_mean = None
    baseline_sqmean = None

    ckpt_dir = REPO_ROOT / cfg.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=dataclasses.asdict(cfg))

    loss_history = []
    l_phys_history = []

    pbar = tqdm(range(cfg.num_updates), desc="train_explore")
    for update in pbar:
        optimizer.zero_grad()

        idx = [random.randrange(len(dataset)) for _ in range(cfg.batch_size)]
        samples = [dataset[i] for i in idx]
        z0_t = torch.tensor(np.stack([s["z0"] for s in samples]), dtype=torch.float32, device=cfg.device)

        episode = rollout_batch_explore(model, action_net, env, z0_t, cfg)
        costs_np, l_physes, terms_list = compute_batch_cost_explore(fk_model, cfg, episode["qpos_beta"])
        costs_t = torch.tensor(costs_np, dtype=torch.float32, device=cfg.device)
        mean_terms = {k: float(np.mean([t[k] for t in terms_list])) for k in terms_list[0]}

        if baseline_mean is None:
            baseline_mean = costs_t.mean()
            baseline_sqmean = (costs_t ** 2).mean()
        baseline_std = (baseline_sqmean - baseline_mean ** 2).clamp(min=1e-6).sqrt()
        advantage = (costs_t - baseline_mean) / (baseline_std + 1e-6)
        baseline_mean = cfg.baseline_momentum * baseline_mean + (1 - cfg.baseline_momentum) * costs_t.mean()
        baseline_sqmean = cfg.baseline_momentum * baseline_sqmean + (1 - cfg.baseline_momentum) * (costs_t ** 2).mean()

        pg_loss = (episode["log_prob_mean"] * advantage.detach()).mean()
        total_loss = pg_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(action_net.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        loss_history.append(total_loss.item())
        l_phys_history.append(float(l_physes.mean()))
        pbar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            L_phys=f"{l_physes.mean():.4f}",
        )

        if cfg.use_wandb:
            wandb.log(
                {
                    "total_loss": total_loss.item(),
                    "L_phys": float(l_physes.mean()),
                    "advantage_mean": advantage.mean().item(),
                    "baseline_mean": baseline_mean.item(),
                    "baseline_std": baseline_std.item(),
                    **{f"terms/{k}": v for k, v in mean_terms.items()},
                },
                step=update,
            )

        if update % cfg.log_every == 0:
            terms_str = " ".join(f"{k}={v:.4f}" for k, v in mean_terms.items())
            tqdm.write(
                f"[{update:04d}/{cfg.num_updates}] total_loss={total_loss.item():.4f} "
                f"L_phys={l_physes.mean():.4f} ({terms_str})"
            )

        if (update + 1) % cfg.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"update_{update + 1:05d}.pt"
            torch.save(
                {"action_net": action_net.state_dict(), "update": update + 1, "cfg": cfg},
                ckpt_path,
            )
            tqdm.write(f"saved checkpoint -> {ckpt_path}")

    env.close()

    plot_loss_curve_explore(loss_history, l_phys_history, REPO_ROOT / cfg.loss_curve_path)

    if cfg.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default=None,
                         help="restrict training to a single reward_name's trials "
                              "(debug mode -- isolates cross-task L_phys-scale noise from "
                              "actual learning signal, see model/diagnose_single_task.py "
                              "for the same idea applied to the full adapter pipeline). "
                              "Omit to use the full train_tasks.txt split (default).")
    parser.add_argument("--num-updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    cfg = ExploreConfig()
    if args.num_updates is not None:
        cfg.num_updates = args.num_updates
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if args.no_wandb:
        cfg.use_wandb = False
    if args.wandb_run_name is not None:
        cfg.wandb_run_name = args.wandb_run_name

    task_filter = [args.task_name] if args.task_name is not None else None
    train_explore(cfg, task_filter=task_filter)


if __name__ == "__main__":
    main()
