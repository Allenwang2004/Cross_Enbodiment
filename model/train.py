"""Single-body cross-embodiment adapter training (model.md's "單一身材訓練流程").

Pipeline per episode:
    z_beta = G_theta(beta, z0)                      # LatentAdapter
    a_t    = ActionHead(actor(obs_t, z_beta), beta)  # residual action correction
    tau_beta = env.step(a_t) for t in 0..T           # rollout on the target body

    L = -R_task(tau_beta) + lambda_rtg * D(tau_beta, tau_beta_ref)
        + lambda_z * ||z_beta - z0||^2 + lambda_phys * L_phys(tau_beta)

Why REINFORCE, not backprop-through-the-rollout
-------------------------------------------------
model._actor() itself IS differentiable w.r.t. z (verified: model.act()/
model.actor() are wrapped in @torch.no_grad(), but calling model._actor()
and model._normalize() directly is not -- gradients flow through the frozen
actor's activations into z just fine, its *weights* are just frozen).

The blocker is MuJoCo: env.step() runs physics (contacts, integration) with
no autodiff support, so R_task/D/L_phys -- all computed from the resulting
qpos trajectory -- cannot be backpropagated through. We use a score-function
(REINFORCE) estimator instead: sample actions from a Gaussian around the
(differentiable) mean, accumulate log-probabilities, and after the episode
weight them by the realized cost. This only needs episode-level returns, not
a differentiable simulator.

The lambda_z * ||z_beta - z0||^2 term is the one exception: it never touches
the environment, so it's added as a normal backprop term for a much
lower-variance gradient on that part of the loss. Both paths write into the
same z_beta leaf tensor and their gradients simply add.

qpos_ref (retargeted_motion) is produced by scripts/qpos_retarget.py, which
retargets each origin_motion trajectory directly onto the same
robot_<label>.xml skeleton used for the live rollout (target_xml in
config.py) -- so D() is comparing two trajectories on the same bone lengths,
as intended. That script's own docstring covers its retargeting assumptions
(same body/joint topology across robot.xml variants, root-height rescaling
only, no per-limb IK correction).

Usage (from project root, once datasets/crossenbodiment-1-datasets exists):
    uv run model/train.py
"""

import os
import random
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from torch.distributions import Normal

from humenv import make_humenv
from humenv.env import make_from_name
from metamotivo.fb_cpr.huggingface import FBcprModel

from model import losses
from model.config import TrainConfig
from model.dataset import CrossEmbodimentDataset, load_beta, load_task_list
from model.networks import ActionHead, LatentAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent


def rollout_episode(model, adapter, action_head, env, reward_fn, z0_t, beta_t, cfg):
    """Runs one episode on the target body, returns everything needed to
    compute the loss: the qpos trajectory, summed task reward, physics
    penalty, and the summed log-prob (for the REINFORCE term)."""
    z_beta = adapter(beta_t, z0_t)  # (1, z_dim), differentiable

    obs, _ = env.reset()
    qpos_hist = []
    log_prob_sum = torch.zeros((), device=cfg.device)
    r_task = 0.0

    for t in range(cfg.steps_per_episode):
        obs_t = torch.tensor(obs["proprio"], dtype=torch.float32, device=cfg.device).unsqueeze(0)

        # Bypass model.act()/model.actor() (both @torch.no_grad()-wrapped) so
        # gradients can reach z_beta through the frozen actor's activations.
        obs_norm = model._normalize(obs_t)
        dist = model._actor(obs_norm, z_beta, model.cfg.actor_std)
        raw_mean = dist.mean  # (1, action_dim), differentiable via z_beta

        action_mean = action_head(raw_mean, beta_t)
        action_dist = Normal(action_mean, model.cfg.actor_std)
        action = action_dist.sample()
        log_prob_sum = log_prob_sum + action_dist.log_prob(action).sum()

        action_np = action.detach().cpu().numpy().ravel()
        obs, _, terminated, truncated, info = env.step(action_np)
        qpos_hist.append(info["qpos"].copy())

        r_task += reward_fn(
            env.unwrapped.model,
            qpos=info["qpos"],
            qvel=info["qvel"],
            ctrl=action_np,
        )

        if terminated or truncated:
            obs, _ = env.reset()

    qpos_beta = np.stack(qpos_hist)
    return {
        "z_beta": z_beta,
        "qpos_beta": qpos_beta,
        "r_task": r_task,
        "log_prob_sum": log_prob_sum,
    }


def compute_episode_cost(env, cfg, episode, qpos_ref):
    """Everything except the lambda_z term (that one's added separately in
    the training loop as a direct backprop term, see module docstring)."""
    d_weights = {
        "root": cfg.d_root_weight,
        "ee": cfg.d_ee_weight,
        "contact": cfg.d_contact_weight,
        "pose": cfg.d_pose_weight,
        "velocity": cfg.d_velocity_weight,
    }
    d_total, d_terms = losses.functional_equivalence(
        env.unwrapped.model, episode["qpos_beta"], qpos_ref, d_weights
    )
    l_phys = losses.physics_penalty(env.unwrapped.model, episode["qpos_beta"])

    cost = -episode["r_task"] + cfg.lambda_rtg * d_total + cfg.lambda_phys * l_phys
    return cost, {"r_task": episode["r_task"], "d_total": d_total, "l_phys": l_phys, **d_terms}


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
        num_envs=1,
        task=None,
        xml=str(REPO_ROOT / cfg.target_xml),
        state_init="Default",
    )

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

    baseline = None  # EMA baseline for REINFORCE variance reduction

    ckpt_dir = REPO_ROOT / cfg.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for update in range(cfg.num_updates):
        optimizer.zero_grad()

        costs, r_tasks, d_totals, l_physes = [], [], [], []
        pg_loss = torch.zeros((), device=cfg.device)
        z_reg_loss = torch.zeros((), device=cfg.device)

        for _ in range(cfg.episodes_per_update):
            sample = dataset[random.randrange(len(dataset))]
            reward_fn = make_from_name(sample["reward_name"])
            z0_t = torch.tensor(sample["z0"], dtype=torch.float32, device=cfg.device).unsqueeze(0)
            beta_t = torch.tensor(sample["beta"], dtype=torch.float32, device=cfg.device).unsqueeze(0)

            episode = rollout_episode(model, adapter, action_head, env, reward_fn, z0_t, beta_t, cfg)
            cost, info = compute_episode_cost(env, cfg, episode, sample["qpos_ref"])

            if baseline is None:
                baseline = cost
            advantage = cost - baseline
            baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * cost

            pg_loss = pg_loss + episode["log_prob_sum"] * advantage
            z_reg_loss = z_reg_loss + ((episode["z_beta"] - z0_t) ** 2).mean()

            costs.append(cost)
            r_tasks.append(info["r_task"])
            d_totals.append(info["d_total"])
            l_physes.append(info["l_phys"])

        pg_loss = pg_loss / cfg.episodes_per_update
        z_reg_loss = cfg.lambda_z * (z_reg_loss / cfg.episodes_per_update)
        total_loss = pg_loss + z_reg_loss

        total_loss.backward()
        optimizer.step()

        if update % cfg.log_every == 0:
            print(
                f"[{update:04d}/{cfg.num_updates}] "
                f"cost={np.mean(costs):.4f} r_task={np.mean(r_tasks):.4f} "
                f"D={np.mean(d_totals):.4f} L_phys={np.mean(l_physes):.4f} "
                f"pg_loss={pg_loss.item():.4f} z_reg={z_reg_loss.item():.4f}"
            )

        if (update + 1) % cfg.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"update_{update + 1:05d}.pt"
            torch.save(
                {"adapter": adapter.state_dict(), "action_head": action_head.state_dict(),
                 "update": update + 1, "cfg": cfg},
                ckpt_path,
            )
            print(f"saved checkpoint -> {ckpt_path}")

    env.close()


if __name__ == "__main__":
    train(TrainConfig())
