"""Diagnostic: is the adapter/action_head actually learning anything, or is
the D/L_phys noise seen in normal training (see outputs/train_logs/loss_curve.png)
coming entirely from a different confound -- a NEW random set of tasks being
sampled every update?

Normal training draws cfg.batch_size fresh random (z0, beta, qpos_ref) rows
every update. Since the 43 train tasks likely have very different intrinsic D
scales (a standing pose task is probably much easier to match than crawl or
headstand), the batch-mean D swinging an order of magnitude update to update
could just reflect "which tasks got sampled this time", with the actual
policy-quality signal buried underneath.

This script removes that confound: it repeats a SINGLE fixed dataset row
cfg.batch_size times every update (all sub-envs run the exact same z0/beta/
qpos_ref -- the only thing that differs between sub-envs is the stochastic
action noise). If D trends down here, task-composition noise was the real
problem in normal training. If it still doesn't move, the issue is deeper
(e.g. reward/cost signal too weak, lr, or REINFORCE variance itself).

Usage (from project root):
    uv run model/diagnose_single_task.py
    uv run model/diagnose_single_task.py --task-idx 0 --num-updates 100 --gpu 0
"""

import argparse
import os
import random
import sys
from pathlib import Path

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from humenv import make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel

from model.config import TrainConfig
from model.dataset import CrossEmbodimentDataset, load_beta, load_task_list
from model.networks import ActionHead, LatentAdapter
from model.train import compute_batch_cost, plot_loss_curve, rollout_batch

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-idx", type=int, default=0, help="dataset row index to repeat every update")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-updates", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", default="outputs/train_logs/diagnose_single_task.png")
    args = parser.parse_args()

    cfg = TrainConfig(device=f"cuda:{args.gpu}", batch_size=args.batch_size, num_updates=args.num_updates)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_tasks_path = REPO_ROOT / cfg.dataset_dir / "splits" / "train_tasks.txt"
    train_tasks = load_task_list(train_tasks_path) if train_tasks_path.exists() else None
    dataset = CrossEmbodimentDataset(REPO_ROOT / cfg.dataset_dir, task_filter=train_tasks)
    beta_dim = len(load_beta(REPO_ROOT / cfg.target_morphology_json))

    sample = dataset[args.task_idx]
    print(f"repeating fixed task: {sample['reward_name']} trial {sample['trial']} "
          f"(has qpos_ref: {sample['qpos_ref'] is not None})")

    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(cfg.device)
    model.eval()

    env, _ = make_humenv(
        num_envs=cfg.batch_size, vectorization_mode=cfg.vectorization_mode,
        task=None, xml=str(REPO_ROOT / cfg.target_xml), state_init="Default",
    )
    fk_model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / cfg.target_xml))

    adapter = LatentAdapter(
        beta_dim=beta_dim, z_dim=model.cfg.archi.z_dim,
        hidden_dims=cfg.adapter_hidden_dims, alpha=cfg.adapter_alpha,
        alpha_learnable=cfg.adapter_alpha_learnable,
    ).to(cfg.device)
    action_head = ActionHead(
        action_dim=model.cfg.action_dim, beta_dim=beta_dim,
        hidden_dims=cfg.action_head_hidden_dims,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(list(adapter.parameters()) + list(action_head.parameters()), lr=cfg.lr)

    z0_t = torch.tensor(sample["z0"], dtype=torch.float32, device=cfg.device).unsqueeze(0).repeat(cfg.batch_size, 1)
    beta_t = torch.tensor(sample["beta"], dtype=torch.float32, device=cfg.device).unsqueeze(0).repeat(cfg.batch_size, 1)
    qpos_refs = [sample["qpos_ref"]] * cfg.batch_size

    baseline_mean = None
    baseline_sqmean = None
    loss_history, d_history, l_phys_history = [], [], []

    pbar = tqdm(range(cfg.num_updates), desc="diagnose")
    for update in pbar:
        optimizer.zero_grad()

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
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) + list(action_head.parameters()), cfg.grad_clip_norm)
        optimizer.step()

        loss_history.append(total_loss.item())
        d_history.append(float(d_totals.mean()))
        l_phys_history.append(float(l_physes.mean()))
        pbar.set_postfix(D=f"{d_totals.mean():.4f}", L_phys=f"{l_physes.mean():.4f}")

        if update % 10 == 0:
            tqdm.write(f"[{update:04d}/{cfg.num_updates}] D={d_totals.mean():.4f} L_phys={l_physes.mean():.4f}")

    env.close()
    plot_loss_curve(loss_history, d_history, l_phys_history, REPO_ROOT / args.out)

    first10 = np.mean(d_history[:10])
    last10 = np.mean(d_history[-10:])
    print(f"\nD mean, first 10 updates: {first10:.4f}")
    print(f"D mean, last 10 updates:  {last10:.4f}")
    print(f"{'IMPROVED' if last10 < first10 else 'DID NOT IMPROVE'} ({(1 - last10/first10)*100:+.1f}%)")


if __name__ == "__main__":
    main()
