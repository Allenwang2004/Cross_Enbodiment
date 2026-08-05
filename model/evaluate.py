"""Evaluate a trained LatentAdapter + ActionHead checkpoint across a task
set -- by default the held-out test split (datasets/crossenbodiment-1-datasets/
splits/test_tasks.txt, see scripts/split_tasks.py) so scores reflect
generalization to tasks the adapter never saw during training, not just
tasks it memorized. Falls back to the full official task list
(docs/humenv_all_tasks_official.txt) with a warning if no split exists yet.

Unlike train.py (which samples actions for REINFORCE), this rolls out
deterministically (mean action, no ActionHead-induced noise) since eval
should be reproducible.

For each task: R_task (task reward), D (vs the retargeted reference motion,
skipped if a row has no retargeted_motion), L_phys -- reported per-task and
aggregated, plus optional per-task comparison videos (target-body rollout
side by side with its retargeted reference).

Usage (from project root):
    uv run model/evaluate.py --checkpoint model/checkpoints/update_00200.pt
    uv run model/evaluate.py --checkpoint ... --render-videos --out-dir outputs/eval
    uv run model/evaluate.py --checkpoint ... --tasks-file docs/humenv_all_tasks_official.txt
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

# Running this file directly (not `-m model.evaluate`) puts model/ itself on
# sys.path, not its parent -- see the same fix in run_train.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from humenv import make_humenv
from humenv.env import make_from_name
from metamotivo.fb_cpr.huggingface import FBcprModel

from model import losses
from model.config import TrainConfig
from model.dataset import BETA_AXES, CrossEmbodimentDataset, load_beta, load_task_list
from model.networks import ActionHead, LatentAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_TASKS = REPO_ROOT / "datasets" / "crossenbodiment-1-datasets" / "splits" / "test_tasks.txt"
OFFICIAL_TASKS = REPO_ROOT / "docs" / "humenv_all_tasks_official.txt"


def rollout_deterministic(model, adapter, action_head, env, reward_fn, z0_t, beta_t, cfg, record_video=False):
    z_beta = adapter(beta_t, z0_t)

    obs, _ = env.reset()
    qpos_hist = []
    frames = [] if record_video else None
    r_task = 0.0

    with torch.no_grad():
        for t in range(cfg.steps_per_episode):
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32, device=cfg.device).unsqueeze(0)
            obs_norm = model._normalize(obs_t)
            dist = model._actor(obs_norm, z_beta, model.cfg.actor_std)
            action = action_head(dist.mean, beta_t)
            action_np = action.cpu().numpy().ravel()

            obs, _, terminated, truncated, info = env.step(action_np)
            qpos_hist.append(info["qpos"].copy())
            if record_video:
                frames.append(env.render())

            r_task += reward_fn(env.unwrapped.model, qpos=info["qpos"], qvel=info["qvel"], ctrl=action_np)

            if terminated or truncated:
                obs, _ = env.reset()

    return {
        "qpos_beta": np.stack(qpos_hist),
        "r_task": r_task,
        "frames": frames,
    }


def evaluate(checkpoint_path, tasks_file=None, trials_per_task=None,
             out_dir="outputs/eval", render_videos=False, device="cuda:0"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: TrainConfig = ckpt["cfg"]
    cfg.device = device

    if tasks_file is None:
        if DEFAULT_TEST_TASKS.exists():
            tasks_file = DEFAULT_TEST_TASKS
            print(f"using held-out test split: {tasks_file}")
        else:
            tasks_file = OFFICIAL_TASKS
            print(f"WARNING: no train/test split found at {DEFAULT_TEST_TASKS} "
                  f"(run scripts/split_tasks.py) -- evaluating on the full "
                  f"official task list instead, which may include tasks the "
                  f"adapter was trained on: {tasks_file}")
    task_list = load_task_list(tasks_file)

    dataset = CrossEmbodimentDataset(REPO_ROOT / cfg.dataset_dir, task_filter=task_list)
    beta_dim = len(BETA_AXES)

    model = FBcprModel.from_pretrained(cfg.metamotivo_repo).to(device)
    model.eval()

    env, _ = make_humenv(
        num_envs=1, task=None, xml=str(REPO_ROOT / cfg.target_xml), state_init="Default",
    )

    adapter = LatentAdapter(
        beta_dim=beta_dim, z_dim=model.cfg.archi.z_dim,
        hidden_dims=cfg.adapter_hidden_dims, alpha=cfg.adapter_alpha,
        alpha_learnable=cfg.adapter_alpha_learnable,
    ).to(device)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()

    action_head = ActionHead(
        action_dim=model.cfg.action_dim, beta_dim=beta_dim,
        hidden_dims=cfg.action_head_hidden_dims,
    ).to(device)
    action_head.load_state_dict(ckpt["action_head"])
    action_head.eval()

    d_weights = {
        "root": cfg.d_root_weight, "ee": cfg.d_ee_weight, "contact": cfg.d_contact_weight,
        "pose": cfg.d_pose_weight, "velocity": cfg.d_velocity_weight,
    }

    out_dir = Path(out_dir)
    video_dir = out_dir / "video"
    if render_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    per_row = []
    trials_seen = defaultdict(int)
    rendered_tasks = set()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        reward_name = sample["reward_name"]
        if trials_per_task is not None and trials_seen[reward_name] >= trials_per_task:
            continue
        trials_seen[reward_name] += 1

        reward_fn = make_from_name(reward_name)
        z0_t = torch.tensor(sample["z0"], dtype=torch.float32, device=device).unsqueeze(0)
        beta_t = torch.tensor(sample["beta"], dtype=torch.float32, device=device).unsqueeze(0)

        record_video = render_videos and reward_name not in rendered_tasks
        episode = rollout_deterministic(model, adapter, action_head, env, reward_fn,
                                         z0_t, beta_t, cfg, record_video=record_video)

        d_total, d_terms = losses.functional_equivalence(
            env.unwrapped.model, episode["qpos_beta"], sample["qpos_ref"], d_weights
        )
        l_phys, _ = losses.physics_penalty(env.unwrapped.model, episode["qpos_beta"])

        per_row.append({
            "reward_name": reward_name, "trial": sample["trial"],
            "r_task": episode["r_task"], "d_total": d_total, "l_phys": l_phys, **d_terms,
        })

        if record_video:
            import imageio
            imageio.mimsave(video_dir / f"{reward_name}.mp4", episode["frames"], fps=30)
            rendered_tasks.add(reward_name)

        print(f"[{reward_name} trial {sample['trial']}] "
              f"r_task={episode['r_task']:.4f} D={d_total:.4f} L_phys={l_phys:.4f}")

    env.close()

    per_task = defaultdict(list)
    for row in per_row:
        per_task[row["reward_name"]].append(row)

    summary = {}
    for reward_name, rows in per_task.items():
        summary[reward_name] = {
            "n_trials": len(rows),
            "r_task_mean": float(np.mean([r["r_task"] for r in rows])),
            "r_task_std": float(np.std([r["r_task"] for r in rows])),
            "d_total_mean": float(np.mean([r["d_total"] for r in rows])),
            "l_phys_mean": float(np.mean([r["l_phys"] for r in rows])),
        }

    overall = {
        "n_tasks": len(per_task),
        "n_rows": len(per_row),
        "r_task_mean": float(np.mean([r["r_task"] for r in per_row])) if per_row else None,
        "d_total_mean": float(np.mean([r["d_total"] for r in per_row])) if per_row else None,
        "l_phys_mean": float(np.mean([r["l_phys"] for r in per_row])) if per_row else None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(
        {"checkpoint": str(checkpoint_path), "tasks_file": str(tasks_file),
         "overall": overall, "per_task": summary, "per_row": per_row},
        indent=2,
    ))

    print("\n=== summary ===")
    print(f"{overall['n_tasks']} tasks, {overall['n_rows']} rows")
    print(f"r_task mean: {overall['r_task_mean']}")
    print(f"D mean:      {overall['d_total_mean']}")
    print(f"L_phys mean: {overall['l_phys_mean']}")
    print(f"full report -> {report_path}")

    return overall, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks-file", default=None,
                         help="defaults to the held-out test split if "
                              "scripts/split_tasks.py has been run, else "
                              "the full official task list")
    parser.add_argument("--trials-per-task", type=int, default=None,
                         help="cap trials evaluated per task (default: all "
                              "available in the dataset)")
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument("--render-videos", action="store_true",
                         help="save one comparison video per task (first trial only)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        tasks_file=args.tasks_file,
        trials_per_task=args.trials_per_task,
        out_dir=args.out_dir,
        render_videos=args.render_videos,
        device=args.device,
    )


if __name__ == "__main__":
    main()
