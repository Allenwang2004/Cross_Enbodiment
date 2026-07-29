"""Zero-shot baseline: run the ORIGINAL frozen Metamotivo model directly on
robot_child.xml -- no adapter, no action head, z_beta = z0 unchanged, action
= the frozen actor's raw mean action (model.act(obs, z0, mean=True)). This
is "what happens if you just force the pretrained policy to walk on a body
it was never trained for" -- the reference point the trained adapter
(model/evaluate.py) needs to beat to be worth anything.

Uses the same task set, reward functions, and D/L_phys scoring as
model/evaluate.py so the two reports are directly comparable.

Usage (from project root):
    uv run model/baseline.py
    uv run model/baseline.py --render-videos --out-dir outputs/baseline
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

# Running this file directly (not `-m model.baseline`) puts model/ itself on
# sys.path, not its parent -- see the same fix in run_train.py/evaluate.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from humenv import make_humenv
from humenv.env import make_from_name
from metamotivo.fb_cpr.huggingface import FBcprModel

from model import losses
from model.dataset import CrossEmbodimentDataset, load_task_list

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_TASKS = REPO_ROOT / "datasets" / "crossenbodiment-1-datasets" / "splits" / "test_tasks.txt"
OFFICIAL_TASKS = REPO_ROOT / "docs" / "humenv_all_tasks_official.txt"


def rollout_baseline(model, env, reward_fn, z0_t, device, steps_per_episode, record_video=False):
    obs, _ = env.reset()
    qpos_hist = []
    frames = [] if record_video else None
    r_task = 0.0

    with torch.no_grad():
        for t in range(steps_per_episode):
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32, device=device).unsqueeze(0)
            action = model.act(obs_t, z0_t, mean=True)
            action_np = action.cpu().numpy().ravel()

            obs, _, terminated, truncated, info = env.step(action_np)
            qpos_hist.append(info["qpos"].copy())
            if record_video:
                frames.append(env.render())

            r_task += reward_fn(env.unwrapped.model, qpos=info["qpos"], qvel=info["qvel"], ctrl=action_np)

            if terminated or truncated:
                obs, _ = env.reset()

    return {"qpos_beta": np.stack(qpos_hist), "r_task": r_task, "frames": frames}


def run_baseline(dataset_dir="datasets/crossenbodiment-1-datasets",
                  target_xml="assets/robots/robot_child.xml",
                  metamotivo_repo="facebook/metamotivo-M-1",
                  tasks_file=None, trials_per_task=None, steps_per_episode=300,
                  out_dir="outputs/baseline", render_videos=False, device="cuda:0"):
    if tasks_file is None:
        if DEFAULT_TEST_TASKS.exists():
            tasks_file = DEFAULT_TEST_TASKS
            print(f"using held-out test split: {tasks_file}")
        else:
            tasks_file = OFFICIAL_TASKS
            print(f"WARNING: no train/test split found at {DEFAULT_TEST_TASKS} "
                  f"(run scripts/split_tasks.py) -- using the full official "
                  f"task list instead: {tasks_file}")
    task_list = load_task_list(tasks_file)

    dataset = CrossEmbodimentDataset(REPO_ROOT / dataset_dir, task_filter=task_list)

    model = FBcprModel.from_pretrained(metamotivo_repo).to(device)
    model.eval()

    env, _ = make_humenv(
        num_envs=1, task=None, xml=str(REPO_ROOT / target_xml), state_init="Default",
    )

    d_weights = {"root": 1.0, "ee": 1.0, "contact": 1.0, "pose": 1.0, "velocity": 1.0}

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

        record_video = render_videos and reward_name not in rendered_tasks
        episode = rollout_baseline(model, env, reward_fn, z0_t, device, steps_per_episode,
                                    record_video=record_video)

        d_total, d_terms = losses.functional_equivalence(
            env.unwrapped.model, episode["qpos_beta"], sample["qpos_ref"], d_weights
        )
        l_phys = losses.physics_penalty(env.unwrapped.model, episode["qpos_beta"])

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
        {"target_xml": target_xml, "tasks_file": str(tasks_file),
         "overall": overall, "per_task": summary, "per_row": per_row},
        indent=2,
    ))

    print("\n=== baseline summary (no adapter, raw z0 on robot_child.xml) ===")
    print(f"{overall['n_tasks']} tasks, {overall['n_rows']} rows")
    print(f"r_task mean: {overall['r_task_mean']}")
    print(f"D mean:      {overall['d_total_mean']}")
    print(f"L_phys mean: {overall['l_phys_mean']}")
    print(f"full report -> {report_path}")

    return overall, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/crossenbodiment-1-datasets")
    parser.add_argument("--target-xml", default="assets/robots/robot_child.xml")
    parser.add_argument("--tasks-file", default=None,
                         help="defaults to the held-out test split if "
                              "scripts/split_tasks.py has been run, else "
                              "the full official task list")
    parser.add_argument("--trials-per-task", type=int, default=None)
    parser.add_argument("--steps-per-episode", type=int, default=300)
    parser.add_argument("--out-dir", default="outputs/baseline")
    parser.add_argument("--render-videos", action="store_true",
                         help="save one video per task (first trial only)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_baseline(
        dataset_dir=args.dataset_dir,
        target_xml=args.target_xml,
        tasks_file=args.tasks_file,
        trials_per_task=args.trials_per_task,
        steps_per_episode=args.steps_per_episode,
        out_dir=args.out_dir,
        render_videos=args.render_videos,
        device=args.device,
    )


if __name__ == "__main__":
    main()
