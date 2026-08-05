"""Test "route A": instead of trusting a purely kinematic IK-retargeted
qpos_ref (data/retargeted_motion, see qpos_retarget.py -- never touches
<actuator>, never simulated), let the frozen Metamotivo backward map DEFINE
the target motion on the child body by dynamically-feasible construction:

  1. Take the ORIGIN body's real recorded motion (data/origin_motion, itself
     produced by actually stepping robot.xml's physics -- see
     metamotivo_motion_rollout.py).
  2. Reconstruct its per-frame proprio observation (same
     compute_humanoid_self_obs_v2 the env itself uses).
  3. model.tracking_inference(obs_seq) -> a per-step z[t] (FB-CPR's built-in
     zero-shot state-tracking mechanism: z[t] = mean of backward_map(obs)
     over a sliding window, see metamotivo's FBcprModel.tracking_inference).
  4. Actually mj_step the CHILD body (robot_child.xml, with scale_robot.py's
     corrected per-joint-load actuator scaling) driven by model.act(obs_t,
     z[t], mean=True) at each step, and record whatever qpos results.

Whatever qpos comes out of step 4 is dynamically feasible BY CONSTRUCTION
(it's a real mj_step rollout) -- unlike check_retarget_actuator_feasibility.py,
which tries to validate a kinematic reference after the fact via finite-
differenced qacc + mj_inverse (found to be numerically unreliable: even
ORIGIN motion's own recorded qpos showed ~34% "infeasible" frames from
double-differentiation noise alone).

IMPORTANT caveat this script exists to quantify, not assume: "dynamically
feasible" != "resembles the origin motion" -- a fall is also feasible. z[t]
here is an OPEN-LOOP schedule (computed once from the origin trajectory,
not replanned against the child body's actual, possibly-diverging state),
and backward_map/forward_map/actor were all trained on the ORIGINAL body's
action semantics -- same actuator-mismatch risk already measured in
model/baseline.py (raw z0 on robot_child.xml: L_phys 2.42 unscaled actuators,
2.71 even with corrected per-joint-load actuator scaling -- both WORSE than
1.70 for the unmodified original body). This script measures whether
tracking-inferred z (continuously re-reading the target motion) does better
than a single static z0, using the exact same L_phys/D metrics as
model/baseline.py and model/evaluate.py for direct comparability.

Usage (from project root):
    uv run scripts/tracking_retarget_check.py --limit 5 --device cuda:0
    uv run scripts/tracking_retarget_check.py --limit 5 --render-videos --out-dir outputs/tracking_retarget
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np
import torch

from humenv import make_humenv
from humenv.env import compute_humanoid_self_obs_v2
from metamotivo.fb_cpr.huggingface import FBcprModel

from model import losses
from model.dataset import CrossEmbodimentDataset, load_task_list

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_TASKS = REPO_ROOT / "datasets" / "crossenbodiment-1-datasets" / "splits" / "test_tasks.txt"
OFFICIAL_TASKS = REPO_ROOT / "docs" / "humenv_all_tasks_official.txt"
ORIGIN_XML = REPO_ROOT / "assets" / "robots" / "robot.xml"


def qpos_to_obs_seq(model, data, qpos_seq, dt):
    """Reconstruct the proprio obs sequence humenv's own get_obs() would
    have produced while recording this motion. Needs qvel (compute_humanoid_
    self_obs_v2 reads body linear/angular velocity from data.sensordata,
    populated by mj_forward) -- origin_motion npz only stores qpos, so qvel
    is reconstructed via mj_differentiatePos (correctly handles the free
    joint's quaternion, unlike a naive qpos diff). This is a SINGLE finite
    difference (qpos -> qvel), not the double-differentiation (-> qacc) that
    was found to blow up into numerical noise in
    check_retarget_actuator_feasibility.py -- much better conditioned."""
    T = qpos_seq.shape[0]
    qvel_seq = np.zeros((T, model.nv))
    for t in range(T - 1):
        mujoco.mj_differentiatePos(model, qvel_seq[t], dt, qpos_seq[t], qpos_seq[t + 1])
    if T > 1:
        qvel_seq[-1] = qvel_seq[-2]

    obs_seq = np.zeros((T, 358), dtype=np.float64)
    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        data.qvel[:] = qvel_seq[t]
        mujoco.mj_forward(model, data)
        obs_dict = compute_humanoid_self_obs_v2(
            model, data, upright_start=False, root_height_obs=True, humanoid_type="smpl",
        )
        obs_seq[t] = np.concatenate([v.ravel() for v in obs_dict.values()], axis=0)
    return obs_seq


def rollout_tracking(model, env, z_track, device, steps, record_video=False):
    obs, _ = env.reset()
    qpos_hist = []
    frames = [] if record_video else None

    with torch.no_grad():
        for t in range(steps):
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32, device=device).unsqueeze(0)
            z_t = z_track[min(t, z_track.shape[0] - 1):min(t, z_track.shape[0] - 1) + 1]
            action = model.act(obs_t, z_t, mean=True)
            action_np = action.cpu().numpy().ravel()

            obs, _, terminated, truncated, info = env.step(action_np)
            qpos_hist.append(info["qpos"].copy())
            if record_video:
                frames.append(env.render())

            if terminated or truncated:
                obs, _ = env.reset()

    return {"qpos_beta": np.stack(qpos_hist), "frames": frames}


def run(dataset_dir="datasets/crossenbodiment-1-datasets",
        target_xml="assets/robots/child/robot.xml",
        metamotivo_repo="facebook/metamotivo-M-1",
        tasks_file=None, trials_per_task=1, steps_per_episode=300,
        out_dir="outputs/tracking_retarget", render_videos=False, device="cuda:0"):
    if tasks_file is None:
        if DEFAULT_TEST_TASKS.exists():
            tasks_file = DEFAULT_TEST_TASKS
            print(f"using held-out test split: {tasks_file}")
        else:
            tasks_file = OFFICIAL_TASKS
            print(f"WARNING: no train/test split found at {DEFAULT_TEST_TASKS} "
                  f"(run scripts/split_tasks.py) -- using the full official task list instead: {tasks_file}")
    task_list = load_task_list(tasks_file)

    dataset = CrossEmbodimentDataset(REPO_ROOT / dataset_dir, task_filter=task_list)

    model = FBcprModel.from_pretrained(metamotivo_repo).to(device)
    model.eval()

    origin_model = mujoco.MjModel.from_xml_path(str(ORIGIN_XML))
    origin_data = mujoco.MjData(origin_model)
    origin_dt = 1.0 / 30.0  # matches origin_motion's recording rate (1 saved frame per env.step)

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
    data_dir = REPO_ROOT / "data" / "origin_motion"

    for idx in range(len(dataset)):
        sample = dataset[idx]
        reward_name = sample["reward_name"]
        trial = sample["trial"]
        if trials_seen[reward_name] >= trials_per_task:
            continue
        origin_npz = data_dir / reward_name / f"{reward_name}_{trial}.npz"
        if not origin_npz.exists() or sample["qpos_ref"] is None:
            continue
        trials_seen[reward_name] += 1

        origin_qpos = np.load(origin_npz)["qpos"]
        obs_seq = qpos_to_obs_seq(origin_model, origin_data, origin_qpos, origin_dt)
        with torch.no_grad():
            z_track = model.tracking_inference(
                torch.tensor(obs_seq, dtype=torch.float32, device=device)
            )

        record_video = render_videos and reward_name not in rendered_tasks
        steps = min(steps_per_episode, origin_qpos.shape[0])
        episode = rollout_tracking(model, env, z_track, device, steps, record_video=record_video)

        d_total, d_terms = losses.functional_equivalence(
            env.unwrapped.model, episode["qpos_beta"], sample["qpos_ref"], d_weights
        )
        l_phys, _ = losses.physics_penalty(env.unwrapped.model, episode["qpos_beta"])

        per_row.append({
            "reward_name": reward_name, "trial": trial,
            "d_total": d_total, "l_phys": l_phys, **d_terms,
        })

        if record_video:
            import imageio
            imageio.mimsave(video_dir / f"{reward_name}.mp4", episode["frames"], fps=30)
            rendered_tasks.add(reward_name)

        print(f"[{reward_name} trial {trial}] D={d_total:.4f} L_phys={l_phys:.4f}")

    env.close()

    per_task = defaultdict(list)
    for row in per_row:
        per_task[row["reward_name"]].append(row)

    summary = {}
    for reward_name, rows in per_task.items():
        summary[reward_name] = {
            "n_trials": len(rows),
            "d_total_mean": float(np.mean([r["d_total"] for r in rows])),
            "l_phys_mean": float(np.mean([r["l_phys"] for r in rows])),
        }

    overall = {
        "n_tasks": len(per_task),
        "n_rows": len(per_row),
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

    print("\n=== tracking-inference retarget check (z[t] from tracking_inference on origin motion, "
          "rolled out on child body) ===")
    print(f"{overall['n_tasks']} tasks, {overall['n_rows']} rows")
    print(f"D mean:      {overall['d_total_mean']}")
    print(f"L_phys mean: {overall['l_phys_mean']}")
    print(f"(compare against outputs/baseline_child_v2/report.json L_phys=2.71 [static z0, scaled actuators], "
          f"outputs/baseline_origin/report.json L_phys=1.70 [original body])")
    print(f"full report -> {report_path}")

    return overall, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/crossenbodiment-1-datasets")
    parser.add_argument("--target-xml", default="assets/robots/child/robot.xml")
    parser.add_argument("--tasks-file", default=None)
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--steps-per-episode", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None, help="only check the first N tasks in the task list")
    parser.add_argument("--out-dir", default="outputs/tracking_retarget")
    parser.add_argument("--render-videos", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tasks_file = args.tasks_file
    if args.limit is not None:
        src = Path(tasks_file) if tasks_file else (DEFAULT_TEST_TASKS if DEFAULT_TEST_TASKS.exists() else OFFICIAL_TASKS)
        names = load_task_list(src)[: args.limit]
        tmp_path = REPO_ROOT / "outputs" / "_tracking_retarget_task_subset.txt"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text("\n".join(names))
        tasks_file = str(tmp_path)

    run(
        dataset_dir=args.dataset_dir,
        target_xml=args.target_xml,
        tasks_file=tasks_file,
        trials_per_task=args.trials_per_task,
        steps_per_episode=args.steps_per_episode,
        out_dir=args.out_dir,
        render_videos=args.render_videos,
        device=args.device,
    )


if __name__ == "__main__":
    main()
