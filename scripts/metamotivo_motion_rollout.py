"""
Rollout skills from the Metamotivo-M-1 pre-trained FB-CPR model inside the
MuJoCo-based HumEnv humanoid environment.

Two ways to pick the context vector z:
  --z-mode random  : z ~ N(0, I), projected to the unit sphere (default).
                      Produces arbitrary, not necessarily meaningful motion.
  --z-mode reward  : z is inferred (zero-shot, no fine-tuning) from a named
                      humenv reward function (e.g. "move-ego-0-2", "jump-2",
                      "rotate-x-5-l-0.8") using Meta Motivo's offline
                      inference buffer. This is how you get purposeful
                      motions like walking, running, jumping, etc.

Batch mode (--tasks-file): run --rollouts-per-task rollouts for every reward
name listed in a text file (one name per line, e.g.
docs/humenv_all_tasks_official.txt). Only the first rollout of each task is
rendered/saved as video (video is expensive); every rollout's qpos
trajectory, ACTION stream and z are saved separately under --data-dir.

The action stream exists for open-loop cross-embodiment transfer: replay the
adult's ctrl on a different body and see what the body alone changes. Three
things it needs that a qpos-only dump cannot give:

  * `action` itself -- what actually went into data.ctrl each step
  * `qpos_init`/`qvel_init` -- the state AFTER reset and BEFORE the first
    action. qpos is appended after env.step, so frame 0 of the trajectory is
    already one action in and the true starting point was never written down.
  * `qvel` -- needed for any velocity-referenced metric on the replay

Numerics note: torch partitions a matmul by thread count, so the thread count
is part of the result. Measured on this humanoid, a 1e-7 change in the actor's
float32 output grows to O(1) rad within 300 steps on contact-unstable tasks
(headstand, rotate-x). The count used is therefore RECORDED into every action
.npz, and --torch-threads defaults to 1 -- which is also ~11x faster than
torch's own default here, because the frozen actor runs at batch 1 and 32
threads is pure synchronization overhead.

Install (see https://github.com/facebookresearch/metamotivo):
    pip install metamotivo humenv[all] gymnasium mujoco torch imageio h5py

Usage (run from the project root):
    uv run scripts/metamotivo_motion_rollout.py --z-mode random --num-rollouts 5 --steps 300
    uv run scripts/metamotivo_motion_rollout.py --z-mode reward --reward-name "move-ego-0-2" --steps 300
    uv run scripts/metamotivo_motion_rollout.py --tasks-file docs/humenv_all_tasks_official.txt --steps 300

Batch mode writes qpos to data/origin_motion/<reward_name>/, the action stream
to data/origin_action/<reward_name>/, z to data/z/<reward_name>/, and video to
outputs/robot_motion_video/ (--data-dir/--out-dir to change either root).
Single/random-rollout mode just writes rollout_<i>.mp4 under --out-dir.
"""

import argparse
import os
from pathlib import Path

# Must be set before mujoco/humenv are imported: headless machines have no
# X11 display, so default to the EGL backend for offscreen rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from metamotivo.fb_cpr.huggingface import FBcprModel
from humenv import make_humenv

import imageio


def load_inference_buffer(args):
    """Download Meta Motivo's offline inference buffer and load it into a
    DictBuffer we can repeatedly re-sample from (see the official
    tutorial.ipynb)."""
    from huggingface_hub import hf_hub_download
    import h5py
    from metamotivo.buffers.buffers import DictBuffer

    buffer_path = hf_hub_download(
        repo_id=args.model,
        filename="data/buffer_inference_500000.hdf5",
        repo_type="model",
    )
    hf = h5py.File(buffer_path, "r")
    data = {k: v[:] for k, v in hf.items()}
    buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
    buffer.extend(data)
    return buffer


def compute_reward_z(model, env, buffer, args, seed, reward_name=None):
    """Infer a z from a named humenv reward function, drawing a fresh random
    subsample of the inference buffer (seeded, so it differs per call)."""
    from metamotivo.wrappers.humenvbench import relabel
    from humenv.env import make_from_name

    torch.manual_seed(seed)
    batch = buffer.sample(args.relabel_samples)

    reward_fn = make_from_name(reward_name or args.reward_name)
    rewards = relabel(
        env,
        qpos=batch["next_qpos"],
        qvel=batch["next_qvel"],
        action=batch["action"],
        reward_fn=reward_fn,
        max_workers=8,
    )

    z = model.reward_wr_inference(
        next_obs=batch["next_observation"],
        reward=torch.tensor(rewards, device=args.device, dtype=torch.float32),
    )
    return z.to(args.device)


# Periodic skills (walking/crawling/spinning) need enough steps to cover
# several cycles; pose skills (reach-and-hold) converge quickly and just
# repeat a static frame afterwards, so a shorter episode avoids wasted,
# redundant storage. First matching prefix wins; anything unmatched falls
# back to --steps.
TASK_STEP_RULES = [
    ("move-ego", 300),   # covers move-ego-*, move-ego-low-*, move-ego-*-raisearms-*
    ("crawl", 300),
    ("rotate", 300),
    ("jump", 150),
    ("headstand", 120),
    ("raisearms", 120),
    ("sitonground", 120),
    ("lieonground", 120),
    ("crouch", 120),
    ("split", 120),
]


def steps_for_task(reward_name, default_steps):
    for prefix, steps in TASK_STEP_RULES:
        if reward_name.startswith(prefix):
            return steps
    return default_steps


def rollout_once(model, env, args, seed, z, steps=None, record_video=True):
    torch.manual_seed(seed)
    steps = steps or args.steps

    obs, _ = env.reset(seed=seed)
    # The state the action stream starts FROM. qpos_hist is appended AFTER
    # env.step, so its frame 0 is already one action in; without this an
    # open-loop replay has nowhere to begin.
    qpos_init = env.unwrapped.data.qpos.copy()
    qvel_init = env.unwrapped.data.qvel.copy()
    frames = [] if record_video else None
    qpos_hist, qvel_hist, action_hist, resets = [], [], [], []
    for t in range(steps):
        obs_t = torch.tensor(obs["proprio"], dtype=torch.float32, device=args.device)
        obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            action = model.act(obs_t, z, mean=True)
            '''
            def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
                dist = self.actor(obs, z, self.cfg.actor_std)
                if mean:
                    return dist.mean
                return dist.sample()
            '''
        action_np = action.cpu().numpy().ravel()

        obs, reward, terminated, truncated, info = env.step(action_np)
        action_hist.append(action_np.copy())
        qpos_hist.append(info["qpos"].copy())
        qvel_hist.append(info["qvel"].copy())
        if record_video:
            frames.append(env.render())

        if terminated or truncated:
            # Currently dead: humenv's is_terminated() is hardcoded
            # `return False` and truncated is never set, so this has never
            # fired. It matters now that actions are being saved -- a reset
            # here would splice two episodes into one action stream and make an
            # open-loop replay of it meaningless. Recorded rather than removed,
            # so if humenv ever changes, `resets` is non-empty instead of the
            # corpus being silently wrong.
            resets.append(t + 1)
            obs, _ = env.reset()

    return {
        "frames": frames,
        "qpos": np.stack(qpos_hist),
        "qvel": np.stack(qvel_hist),
        "action": np.stack(action_hist).astype(np.float32),
        "qpos_init": qpos_init,
        "qvel_init": qvel_init,
        "resets": np.array(resets, dtype=np.int64),
    }


def run_batch(model, env, args):
    """--tasks-file mode: for every reward name in the file, run
    --rollouts-per-task rollouts, save video only for the first. Each
    rollout writes a qpos-only .npz (trajectory data, for e.g. FBX
    conversion) under --data-dir/origin_motion/, the full replay record
    (action + qpos + qvel + initial state + provenance) under
    --data-dir/origin_action/, and the z vector used (a per-rollout latent,
    not a per-frame trajectory) under --data-dir/z/.

    origin_motion stays qpos-only on purpose: model/bilevel/data.py:255 reads
    it as np.load(path)["qpos"] and it is already ~98 MB resident across 540
    clips. The replay-only arrays go in their own tree rather than fattening
    the one every training iteration loads.

    args.z_only skips rollout_once (and hence the qpos .npz/video) entirely:
    z comes from compute_reward_z(), which only needs a buffer sample + a
    reward-function relabel, NOT a simulated rollout -- measured ~1.2s/trial
    vs ~60-90s/trial with a full 300-step rollout. Use this when only z
    itself is needed (e.g. to grow a task's z library for training)."""
    buffer = load_inference_buffer(args)

    reward_names = [
        line.strip()
        for line in Path(args.tasks_file).read_text().splitlines()
        if line.strip()
    ]

    data_dir = Path(args.data_dir)
    video_dir = Path(args.out_dir)
    npz_dir = data_dir / "origin_motion"
    act_dir = data_dir / "origin_action"
    z_dir = data_dir / "z"
    video_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)
    z_dir.mkdir(parents=True, exist_ok=True)

    for task_idx, reward_name in enumerate(reward_names):
        task_npz_dir = npz_dir / reward_name
        task_act_dir = act_dir / reward_name
        task_z_dir = z_dir / reward_name
        task_npz_dir.mkdir(parents=True, exist_ok=True)
        task_act_dir.mkdir(parents=True, exist_ok=True)
        task_z_dir.mkdir(parents=True, exist_ok=True)

        steps = steps_for_task(reward_name, args.steps) if args.auto_steps else args.steps

        for trial in range(args.rollouts_per_task):
            seed = args.seed + trial
            z = compute_reward_z(model, env, buffer, args, seed, reward_name=reward_name)
            z_path = task_z_dir / f"{reward_name}_{trial}.npy"
            np.save(z_path, z.cpu().numpy())

            if args.z_only:
                print(f"[{task_idx + 1}/{len(reward_names)} {reward_name}] "
                      f"trial {trial + 1}/{args.rollouts_per_task} -> {z_path}")
                continue

            record_video = trial == 0
            result = rollout_once(model, env, args, seed, z, steps=steps, record_video=record_video)

            npz_path = task_npz_dir / f"{reward_name}_{trial}.npz"
            np.savez(npz_path, qpos=result["qpos"])

            # Everything an open-loop replay on another body needs, plus the
            # provenance that makes this rollout reproducible: the exact z, the
            # seed, and the torch thread count (which IS part of the numerics --
            # see the module docstring).
            act_path = task_act_dir / f"{reward_name}_{trial}.npz"
            np.savez(
                act_path,
                action=result["action"], qpos=result["qpos"], qvel=result["qvel"],
                qpos_init=result["qpos_init"], qvel_init=result["qvel_init"],
                resets=result["resets"], z=z.cpu().numpy().ravel(),
                seed=seed, steps=steps, model=args.model,
                torch_threads=torch.get_num_threads(),
            )

            if record_video:
                video_path = video_dir / f"{reward_name}.mp4"
                imageio.mimsave(video_path, result["frames"], fps=30)

            print(f"[{task_idx + 1}/{len(reward_names)} {reward_name}] "
                  f"trial {trial + 1}/{args.rollouts_per_task} -> {npz_path}, "
                  f"{act_path}, {z_path}"
                  + (f" (+ video)" if record_video else ""), flush=True)
            # A reset AT `steps` is gymnasium's TimeLimit firing on the final
            # iteration (make_humenv's DEFAULT_MAX_EPISODE_STEPS is 300, which
            # is also TASK_STEP_RULES' longest). qpos is appended before the
            # reset and the loop ends immediately after, so nothing recorded is
            # affected. Only a reset STRICTLY INSIDE the window splices two
            # episodes into one action stream and breaks open-loop replay.
            mid = [int(r) for r in result["resets"] if r < steps]
            if mid:
                print(f"    WARNING: {len(mid)} mid-episode reset(s) at {mid} -- this "
                      f"action stream splices episodes and is NOT replayable "
                      f"open-loop", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/metamotivo-M-1",
                         help="HuggingFace repo id of the pre-trained model")
    parser.add_argument("--z-mode", choices=["random", "reward"], default="random",
                         help="how to pick z: random sampling, or inferred "
                              "from a named humenv reward function")
    parser.add_argument("--reward-name", default=None,
                         help="humenv reward name, required when --z-mode reward "
                              "(e.g. 'move-ego-0-2', 'jump-2', 'rotate-x-5-l-0.8')")
    parser.add_argument("--tasks-file", default=None,
                         help="path to a text file of reward names (one per line); "
                              "if set, runs batch mode over all of them and "
                              "ignores --z-mode/--reward-name/--num-rollouts")
    parser.add_argument("--rollouts-per-task", type=int, default=10,
                         help="rollouts per reward name in --tasks-file mode; "
                              "only the first is saved as video")
    parser.add_argument("--auto-steps", action="store_true",
                         help="in --tasks-file mode, pick episode length per "
                              "task from TASK_STEP_RULES (periodic skills get "
                              "more steps, pose skills fewer) instead of a "
                              "fixed --steps for every task")
    parser.add_argument("--relabel-samples", type=int, default=1_000,
                         help="number of buffer transitions used to infer z "
                              "when --z-mode reward / --tasks-file")
    parser.add_argument("--z-only", action="store_true",
                         help="--tasks-file mode only: skip the simulated "
                              "rollout (qpos .npz + video) entirely, only "
                              "compute and save z -- much faster (~1.2s vs "
                              "~60-90s per trial) since z-inference doesn't "
                              "need a rollout")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-rollouts", type=int, default=5,
                         help="number of independent rollouts to run "
                              "(each with a fresh random z when --z-mode random)")
    parser.add_argument("--seed", type=int, default=0,
                         help="base seed; rollout i uses seed + i")
    parser.add_argument("--out-dir", default="outputs/robot_motion_video",
                         help="directory to write video into (single-rollout "
                              "mode: rollout_<i>.mp4 directly here; "
                              "--tasks-file mode: <reward_name>.mp4 here)")
    parser.add_argument("--data-dir", default="data",
                         help="--tasks-file mode only: qpos goes to "
                              "<data-dir>/origin_motion/<reward_name>/, the "
                              "action stream to <data-dir>/origin_action/"
                              "<reward_name>/, z to <data-dir>/z/<reward_name>/")
    parser.add_argument("--torch-threads", type=int, default=1,
                         help="torch intra-op threads. PART OF THE NUMERICS: the "
                              "thread count sets the matmul reduction order, and "
                              "this system is chaotic enough that a 1e-7 change "
                              "grows to O(1) rad in 300 steps on headstand / "
                              "rotate-x. Recorded into every action .npz. 1 is "
                              "also ~11x faster than torch's default here (14 vs "
                              "155 ms/step) because the actor runs at batch 1. "
                              "0 leaves torch's default untouched.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.tasks_file is None and args.z_mode == "reward" and args.reward_name is None:
        parser.error("--reward-name is required when --z-mode reward")

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    print(f"torch threads: {torch.get_num_threads()} (recorded into each action .npz)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # Load the pre-trained FB-CPR model and build the env once, reuse across rollouts.
    model = FBcprModel.from_pretrained(args.model)
    model = model.to(args.device)
    model.eval()

    env, _ = make_humenv(
        num_envs=1,
        task=None,
        state_init="Default",
        render_width=640,
        render_height=480,
    )

    if args.tasks_file is not None:
        run_batch(model, env, args)
        env.close()
        return

    if args.z_mode == "reward":
        buffer = load_inference_buffer(args)

    for i in range(args.num_rollouts):
        seed = args.seed + i
        if args.z_mode == "random":
            z = model.sample_z(1)
            if isinstance(z, np.ndarray):
                z = torch.tensor(z, dtype=torch.float32)
            z = z.to(args.device)
        else:
            # Fresh buffer.sample(N) each rollout -> a slightly different z
            # estimate of the same reward each time.
            z = compute_reward_z(model, env, buffer, args, seed)

        result = rollout_once(model, env, args, seed, z, record_video=True)

        out_path = out_dir / f"rollout_{i}.mp4"
        imageio.mimsave(out_path, result["frames"], fps=30)
        print(f"[{i + 1}/{args.num_rollouts}] seed={seed} "
              f"saved {len(result['frames'])} frames -> {out_path}")

    env.close()


if __name__ == "__main__":
    main()
