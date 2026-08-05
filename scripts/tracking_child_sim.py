"""Track a naive kinematic retarget (data/retargeted_motion, produced by
qpos_retarget.py via IK only -- never simulated, see that script's
docstring) with the CHILD body's own real physics, and record whatever qpos
actually results. That recorded qpos is dynamically feasible by
construction (it's a real mj_step rollout on robot_child.xml), unlike the
naive reference itself.

z[t] (what tells the frozen actor "what to do" each step) comes from
FBcprModel.tracking_inference(obs_seq) -- FB-CPR's built-in zero-shot
state-tracking mechanism (z[t] = mean of backward_map(obs) over a sliding
window; see metamotivo's FBcprModel.tracking_inference source). obs_seq is
built directly from the naive retargeted qpos on the CHILD skeleton itself
(same skeleton being simulated, so no cross-body obs conversion is needed --
contrast scripts/tracking_retarget_check.py, which had to reconstruct obs
from the ORIGIN body since it tracked data/origin_motion instead).

At each step: current obs comes from the actual simulated state (closed-loop
w.r.t. the frozen actor), action = model.act(obs_t, z[t], mean=True), then a
real env.step() (mj_step, not just kinematic playback).

Usage (from project root):
    uv run scripts/tracking_child_sim.py --task move-ego--90-2 --trial 0 \
        --out outputs/tracking_child_sim/move-ego--90-2
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio
import mujoco
import numpy as np
import torch

from humenv import make_humenv
from humenv.env import compute_humanoid_self_obs_v2
from metamotivo.fb_cpr.huggingface import FBcprModel

from model import losses

REPO_ROOT = Path(__file__).resolve().parent.parent


def qpos_to_obs_seq(model, data, qpos_seq, dt):
    """Single finite difference (qpos -> qvel via mj_differentiatePos, which
    correctly handles the free joint's quaternion) to get the velocity
    compute_humanoid_self_obs_v2 needs -- much better conditioned than the
    double difference (-> qacc) that was found to be numerically unreliable
    in check_retarget_actuator_feasibility.py."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="move-ego--90-2")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--retarget-dir", default="data/retargeted_motion")
    parser.add_argument("--xml", default="assets/robots/child/robot.xml")
    parser.add_argument("--metamotivo-repo", default="facebook/metamotivo-M-1")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera", default="front_side")
    parser.add_argument("--out", default="outputs/tracking_child_sim/move-ego--90-2")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    naive_path = Path(args.retarget_dir) / f"{args.task}_{args.trial}.npz"
    qpos_naive = np.load(naive_path)["qpos"]
    T = qpos_naive.shape[0]
    dt = 1.0 / args.fps
    print(f"tracking {naive_path} ({T} frames)")

    model = FBcprModel.from_pretrained(args.metamotivo_repo).to(args.device)
    model.eval()

    child_model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / args.xml))
    child_data = mujoco.MjData(child_model)

    obs_seq = qpos_to_obs_seq(child_model, child_data, qpos_naive, dt)
    with torch.no_grad():
        z_track = model.tracking_inference(
            torch.tensor(obs_seq, dtype=torch.float32, device=args.device)
        )
    print(f"tracking_inference -> z_track {tuple(z_track.shape)}")

    env, _ = make_humenv(num_envs=1, task=None, xml=str(REPO_ROOT / args.xml), state_init="Default")
    env.reset()
    env.unwrapped.set_physics(qpos_naive[0])  # start exactly where the naive reference starts

    qpos_child = []
    frames = []
    with torch.no_grad():
        obs = env.unwrapped.get_obs()
        for t in range(T):
            obs_t = torch.tensor(obs["proprio"], dtype=torch.float32, device=args.device).unsqueeze(0)
            z_t = z_track[t : t + 1]
            action = model.act(obs_t, z_t, mean=True)
            action_np = action.cpu().numpy().ravel()

            obs, _, terminated, truncated, info = env.step(action_np)
            qpos_child.append(info["qpos"].copy())
            frames.append(env.render())

    env.close()
    qpos_child = np.array(qpos_child)  # this is the GT: dynamically feasible by construction

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(f"{args.out}.mp4", frames, fps=args.fps)
    np.savez(f"{args.out}_qpos_child.npz", qpos=qpos_child)
    print(f"wrote {T} frames -> {args.out}.mp4, {args.out}_qpos_child.npz")

    root_naive = qpos_naive[:, 2]
    root_child = qpos_child[:, 2]
    print(f"root z (pelvis height) -- naive retarget: mean={root_naive.mean():.4f} min={root_naive.min():.4f}")
    print(f"root z (pelvis height) -- child sim:       mean={root_child.mean():.4f} min={root_child.min():.4f}")
    print(f"root z tracking RMSE: {np.sqrt(np.mean((root_naive - root_child) ** 2)):.4f}")

    d_weights = {"root": 1.0, "ee": 1.0, "contact": 1.0, "pose": 1.0, "velocity": 1.0}
    d_total, d_terms = losses.functional_equivalence(child_model, qpos_child, qpos_naive, d_weights)
    l_phys, _ = losses.physics_penalty(child_model, qpos_child)
    print(f"D (vs naive retarget) = {d_total:.4f}  ({d_terms})")
    print(f"L_phys (child sim)    = {l_phys:.4f}")


if __name__ == "__main__":
    main()
