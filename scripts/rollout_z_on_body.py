"""Drive the frozen Metamotivo FB-CPR actor with a z sequence read from disk,
on an arbitrary body, and render the result.

`metamotivo_motion_rollout.py` can only get its z from a named humenv reward
or from noise, and it always builds the env on humenv's own (adult) model.
This script covers the other direction: a z that was inferred back OUT of a
motion (`batch_infer_z.py`'s per-frame `tracking_inference` output) is fed to
the same frozen actor, but the physics runs on `--xml` -- so the actor is
adult-trained while the body it is closing the loop on is not.

The z file may be (T, 256) (per-frame, from `tracking_inference`) or
(1, 256) (a single context, e.g. `data/z/`'s reward-inferred z). Per-frame z
is applied step-for-step; if the rollout runs longer than T, the last z is
held. `--z-reduce mean|first|last` collapses a (T, 256) file to a single
constant z instead, which is the apples-to-apples comparison against how
`data/z/` was used to generate the original motion.

Usage (from project root):
    python3 scripts/rollout_z_on_body.py \
        --z data/retargeting_z/move-ego--90-2/move-ego--90-2_0.npy \
        --xml assets/robots_calib/child/robot.xml \
        --reference data/retargeting_motion/move-ego--90-2/move-ego--90-2_0.npz \
        --out outputs/rollout_z_on_body/move-ego--90-2_0_child.mp4

Writes <out> (mp4) and <out>.npz (qpos/qvel/action of the physical rollout).
With --reference, the video is side by side: LEFT = the retargeted reference
played back kinematically (no physics), RIGHT = this physical rollout.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
import torch

from humenv import make_humenv
from metamotivo.fb_cpr.huggingface import FBcprModel


def render_reference(xml_path, qpos_seq, width, height, camera):
    """Kinematic playback of a qpos sequence (mj_forward only, no stepping) --
    same math as render_qpos_playback.render_qpos, inlined so this script has
    no cross-script import."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = []
    for t in range(qpos_seq.shape[0]):
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z", required=True, help="(T,256) or (1,256) .npy context vector(s)")
    parser.add_argument("--xml", required=True, help="MJCF of the body to roll out on")
    parser.add_argument("--out", required=True, help="output .mp4 path")
    parser.add_argument("--reference", default=None,
                        help="optional qpos .npz to play back side by side on --xml")
    parser.add_argument("--init-from-reference", action="store_true",
                        help="start the physics from the reference's frame 0 (qvel=0) instead "
                             "of humenv's default standing reset")
    parser.add_argument("--steps", type=int, default=None,
                        help="rollout length (default: len(z), or 300 for a single z)")
    parser.add_argument("--z-reduce", choices=["none", "mean", "first", "last"], default="none",
                        help="collapse a (T,256) z file to one constant z (default: per-frame)")
    parser.add_argument("--model", default="facebook/metamotivo-M-1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="thread count changes float32 reduction order and so changes the "
                             "trajectory; pinned to 1 like metamotivo_motion_rollout.py")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera", default="front_side")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    z_np = np.load(args.z)
    if z_np.ndim == 1:
        z_np = z_np[None]
    if args.z_reduce == "mean":
        # renormalize to the sphere of radius sqrt(d) the actor was trained on
        z_np = z_np.mean(axis=0, keepdims=True)
        z_np = z_np / np.linalg.norm(z_np) * np.sqrt(z_np.shape[1])
    elif args.z_reduce == "first":
        z_np = z_np[:1]
    elif args.z_reduce == "last":
        z_np = z_np[-1:]
    z_all = torch.tensor(z_np, dtype=torch.float32, device=args.device)
    steps = args.steps or (z_all.shape[0] if z_all.shape[0] > 1 else 300)
    print(f"z {tuple(z_all.shape)} (|z|={np.linalg.norm(z_np, axis=1).mean():.2f}), "
          f"rollout {steps} steps on {args.xml}")

    model = FBcprModel.from_pretrained(args.model).to(args.device)
    model.eval()

    env, _ = make_humenv(
        num_envs=1,
        task=None,
        xml=args.xml,
        state_init="Default",
        render_width=args.width,
        render_height=args.height,
        camera=args.camera,
    )

    # One env.step advances 1/render_fps of simulated time (humenv substeps
    # its 1/450 s physics timestep internally), so `steps` and the reference
    # clip's frames are the same clock -- 300 steps == 10 s == 300 ref frames.
    env_fps = env.unwrapped.metadata["render_fps"]

    torch.manual_seed(args.seed)
    obs, _ = env.reset(seed=args.seed)
    ref_qpos = np.load(args.reference)["qpos"] if args.reference else None
    if args.init_from_reference:
        if ref_qpos is None:
            raise SystemExit("--init-from-reference needs --reference")
        env.unwrapped.set_physics(qpos=ref_qpos[0], qvel=np.zeros(env.unwrapped.model.nv))
        obs = env.unwrapped.get_obs()

    frames, qpos_hist, qvel_hist, action_hist = [], [], [], []
    diverged_at = None
    for t in range(steps):
        z_t = z_all[min(t, z_all.shape[0] - 1)].unsqueeze(0)
        obs_t = torch.tensor(obs["proprio"], dtype=torch.float32, device=args.device).unsqueeze(0)
        with torch.no_grad():
            action = model.act(obs_t, z_t, mean=True)
        action_np = action.cpu().numpy().ravel()
        try:
            obs, _, terminated, truncated, info = env.step(action_np)
        except ValueError as e:
            # humenv raises on mjWARN_BADQACC. A body the frozen adult actor
            # cannot stabilise diverges rather than merely falling over, and
            # that IS the result -- keep the frames up to that point instead
            # of losing the whole run.
            print(f"  DIVERGED at t={t} ({t / env_fps:.2f}s): {e}")
            diverged_at = t
            break
        action_hist.append(action_np.copy())
        qpos_hist.append(info["qpos"].copy())
        qvel_hist.append(info["qvel"].copy())
        frames.append(env.render())
        if terminated or truncated:
            print(f"  episode ended at t={t}")
            break
    env.close()

    qpos = np.stack(qpos_hist)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ref_qpos is not None:
        ref_frames = render_reference(args.xml, ref_qpos, args.width, args.height, args.camera)
        n = min(len(ref_frames), len(frames))
        video = [np.concatenate([ref_frames[i], frames[i]], axis=1) for i in range(n)]
    else:
        video = frames
    imageio.mimsave(out_path, video, fps=args.fps)

    np.savez(out_path.with_suffix(".npz"), qpos=qpos, qvel=np.stack(qvel_hist),
             action=np.stack(action_hist).astype(np.float32), fps=args.fps)

    print(f"wrote {len(video)} frames -> {out_path}"
          + (f" (physics diverged at t={diverged_at})" if diverged_at is not None else ""))
    print(f"wrote rollout state -> {out_path.with_suffix('.npz')}")
    print(f"pelvis z: start {qpos[0, 2]:.3f} end {qpos[-1, 2]:.3f} "
          f"min {qpos[:, 2].min():.3f} | xy displacement "
          f"{np.linalg.norm(qpos[-1, :2] - qpos[0, :2]):.3f} m")
    if ref_qpos is not None:
        n = min(len(ref_qpos), len(qpos))
        print(f"reference xy displacement {np.linalg.norm(ref_qpos[n - 1, :2] - ref_qpos[0, :2]):.3f} m")


if __name__ == "__main__":
    main()
