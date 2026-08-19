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

--obs-scale: showing the actor an adult-sized body
--------------------------------------------------
The obs is 358 features and its DIMENSION does not change with the body -- a
scaled skeleton keeps humenv's 24 rigid bodies, so no reshaping is needed to run
the child. What changes is the units. Measured on move-ego--90-2, adult vs child
at qvel=0:

    root_h_obs             1 dim    child/adult 0.6188   cos 1.0000
    local_body_pos        69 dims               0.6657   cos 0.9965
    local_body_rot_obs   144 dims               1.0000   cos 1.0000

The rotations are bit-identical, because the retarget copies every hinge angle
verbatim; only the features carrying a METRE move, and they move by very nearly
one scalar -- cos 0.9965 says the child's pose vector points the same way and is
simply shorter. So the mismatch the frozen actor sees is a units mismatch, not a
different pose, and it is worth being able to switch off: the obs normalizer is
a BatchNorm holding adult-scale running statistics, so those 70 features arrive
at a systematic offset rather than merely "smaller".

--obs-scale divides every length-carrying feature by its ratio, which is what
makes the child's obs read as an adult's. The ratio is PER BODY, not one number:
scale_robot.py scales legs / arms / torso / head independently (child uses 0.62 /
0.65 / 0.75 / 1.05), so the pelvis-to-body distance scales by 0.6200 in the legs,
0.7500 up the torso, 0.7847 at the head, and along a 0.7181 -> 0.6676 gradient
down the arm as the chain leaves the torso. A single scalar taken from the rest
pelvis height would be 0.6110 -- below every one of them, and 28% wrong at the
head. `--obs-scale <float>` still forces the uniform version, as the ablation.

--obs-scale-parts picks how far to take it: `length` also rescales
local_body_vel, the dimensionally consistent choice under a kinematic retarget
(same angles, same clock, so linear velocity carries the same metre as position
while angular velocity does not); `pose` rescales only the static features and
leaves all 144 velocity dims alone, which is the right choice if the motion is
gravity-driven, where speeds scale like sqrt(L) rather than L.

This is a canonicalisation, and it removes part of the problem rather than
solving it: it tells the actor the body is adult-sized when it is not, so the
motions it commands are calibrated for adult limb lengths. Its use is as a
BASELINE -- run with and without to split "how much of the gap is pure scale"
from "how much is real dynamics" -- not as the default. model/train.py
deliberately does the opposite, feeding raw child obs and letting the adapter
learn the compensation, because canonicalised obs would leave beta nothing to
explain.

Usage (from project root):
    uv run scripts/rollout_z_on_body.py \
        --z data/z/move-ego-0-4/move-ego-0-4_0.npy \
        --xml assets/robot_torque/robot_torque.xml \
        --reference data/retargeting_motion/move-ego-0-4/move-ego-0-4_0.npz \
        --out outputs/rollout_z_on_body/move-ego-0-4_0_robot_torque_init_sym.mp4 \
        --init-from-reference

    # same rollout, but the actor is shown an adult-sized obs
    uv run scripts/rollout_z_on_body.py ... --obs-scale auto

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


# humenv/env.py:compute_humanoid_self_obs_v2 concatenates an OrderedDict in this
# order over 24 rigid bodies; local_body_pos drops the root's own 3, and
# local_body_rot_obs is 6D tan-norm rather than quaternions.
OBS_SEGMENTS = {
    "root_h":       (0, 1),      # metre
    "body_pos":     (1, 70),     # metre
    "body_rot":     (70, 214),   # unitless
    "body_vel":     (214, 286),  # metre / second
    "body_ang_vel": (286, 358),  # radian / second
}
SCALED_BY_PARTS = {
    "pose":   ("root_h", "body_pos"),
    "length": ("root_h", "body_pos", "body_vel"),
}


def body_scale_ratios(xml: str, ref_xml: str) -> np.ndarray:
    """(24,) per-body length ratio of --xml against the reference body.

    One scalar is NOT enough. scale_robot.py scales four groups independently
    and assets/robots/child/parameter.json uses leg 0.62, torso 0.75, arm 0.65,
    head 1.05, so the distance from the pelvis to each body scales by a
    different amount -- measured 0.6200 for every leg body, 0.7500 for the torso
    chain, 0.7847 for the head, and a gradient 0.7181 -> 0.6676 down the arm as
    the chain leaves the torso and accumulates arm segments. The rest pelvis
    height ratio alone is 0.6110, below all of them and 28% wrong at the head.

    Measuring at the rest pose is enough because the ratios barely move: across
    100 frames of move-ego--90-2 the leg and torso ratios are constant to
    0.0000 (single-scale chains) and the mixed arm/head chains hold to a std of
    0.005, differing from their rest value by at most 0.019.

    Index 0 is the root, whose local_body_pos is identically zero and carries no
    length of its own; it gets the rest pelvis height ratio instead, which is
    what phi0_retarget scaled the root translation by.
    """
    out = []
    for path in (ref_xml, xml):
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        d.qpos[:] = m.qpos0
        mujoco.mj_forward(m, d)
        pos = d.xpos[1:25].copy()
        out.append((pos - pos[0], float(m.qpos0[2])))
    (pa, ha), (pb, hb) = out
    ratios = np.ones(24)
    ratios[0] = hb / ha
    na, nb = np.linalg.norm(pa, axis=1), np.linalg.norm(pb, axis=1)
    ok = na > 1e-9
    ratios[1:][ok[1:]] = (nb[1:] / na[1:])[ok[1:]]
    return ratios


def obs_rescaler(ratios: np.ndarray, parts: str) -> np.ndarray:
    """(358,) multiplier making this body's length features read as the
    reference body's. Every feature carrying a metre is divided by its OWN
    body's ratio; rotations and angular velocities are left at 1.0.

    body_pos covers bodies 1..23 (the root's own offset is dropped by humenv),
    while body_vel covers all 24 -- the sensors are world-frame velocities, not
    root-relative ones, so the root has a real velocity to rescale. That makes
    the velocity term the approximate one: a body's world velocity mixes the
    root's translation with its own local motion, and those two carry different
    ratios. Use --obs-scale-parts pose to leave it out.
    """
    scaled = SCALED_BY_PARTS[parts]
    mul = np.ones(358)
    if "root_h" in scaled:
        mul[0] = 1.0 / ratios[0]
    if "body_pos" in scaled:
        mul[1:70] = 1.0 / np.repeat(ratios[1:24], 3)
    if "body_vel" in scaled:
        mul[214:286] = 1.0 / np.repeat(ratios[0:24], 3)
    return mul


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
    parser.add_argument("--obs-scale", default="none",
                        help="rescale the actor's obs to a reference body size: "
                             "'none', 'auto' (PER-BODY ratios measured from the "
                             "two rest poses -- the bodies are scaled by group, "
                             "not uniformly), or an explicit float to force one "
                             "uniform ratio, which is the ablation 'auto' "
                             "replaces. Affects ONLY what the actor is shown -- "
                             "the physics and the saved rollout are untouched")
    parser.add_argument("--obs-scale-parts", choices=["length", "pose"],
                        default="length",
                        help="'length' also rescales local_body_vel (consistent "
                             "under a kinematic retarget); 'pose' leaves all 144 "
                             "velocity features alone")
    parser.add_argument("--obs-scale-ref-xml",
                        default="assets/robots/adult/robot.xml",
                        help="the body whose size the actor was trained on")
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

    if args.obs_scale == "none":
        obs_mul = None
    else:
        if args.obs_scale == "auto":
            ratios = body_scale_ratios(args.xml, args.obs_scale_ref_xml)
            how = (f"per-body ratios {ratios.min():.4f}..{ratios.max():.4f} "
                   f"(root {ratios[0]:.4f})")
        else:
            uniform = float(args.obs_scale)
            if not uniform > 0:
                raise SystemExit(f"--obs-scale must be positive, got {uniform}")
            ratios = np.full(24, uniform)
            how = f"uniform ratio {uniform:.4f}"
        obs_mul = obs_rescaler(ratios, args.obs_scale_parts)
        scaled = ", ".join(SCALED_BY_PARTS[args.obs_scale_parts])
        print(f"obs rescale: {how} -> {int((obs_mul != 1.0).sum())}/358 features "
              f"({scaled}) multiplied by {1.0 / ratios.max():.4f}"
              f"..{1.0 / ratios.min():.4f}")

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
        # Only the actor's view is rescaled. env.step still receives the real
        # physics, and qpos/qvel are recorded unscaled, so the saved rollout
        # stays comparable with runs that did not use --obs-scale.
        proprio = obs["proprio"] if obs_mul is None else obs["proprio"] * obs_mul
        obs_t = torch.tensor(proprio, dtype=torch.float32, device=args.device).unsqueeze(0)
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
