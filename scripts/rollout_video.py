"""Roll one named clip through a trained checkpoint on one body, and look at it.

The rollout itself is model/bilevel/longeval.rollout_one -- the SAME function the
training loop calls every 100 iterations for its `long/` metrics, so the number
printed under the video and the number on the W&B curve cannot drift apart.

Answers a question no aggregate metric can: does the policy actually PERFORM the
motion on this morphology, or does it merely score well? Runs a single
long rollout -- the whole clip from one RSI, not the 24-step windows training
optimizes -- and writes an mp4 with the reference on the left and the physical
robot on the right, plus per-step tracking numbers.

Two deliberate differences from training, both so the video shows the thing we
would actually ship:

  --wrench 0   The external root wrench is a training crutch (proposal.md R5).
               Default is OFF. Pass --wrench 1 to see how much the policy is
               still leaning on it; the honest video is the one without.

  full clip    Training sees 0.8 s windows. Stitching locally-good but globally
               incoherent motion is invisible at that length and obvious here.

Usage:
    python scripts/rollout_video.py --ckpt model/bilevel/checkpoints/stage1_002000.pt \
        --task move-ego-0-2 --body child
    python scripts/rollout_video.py --ckpt ... --task headstand --trial 3 --wrench 1
    python scripts/rollout_video.py --ckpt ... --list-tasks
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
import torch

from model.bilevel.config import BilevelConfig
from model.bilevel.data import WindowDataset
from model.bilevel.longeval import rollout_one
from model.bilevel.policy import LowerPolicy, load_frozen_model
from model.bilevel.retarget import U_DIM, RetargetNet, Retargeter
from model.bilevel.sim.worker import _make_obs_fn
import model.bilevel.rewards as R


def render(xml_path, qpos_seq, width, height, camera):
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=height, width=width)
    frames = []
    for t in range(qpos_seq.shape[0]):
        d.qpos[:] = qpos_seq[t]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=camera)
        frames.append(r.render().copy())
    r.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--task", help="task name, e.g. move-ego-0-2 (see --list-tasks)")
    ap.add_argument("--trial", type=int, default=0, help="which trial of that task")
    ap.add_argument("--body", default="child")
    ap.add_argument("--steps", type=int, default=None,
                    help="cap the rollout length (default: the whole clip)")
    ap.add_argument("--wrench", type=float, default=0.0,
                    help="fraction of the training wrench to allow, 0 = none (default)")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of using the mean")
    ap.add_argument("--device", default=None)
    ap.add_argument("--camera", default="front_side")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--list-tasks", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: BilevelConfig = ck["cfg"]
    if args.device:
        cfg.device = args.device
    dev = torch.device(cfg.device)

    ds = WindowDataset(cfg, bodies=[args.body], verbose=False)
    if args.list_tasks:
        for t in sorted(ds.tasks()):
            print(t)
        return
    if not args.task:
        raise SystemExit("--task is required (or --list-tasks)")

    hits = [c for c in ds.clips if c.task == args.task]
    if not hits:
        raise SystemExit(f"no clip for task {args.task!r}; try --list-tasks")
    if args.trial >= len(hits):
        raise SystemExit(f"task {args.task} has {len(hits)} trials, --trial {args.trial} "
                         f"is out of range")
    clip = sorted(hits, key=lambda c: c.trial)[args.trial]

    frozen = load_frozen_model(cfg)
    policy = LowerPolicy(cfg, frozen, ds.beta_dim).to(dev)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    net = RetargetNet(ds.beta_dim, cfg.retarget_hidden_dims).to(dev).double()
    net.load_state_dict(ck["retarget_net"])
    net.eval()

    spec = ds.bodies[0]
    spec.kin.to(dev)
    rt = Retargeter(cfg, net, spec.kin, ds.source.rest_h).to(dev).double()

    T = clip.qpos.shape[0] - 1
    if args.steps:
        T = min(T, args.steps)
    src = torch.as_tensor(clip.qpos[: T + 1], dtype=torch.float64, device=dev).unsqueeze(0)
    beta = torch.as_tensor(spec.beta, dtype=torch.float64, device=dev).unsqueeze(0)
    with torch.no_grad():
        _, ref, u = rt(src, beta, n_out=T + 1)
    ref = ref[0].cpu().numpy()

    print(f"checkpoint : {args.ckpt}  (stage {ck.get('stage','?')}, iter {ck.get('iter','?')})")
    print(f"body       : {args.body}   (leg {spec.leg_len:.3f} m, "
          f"{spec.model.body_mass.sum():.1f} kg)")
    print(f"clip       : {args.task} trial {args.trial}  "
          f"{T} steps @ 30 Hz = {T/30:.1f} s")
    print(f"wrench     : {args.wrench:.0%} of the training crutch")
    print(f"u (p)    : absmax {float(u.abs().max()):.4f}, "
          f"dz_root {float(rt.params(u)['root_dz'][0]):+.5f} m\n")

    spec.model.opt.timestep = cfg.physics_dt
    qpos, st = rollout_one(
        cfg, spec.model, spec.leg_len, ref, policy, clip.z0, spec.beta,
        _make_obs_fn(), wrench_frac=args.wrench,
        deterministic=not args.stochastic, device=dev,
    )

    print(f"survived   : {st['alive_steps']}/{st['steps']} steps "
          f"({st['alive_steps']/st['steps']:.0%})"
          + (f", first failure at step {st['first_failure']}"
             if st["first_failure"] is not None else ", never failed"))
    print(f"pose_err   : {st['pose_err']:.4f} rad^2 mean "
          f"(termination threshold {cfg.term_pose_err})")
    print(f"root drift : {st['root_dist']:.3f} m mean, {st['root_dist_max']:.3f} m max")

    out = args.out or (REPO_ROOT / "outputs" / "rollout_video" /
                       f"{args.body}_{args.task}_{args.trial}"
                       f"{'_wrench' if args.wrench else ''}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    left = render(spec.xml_path, ref, args.width, args.height, args.camera)
    right = render(spec.xml_path, qpos, args.width, args.height, args.camera)
    imageio.mimsave(str(out), [np.concatenate([a, b], axis=1)
                               for a, b in zip(left, right)], fps=30)
    print(f"\nwrote {out}   (left: reference   right: physical robot)")


if __name__ == "__main__":
    main()
