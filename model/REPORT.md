# model/ — Single-Body Cross-Embodiment Adapter Training

Implements the pipeline described in [`model.md`](../model.md) (單一身材訓練流程):
adapt a frozen, pretrained Metamotivo policy — trained on one humanoid body
(`assets/robots/robot.xml`) — to control a different body (`robot_child.xml`)
by learning two small networks on top of it, instead of retraining the whole
policy.

## 1. What's being learned

Two networks, both small relative to the frozen ~hundreds-of-millions-parameter
Metamotivo actor:

| Network | File | Role |
|---|---|---|
| `LatentAdapter` (G_θ) | `networks.py` | `z_beta = z0 + alpha * MLP([beta, z0])` — nudges the pretrained skill embedding `z0` toward one appropriate for body `beta`. |
| `ActionHead` | `networks.py` | Residual correction on the frozen actor's output action, conditioned on `beta` — compensates for the target body's different mass/limb-length dynamics even though the action space size is unchanged. |

`beta` is an 8-dim vector: `leg_scale, arm_scale, torso_scale, head_scale,
leg_girth, arm_girth, torso_girth, head_girth` (`dataset.py:BETA_AXES`), read
straight from `assets/robots/robot_<label>_parameter.json`
(`scripts/scale_robot.py`'s output — see that script for what each axis means).

Everything else — the pretrained F/B/actor/critic networks — stays frozen
(`FBModel.__init__` already sets `requires_grad_(False)` on the whole model).

## 2. Loss

```
L = -R_task(tau_beta) + lambda_rtg * D(tau_beta, tau_beta_ref)
    + lambda_z * ||z_beta - z0||^2 + lambda_phys * L_phys(tau_beta)

D = D_root + D_ee + D_contact + D_pose + D_velocity
```

computed per episode in `train.py:compute_episode_cost` using `losses.py`.

- **R_task**: the original humenv reward function for that row's `reward_name`
  (`humenv.env.make_from_name`), evaluated step-by-step on the target-body
  rollout.
- **D**: how closely the target-body rollout matches a *reference* motion —
  the same origin motion retargeted onto the target body
  (`scripts/qpos_retarget.py` output, `retargeted_motion` column in the
  dataset). If a row has no retargeted reference yet, `D` is just `0` for
  that sample (`losses.functional_equivalence` returns `(0.0, {})` when
  `qpos_ref is None`).
- **L_phys**: joint-limit violations + a crude fall detector, computed from
  the rollout alone (no reference needed).
- **λ_z‖z_β−z_0‖²**: keeps the adapted latent close to the original skill.

All five `D_*` sub-terms and `kinematics.py`'s forward-kinematics helpers are
plain NumPy/MuJoCo, not PyTorch — they're not meant to be backpropagated
through directly (see §3).

## 3. Why REINFORCE, not backprop through the rollout

This is the one design decision in `train.py` worth understanding before
touching the training loop.

`model._actor()` (the frozen actor's underlying `nn.Module`, as opposed to
the public `model.actor()`/`model.act()` convenience methods) **is**
differentiable w.r.t. `z` — verified directly: `model.act()`/`model.actor()`
are wrapped in `@torch.no_grad()`, but calling `model._actor()` and
`model._normalize()` yourself is not. Frozen *weights* don't block gradients
to an *input* tensor.

The actual blocker is MuJoCo: `env.step()` runs physics (contacts,
integration) with no autodiff support, so anything computed from the
resulting `qpos` trajectory — `R_task`, `D`, `L_phys` — cannot be
backpropagated through. `train.py:rollout_episode` instead:

1. Samples actions from `Normal(mean, actor_std)`, where `mean` comes from
   the (differentiable) actor + `ActionHead`, both conditioned on `z_beta`.
2. Accumulates `log_prob_sum` over the episode.
3. After the episode, weights `log_prob_sum` by the realized cost (score-
   function / REINFORCE estimator), with an EMA baseline
   (`cfg.baseline_momentum`) for variance reduction.

The one exception is `λ_z‖z_β−z_0‖²`: it never touches the environment, so
it's added as an ordinary backprop term (`z_reg_loss` in `train.py`) — much
lower variance than REINFORCE for that part of the gradient. Both paths
write into the same `z_beta` leaf tensor; PyTorch just sums the two gradient
contributions.

## 4. Files

```
model/
├── config.py       TrainConfig — every hyperparameter (loss weights, lr,
│                   episode/update counts, network sizes, paths)
├── networks.py      LatentAdapter, ActionHead
├── kinematics.py     forward-kinematics helpers (world body position/
│                   orientation from a raw qpos array, no physics stepping)
├── losses.py         D_root/D_ee/D_contact/D_pose/D_velocity, L_phys
├── dataset.py         CrossEmbodimentDataset — reads
│                   data/crossenbodiment-1-datasets/manifest.jsonl
├── train.py            rollout_episode, compute_episode_cost, train()
├── run_train.py         entry point: `uv run model/run_train.py`
└── checkpoints/          adapter.state_dict() + action_head.state_dict(),
                        written every cfg.ckpt_every updates
```

Data flow: `scripts/metamotivo_rollout.py` (origin motion + z) →
`scripts/qpos_retarget.py` (retargeted motion) → `scripts/build_dataset.py`
(manifest) → `model/dataset.py` → `model/train.py`.

## 5. A bug that was found and fixed while first running this on real data

`D_root`'s trajectory-curvature term (`dheading / speed`) is numerically
ill-conditioned as `speed -> 0` — exactly what happens during near-stationary
motion (crawling, crouching, sitting). A bounded heading change divided by a
near-zero speed exploded to values in the tens of thousands to millions,
completely swamping every other loss term (`cost` was printing as `233884.1`
instead of an `O(1)` number). Fixed in `losses.py:d_root` by:
- masking out samples where `speed <= 1e-3` instead of dividing by it,
- clipping the remaining curvature to `±50 rad/m` (generous — covers a sharp
  human U-turn) before it can blow up from borderline-small speeds,
- normalizing by that same clip so curvature's natural units (rad/m) don't
  dominate the other sub-terms' units (rad² or m²) by construction,
- wrapping `dheading` to `[-π, π]` (arctan2's branch cut otherwise turns a
  turn crossing ±π into a fake ~2π jump).

Also worth knowing about `D_pose`/`D_velocity` (documented in `losses.py`'s
module docstring, not bugs, just simplifications): trajectories of different
length are compared by truncating to the shorter one (no DTW/temporal
alignment), and `D_velocity` uses finite-difference `qpos` deltas as a
stand-in for `qvel`, since reference trajectories only store `qpos`.

## 6. Known limitations / things to revisit

- **Loss weights are un-tuned.** `TrainConfig`'s `lambda_*`/`d_*_weight`
  defaults are a starting point, not calibrated values — model.md itself
  expects this ("神經網路設計等實際訓練後會再調整"). Watch the per-term
  breakdown each `train.py` log line prints (`r_task`, `D`, `L_phys`
  separately) and rebalance if one term dominates like curvature originally
  did.
- **REINFORCE has no baseline network**, just an EMA scalar — fine for a
  first pass, but a learned value-function baseline would reduce gradient
  variance further if training is unstable.
- **`morphology` is fixed to `child` for every row** even though
  `origin_motion`/`origin_z` were generated on the baseline body — this is
  intentional per current scope (single target-body training), not a bug,
  but it means `beta` doesn't currently vary across the dataset. Adapting to
  multiple target bodies in one training run would need per-row env
  switching (not implemented — `train.py` builds one env for `cfg.target_xml`
  up front).
- **`is_terminated()` in HumEnv always returns `False`** (upstream
  `humenv` behavior, not ours), so `L_phys`'s fall detector (pelvis height
  threshold) is currently the only thing that penalizes a collapsed episode.

## 7. Running it

```bash
uv run model/run_train.py                    # default TrainConfig, cuda:0
```

or with custom hyperparameters:

```python
from model.config import TrainConfig
from model.train import train

train(TrainConfig(device="cuda:0", num_updates=500, lr=1e-4))
```

Checkpoints land in `model/checkpoints/update_<N>.pt` as
`{"adapter": ..., "action_head": ..., "update": N, "cfg": TrainConfig}`.
