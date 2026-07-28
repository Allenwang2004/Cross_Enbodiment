# HumEnv reward function reference

Source: `humenv/rewards.py` (`reward_from_name` regexes) and `humenv/__init__.py`
(`ALL_TASKS`) in the installed `humenv` package.

Every entry below is a **name pattern** accepted by `humenv.env.make_from_name(name)`,
which is what `--reward-name` in `metamotivo_rollout.py` passes through to
`model.reward_wr_inference` after relabeling the buffer.

## 1. Full parameter space (continuous params, no fixed range)

These are the raw regexes. Continuous parameters (angle/speed/height/distance/
ang_vel) accept **any** float matching the regex, so the combination count is
technically infinite unless you pick a discretization step. Use
`humenv_reward_combinations.py` (below) to enumerate a chosen grid.

| # | Reward class | Name pattern | Params | Notes |
|---|---|---|---|---|
| 1 | LocomotionReward | `move-ego-<angle>-<speed>` | angle: float (deg), speed: float (m/s, signed) | walk/run in a direction |
| 2 | LocomotionReward (low) | `move-ego-low-<angle>-<speed>` | angle, speed | same but crouched/low posture |
| 3 | JumpReward | `jump-<height>` | height: float ≥ 0 | jump to target head height |
| 4 | HeadstandReward | `headstand` | — (no params) | fixed pose reward |
| 5 | RotationReward | `rotate-<axis>-<ang_vel>-<pelvis_height>` | axis: {x,y,z}, ang_vel: float (signed), pelvis_height: float ≥ 0 | spin around an axis |
| 6 | ArmsReward | `raisearms-<left>-<right>` | left, right: {l,m,h,x} | arm pose, low/mid/high/cross |
| 7 | LieDownReward | `lieonground-up` \| `lieonground-down` | — (fixed strings only) | lie on back / front |
| 8 | SplitReward | `split-<distance>` | distance: float ≥ 0 | leg split distance |
| 9 | SitOnGroundReward | `sitonground` | — (no params) | fixed pose reward |
| 10 | SitOnGroundReward (crouch) | `crouch-<height_th>` | height_th: float ≥ 0 | crouch below pelvis height threshold |
| 11 | CrawlReward | `crawl-<speed>-<angle>-<u\|d>` | speed: float ≥ 0, angle: float (deg), u/d: spine up/down | crawling |
| 12 | MoveAndRaiseArmsReward | `move-ego-<angle>-<speed>-raisearms-<left>-<right>` | angle, speed, left, right: {l,m,h,x} | walk while raising arms |
| 13 | ZeroReward | `none` \| `zero` \| `rewardfree` | — | no-op reward (same as `task=None`) |

Regex source lines (for reference, `humenv/rewards.py`):
```
move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)
move-ego-low-(-?\d+\.*\d*)-(-?\d+\.*\d*)
jump-(\d+\.*\d*)
headstand                                   (exact string)
rotate-(x|y|z)-(-?\d+\.*\d*)-(\d+\.*\d*)
raisearms-(l|m|h|x)-(l|m|h|x)
lieonground-up / lieonground-down           (exact strings)
split-(\d+\.*\d*)
sitonground                                 (exact string)
crouch-(\d+\.*\d*)
crawl-(\d+\.*\d*)-(\d+\.*\d*)-(u|d)
move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)-raisearms-(l|m|h|x)-(l|m|h|x)
none / zero / rewardfree                    (exact strings)
```

## 2. Official pre-discretized task set (`humenv.ALL_TASKS`)

This is the exact grid Meta Motivo's own benchmark evaluates against — i.e.
values the model was actually scored on, so they're a safe/meaningful
starting point (defined in `humenv/__init__.py`):

| Group | Grid | Count |
|---|---|---|
| STAND_TASKS | `move-ego-0-0`, `move-ego-low-0-0`, `headstand` | 3 |
| LOCOMOTION_TASKS | angle ∈ {0, -90, 90, 180} × speed ∈ {2, 4} | 8 |
| LOCOMOTION_LOW_TASKS | angle ∈ {0, -90, 90, 180} × speed ∈ {2} | 4 |
| JUMP_TASKS | `jump-2` | 1 |
| ROTATION_TASKS | axis ∈ {x,y,z} × ang_vel ∈ {-5, 5} (pelvis_height fixed 0.8) | 6 |
| RAISE_ARMS_TASKS | left ∈ {l,m,h} × right ∈ {l,m,h} | 9 |
| SITTING_LIEONGROUND_TASKS | `crouch-0`, `sitonground`, `lieonground-up`, `lieonground-down`, `split-0.5`, `split-1` | 6 |
| CRAWL_TASKS | direction ∈ {u,d} × height ∈ {0.4,0.5} × speed ∈ {0,2} | 8 |
| **STANDARD_TASKS** (sum of the above) | | **45** |
| MOVE_AND_RAISE_HANDS_TASKS | angle ∈ {0} × speed ∈ {2} × left ∈ {l,m,h} × right ∈ {l,m,h} | 9 |
| **ALL_TASKS** (STANDARD_TASKS + MOVE_AND_RAISE_HANDS_TASKS) | | **54** |

Get it directly in Python:
```python
import humenv
print(len(humenv.ALL_TASKS))   # 54
print(humenv.ALL_TASKS)
```
