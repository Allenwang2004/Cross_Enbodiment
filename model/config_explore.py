import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class ExploreConfig:
    """Config for model/train_explore.py -- kinematics-only action-network
    exploration with a fixed, unmodified standard-body z (no LatentAdapter,
    no beta, no retargeted-motion reference). See TrainConfig (model/config.py)
    for the full adapter pipeline this is a stripped-down control experiment
    against."""

    metamotivo_repo: str = "facebook/metamotivo-M-1"

    # Reused only to enumerate existing origin_z entries (and the
    # train/test task split) -- beta/retargeted_motion fields in this
    # dataset are ignored entirely by train_explore.py.
    dataset_dir: str = "datasets/crossenbodiment-1-datasets"

    # Child body geometry/inertia, but robot.xml's original (unscaled)
    # actuator gains -- see scripts/make_child_origact_xml.py.
    target_xml: str = "assets/robots/child/robot.xml"
    device: str = "cuda:0"

    # ActionResidual: residual correction on the frozen actor's output, no beta
    action_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 128])

    # loss weight -- L = lambda_phys * L_phys (kinematics-only, no D/qpos_ref;
    # see model/losses.physics_penalty)
    lambda_phys: float = 1.0

    # optimization (REINFORCE with a moving-average baseline; same reasoning
    # as TrainConfig -- the MuJoCo rollout is not autodiff-differentiable)
    lr: float = 3e-4
    exploration_std: float = 0.05
    batch_size: int = 16
    vectorization_mode: str = "sync"
    num_updates: int = 50
    steps_per_episode: int = 300
    baseline_momentum: float = 0.95
    grad_clip_norm: float = 5.0
    seed: int = 0

    log_every: int = 1
    ckpt_dir: str = "model/checkpoints_explore"
    ckpt_every: int = 20
    loss_curve_path: str = "outputs/train_logs/loss_curve_explore.png"

    use_wandb: bool = True
    wandb_project: str = "crossenbodiment-explore"
    wandb_run_name: Optional[str] = None
