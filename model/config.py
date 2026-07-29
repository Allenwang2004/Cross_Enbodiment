import dataclasses
from typing import List


@dataclasses.dataclass
class TrainConfig:
    metamotivo_repo: str = "facebook/metamotivo-M-1"
    dataset_dir: str = "datasets/crossenbodiment-1-datasets"
    target_morphology_json: str = "assets/robots/robot_child_parameter.json"
    target_xml: str = "assets/robots/robot_child.xml"
    device: str = "cpu"

    # G_theta: z_beta = z0 + alpha * MLP([beta, z0])
    adapter_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 512, 512, 256])
    adapter_alpha: float = 0.1
    adapter_alpha_learnable: bool = False

    # Action_Head: residual correction on the frozen actor's output, conditioned on beta
    action_head_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 128])

    # loss weights, L = -R_task + lambda_rtg * D + lambda_z * ||z_beta - z0||^2 + lambda_phys * L_phys
    lambda_rtg: float = 1.0
    lambda_z: float = 0.1
    lambda_phys: float = 1.0
    d_root_weight: float = 1.0
    d_ee_weight: float = 1.0
    d_contact_weight: float = 1.0
    d_pose_weight: float = 1.0
    d_velocity_weight: float = 1.0

    # optimization (REINFORCE with a moving-average baseline; see train.py docstring
    # for why -- the MuJoCo rollout is not autodiff-differentiable)
    lr: float = 3e-4
    episodes_per_update: int = 4
    num_updates: int = 200
    steps_per_episode: int = 300
    baseline_momentum: float = 0.95
    seed: int = 0

    log_every: int = 1
    ckpt_dir: str = "model/checkpoints"
    ckpt_every: int = 20
