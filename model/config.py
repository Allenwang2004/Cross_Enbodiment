import dataclasses
from typing import List


@dataclasses.dataclass
class TrainConfig:
    metamotivo_repo: str = "facebook/metamotivo-M-1"
    dataset_dir: str = "datasets/crossenbodiment-1-datasets"
    target_morphology_json: str = "assets/robots/child/parameter.json"
    target_xml: str = "assets/robots/child/robot.xml"
    device: str = "cuda:0"

    # G_theta: z_beta = z0 + alpha * MLP([beta, z0])
    adapter_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [256, 512, 512, 256])
    adapter_alpha: float = 0.1
    adapter_alpha_learnable: bool = False

    # Action_Head: residual correction on the frozen actor's output, conditioned on beta
    action_head_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 128])

    # loss weights, L = lambda_rtg * D + lambda_z * ||z_beta - z0||^2 + lambda_phys * L_phys
    # (no R_task term -- see train.py module docstring for why)
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
    exploration_std: float = 0.05  # std used to SAMPLE actions for REINFORCE (see train.py) --
                                    # deliberately decoupled from the frozen model's own
                                    # actor_std (0.2): that noise compounds over 300 MuJoCo
                                    # steps and was swamping the D/L_phys learning signal
    batch_size: int = 16  # episodes per update, run as one vectorized HumEnv (see train.py)
    vectorization_mode: str = "sync"  # gymnasium VectorEnv mode for the batched rollout
    num_updates: int = 50
    steps_per_episode: int = 300
    baseline_momentum: float = 0.95  # EMA momentum for both the cost mean AND std used to
                                      # standardize the REINFORCE advantage (see train.py)
    grad_clip_norm: float = 5.0
    seed: int = 0

    log_every: int = 1
    ckpt_dir: str = "model/checkpoints"
    ckpt_every: int = 20
    loss_curve_path: str = "outputs/train_logs/loss_curve.png"
