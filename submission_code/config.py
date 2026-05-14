"""Centralized configuration for the evidential ACL framework.

Modify parameters here instead of hunting through source files.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# =========================================================================
# Simulation
# =========================================================================
@dataclass
class SimConfig:
    dt: float = 0.02                    # integration timestep (s)
    total_time: float = 30.0            # total simulation time (s)
    switch_time: float = 10.0           # time at which perturbation activates (s)

    @property
    def n_steps(self) -> int:
        return int(self.total_time / self.dt)


# =========================================================================
# Trajectory
# =========================================================================
@dataclass
class TrajectoryConfig:
    shape: str = "lemniscate_2d"        # "lemniscate_2d" or "lemniscate_3d"
    scale: float = 4.0                  # spatial scale (meters)
    speed: float = 0.5                  # angular speed of the parametric curve


# =========================================================================
# Unicycle Robot
# =========================================================================
@dataclass
class UnicycleConfig:
    mass: float = 5.0                   # kg
    wheel_radius: float = 0.1          # m
    inertia_y: float = 0.5             # kg·m²  (translational)
    inertia_z: float = 0.2             # kg·m²  (rotational)
    friction: float = 0.1              # viscous friction coefficient
    feature_mode: str = "invariant"    # "invariant" (10-dim, body-frame) or "full" (12-dim)
    init_state: Tuple[float, ...] = (0.5, 0.0, 0.0, 0.0, 0.0)


# =========================================================================
# PID Controller
# =========================================================================
@dataclass
class PIDConfig:
    kp: List[float] = field(default_factory=lambda: [12.0, 8.0])
    ki: List[float] = field(default_factory=lambda: [0.3, 0.1])
    kd: List[float] = field(default_factory=lambda: [4.0, 2.5])
    output_limits: List[List[float]] = field(
        default_factory=lambda: [[-20.0, 20.0], [-10.0, 10.0]]
    )


# =========================================================================
# Neural Network Architecture
# =========================================================================
@dataclass
class NNConfig:
    hidden_dims: Tuple[int, ...] = (128, 128)   # hidden layer sizes
    activation: str = "relu"                      # "relu", "tanh", "elu"
    dropout: float = 0.0                          # dropout probability (0 = off)
    use_batch_norm: bool = False                  # batch normalization


# =========================================================================
# Offline Training (Phase 1)
# =========================================================================
@dataclass
class OfflineConfig:
    n_collection_steps: int = 5000      # PID data collection steps
    epochs: int = 100                    # training epochs
    batch_size: int = 64                 # mini-batch size
    lr: float = 1e-3                     # learning rate
    weight_decay: float = 0.0           # L2 regularization
    lambda_reg: float = 0.01            # evidential regularization weight
    train_val_split: float = 0.9        # fraction for training


# =========================================================================
# Online / Continual Learning (Phase 2)
# =========================================================================
@dataclass
class OnlineConfig:
    lr: float = 5e-4                     # online learning rate
    batch_size: int = 32                 # replay mini-batch size
    lambda_reg: float = 0.01            # evidential regularization weight
    update_every: int = 1               # gradient steps per env step
    warmup_steps: int = 8               # min buffer size before training


# =========================================================================
# Replay Buffer
# =========================================================================
@dataclass
class BufferConfig:
    capacity: int = 500                  # max stored transitions
    buffer_type: str = "random"          # "random", "priority_epistemic", "priority_total"


# =========================================================================
# Active Learning (Phase 3 — to be enabled later)
# =========================================================================
@dataclass
class ActiveLearningConfig:
    enabled: bool = False                # toggle active selection
    score_type: str = "epistemic"        # "epistemic", "aleatoric", "total"
    threshold: float = 0.0              # min score to store (0 = store all)


# =========================================================================
# Visualization / Animation
# =========================================================================
@dataclass
class VisConfig:
    fps: int = 60                        # pygame frames per second
    window_width: int = 900              # pixels
    window_height: int = 700             # pixels
    trail_length: int = 300              # number of past positions to draw
    pixels_per_meter: float = 50.0       # zoom level
    show_heading: bool = True            # draw heading arrow


# =========================================================================
# Adaptive EDL Configuration (NeurIPS 2025)
# =========================================================================
@dataclass
class AdaptiveConfig:
    lambda_0: float = 0.01          # base regularization coefficient λ₀
    lambda_tau: float = 1.0          # temperature τ for λ_i = λ_0·ν/(ν+τ)
    use_full_nig_loss: bool = True   # use adaptive NIG loss (vs MSE on γ only)


# =========================================================================
# Calibration / ACI Configuration
# =========================================================================
@dataclass
class ACIConfig:
    target_coverage: float = 0.90   # desired marginal coverage
    step_size: float = 0.05         # ACI adaptation step size γ
    window_size: int = 200          # sliding window for empirical coverage
    q_init: float = 2.0             # initial quantile multiplier


# =========================================================================
# EWC Configuration
# =========================================================================
@dataclass
class EWCConfig:
    lambda_ewc: float = 500.0       # EWC regularization strength
    ema_gamma: float = 0.99         # Fisher EMA decay for online EWC


# =========================================================================
# Master Config (updated)
# =========================================================================
@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    traj: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    unicycle: UnicycleConfig = field(default_factory=UnicycleConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    active: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    vis: VisConfig = field(default_factory=VisConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    aci: ACIConfig = field(default_factory=ACIConfig)
    ewc: EWCConfig = field(default_factory=EWCConfig)
    device: str = "cpu"
    seed: int = 42
    scenario_name: str = "Sudden Payload"  # scenario for online evaluation


# Singleton default config
DEFAULT_CONFIG = Config()
