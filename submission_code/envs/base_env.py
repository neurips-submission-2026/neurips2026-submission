"""Abstract base class for robot environments."""
from abc import ABC, abstractmethod
import numpy as np


class BaseRobotEnv(ABC):
    """Base class for all robot simulation environments.

    Each environment wraps either a PyBullet simulation or a
    numerical dynamics model and provides a uniform interface.
    """

    def __init__(self, dt=0.01, use_pybullet=True):
        self.dt = dt
        self.use_pybullet = use_pybullet
        self.time_step = 0

        # Perturbation flags
        self._uncertainty_active = False
        self._disturbance_active = False
        self._noise_active = False
        self._noise_std_pos = 0.0
        self._noise_std_ori = 0.0
        self._disturbance_force = np.zeros(3)
        self._mass_multiplier = 1.0

    # ---- Abstract interface ----

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of full state vector."""

    @property
    @abstractmethod
    def control_dim(self) -> int:
        """Dimension of control input."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of observed output (position/attitude)."""

    @property
    @abstractmethod
    def r_bar(self) -> int:
        """Maximum relative degree."""

    @abstractmethod
    def reset(self, state=None) -> np.ndarray:
        """Reset environment and return initial state."""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple:
        """Apply action, advance one timestep.

        Returns:
            state: full state vector
            output: observed output (possibly noisy)
            reward: negative tracking error (optional)
            info: dict with extra data
        """

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Get current full state (no noise)."""

    @abstractmethod
    def get_output(self) -> np.ndarray:
        """Get current observed output (with noise if active)."""

    @abstractmethod
    def build_nn_input(self, state, desired_future) -> np.ndarray:
        """Build the physics-informed NN input vector.

        Applies invariances: translation, rotation, periodicity.
        """

    @abstractmethod
    def build_inverse_dynamics_input(self, state_t, state_tp1,
                                      feature_mode="invariant") -> np.ndarray:
        """Build input for the inverse dynamics NN.

        The NN learns: f(state_t, delta_state) -> action_t

        Should return a body-frame, physics-informed feature vector encoding
        the transition from state_t to state_tp1.

        Parameters
        ----------
        state_t : array — current state
        state_tp1 : array — next state
        feature_mode : str — "full" or "invariant"
        """

    def build_inverse_dynamics_input_for_control(self, state_t, desired_state,
                                                  feature_mode="invariant"):
        """Build input for using the inverse dynamics NN as a controller.

        At inference time, we substitute state_{t+1} with the desired state.
        """
        return self.build_inverse_dynamics_input(state_t, desired_state,
                                                  feature_mode=feature_mode)

    @abstractmethod
    def get_pid_error(self, desired_output) -> np.ndarray:
        """Compute error vector for PID control."""

    # ---- Perturbation controls ----

    def set_uncertainty(self, active, mass_multiplier=2.0):
        """Toggle internal parameter uncertainty (e.g. doubled mass)."""
        self._uncertainty_active = active
        self._mass_multiplier = mass_multiplier if active else 1.0

    def set_disturbance(self, active, force=None):
        """Toggle external disturbance (constant force)."""
        self._disturbance_active = active
        if active and force is not None:
            self._disturbance_force = np.asarray(force)
        elif not active:
            self._disturbance_force = np.zeros(3)

    def set_noise(self, active, std_pos=0.1, std_ori=0.1):
        """Toggle measurement noise."""
        self._noise_active = active
        self._noise_std_pos = std_pos if active else 0.0
        self._noise_std_ori = std_ori if active else 0.0

    def _apply_noise(self, output, n_pos, n_ori):
        """Add Gaussian noise to output measurements."""
        if not self._noise_active:
            return output
        noisy = output.copy()
        noisy[:n_pos] += np.random.normal(0, self._noise_std_pos, n_pos)
        if n_ori > 0:
            noisy[n_pos : n_pos + n_ori] += np.random.normal(0, self._noise_std_ori, n_ori)
        return noisy