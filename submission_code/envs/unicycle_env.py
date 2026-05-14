"""Unicycle environment for simulation.

State: [x, y, theta, v, omega]  (5-dim)
Action: [tau_drive, tau_steer]   (2-dim)
"""
import numpy as np


class UnicycleEnv:
    # --- Robot physical parameters ---
    MASS = 2.0           # kg
    INERTIA_Y = 0.01     # kg·m² (wheel)
    INERTIA_Z = 0.05     # kg·m² (yaw)
    WHEEL_RADIUS = 0.1   # m
    FRICTION = 0.5       # viscous friction coefficient

    # Box limits used by platform-agnostic adapters for action clipping
    ACTION_LIMITS = ((-20.0, 20.0), (-10.0, 10.0))

    def __init__(self, dt=0.02):
        self.dt = dt
        self.state = np.zeros(5)
        self._mass_multiplier = 1.0
        self._friction_multiplier = 1.0
        # Observation-noise parameters (applied only by get_observation()):
        #   std_pos  → σ added to x, y
        #   std_ori  → σ added to θ
        #   std_vel  → σ added to v, ω
        self._noise_std_pos = 0.0
        self._noise_std_ori = 0.0
        self._noise_std_vel = 0.0
        # Per-env RNG. Built fresh on reset(seed=...). Drives observation
        # noise so multi-seed sweeps produce genuinely different rollouts.
        self._rng = np.random.default_rng(0)
        # World-frame wind force (N), unmodeled disturbance.  Added to the
        # commanded torque after rotation into body frame.  The same
        # signal is what causes "double wind" if a scenario stores the
        # post-wind torque as an adaptation label — adapters MUST store
        # the commanded action only.
        self.wind_x = 0.0
        self.wind_y = 0.0
        # Internal time used for gust shaping
        self._time = 0.0
        # Last actually-applied torque (commanded + wind), recorded for
        # diagnostics and the animation; never used as an adapter label.
        self.last_actual_torque = np.zeros(2)

    def reset(self, state=None, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if state is not None:
            self.state = state.copy()
        else:
            self.state = np.zeros(5)
            if seed is not None:
                # Randomise initial pose under the seeded RNG so that the
                # baseline controllers (LQR / Frozen) actually produce
                # seed-to-seed variation. Velocity terms are left at zero
                # to stay inside each baseline's small-angle linearisation.
                self.state[:3] += self._rng.uniform(-0.05, 0.05, size=3)
        self._time = 0.0
        return self.state.copy()

    def get_state(self):
        return self.state.copy()

    def set_noise(self, std_pos=0.0, std_ori=0.0, std_vel=0.0):
        """Enable Gaussian observation noise on the state returned by ``get_observation``.

        The true internal state is unaffected.
        """
        self._noise_std_pos = float(std_pos)
        self._noise_std_ori = float(std_ori)
        self._noise_std_vel = float(std_vel)

    def get_observation(self):
        """Noisy observation of the current state (noise-free if ``set_noise`` is disabled)."""
        s = self.state.copy()
        if self._noise_std_pos > 0.0:
            s[0] += self._rng.normal(0.0, self._noise_std_pos)
            s[1] += self._rng.normal(0.0, self._noise_std_pos)
        if self._noise_std_ori > 0.0:
            s[2] += self._rng.normal(0.0, self._noise_std_ori)
        if self._noise_std_vel > 0.0:
            s[3] += self._rng.normal(0.0, self._noise_std_vel)
            s[4] += self._rng.normal(0.0, self._noise_std_vel)
        return s

    def set_uncertainty(self, active, mass_multiplier=1.0, friction_multiplier=1.0):
        if active:
            self._mass_multiplier = mass_multiplier
            self._friction_multiplier = friction_multiplier
        else:
            self._mass_multiplier = 1.0
            self._friction_multiplier = 1.0

    @property
    def effective_mass(self):
        m = self.MASS * self._mass_multiplier
        return m + self.INERTIA_Y / (self.WHEEL_RADIUS ** 2)

    @property
    def effective_Iz(self):
        return self.INERTIA_Z * self._mass_multiplier

    @property
    def effective_friction(self):
        return self.FRICTION * self._friction_multiplier

    def set_wind(self, wind_x: float = 0.0, wind_y: float = 0.0):
        """World-frame constant + small-gust wind force (N).

        The wind is applied AFTER the commanded torque clip, so the
        actually-experienced torque can exceed the commanded box by the
        wind amplitude — this is what makes wind genuinely OOD for the
        NN inverse-dynamics model.
        """
        self.wind_x = float(wind_x)
        self.wind_y = float(wind_y)

    def step(self, action):
        """Advance one timestep. Returns next_state."""
        action = np.array(action, dtype=np.float64)
        tau_drive_cmd = float(np.clip(action[0], -20.0, 20.0))
        tau_steer_cmd = float(np.clip(action[1], -10.0, 10.0))

        x, y, theta, v, omega = self.state
        m_eff = self.effective_mass
        I_z = self.effective_Iz
        friction = self.effective_friction
        dt = self.dt

        # ── Wind / disturbance (world-frame -> body-frame) ───────────────
        if self.wind_x != 0.0 or self.wind_y != 0.0:
            # ±50 % gust modulation at 0.2-0.3 Hz — matches the
            # `ExtendedEnv` used by `animate.py` so the benchmark
            # exercises the same disturbance regime the user sees in
            # the live visualisation.  Earlier benchmarks used ±25 %
            # which made the disturbance too gentle for the K-feedback
            # to require the NN's adaptive feedforward.
            gx = self.wind_x * (1.0 + 0.50 * np.sin(2 * np.pi * 0.30 * self._time))
            gy = self.wind_y * (1.0 + 0.50 * np.cos(2 * np.pi * 0.20 * self._time + 0.7))
            wb_drive = np.cos(theta) * gx + np.sin(theta) * gy
            wb_steer = (-np.sin(theta) * gx + np.cos(theta) * gy) * 0.5
        else:
            wb_drive = wb_steer = 0.0

        # Actually applied torque (commanded + wind)
        tau_drive = float(np.clip(tau_drive_cmd + wb_drive, -25.0, 25.0))
        tau_steer = float(np.clip(tau_steer_cmd + wb_steer, -12.0, 12.0))
        # Diagnostics only — adapters MUST keep using the commanded
        # action as their self-supervised label (otherwise wind doubles).
        self.last_actual_torque = np.array([tau_drive, tau_steer])

        # Velocity dynamics
        v_dot = (tau_drive - friction * v) / m_eff
        omega_dot = (tau_steer - friction * omega) / I_z

        # Euler integration
        v_new = v + v_dot * dt
        omega_new = omega + omega_dot * dt
        theta_new = theta + omega * dt
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt

        self.state = np.array([x_new, y_new, theta_new, v_new, omega_new])
        self._time += dt
        return self.state.copy()

    # ------------------------------------------------------------------
    # NN input builders
    # ------------------------------------------------------------------

    def build_inverse_dynamics_input(self, state_t, state_tp1, feature_mode="invariant"):
        """Build input for the inverse dynamics NN.

        The NN learns:  f(state_t, delta_state) -> action_t

        feature_mode="full" (12-dim):
            [0]  v_t              current forward velocity
            [1]  omega_t          current angular velocity
            [2]  cos(theta_t)     current heading (cos)
            [3]  sin(theta_t)     current heading (sin)
            [4]  dv / dt          velocity change (scaled by 1/dt)
            [5]  domega / dt      angular velocity change (scaled by 1/dt)
            [6]  dx_body / dt     body-frame x displacement rate
            [7]  dy_body / dt     body-frame y displacement rate
            [8]  dtheta / dt      heading change rate (= omega effectively)
            [9]  v_tp1            next forward velocity
            [10] omega_tp1        next angular velocity
            [11] dt               timestep (constant, but included for generality)

        feature_mode="invariant" (10-dim):
            Omits cos(theta_t) and sin(theta_t). The unicycle dynamics
            v_dot = (tau_drive - friction*v) / m_eff
            omega_dot = (tau_steer - friction*omega) / I_z
            are independent of heading, so removing these achieves full
            orientation invariance. cos/sin are still used internally
            for the body-frame rotation.

        Using rates (delta/dt) rather than raw deltas makes the input
        scale-invariant to the timestep.
        """
        x_t, y_t, th_t, v_t, om_t = state_t
        x_tp1, y_tp1, th_tp1, v_tp1, om_tp1 = state_tp1

        dt = self.dt

        # Velocity derivatives
        dv_dt = (v_tp1 - v_t) / dt
        dom_dt = (om_tp1 - om_t) / dt

        # Position change in body frame
        dx_world = x_tp1 - x_t
        dy_world = y_tp1 - y_t
        cos_t = np.cos(th_t)
        sin_t = np.sin(th_t)
        dx_body = cos_t * dx_world + sin_t * dy_world
        dy_body = -sin_t * dx_world + cos_t * dy_world

        # Heading change
        dth = (th_tp1 - th_t + np.pi) % (2 * np.pi) - np.pi

        if feature_mode == "invariant":
            return np.array([
                v_t, om_t,
                dv_dt, dom_dt,
                dx_body / dt, dy_body / dt,
                dth / dt,
                v_tp1, om_tp1,
                dt,
            ], dtype=np.float64)

        if feature_mode == "global":
            # Ablation: replace body-frame relative displacement with world coordinates
            # and heading change with raw heading.
            # Must remain 12-dim to match the 'full' model architecture.
            return np.array([
                v_t, om_t,
                x_t, y_t,           # World coords (non-invariant)
                dv_dt, dom_dt,
                x_tp1, y_tp1,       # Next world coords
                th_t,               # Raw heading
                v_tp1, om_tp1,
                dt,
            ], dtype=np.float64)

        return np.array([
            v_t, om_t,
            cos_t, sin_t,
            dv_dt, dom_dt,
            dx_body / dt, dy_body / dt,
            dth / dt,
            v_tp1, om_tp1,
            dt,
        ], dtype=np.float64)

    def build_inverse_dynamics_input_for_control(self, state_t, desired_state,
                                                   feature_mode="invariant"):
        """Build input for using the inverse dynamics NN as a controller.

        At inference time, we substitute state_{t+1} with the desired state.
        The NN then outputs: "what torques would cause this transition?"

        Args:
            state_t: current state [x, y, theta, v, omega]
            desired_state: desired next state [x, y, theta, v, omega]
            feature_mode: "full" or "invariant"

        Returns:
            nn_input, same format as build_inverse_dynamics_input
        """
        return self.build_inverse_dynamics_input(state_t, desired_state,
                                                  feature_mode=feature_mode)

    # Legacy methods for backward compatibility
    def build_nn_input(self, state, desired_future, ref_v=0.0, ref_omega=0.0):
        """Legacy 11-dim input for behavioral cloning models."""
        x, y, theta, v, omega = state
        ref_x, ref_y, ref_theta = desired_future[0], desired_future[1], desired_future[2]

        dx = ref_x - x
        dy = ref_y - y
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        e_x = cos_t * dx + sin_t * dy
        e_y = -sin_t * dx + cos_t * dy
        e_theta = (ref_theta - theta + np.pi) % (2 * np.pi) - np.pi
        dist = np.sqrt(dx**2 + dy**2)

        return np.array([
            v, omega, e_x, e_y,
            np.sin(e_theta), np.cos(e_theta), e_theta,
            ref_v, ref_omega, dist, cos_t,
        ], dtype=np.float64)

    def build_nn_input_legacy(self, state, desired_future):
        """Legacy 7-dim input."""
        x, y, theta, v, omega = state
        ref_x, ref_y, ref_theta = desired_future[0], desired_future[1], desired_future[2]

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dx = ref_x - x
        dy = ref_y - y
        e_x = cos_t * dx + sin_t * dy
        e_y = -sin_t * dx + cos_t * dy
        dist = np.sqrt(dx**2 + dy**2)

        return np.array([cos_t, sin_t, v, omega, e_x, e_y, dist], dtype=np.float64)

# Verify UnicycleEnv has build_inverse_dynamics_input
# If it doesn't exist, the NN controller will crash.
# This should already be defined in envs/unicycle_env.py