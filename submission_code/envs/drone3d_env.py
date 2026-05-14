"""Quadrotor environment with proper torque/thrust actuation.

State (12-d):
    [x, y, z,           world-frame position (m)
     phi, theta, psi,   body-axis Euler angles, ZYX convention (rad)
     vx, vy, vz,        world-frame linear velocity (m/s)
     p, q, r]           body-frame angular velocity (rad/s)

Action (4-d):
    [T, tau_x, tau_y, tau_z]
        T          total upward thrust along body-z axis (N), >= 0
        tau_x, ..  body-frame torques about x, y, z (N·m)

Dynamics (continuous-time):
    Position:  d(x,y,z)/dt = (vx, vy, vz)
    Velocity:  m * d(v)/dt = T * R(phi,theta,psi) @ ez_body
                              - m*g*ez_world + drag(v) + wind
    Attitude:  d(phi,theta,psi)/dt = W(phi,theta) @ (p, q, r)
    Angular:   I * d(omega)/dt = tau - omega x (I*omega)
    where I = diag(Ixx, Iyy, Izz).

The class uses a 12-state quadrotor interface. Older 8-state
checkpoints are not compatible with this environment.
"""
from __future__ import annotations

import numpy as np


def _ssa(angle: float) -> float:
    return angle - 2.0 * np.pi * np.floor_divide(angle + np.pi, 2.0 * np.pi)


def _Rzyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) rotation matrix body->world."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    R = np.array([
        [cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi],
        [spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi],
        [   -sth,                cth*sphi,                cth*cphi    ],
    ])
    return R


def _euler_rates(phi: float, theta: float,
                 p: float, q: float, r: float) -> np.ndarray:
    """Map body angular rates (p,q,r) -> Euler rates (phi_dot,theta_dot,psi_dot)."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth = max(np.cos(theta), 1e-3)  # avoid singularity at +/- pi/2
    tth = np.tan(theta)
    return np.array([
        p + sphi*tth*q + cphi*tth*r,
        cphi*q - sphi*r,
        (sphi/cth)*q + (cphi/cth)*r,
    ])


class Drone3DEnv:
    """Quadrotor with (T, tau_x, tau_y, tau_z) actuation."""

    # --- Physical parameters (1.5 kg quadrotor) ---
    MASS = 1.5
    G    = 9.81
    Ixx  = 0.020
    Iyy  = 0.020
    Izz  = 0.040

    # Linear aerodynamic drag (world frame) and rotational damping (body)
    K_drag_lin = 0.30      # F_drag = -k * v
    K_drag_ang = 0.05      # tau_drag = -k * omega

    # Action limits.  Total thrust spans 0 to ~3*hover (m*g ≈ 14.7 N → up to 45 N).
    # Torques cover a few times the moment that a small mass shift introduces.
    ACTION_LIMITS = (
        (0.0, 45.0),    # T (N), motor thrust never reverses
        (-2.0, 2.0),    # tau_x (N·m)
        (-2.0, 2.0),    # tau_y (N·m)
        (-1.0, 1.0),    # tau_z (N·m)
    )

    # Backwards-compat aliases used by LQR3D / data_collection / tests.  They
    # describe the quadrotor's first-order linearisation around hover so the
    # generic 8-state damping fields still work for code that hasn't been
    # ported to the 12-state interface.
    X_u = -K_drag_lin
    Y_v = -K_drag_lin
    Z_w = -K_drag_lin
    N_r = -K_drag_ang
    X_uc = 0.0
    Y_vc = 0.0
    Z_wc = 0.0
    N_rc = 0.0
    I_zz = Izz

    # State dimensionality, exposed for discovery by adapters.
    STATE_DIM = 12
    ACTION_DIM = 4

    def __init__(self, dt: float = 0.02):
        self.dt = float(dt)
        self.state = np.zeros(12)
        self._mass_mult = 1.0
        self._drag_mult = 1.0
        self._wind = np.zeros(3)

        self._noise_std_pos = 0.0
        self._noise_std_ori = 0.0
        self._noise_std_vel = 0.0

        # Per-env RNG. Built fresh on reset(seed=...).
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def reset(self, state: np.ndarray | None = None,
              seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if state is not None:
            self.state = np.asarray(state, dtype=np.float64).copy()
        else:
            # Hover at z=3 m, level attitude, zero velocities.
            self.state = np.array([
                0.0, 0.0, 3.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ])
            if seed is not None:
                self.state[:3] += self._rng.uniform(-0.10, 0.10, size=3)
                self.state[3:6] += self._rng.uniform(-0.087, 0.087, size=3)
        return self.state.copy()

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def set_perturbation(self, mass_mult: float = 1.0, drag_mult: float = 1.0,
                         wind_x: float = 0.0, wind_y: float = 0.0,
                         wind_z: float = 0.0):
        self._mass_mult = float(mass_mult)
        self._drag_mult = float(drag_mult)
        self._wind = np.array([wind_x, wind_y, wind_z], dtype=np.float64)

    def set_uncertainty(self, active: bool, mass_multiplier: float = 1.0,
                        friction_multiplier: float = 1.0):
        if active:
            self.set_perturbation(mass_mult=mass_multiplier,
                                  drag_mult=friction_multiplier)
        else:
            self.set_perturbation(1.0, 1.0, 0.0, 0.0, 0.0)

    def set_noise(self, std_pos: float = 0.0, std_ori: float = 0.0,
                  std_vel: float = 0.0):
        self._noise_std_pos = float(std_pos)
        self._noise_std_ori = float(std_ori)
        self._noise_std_vel = float(std_vel)

    def get_observation(self) -> np.ndarray:
        s = self.state.copy()
        if self._noise_std_pos > 0.0:
            s[0:3] += self._rng.normal(0.0, self._noise_std_pos, size=3)
        if self._noise_std_ori > 0.0:
            s[3:6] += self._rng.normal(0.0, self._noise_std_ori, size=3)
        if self._noise_std_vel > 0.0:
            s[6:9]  += self._rng.normal(0.0, self._noise_std_vel, size=3)
            s[9:12] += self._rng.normal(0.0, self._noise_std_vel, size=3)
        return s

    # ------------------------------------------------------------------
    # Effective parameters under perturbation
    # ------------------------------------------------------------------

    @property
    def _m_eff(self):  return self.MASS * self._mass_mult
    @property
    def _Iz_eff(self): return self.Izz  * self._mass_mult

    @property
    def _Ixx_eff(self): return self.Ixx * self._mass_mult
    @property
    def _Iyy_eff(self): return self.Iyy * self._mass_mult

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64)
        T, tx, ty, tz = [float(np.clip(a[i], lo, hi))
                         for i, (lo, hi) in enumerate(self.ACTION_LIMITS)]

        x, y, z, phi, th, psi, vx, vy, vz, p, q, r = self.state
        m = self._m_eff
        dmu = self._drag_mult

        # ---- Translational dynamics (world frame) ----
        R = _Rzyx(phi, th, psi)
        thrust_world = R @ np.array([0.0, 0.0, T])
        gravity = np.array([0.0, 0.0, -m * self.G])
        drag = -dmu * self.K_drag_lin * np.array([vx, vy, vz])
        wind = self._wind.copy()
        a_lin = (thrust_world + gravity + drag + wind) / m

        # ---- Rotational dynamics (body frame) ----
        Ix, Iy, Iz = self._Ixx_eff, self._Iyy_eff, self._Iz_eff
        # Euler equations: I·ω̇ = τ − ω × (I·ω) − damping·ω
        gyro_x = (Iy - Iz) * q * r
        gyro_y = (Iz - Ix) * p * r
        gyro_z = (Ix - Iy) * p * q
        damp = dmu * self.K_drag_ang
        p_dot = (tx - gyro_x - damp * p) / Ix
        q_dot = (ty - gyro_y - damp * q) / Iy
        r_dot = (tz - gyro_z - damp * r) / Iz

        # ---- Euler-angle rates ----
        euler_d = _euler_rates(phi, th, p, q, r)

        dt = self.dt
        new = np.array([
            x + vx * dt,
            y + vy * dt,
            z + vz * dt,
            _ssa(phi + euler_d[0] * dt),
            _ssa(th  + euler_d[1] * dt),
            _ssa(psi + euler_d[2] * dt),
            vx + a_lin[0] * dt,
            vy + a_lin[1] * dt,
            vz + a_lin[2] * dt,
            p + p_dot * dt,
            q + q_dot * dt,
            r + r_dot * dt,
        ])
        # Soft-clamp tilt to avoid integration blow-ups under wild actions.
        new[3] = float(np.clip(new[3], -np.pi/2 + 0.05, np.pi/2 - 0.05))
        new[4] = float(np.clip(new[4], -np.pi/2 + 0.05, np.pi/2 - 0.05))
        self.state = new
        return self.state.copy()

    # ------------------------------------------------------------------
    # Inverse-dynamics features
    # ------------------------------------------------------------------

    def build_inverse_dynamics_input(self, state_t, state_tp1,
                                     feature_mode: str = "invariant"):
        """Body-relative features that the network sees.

        Layout (invariant, 18-d):
            [phi, th, vx, vy, vz, p, q, r,
             dvx, dvy, dvz, dp, dq, dr,
             vx_next, vy_next, vz_next, dt]
        Adds (cos psi, sin psi, z, x, y) when feature_mode == "full" (23-d)
        so the full version can express world-frame setpoints if needed.
        """
        x_t, y_t, z_t, phi_t, th_t, psi_t, vx_t, vy_t, vz_t, p_t, q_t, r_t = state_t
        _, _, _, _, _, _, vx1, vy1, vz1, p1, q1, r1 = state_tp1
        dt = self.dt
        inv = np.array([
            phi_t, th_t,
            vx_t, vy_t, vz_t, p_t, q_t, r_t,
            (vx1 - vx_t) / dt, (vy1 - vy_t) / dt, (vz1 - vz_t) / dt,
            (p1  - p_t)  / dt, (q1  - q_t)  / dt, (r1  - r_t)  / dt,
            vx1, vy1, vz1,
            dt,
        ], dtype=np.float64)  # 18-d
        if feature_mode == "invariant":
            return inv
        return np.concatenate([
            inv,
            np.array([np.cos(psi_t), np.sin(psi_t), z_t, x_t, y_t]),
        ])  # 23-d

    def build_inverse_dynamics_input_for_control(self, state_t, desired_state,
                                                 feature_mode="invariant"):
        return self.build_inverse_dynamics_input(state_t, desired_state,
                                                 feature_mode=feature_mode)

    # ------------------------------------------------------------------
    # Waypoint follower — produces a 12-d desired next state used both
    # as a reference for adapters and as the target the inverse-dynamics
    # NN must learn to produce.  Cascaded: position error → desired
    # acceleration → desired tilt + thrust setpoint via the small-angle
    # inverse  acc_x ≈ g·theta,  acc_y ≈ −g·phi,  acc_z ≈ (T−m·g)/m.
    # ------------------------------------------------------------------

    def compute_desired_next_state(self, state, positions, step, dt):
        positions = np.asarray(positions, dtype=np.float64)
        n = len(positions)
        i  = min(step, n - 2)
        i1 = min(i + 1, n - 1)
        i2 = min(i + 2, n - 1)

        x, y, z, phi, th, psi, vx, vy, vz, p, q, r = state

        # World-frame reference velocity from the trajectory.
        ref_vel = (positions[i1] - positions[i]) / dt
        ref_vel_next = (positions[i2] - positions[i1]) / dt
        ref_acc = (ref_vel_next - ref_vel) / dt

        # Position error (world frame).
        err = positions[i] - np.array([x, y, z])

        # Desired acceleration = trajectory accel + PD on position/velocity
        # error.  Saturated to avoid blow-up under large tracking error.
        kp = np.array([1.5, 1.5, 2.0])
        kv = np.array([1.2, 1.2, 1.5])
        a_des = ref_acc + kp * err + kv * (ref_vel - np.array([vx, vy, vz]))
        a_des = np.clip(a_des, -8.0, 8.0)

        # Desired velocity one step ahead.
        v_des = np.array([vx, vy, vz]) + a_des * dt
        v_des = np.clip(v_des, -3.0, 3.0)

        # Desired heading: align with horizontal motion (zero if stationary).
        ref_psi = float(np.arctan2(ref_vel[1], ref_vel[0] + 1e-9))
        psi_err = _ssa(ref_psi - psi)
        r_des = float(np.clip(r + 0.5 * (3.0 * np.tanh(psi_err) - r), -3.0, 3.0))
        psi_next = _ssa(psi + r_des * dt)

        # Map desired acceleration to desired tilt (small-angle inverse around
        # nominal heading).  Rotate world-frame target acceleration into the
        # frame whose x-axis is along the desired heading psi_next.
        cpsi, spsi = np.cos(psi_next), np.sin(psi_next)
        a_body_x =  cpsi * a_des[0] + spsi * a_des[1]
        a_body_y = -spsi * a_des[0] + cpsi * a_des[1]
        # Small-angle inverse: a_body_x ≈ g·theta, a_body_y ≈ −g·phi
        theta_des = float(np.clip(a_body_x / self.G, -0.4, 0.4))
        phi_des   = float(np.clip(-a_body_y / self.G, -0.4, 0.4))
        # Body angular rates that move attitude toward the desired tilt
        p_des = float(np.clip((phi_des - phi) / max(dt, 1e-3) * 0.5, -3.0, 3.0))
        q_des = float(np.clip((theta_des - th) / max(dt, 1e-3) * 0.5, -3.0, 3.0))

        # Integrate one step forward.
        x_next, y_next, z_next = np.array([x, y, z]) + v_des * dt
        phi_next   = _ssa(phi + p_des * dt)
        theta_next = _ssa(th  + q_des * dt)

        return np.array([x_next, y_next, z_next,
                         phi_next, theta_next, psi_next,
                         v_des[0], v_des[1], v_des[2],
                         p_des, q_des, r_des])
