"""BlueROV2-style 3D AUV environment.

4-DOF maneuvering model (surge, sway, heave, yaw) — roll and pitch are
neutrally stable and are not actively controlled, matching the BlueROV2
configuration and the dynamics provided by the user in the issue.

State (8-dim):   ``[x, y, z, psi, u, v, w, r]``
Action (4-dim):  ``[X, Y, Z, M_z]``   (body-frame forces + yaw moment, N, N·m)
dt default:      0.05 s

The build_inverse_dynamics_input / compute_desired_next_state / set_noise /
set_uncertainty / ACTION_LIMITS interfaces match the rest of the project
(unicycle, drone) so the same adapters and experiment runner work
unchanged.
"""
from __future__ import annotations

import numpy as np


def _ssa(angle: float) -> float:
    """Smallest signed angle: wrap to (-pi, pi]."""
    return angle - 2.0 * np.pi * np.floor_divide(angle + np.pi, 2.0 * np.pi)


class AUV3DEnv:
    # --- BlueROV2 physical parameters (from the user's reference) ---
    MASS = 11.4            # kg
    G    = 9.82            # m/s²
    F_BOUY = 1026.0 * 0.0115 * 9.82   # buoyancy force (N)

    # Added mass
    X_ud = -2.6
    Y_vd = -18.5
    Z_wd = -13.3
    N_rd = -0.28

    # Moment of inertia (yaw)
    I_zz = 0.245

    # Linear damping
    X_u = -0.09
    Y_v = -0.26
    Z_w = -0.19
    N_r = -4.64

    # Quadratic damping
    X_uc = -34.96
    Y_vc = -103.25
    Z_wc = -74.23
    N_rc = -0.43

    # Actuator box — chosen to span realistic commands
    ACTION_LIMITS = ((-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0), (-10.0, 10.0))

    def __init__(self, dt: float = 0.05):
        self.dt = float(dt)
        self.state = np.zeros(8)

        # Perturbation multipliers
        self._mass_mult = 1.0
        self._drag_mult = 1.0
        self._added_mult = 1.0

        # Observation-noise parameters (applied only to get_observation)
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
            # Neutrally buoyant at depth 3 m
            self.state = np.array([0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            if seed is not None:
                self.state[:3] += self._rng.uniform(-0.10, 0.10, size=3)
                self.state[3]  += self._rng.uniform(-0.087, 0.087)
        return self.state.copy()

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    # ---- Perturbation / noise ----------------------------------------

    def set_perturbation(self, mass_mult: float = 1.0, drag_mult: float = 1.0,
                         added_mult: float = 1.0):
        """Scale effective mass, drag, and added-mass."""
        self._mass_mult = float(mass_mult)
        self._drag_mult = float(drag_mult)
        self._added_mult = float(added_mult)

    def set_uncertainty(self, active: bool, mass_multiplier: float = 1.0,
                        friction_multiplier: float = 1.0):
        """Scenario-compatible shim: mass + drag multipliers."""
        if active:
            self.set_perturbation(mass_mult=mass_multiplier,
                                  drag_mult=friction_multiplier)
        else:
            self.set_perturbation(1.0, 1.0, 1.0)

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
            s[3] += self._rng.normal(0.0, self._noise_std_ori)
        if self._noise_std_vel > 0.0:
            s[4:8] += self._rng.normal(0.0, self._noise_std_vel, size=4)
        return s

    # ---- Dynamics ----------------------------------------------------

    @property
    def _m_eff(self):  return self.MASS * self._mass_mult
    @property
    def _Iz_eff(self): return self.I_zz * self._mass_mult

    def step(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64)
        X, Y, Z, M_z = [float(np.clip(a[i], lo, hi))
                        for i, (lo, hi) in enumerate(self.ACTION_LIMITS)]

        x, y, z, psi, u, v, w, r = self.state
        m = self._m_eff
        X_ud = self.X_ud * self._added_mult
        Y_vd = self.Y_vd * self._added_mult
        Z_wd = self.Z_wd * self._added_mult
        N_rd = self.N_rd * self._added_mult
        dmu  = self._drag_mult

        # Body-frame accelerations (BlueROV2 equations)
        u_d = (X + (m - Y_vd) * v * r
               + dmu * (self.X_u + self.X_uc * abs(u)) * u) / (m - X_ud)
        v_d = (Y - (m - X_ud) * u * r
               + dmu * (self.Y_v + self.Y_vc * abs(v)) * v) / (m - Y_vd)
        w_d = (Z + dmu * (self.Z_w + self.Z_wc * abs(w)) * w
               + m * self.G - self.F_BOUY) / (m - Z_wd)
        r_d = (M_z - (X_ud - Y_vd) * u * v
               + dmu * (self.N_r + self.N_rc * abs(r)) * r) / (self._Iz_eff - N_rd)

        # Kinematics (yaw-only rotation)
        x_d = np.cos(psi) * u - np.sin(psi) * v
        y_d = np.sin(psi) * u + np.cos(psi) * v
        z_d = w
        psi_d = r

        dt = self.dt
        self.state = np.array([
            x + x_d * dt,
            y + y_d * dt,
            z + z_d * dt,
            _ssa(psi + psi_d * dt),
            u + u_d * dt,
            v + v_d * dt,
            w + w_d * dt,
            r + r_d * dt,
        ])
        return self.state.copy()

    # ------------------------------------------------------------------
    # Feature builder (orientation-invariant 12-dim; 16-dim full)
    # ------------------------------------------------------------------

    def build_inverse_dynamics_input(self, state_t: np.ndarray,
                                     state_tp1: np.ndarray,
                                     feature_mode: str = "invariant"):
        """Body-frame feature vector (orientation-invariant for feature_mode='invariant').

        ``invariant`` (12-dim):
            [u_t, v_t, w_t, r_t,
             du/dt, dv/dt, dw/dt, dr/dt,
             u_tp1, v_tp1, w_tp1, r_tp1, dt]  → 13 elements actually
        ``full`` (15-dim):
            Prepends [cos(psi), sin(psi), z_t] — absolute depth & heading.
        """
        _, _, z_t, psi_t, u_t, v_t, w_t, r_t = state_t
        _, _, _, _, u1, v1, w1, r1 = state_tp1
        dt = self.dt
        inv = np.array([
            u_t, v_t, w_t, r_t,
            (u1 - u_t) / dt,
            (v1 - v_t) / dt,
            (w1 - w_t) / dt,
            (r1 - r_t) / dt,
            u1, v1, w1, r1,
            dt,
        ], dtype=np.float64)   # 13-dim
        if feature_mode == "invariant":
            return inv
        return np.concatenate([
            inv,
            np.array([np.cos(psi_t), np.sin(psi_t), z_t]),
        ])  # 16-dim

    def build_inverse_dynamics_input_for_control(self, state_t, desired_state,
                                                 feature_mode="invariant"):
        return self.build_inverse_dynamics_input(state_t, desired_state,
                                                 feature_mode=feature_mode)

    # ------------------------------------------------------------------
    # Reference / desired-next-state generator
    # ------------------------------------------------------------------

    def compute_desired_next_state(self, state: np.ndarray,
                                   positions: np.ndarray, step: int,
                                   dt: float) -> np.ndarray:
        """Smooth waypoint follower for 3-D position tracking.

        ``positions`` is an ``(N, 3)`` array of ``(x, y, z)`` waypoints.
        Heading is aligned with the segment tangent in the x-y plane.
        """
        positions = np.asarray(positions, dtype=np.float64)
        n = len(positions)
        i = min(step, n - 2)
        i1 = min(i + 1, n - 1)
        i2 = min(i + 2, n - 1)

        x, y, z, psi, u, v, w, r = state

        seg_xy = positions[i1, :2] - positions[i, :2]
        seg_z = positions[i1, 2] - positions[i, 2]
        ref_psi = float(np.arctan2(seg_xy[1], seg_xy[0] + 1e-9))
        ref_speed_xy = float(np.linalg.norm(seg_xy) / dt)

        # Next segment for yaw rate
        seg2_xy = positions[i2, :2] - positions[i1, :2]
        ref_psi_next = float(np.arctan2(seg2_xy[1], seg2_xy[0] + 1e-9))
        dpsi_ref = _ssa(ref_psi_next - ref_psi) / dt

        # Position error
        ex = positions[i, 0] - x
        ey = positions[i, 1] - y
        ez = positions[i, 2] - z
        dist_xy = float(np.hypot(ex, ey))

        # Heading error -> desired yaw rate (tanh-saturated)
        psi_err = _ssa(ref_psi - psi)
        r_target = dpsi_ref + 2.5 * np.tanh(2.0 * psi_err)
        r_target = float(np.clip(r_target, -2.5, 2.5))
        r_des = float(np.clip(r + 0.5 * (r_target - r), -2.5, 2.5))

        # Surge tracks reference speed + distance correction
        k_pos = 1.2
        u_target = float(np.clip(ref_speed_xy + k_pos * dist_xy, 0.0, 2.0))
        u_des = float(np.clip(u + 0.5 * (u_target - u), 0.0, 2.0))

        # Sway damped (decoupled)
        v_des = float(np.clip(0.6 * (-0.4 * ey) + 0.4 * v, -1.0, 1.0))

        # Heave tracks z reference
        w_target = float(np.clip(seg_z / dt + 0.8 * ez, -1.0, 1.0))
        w_des = float(np.clip(w + 0.5 * (w_target - w), -1.0, 1.0))

        # Desired next position (Euler)
        x_next = x + (np.cos(psi) * u_des - np.sin(psi) * v_des) * dt
        y_next = y + (np.sin(psi) * u_des + np.cos(psi) * v_des) * dt
        z_next = z + w_des * dt
        psi_next = _ssa(psi + r_des * dt)

        return np.array([x_next, y_next, z_next, psi_next,
                         u_des, v_des, w_des, r_des])
