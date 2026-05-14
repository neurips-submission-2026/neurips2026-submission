"""Stabilising LQR controllers for the 3-D AUV and quadrotor drone.

The AUV is a fully-actuated 4-DOF underwater vehicle with state
``[x, y, z, psi, u, v, w, r]`` (8-dim) and action ``[X, Y, Z, M_z]``
(body-frame thrust forces + yaw moment).

The quadrotor uses the proper 12-d state and 4-d torque/thrust
action (see ``envs/drone3d_env.py``).  We linearise the quadrotor
around the hover equilibrium ``T = m·g``, ``phi = theta = 0`` and
solve a single 12-state / 4-input LQR; the (lateral motion ↔ tilt)
coupling collapses to ``a_x ≈ g·theta``, ``a_y ≈ −g·phi`` at first
order, giving a fully linear small-signal controller without any
explicit cascading.

Both controllers are used as (i) data-collection experts during
offline pre-training and (ii) baseline references in the online
experiment suite.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import solve_continuous_are
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _solve_are(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Solve the continuous-time ARE; fallback is a few iterations of the
    discrete-time DP if scipy is unavailable.  ``solve_continuous_are`` is
    standard in scipy.
    """
    if _HAS_SCIPY:
        return solve_continuous_are(A, B, Q, R)
    P = np.eye(A.shape[0])
    for _ in range(500):
        P = Q + A.T @ P + P @ A - P @ B @ np.linalg.inv(R) @ B.T @ P
    return P


class _LQRAUV:
    """LQR for the 8-state AUV with (X, Y, Z, M_z) action."""

    def __init__(self, env, setpoint=None, Q=None, R=None):
        self.env = env
        self.setpoint = (np.zeros(3) if setpoint is None
                         else np.asarray(setpoint, dtype=np.float64))
        # Constant feedforward trim for the Z (heave) channel: AUV → buoyancy.
        self.hover_Z = env.F_BOUY - env.MASS * env.G

        m = env.MASS
        Iz = env.I_zz
        Xu, Yv, Zw, Nr = env.X_u, env.Y_v, env.Z_w, env.N_r
        Xud, Yvd, Zwd, Nrd = env.X_ud, env.Y_vd, env.Z_wd, env.N_rd
        m_u, m_v, m_w, I_r = m - Xud, m - Yvd, m - Zwd, Iz - Nrd

        # State order: [x, y, z, psi, u, v, w, r]
        A = np.zeros((8, 8))
        A[0, 4] = 1.0; A[1, 5] = 1.0; A[2, 6] = 1.0; A[3, 7] = 1.0
        A[4, 4] = Xu / m_u
        A[5, 5] = Yv / m_v
        A[6, 6] = Zw / m_w
        A[7, 7] = Nr / I_r

        B = np.zeros((8, 4))
        B[4, 0] = 1.0 / m_u
        B[5, 1] = 1.0 / m_v
        B[6, 2] = 1.0 / m_w
        B[7, 3] = 1.0 / I_r

        # iter-28: position weights softened a further 2× on top of
        # iter-24's softening.  After Track C eliminated the offline
        # NN's systematic bias, K-feedback no longer needs to carry
        # the AUV's tracking — it can recede to a true residual on top
        # of NN feed-forward.  Same Q used by both the LQR baseline
        # and StateFeedback (every NN method's K), so the comparison
        # stays symmetric — only the feed-forward differs.
        Q = np.diag(Q) if Q is not None else np.diag([
            6.0, 6.0, 8.0, 1.5,        # x, y, z, psi   (iter-24 setting restored)
            0.6, 0.6, 1.0, 0.3,        # u, v, w, r     (unchanged)
        ])
        R = np.diag(R) if R is not None else np.diag([0.2, 0.2, 0.2, 0.5])
        P = _solve_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P

    def compute(self, state: np.ndarray, setpoint=None):
        sp = np.asarray(setpoint if setpoint is not None else self.setpoint,
                        dtype=np.float64)
        x_ref = np.zeros(8)
        x_ref[0:3] = sp
        err = state - x_ref
        err[3] = (err[3] + np.pi) % (2.0 * np.pi) - np.pi
        u = -self.K @ err
        u[2] += self.hover_Z
        return u


class _LQRQuadrotor:
    """LQR for the 12-state quadrotor with (T, tau_x, tau_y, tau_z) action.

    Linearised around hover (phi=theta=0, all velocities and angular
    rates zero, T_trim = m·g, tau_trim = 0).  The state is shifted by
    the hover reference and the input is shifted by the trim;
    closed-loop the small-angle dynamics decouple into two independent
    pitch/roll-translation channels and a yaw + altitude channel.
    """

    def __init__(self, env, setpoint=None, Q=None, R=None):
        self.env = env
        self.setpoint = (np.zeros(3) if setpoint is None
                         else np.asarray(setpoint, dtype=np.float64))
        self.hover_T = env.MASS * env.G

        m  = env.MASS
        Ix, Iy, Iz = env.Ixx, env.Iyy, env.Izz
        kv = env.K_drag_lin
        kw = env.K_drag_ang
        g  = env.G

        # State order:
        #   0..2  position    (x, y, z)
        #   3..5  attitude    (phi, theta, psi)
        #   6..8  velocity    (vx, vy, vz)
        #   9..11 ang. vel.   (p, q, r)
        A = np.zeros((12, 12))
        # Position kinematics
        A[0, 6] = 1.0;  A[1, 7] = 1.0;  A[2, 8] = 1.0
        # Attitude kinematics (small angle)
        A[3, 9] = 1.0;  A[4,10] = 1.0;  A[5,11] = 1.0
        # Velocity dynamics: lateral acc from tilt, drag on velocity
        A[6, 4] =  g            # vx_dot += g·theta
        A[7, 3] = -g            # vy_dot += -g·phi
        A[6, 6] = -kv / m
        A[7, 7] = -kv / m
        A[8, 8] = -kv / m
        # Angular damping
        A[9,  9] = -kw / Ix
        A[10,10] = -kw / Iy
        A[11,11] = -kw / Iz

        B = np.zeros((12, 4))
        B[8,  0] = 1.0 / m         # delta-T → vz_dot
        B[9,  1] = 1.0 / Ix        # tau_x → p_dot
        B[10, 2] = 1.0 / Iy        # tau_y → q_dot
        B[11, 3] = 1.0 / Iz        # tau_z → r_dot

        # iter-28: position weights softened a further 2× on top of
        # iter-24's softening.  Same rationale as the AUV's iter-28
        # softening above — after Track C eliminated the offline NN's
        # systematic bias, K-feedback can recede to a true residual.
        # Attitude / velocity / rate weights kept (stability margin).
        Q = np.diag(Q) if Q is not None else np.diag([
            4.0, 4.0, 6.0,         # x, y, z       (iter-24 setting restored)
            0.5, 0.5, 1.5,         # phi, theta, psi (unchanged)
            0.5, 0.5, 1.0,         # vx, vy, vz   (unchanged)
            0.1, 0.1, 0.2,         # p, q, r      (unchanged)
        ])
        # Input weights — keep T near hover, allow torques freely.
        R = np.diag(R) if R is not None else np.diag([0.05, 0.10, 0.10, 0.20])
        P = _solve_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P

    def compute(self, state: np.ndarray, setpoint=None):
        sp = np.asarray(setpoint if setpoint is not None else self.setpoint,
                        dtype=np.float64)
        x_ref = np.zeros(12)
        x_ref[0:3] = sp
        err = state - x_ref
        # Wrap yaw error
        err[5] = (err[5] + np.pi) % (2.0 * np.pi) - np.pi
        u_delta = -self.K @ err
        u = np.array([self.hover_T, 0.0, 0.0, 0.0]) + u_delta
        # Enforce non-negative thrust
        u[0] = max(u[0], 0.0)
        return u


class LQR3D:
    """Dispatcher that returns the right controller for the env kind.

    ``kind="auv"`` uses the 8-state body-frame model (X, Y, Z, M_z).
    ``kind="drone"`` uses the 12-state quadrotor model
    (T, tau_x, tau_y, tau_z).
    """

    def __new__(cls, env, kind: str,
                setpoint: np.ndarray | None = None,
                Q=None, R=None):
        if kind == "auv":
            return _LQRAUV(env, setpoint=setpoint, Q=Q, R=R)
        if kind == "drone":
            return _LQRQuadrotor(env, setpoint=setpoint, Q=Q, R=R)
        raise ValueError(f"unknown LQR3D kind: {kind!r}")
