"""Full-state trajectory-tracking LQR for the unicycle robot.

Architecture:
    The controller linearizes the nonlinear unicycle dynamics around a reference
    trajectory point (x_ref, u_ref) and solves a discrete-time LQR (DARE) to get
    feedback gains K. The control law is:

        u[k] = u_ref[k] + K @ (x[k] - x_ref[k])

    The A, B matrices depend on the robot's physical parameters (mass, friction,
    inertia). If these change at runtime but the LQR still uses nominal values,
    the gains K are wrong → tracking degrades. This is realistic: a model-based
    controller is tuned once offline and cannot adapt.

State:   x = [x, y, theta, v, omega]  (5-dim)
Control: u = [tau_drive, tau_steer]     (2-dim)

Dynamics (continuous):
    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = omega
    v_dot     = (tau_drive - friction * v) / m_eff
    omega_dot = (tau_steer - friction * omega) / I_z

Linearized around (x_ref, u_ref), discretized with Euler (dt):
    A = I + dt * dF/dx |_{ref}
    B = dt * dF/du |_{ref}
"""

import numpy as np
from scipy.linalg import solve_discrete_are


def solve_dare(A, B, Q, R):
    """Solve discrete-time algebraic Riccati equation.
    Returns gain K such that u = -K @ x minimizes J = sum x'Qx + u'Ru.
    """
    try:
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        return K, P
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: use a simple proportional gain
        return np.zeros((B.shape[1], A.shape[0])), np.eye(A.shape[0])


class UnicycleLQR:
    """Full-state trajectory-tracking LQR for the unicycle.

    Parameters
    ----------
    m_eff : float
        Effective mass (kg). Used to build A, B matrices.
    I_z : float
        Yaw inertia (kg·m²). Used to build A, B matrices.
    friction : float
        Viscous friction coefficient (Ns/m).
    dt : float
        Timestep (s).
    Q_diag : array-like, shape (5,)
        Diagonal of state cost matrix Q = diag(q_x, q_y, q_theta, q_v, q_omega).
    R_diag : array-like, shape (2,)
        Diagonal of input cost matrix R = diag(r_drive, r_steer).
    """

    def __init__(self, m_eff, I_z, friction, dt,
                 Q_diag=None, R_diag=None):
        self.m_eff = m_eff
        self.I_z = I_z
        self.friction = friction
        self.dt = dt

        # Cost weights — calibrated so the K-feedback is a *residual*
        # stabiliser, not the dominant tracking signal.  Earlier
        # iter-23: softened Q from [10,10,5,1,1] to [3,3,1,1,1].  The
        # earlier value still left K-feedback dominant (~65 % of
        # tracking on Nominal), masking the NN feed-forward's true
        # contribution.  At Q_pos=3, K alone is a residual stabiliser
        # (~30 % of tracking) and the inverse-dynamics feed-forward
        # determines absolute tracking.  All NN methods share this Q
        # through StateFeedback so the comparison stays symmetric.
        if Q_diag is None:
            Q_diag = [3.0, 3.0, 1.0, 1.0, 1.0]
        if R_diag is None:
            R_diag = [1.0, 1.0]

        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)

        # Cache for gains
        self._last_K = None
        self._last_v_ref = None
        self._last_theta_ref = None

    def _linearize(self, theta_ref, v_ref):
        """Linearize dynamics around reference point, return discrete A, B."""
        dt = self.dt
        ct = np.cos(theta_ref)
        st = np.sin(theta_ref)
        f_v = self.friction / self.m_eff
        f_w = self.friction / self.I_z

        # Jacobian dF/dx (continuous)
        #   dF/d[x, y, theta, v, omega]
        Ac = np.array([
            [0, 0, -v_ref * st, ct, 0],
            [0, 0,  v_ref * ct, st, 0],
            [0, 0,  0,          0,  1],
            [0, 0,  0,         -f_v, 0],
            [0, 0,  0,          0, -f_w],
        ])

        # Jacobian dF/du (continuous)
        Bc = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1.0 / self.m_eff, 0],
            [0, 1.0 / self.I_z],
        ])

        # Euler discretization
        A = np.eye(5) + dt * Ac
        B = dt * Bc
        return A, B

    def compute_reference(self, positions, step, n_steps):
        """Compute reference state and feedforward torque from trajectory.

        Parameters
        ----------
        positions : array, shape (N, 2)
            Reference (x, y) positions.
        step : int
            Current time step.
        n_steps : int
            Total trajectory length.

        Returns
        -------
        x_ref : array, shape (5,)
            Reference state [x, y, theta, v, omega].
        u_ref : array, shape (2,)
            Feedforward torque.
        """
        dt = self.dt
        i = min(step, n_steps - 2)
        i1 = min(i + 1, n_steps - 1)
        i2 = min(i + 2, n_steps - 1)

        # Reference position
        px, py = positions[i]

        # Reference velocity (finite difference)
        dx = positions[i1][0] - positions[i][0]
        dy = positions[i1][1] - positions[i][1]
        v_ref = np.sqrt(dx**2 + dy**2) / dt
        theta_ref = np.arctan2(dy, dx)

        # Reference angular velocity (finite difference of heading)
        dx2 = positions[i2][0] - positions[i1][0]
        dy2 = positions[i2][1] - positions[i1][1]
        theta_next = np.arctan2(dy2, dx2)
        dtheta = theta_next - theta_ref
        # Wrap to [-pi, pi]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        omega_ref = dtheta / dt

        # Reference accelerations (finite difference of velocities)
        if i > 0:
            dx_prev = positions[i][0] - positions[i-1][0]
            dy_prev = positions[i][1] - positions[i-1][1]
            v_prev = np.sqrt(dx_prev**2 + dy_prev**2) / dt
            v_dot_ref = (v_ref - v_prev) / dt
        else:
            v_dot_ref = 0.0

        # omega acceleration
        if i > 0 and i + 2 < n_steps:
            dx_prev = positions[i][0] - positions[i-1][0]
            dy_prev = positions[i][1] - positions[i-1][1]
            theta_prev = np.arctan2(dy_prev, dx_prev)
            dtheta_prev = theta_ref - theta_prev
            dtheta_prev = (dtheta_prev + np.pi) % (2 * np.pi) - np.pi
            omega_prev = dtheta_prev / dt
            omega_dot_ref = (omega_ref - omega_prev) / dt
        else:
            omega_dot_ref = 0.0

        x_ref = np.array([px, py, theta_ref, v_ref, omega_ref])

        # Feedforward: invert dynamics  u = m*a + friction*v
        u_ref = np.array([
            self.m_eff * v_dot_ref + self.friction * v_ref,
            self.I_z * omega_dot_ref + self.friction * omega_ref,
        ])

        return x_ref, u_ref

    def compute(self, state, positions, step):
        """Compute LQR control action.

        Parameters
        ----------
        state : array, shape (5,)
            Current state [x, y, theta, v, omega].
        positions : array, shape (N, 2)
            Reference trajectory positions.
        step : int
            Current time step.

        Returns
        -------
        action : array, shape (2,)
            Control torques [tau_drive, tau_steer].
        info : dict
        """
        n_steps = len(positions)
        x_ref, u_ref = self.compute_reference(positions, step, n_steps)

        # State error
        e = state - x_ref
        # Wrap heading error to [-pi, pi]
        e[2] = (e[2] + np.pi) % (2 * np.pi) - np.pi

        # Linearize around reference
        A, B = self._linearize(x_ref[2], x_ref[3])

        # Solve DARE for gains
        K, _ = solve_dare(A, B, self.Q, self.R)

        # Control law: u = u_ref - K @ error
        action = u_ref - K @ e

        # Clip to actuator limits
        action[0] = np.clip(action[0], -20.0, 20.0)
        action[1] = np.clip(action[1], -10.0, 10.0)

        return action, {"x_ref": x_ref, "u_ref": u_ref, "K": K}


def compute_reference_from_trajectory(positions, dt, k, lookahead=0):
    """Extract (x_r, y_r, theta_r, v_r, omega_r) from a trajectory array.

    Uses central finite differences for heading and velocities.
    """
    n = len(positions)
    idx = min(k + lookahead, n - 2)
    idx = max(idx, 1)

    ref_x = positions[idx, 0]
    ref_y = positions[idx, 1]

    # Heading from central finite difference
    i_prev = max(idx - 1, 0)
    i_next = min(idx + 1, n - 1)
    dx = positions[i_next, 0] - positions[i_prev, 0]
    dy = positions[i_next, 1] - positions[i_prev, 1]
    ref_theta = np.arctan2(dy, dx)

    # Linear velocity
    dx_dt = (positions[i_next, 0] - positions[idx, 0]) / dt
    dy_dt = (positions[i_next, 1] - positions[idx, 1]) / dt
    ref_v = np.sqrt(dx_dt**2 + dy_dt**2)

    # Angular velocity from heading change
    i_next2 = min(idx + 2, n - 1)
    if i_next2 > idx:
        dx2 = positions[i_next2, 0] - positions[idx, 0]
        dy2 = positions[i_next2, 1] - positions[idx, 1]
        theta_next = np.arctan2(dy2, dx2)
        d_theta = (theta_next - ref_theta + np.pi) % (2 * np.pi) - np.pi
        ref_omega = d_theta / ((i_next2 - idx) * dt)
    else:
        ref_omega = 0.0

    return ref_x, ref_y, ref_theta, ref_v, ref_omega
