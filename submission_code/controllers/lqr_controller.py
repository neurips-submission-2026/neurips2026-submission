"""LQR trajectory-tracking controller for the unicycle robot.

Uses a full-state trajectory-tracking LQR that linearizes the nonlinear unicycle
dynamics around the reference trajectory and solves DARE at each step.

The control law is:
    u[k] = u_ref[k] - K[k] @ (x[k] - x_ref[k])

where K[k] is recomputed at each step from DARE using the (possibly stale/wrong)
dynamics parameters. This means:
    - If mass/friction change and the LQR doesn't know → wrong A, B → wrong K → degraded tracking
    - If the LQR sees delayed state → acts on stale error → oscillation
"""

import numpy as np
from utils.unicycle_controller import UnicycleLQR


class LQRController:
    """Full-state trajectory-tracking LQR controller.

    Parameters
    ----------
    env : environment or NominalEnvWrapper
        Must expose effective_mass, effective_Iz, effective_friction properties.
        If wrapped with NominalEnvWrapper, these return nominal (wrong) values.
    dt : float
        Simulation timestep.
    Q_diag : list of float, optional
        State cost diagonal [q_x, q_y, q_theta, q_v, q_omega].
    R_diag : list of float, optional
        Input cost diagonal [r_drive, r_steer].
    """

    def __init__(self, env, dt, Q_diag=None, R_diag=None):
        self.env = env
        self.dt = dt
        # Build LQR with env's reported parameters
        # (if env is NominalEnvWrapper, these are frozen nominal values)
        self.lqr = UnicycleLQR(
            m_eff=env.effective_mass,
            I_z=env.effective_Iz,
            friction=env.effective_friction,
            dt=dt,
            Q_diag=Q_diag,
            R_diag=R_diag,
        )

    def compute(self, state, positions, step):
        """Compute LQR control action.

        Parameters
        ----------
        state : array, shape (5,)
            Current robot state [x, y, theta, v, omega].
        positions : array, shape (N, 2)
            Reference trajectory waypoints.
        step : int
            Current time step index.

        Returns
        -------
        action : array, shape (2,)
            Control torques [tau_drive, tau_steer].
        info : dict
            Contains x_ref, u_ref, K.
        """
        return self.lqr.compute(state, positions, step)
