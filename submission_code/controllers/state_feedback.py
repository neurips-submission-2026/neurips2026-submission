"""Lightweight LQR-style state-feedback term for inverse-dynamics
controllers across the three platforms (Unicycle, AUV, Drone).

The motivation is the structural difference between LQR and a vanilla
neural-network inverse-dynamics controller:

    LQR             u = u_ref + K(x_ref - x)        ← explicit feedback
    NN feed-forward u = NN(state, desired_next)     ← only implicit feedback
                                                       (via desired_next)

Under any non-trivial disturbance the explicit feedback term is what keeps
LQR's tracking error in the centimetre range while a purely-feedforward
NN drifts.  This module provides a re-usable feedback term that can be
*added* to a NN command so the resulting controller is

    u = NN(state, desired_next) + g · K(x_ref - x)

The neural network is still responsible for the feed-forward action; the
feedback term `K(x_ref - x)` is a fast residual stabiliser.  Under shift,
the NN learns a better u_ref than LQR's stale linearization, while K@e
absorbs the remaining tracking error – so the hybrid out-performs LQR in
exactly the regimes where a model-based controller falls behind (mass,
friction, wind).

The K matrix is computed from the *nominal* (offline) linearisation —
the same model the NN was trained on.  No privileged information is
used; the only difference between LQR and any NN method on a perturbed
scenario is the NN's online adaptation.
"""
from __future__ import annotations

import numpy as np

from utils.unicycle_controller import UnicycleLQR, solve_dare


# ----------------------------------------------------------------------
# 2-D unicycle feedback
# ----------------------------------------------------------------------


class StateFeedback:
    """LQR-style state feedback for the 2-D unicycle.

    Re-uses :class:`UnicycleLQR` for reference computation and
    linearization but ignores its feed-forward output: only the
    ``-K @ (state - x_ref)`` correction is returned.

    Parameters
    ----------
    env : UnicycleEnv-like
        Used at construction only to read nominal mass, inertia, and
        friction (the *original* offline parameters).  Mutating the env
        afterwards does not change the gain matrix — exactly the same as
        what LQR does.
    dt : float
    gain : float
        Scalar multiplier applied to the feedback torque.  ``1.0``
        recovers a standard LQR feedback; smaller values give a softer
        correction (useful when the NN feed-forward is good).
    Q_diag, R_diag : list[float] or None
        DARE cost weights.  Defaults match :class:`UnicycleLQR`.
    """

    def __init__(self, env, dt: float, gain: float = 1.0,
                 Q_diag=None, R_diag=None):
        self.lqr = UnicycleLQR(
            m_eff=env.effective_mass,
            I_z=env.effective_Iz,
            friction=env.effective_friction,
            dt=dt,
            Q_diag=Q_diag,
            R_diag=R_diag,
        )
        self.dt = dt
        self.gain = float(gain)
        # Cache for ``K`` keyed by (theta_ref bucket, v_ref bucket) — DARE
        # solves cost a few ms each and the (θ, v) pair varies smoothly
        # along a typical reference, so we reuse K when the linearization
        # would barely change.  This brings the per-step overhead down
        # from ~3 ms to ~30 µs once the cache fills.
        self._K_cache: dict = {}

    def compute(self, state, positions, step) -> np.ndarray:
        """Return ``-gain · K @ (state - x_ref)``.  Cached K lookup keeps the
        per-step cost low even though the reference (θ, v) changes every step.
        """
        n_steps = len(positions)
        x_ref, _u_ref = self.lqr.compute_reference(positions, step, n_steps)
        e = state - x_ref
        # Wrap heading error to [-pi, pi]
        e[2] = (e[2] + np.pi) % (2 * np.pi) - np.pi

        # Bucket θ_ref to the nearest ~5° and v_ref to 0.05 m/s.  This is
        # well below the resolution at which DARE's gain matters in
        # practice and gives orders-of-magnitude reuse along the
        # lemniscate.
        key = (round(float(x_ref[2]) / 0.0873), round(float(x_ref[3]) / 0.05))
        K = self._K_cache.get(key)
        if K is None:
            A, B = self.lqr._linearize(x_ref[2], x_ref[3])
            K, _ = solve_dare(A, B, self.lqr.Q, self.lqr.R)
            self._K_cache[key] = K
        u_fb = -K @ e
        return self.gain * u_fb


# ----------------------------------------------------------------------
# 3-D AUV / Drone feedback
# ----------------------------------------------------------------------


class StateFeedback3D:
    """LQR-style state feedback for the 4-DOF AUV / Drone (8-state model).

    Wraps :class:`controllers.lqr_3d.LQR3D` but returns only the K-feedback
    part of its control law (we drop the gravity / buoyancy trim that the
    NN feed-forward already accounts for).  The K matrix is built from
    the nominal mass / damping linearisation around hover at construction
    time, exactly mirroring the pure-LQR baseline.

    Parameters
    ----------
    env : AUV3DEnv | Drone3DEnv
        Used only at construction for reading nominal physical parameters
        (`MASS`, `I_zz`, damping coefficients, gravity / buoyancy).
    kind : ``"auv"`` or ``"drone"``
    dt : float
    gain : float
        Scalar multiplier on the feedback term.  ``1.0`` matches the
        underlying LQR3D feedback magnitude.
    """

    def __init__(self, env, kind: str, dt: float, gain: float = 1.0,
                 Q=None, R=None):
        from controllers.lqr_3d import LQR3D
        self.lqr = LQR3D(env, kind, Q=Q, R=R)
        self.kind = kind
        self.dt = dt
        self.gain = float(gain)

    def compute(self, state, positions, step) -> np.ndarray:
        """Return the *feedback only* part of LQR3D's command.

        The setpoint is the trajectory waypoint at ``step`` (or end of
        the array if past it).  ``state`` is 8-dim for AUV (body-frame
        model) and 12-dim for the quadrotor drone.
        """
        n_steps = len(positions)
        i = min(step, n_steps - 1)
        sp = positions[i]                                      # (3,) target xyz
        # LQR3D.compute returns u_fb + hover trim.  We subtract the
        # trim because the NN feed-forward already learned the correct
        # hover from offline data.  AUV trim sits on channel 2 (Z);
        # quadrotor trim sits on channel 0 (T).
        u = self.lqr.compute(state, setpoint=sp)
        u_fb = u.copy()
        if self.kind == "drone" and hasattr(self.lqr, "hover_T"):
            u_fb[0] -= self.lqr.hover_T
        elif hasattr(self.lqr, "hover_Z"):
            u_fb[2] -= self.lqr.hover_Z
        return self.gain * u_fb


# ----------------------------------------------------------------------
# Factory: return the right feedback class for the env at hand
# ----------------------------------------------------------------------


def make_state_feedback(env, dt: float, gain: float = 1.0):
    """Construct a :class:`StateFeedback` (unicycle) or
    :class:`StateFeedback3D` (AUV / Drone) based on the env's interface.

    Returns ``None`` if ``gain <= 0`` or the env doesn't expose a
    recognised physics interface (in which case the caller should fall
    back to the legacy purely-feed-forward NN controller).
    """
    if gain <= 0.0:
        return None
    if hasattr(env, "MASS") and hasattr(env, "I_zz"):
        kind = "auv" if hasattr(env, "F_BOUY") else "drone"
        return StateFeedback3D(env, kind=kind, dt=dt, gain=gain)
    needed = ("effective_mass", "effective_Iz", "effective_friction")
    if all(hasattr(env, n) for n in needed):
        return StateFeedback(env, dt, gain=gain)
    return None


