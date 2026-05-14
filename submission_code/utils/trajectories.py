"""Trajectory generators for robot tracking experiments.

All trajectory functions take a continuous parameter k (which is
typically k = step_index, not time) and return position arrays.
"""
import numpy as np


def lemniscate_2d(k, scale=4.0, speed=0.5):
    """2D lemniscate of Bernoulli (figure-8) for ground robots.

    Parametric form:
        x(t) = scale * cos(t) / (1 + sin²(t))
        y(t) = scale * sin(t) * cos(t) / (1 + sin²(t))

    where t = speed * k.
    """
    t = speed * k
    s = np.sin(t)
    c = np.cos(t)
    denom = 1.0 + s ** 2
    x = scale * c / denom
    y = scale * s * c / denom
    return np.array([x, y])


def lemniscate_3d(k, scale_xy=20.0, scale_z=10.0, z_offset=10.0, speed=0.2):
    """3D lemniscate for aerial/underwater robots.

    Same x-y as 2D but adds a sinusoidal z component.
    """
    t = speed * k
    s = np.sin(t)
    c = np.cos(t)
    denom = 1.0 + s ** 2
    x = scale_xy * c / denom
    y = scale_xy * s * c / denom
    z = z_offset + scale_z * np.sin(2 * t) * 0.5
    return np.array([x, y, z])


def desired_heading_2d(pos_next, pos_current, pos_prev):
    """Compute desired heading from trajectory tangent (forward difference).

    Uses the direction from pos_current to pos_next.
    """
    dy = pos_next[1] - pos_current[1]
    dx = pos_next[0] - pos_current[0]
    return np.arctan2(dy, dx)


def generate_trajectory(traj_fn, dt, n_steps, r_bar=2):
    """Sample trajectory at discrete timesteps.

    Returns:
        positions:  (n_steps + r_bar + 2, D) array
        velocities: (n_steps + r_bar + 2, D) array (finite differences)
    """
    total = n_steps + r_bar + 2
    positions = np.array([traj_fn(k * dt) for k in range(total)])

    # Velocities via central finite differences
    velocities = np.zeros_like(positions)
    for i in range(1, total - 1):
        velocities[i] = (positions[i + 1] - positions[i - 1]) / (2 * dt)
    velocities[0] = velocities[1]
    velocities[-1] = velocities[-2]

    return positions, velocities

def exciting_3d(k, scale=2.0, speed=0.4, seed=0):
    """Sum-of-sinusoids 3-D trajectory with incommensurate frequencies.

    Produces a trajectory that is persistently exciting across x, y, z —
    useful for collecting diverse offline data for 3-D platforms.
    """
    rng = np.random.RandomState(seed)
    freqs = 0.5 + 2.0 * rng.rand(3, 4)   # 4 freq components per axis
    phase = 2.0 * np.pi * rng.rand(3, 4)
    amps  = 1.0 / (1.0 + freqs)           # 1/f amplitude shaping
    t = speed * k
    xs = scale * np.sum(amps[0] * np.sin(freqs[0] * t + phase[0]))
    ys = scale * np.sum(amps[1] * np.sin(freqs[1] * t + phase[1]))
    zs = 3.0 + 1.5 * scale * np.sum(amps[2] * np.sin(freqs[2] * t + phase[2]))
    return np.array([xs, ys, zs])


def smooth_waypoints_3d(k, scale_xy=2.0, scale_z=0.8,
                         period=120.0, z_offset=3.0):
    """Smooth 3-D lemniscate-over-altitude reference (nice for plots)."""
    t = 2 * np.pi * k / period
    s = np.sin(t)
    c = np.cos(t)
    denom = 1.0 + s ** 2
    x = scale_xy * c / denom
    y = scale_xy * s * c / denom
    z = z_offset + scale_z * np.sin(2 * t) * 0.5
    return np.array([x, y, z])
