"""Persistently exciting trajectory generators for system identification.

Generates trajectories that excite all dynamic modes of the unicycle:
  - Multiple frequency components (slow + fast)
  - Varying curvature (straights + tight turns)
  - Speed changes (accelerations + decelerations)
  - Good spatial coverage

Trajectory types:
  1. multi_freq_lemniscate:  Lemniscate + high-freq perturbations
  2. chirp_spiral:           Frequency-sweeping spiral (chirp signal)
  3. random_waypoint:        Smooth random waypoints via cubic spline
  4. composite_exciting:     Concatenation of all above (best for training)
"""
import numpy as np
from scipy.interpolate import CubicSpline


def multi_freq_lemniscate(k, scale=4.0, speed=0.5, n_harmonics=4, amp_ratio=0.15):
    """Lemniscate with multi-frequency perturbations for persistent excitation.

    Base: standard lemniscate (figure-8)
    Perturbation: sum of sinusoids at incommensurate frequencies.
    This excites both low-freq (tracking) and high-freq (velocity) dynamics.
    """
    t = speed * k
    s = np.sin(t)
    c = np.cos(t)
    denom = 1.0 + s ** 2

    # Base lemniscate
    x_base = scale * c / denom
    y_base = scale * s * c / denom

    # Multi-frequency perturbation (incommensurate frequencies for PE)
    freqs = [1.7, 3.1, 5.3, 7.9][:n_harmonics]
    px, py = 0.0, 0.0
    for i, f in enumerate(freqs):
        amp = scale * amp_ratio / (i + 1)
        phase_x = 0.3 * (i + 1)
        phase_y = 0.7 * (i + 1)
        px += amp * np.sin(f * t + phase_x)
        py += amp * np.cos(f * t + phase_y)

    return np.array([x_base + px, y_base + py])


def chirp_spiral(k, dt=0.02, scale=3.0, f_start=0.05, f_end=1.0, duration=20.0):
    """Frequency-sweeping spiral that excites a range of frequencies.

    The radius oscillates while the angle sweeps, and the frequency
    increases linearly from f_start to f_end (chirp).
    """
    t = k * dt
    T = max(duration, 1.0)
    # Chirp: instantaneous frequency increases linearly
    phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t**2 / T)

    # Slowly varying radius for spatial coverage
    r = scale * (0.4 + 0.3 * np.sin(0.2 * t) + 0.2 * np.sin(0.07 * t))
    x = r * np.cos(phase)
    y = r * np.sin(phase)

    return np.array([x, y])


def random_waypoint_trajectory(n_steps, dt=0.02, scale=4.0, n_waypoints=12, seed=42):
    """Smooth random-waypoint trajectory via cubic spline interpolation.

    Generates random waypoints in [-scale, scale]² and connects them
    with a smooth cubic spline. Ensures good spatial coverage and
    varying curvature/speed.
    """
    rng = np.random.RandomState(seed)

    # Generate waypoints in a grid-like pattern for coverage + randomness
    angles = np.linspace(0, 2 * np.pi, n_waypoints, endpoint=False)
    radii = scale * (0.3 + 0.7 * rng.rand(n_waypoints))
    wx = radii * np.cos(angles) + scale * 0.2 * rng.randn(n_waypoints)
    wy = radii * np.sin(angles) + scale * 0.2 * rng.randn(n_waypoints)

    # Close the loop
    wx = np.append(wx, wx[0])
    wy = np.append(wy, wy[0])

    # Parameterize by arc-length-like parameter
    t_wp = np.linspace(0, 1, len(wx))

    cs_x = CubicSpline(t_wp, wx, bc_type='periodic')
    cs_y = CubicSpline(t_wp, wy, bc_type='periodic')

    t_dense = np.linspace(0, 1, n_steps, endpoint=False)
    positions = np.column_stack([cs_x(t_dense), cs_y(t_dense)])

    return positions


def composite_exciting_trajectory(n_steps, dt=0.02, scale=4.0, seed=42):
    """Composite trajectory: best for training NN inverse models.

    Instead of concatenating segments (which causes jumps), we
    superimpose multiple frequency components on a single smooth base.
    This guarantees C2 continuity while exciting all modes.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_steps) * dt

    # Base: lemniscate
    s = np.sin(0.3 * t)
    c = np.cos(0.3 * t)
    denom = 1.0 + s ** 2
    x = scale * c / denom
    y = scale * s * c / denom

    # Add multi-frequency perturbations (incommensurate for PE)
    freqs = [0.7, 1.3, 2.1, 3.7, 5.3]
    for i, f in enumerate(freqs):
        amp = scale * 0.12 / (1 + i * 0.5)
        phi_x = rng.uniform(0, 2 * np.pi)
        phi_y = rng.uniform(0, 2 * np.pi)
        x += amp * np.sin(f * t + phi_x)
        y += amp * np.cos(f * t + phi_y)

    # Add a slow wandering component (spatial coverage)
    x += scale * 0.15 * np.sin(0.11 * t + 0.5)
    y += scale * 0.15 * np.cos(0.07 * t + 1.2)

    # Add chirp component (frequency sweep)
    chirp_phase = 2 * np.pi * (0.05 * t + 0.3 * t**2 / t[-1])
    x += scale * 0.08 * np.sin(chirp_phase)
    y += scale * 0.08 * np.cos(chirp_phase + 0.5)

    positions = np.column_stack([x, y])

    # Smooth to limit maximum acceleration (physical feasibility)
    # Use a small Gaussian-like kernel
    kernel_size = 7
    kernel = np.array([1, 4, 10, 16, 10, 4, 1], dtype=float)
    kernel /= kernel.sum()
    for dim in range(2):
        positions[:, dim] = np.convolve(positions[:, dim], kernel, mode='same')

    return positions


def generate_exciting_trajectory(traj_type, dt, n_steps, r_bar=4, **kwargs):
    """Generate a trajectory with extra points for look-ahead.

    Args:
        traj_type: 'lemniscate', 'multi_freq', 'chirp', 'random', 'composite'
        dt: timestep
        n_steps: number of simulation steps
        r_bar: look-ahead buffer

    Returns:
        positions:  (n_steps + r_bar + 2, 2) array
        velocities: (n_steps + r_bar + 2, 2) array
    """
    total = n_steps + r_bar + 2
    scale = kwargs.get('scale', 4.0)
    speed = kwargs.get('speed', 0.5)
    seed = kwargs.get('seed', 42)

    if traj_type == 'lemniscate':
        from utils.trajectories import lemniscate_2d
        positions = np.array([lemniscate_2d(k * dt, scale=scale, speed=speed)
                              for k in range(total)])
    elif traj_type == 'multi_freq':
        positions = np.array([multi_freq_lemniscate(k * dt, scale=scale, speed=speed,
                                                     amp_ratio=0.12)
                              for k in range(total)])
    elif traj_type == 'chirp':
        positions = np.array([chirp_spiral(k, dt=dt, scale=scale * 0.8,
                                           f_start=0.03, f_end=0.5,
                                           duration=total * dt)
                              for k in range(total)])
    elif traj_type == 'random':
        positions = random_waypoint_trajectory(total, dt=dt, scale=scale,
                                                seed=seed)
    elif traj_type == 'composite':
        positions = composite_exciting_trajectory(total, dt=dt, scale=scale,
                                                   seed=seed)
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    # Velocities via central finite differences
    velocities = np.zeros_like(positions)
    for i in range(1, total - 1):
        velocities[i] = (positions[i + 1] - positions[i - 1]) / (2 * dt)
    velocities[0] = velocities[1]
    velocities[-1] = velocities[-2]

    return positions, velocities
