"""Tests for the demo-clip renderer's pure-data helpers."""
from __future__ import annotations
import os, sys
import numpy as np
import pytest

# Make the scripts package importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.render_demo_clips import build_trajectory


def test_build_trajectory_shape_and_init():
    """build_trajectory(T, dt) returns (positions, n_steps, init_state)
    with the same shape/seed conventions as run_benchmark.py."""
    positions, n_steps, init = build_trajectory(total_time=30.0, dt=0.02)
    assert n_steps == 1500
    assert positions.shape == (1500, 2)
    # Init state: [x, y, theta, v, omega]
    assert init.shape == (5,)
    # Theta is the heading toward positions[2] from positions[0].
    expected_theta = np.arctan2(positions[2][1] - positions[0][1],
                                positions[2][0] - positions[0][0])
    assert init[0] == pytest.approx(positions[0][0])
    assert init[1] == pytest.approx(positions[0][1])
    assert init[2] == pytest.approx(expected_theta)
    assert init[3] == 0.0 and init[4] == 0.0


from scripts.render_demo_clips import load_unicycle_model


def test_load_unicycle_model():
    """Loader returns an EvidentialNet with the normalisation tensors
    and feature_mode attached, matching run_benchmark.py."""
    model = load_unicycle_model()
    assert hasattr(model, "input_mean")
    assert hasattr(model, "input_std")
    assert hasattr(model, "target_mean")
    assert hasattr(model, "target_std")
    assert hasattr(model, "feature_mode")
    # Forward pass on the right input dimension should work without error.
    with __import__("torch").no_grad():
        gamma, nu, alpha, beta = model(__import__("torch").zeros(1, model.input_mean.shape[0]))
    # Each NIG output tensor should have shape (B, output_dim)
    assert gamma.shape[0] == 1
    assert nu.shape == gamma.shape
    assert alpha.shape == gamma.shape
    assert beta.shape == gamma.shape


from scripts.render_demo_clips import make_controller


def test_make_controller_returns_known_types():
    """make_controller returns a (controller, kind) pair per method."""
    model = load_unicycle_model()
    env = __import__("envs.unicycle_env", fromlist=["UnicycleEnv"]).UnicycleEnv(dt=0.02)
    for method, expected_kind in [("LQR", "lqr"), ("Frozen", "nn"), ("ACE", "adapter")]:
        ctrl, kind = make_controller(method, model, env, dt=0.02)
        assert kind == expected_kind, f"{method}: expected kind {expected_kind!r}, got {kind!r}"
        assert ctrl is not None


from scripts.render_demo_clips import SCENARIOS


def test_scenarios_table_matches_spec():
    """The scenario table must contain exactly the five paper-aligned scenarios."""
    expected = {"nominal", "mass_shift", "disturbance",
                "sensor_noise", "actuator_noise"}
    assert set(SCENARIOS.keys()) == expected
    for name, s in SCENARIOS.items():
        assert "title" in s, f"{name} missing title"
        assert "caption" in s, f"{name} missing caption"
        assert "shift" in s, f"{name} missing shift"
        assert isinstance(s["shift"], dict)


from scripts.render_demo_clips import run_scenario


def test_run_scenario_returns_per_method_arrays():
    """run_scenario returns positions, dt, and per-method arrays of the
    right shape; ACE wins (lower mean error) on the disturbance scenario.
    """
    model = load_unicycle_model()
    data = run_scenario("disturbance", model, total_time=30.0, dt=0.02, seed=42)
    assert "positions" in data
    assert "dt" in data and data["dt"] == 0.02
    assert data["positions"].shape == (1500, 2)
    for m in ("LQR", "Frozen", "ACE"):
        assert m in data
        for k in ("errs", "xy", "epist", "aleat", "lamb",
                  "shift_sig", "updates"):
            assert k in data[m], f"missing {m}/{k}"
            assert data[m][k].shape[0] == 1500
    # ACE must actually exercise the shift detector + update path under
    # the disturbance scenario (this is the visible-action story).
    assert data["ACE"]["shift_sig"].max() > 2.0, \
        "ACE shift_signal never crosses threshold under disturbance"
    assert data["ACE"]["updates"][-1] > 0, \
        "ACE made zero gradient updates under disturbance"
    skip = int(5.0 / data["dt"])
    ace_mean = float(data["ACE"]["errs"][skip:].mean())
    frozen_mean = float(data["Frozen"]["errs"][skip:].mean())
    assert ace_mean <= frozen_mean * 1.05, \
        f"ACE {ace_mean:.3f} m unexpectedly worse than Frozen {frozen_mean:.3f} m on disturbance"


def test_sensor_noise_does_no_harm():
    """Under sensor noise, ACE should keep λ high and look like Frozen
    (gate prevents drift). Failure mode: noisy observations flow through
    get_state() unchecked, ACE adapts to noise, error blows up.
    """
    model = load_unicycle_model()
    data = run_scenario("sensor_noise", model,
                        total_time=15.0, dt=0.02, seed=42)
    skip = int(2.0 / data["dt"])
    ace_mean    = float(data["ACE"]["errs"][skip:].mean())
    frozen_mean = float(data["Frozen"]["errs"][skip:].mean())
    assert ace_mean <= frozen_mean * 1.1, \
        f"ACE {ace_mean:.3f} > 1.1 * Frozen {frozen_mean:.3f} under sensor noise"


from scripts.render_demo_clips import SCENARIO_GATES


@pytest.mark.parametrize("scenario", list(SCENARIO_GATES))
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_scenario_gate(scenario, seed):
    """Each scenario must meet its per-scenario ACE-vs-Frozen gate.

    Gates live in SCENARIO_GATES; rationale documented in the v2 spec.
    Skip 2 s post-shift before computing the mean so we measure ACE
    after the schedule has had time to engage.
    """
    model = load_unicycle_model()
    data = run_scenario(scenario, model,
                        total_time=15.0, dt=0.02, seed=seed)
    skip = int(2.0 / data["dt"])
    ace = float(data["ACE"]["errs"][skip:].mean())
    frozen = float(data["Frozen"]["errs"][skip:].mean())
    _, mult = SCENARIO_GATES[scenario]
    assert ace <= frozen * mult, (
        f"{scenario} seed {seed}: ACE {ace:.3f} > Frozen {frozen:.3f} * {mult}"
    )
