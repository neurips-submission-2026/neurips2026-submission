"""Tests for the EWC online adapter."""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envs.unicycle_env import UnicycleEnv
from models.evidential_net import EvidentialNet
from training.ewc import (
    EWCAdapter,
    compute_fisher_diagonal,
    _synthetic_dataset_from_model,
)
from utils.trajectories import generate_trajectory, lemniscate_2d


def _make_model():
    """Tiny evidential net wired to the unicycle's invariant feature."""
    model = EvidentialNet(input_dim=10, output_dim=2,
                          hidden_dims=[32, 32], use_layernorm=True)
    # Provide normalisation stats; otherwise the adapter's standardisation
    # will divide by zero.
    model.input_mean = torch.zeros(10)
    model.input_std = torch.ones(10)
    model.target_mean = torch.zeros(2)
    model.target_std = torch.ones(2)
    model.feature_mode = "invariant"
    return model


def test_fisher_is_nonnegative():
    model = _make_model()
    inputs, targets = _synthetic_dataset_from_model(model, n=128, seed=0)
    fisher = compute_fisher_diagonal(model, inputs, targets,
                                     num_samples=128, batch_size=16)
    assert len(fisher) > 0
    for name, f in fisher.items():
        assert torch.all(f >= 0), f"Fisher entry {name} has negatives"
        assert torch.isfinite(f).all(), f"Fisher entry {name} has non-finite"


def test_adapter_runs_without_nan():
    env = UnicycleEnv()
    env.reset(seed=42)
    model = _make_model()
    positions, _ = generate_trajectory(
        lambda t: lemniscate_2d(t, scale=2.0, speed=0.5),
        dt=env.dt, n_steps=120, r_bar=4)
    adapter = EWCAdapter(model, env, env.dt,
                         lambda_ewc=1.0,
                         num_fisher_samples=128,
                         lr=1e-4, batch_size=8,
                         min_buffer_size=4,
                         buffer_capacity=64,
                         update_every=5)
    state = env.get_state()
    for k in range(50):
        action, info = adapter.step(env, state, positions, k)
        assert np.isfinite(action).all(), f"NaN action at step {k}"
        state = env.get_state()
    # Check the model parameters are still finite after several updates.
    for name, p in adapter.model.named_parameters():
        assert torch.isfinite(p).all(), f"{name} has non-finite values"


def test_adapter_deterministic_given_seed():
    """Two adapters built with the same seed and offline weights produce
    the same sequence of commanded actions on the same env."""
    actions_a = []
    actions_b = []
    for store in (actions_a, actions_b):
        torch.manual_seed(0)
        np.random.seed(0)
        env = UnicycleEnv()
        env.reset(seed=42)
        model = _make_model()
        positions, _ = generate_trajectory(
            lambda t: lemniscate_2d(t, scale=2.0, speed=0.5),
            dt=env.dt, n_steps=80, r_bar=4)
        adapter = EWCAdapter(model, env, env.dt,
                             lambda_ewc=1.0,
                             num_fisher_samples=64,
                             lr=1e-4, batch_size=8,
                             min_buffer_size=4,
                             buffer_capacity=64,
                             update_every=5)
        state = env.get_state()
        for k in range(40):
            a, _ = adapter.step(env, state, positions, k)
            store.append(a.copy())
            state = env.get_state()
    for a, b in zip(actions_a, actions_b):
        np.testing.assert_allclose(a, b, atol=1e-5)
