"""Shared pytest fixtures for unit tests."""
import os
import sys
import pytest
import numpy as np
import torch

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.evidential_net import EvidentialNet
from envs.unicycle_env import UnicycleEnv


def _make_evidential_net(input_dim, hidden_dims):
    """Create EvidentialNet with matching normalization stats."""
    model = EvidentialNet(input_dim=input_dim, output_dim=2, hidden_dims=hidden_dims)
    model.input_mean = torch.zeros(input_dim)
    model.input_std = torch.ones(input_dim)
    model.target_mean = torch.zeros(2)
    model.target_std = torch.ones(2)
    model.feature_mode = "full"
    model.eval()
    return model


@pytest.fixture
def small_evidential_net():
    """Create a small fresh EvidentialNet with full-mode (12-dim) stats."""
    return _make_evidential_net(12, [32, 32])


@pytest.fixture
def unicycle_evidential_net():
    """Create an EvidentialNet sized for unicycle full-mode features (12 -> 2)."""
    return _make_evidential_net(12, [64, 64])


@pytest.fixture
def unicycle_evidential_net_full():
    """Create an EvidentialNet sized for unicycle full-mode features (12 -> 2)."""
    return _make_evidential_net(12, [64, 64])


@pytest.fixture
def unicycle_env():
    """Create a UnicycleEnv with default dt."""
    return UnicycleEnv(dt=0.02)


@pytest.fixture
def seeded_rng():
    """Provide a seeded numpy RNG for reproducibility."""
    return np.random.RandomState(42)
