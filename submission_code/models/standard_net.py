"""Standard DNN for inverse dynamics (deterministic point estimate)."""
import torch
import torch.nn as nn


class StandardNet(nn.Module):
    """Standard feedforward DNN for inverse dynamics regression.

    Outputs a deterministic point estimate of the control input.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)