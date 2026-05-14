"""Evidential Deep Learning for regression (Amini et al., NeurIPS 2020).

Implements the Normal-Inverse-Gamma (NIG) evidential regression framework.
The network outputs 4 parameters per output dimension: (γ, ν, α, β) that
parameterize a NIG prior over the unknown mean and variance of the target.

Reference implementation:
    github.com/aamini/evidential-deep-learning
    (evidential_deep_learning/layers/dense.py, losses/continuous.py)

Theory:
    The target y is modeled as y ~ N(μ, σ²) where:
        μ ~ N(γ, σ²/ν)        (mean has Gaussian prior)
        σ² ~ Inv-Gamma(α, β)  (variance has Inverse-Gamma prior)

    Together, (μ, σ²) ~ NIG(γ, ν, α, β).

    The network predicts (γ, ν, α, β) directly:
        γ  = linear output     (predicted mean, unconstrained)
        ν  = softplus(·)       (evidence for mean, > 0)
        α  = softplus(·) + 1   (evidence for variance, > 1)
        β  = softplus(·)       (scale, > 0)

    Epistemic uncertainty (model uncertainty):
        Var_epi = β / (ν · (α - 1))

    Aleatoric uncertainty (data noise):
        Var_alea = β / (α - 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EvidentialNet(nn.Module):
    """Evidential regression network with NIG output layer.

    Architecture:
        Input(input_dim) → [Dense(h) → ReLU] × L → DenseNormalGamma(output_dim)

    The final layer outputs 4 × output_dim values, split into (γ, ν, α, β).
    This mirrors the original TensorFlow DenseNormalGamma layer.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None,
                 use_layernorm: bool = True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 256]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_layernorm = bool(use_layernorm)

        # LayerNorm by default — BatchNorm running stats drift on the
        # priority-weighted online batches and the L2 anchor only protects
        # parameters, not buffers.  LN normalises per-sample with no running
        # state, eliminating the entire failure mode.  Set use_layernorm=False
        # to load legacy BN-trained checkpoints.
        norm_cls = nn.LayerNorm if self.use_layernorm else nn.BatchNorm1d
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(norm_cls(h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.backbone = nn.Sequential(*layers)

        # NIG output head: 4 params per output dim
        # Mirrors DenseNormalGamma from original EDL
        self.head = nn.Linear(prev_dim, 4 * output_dim)

        # Normalization stats (set after data collection)
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('target_mean', torch.zeros(output_dim))
        self.register_buffer('target_std', torch.ones(output_dim))

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, input_dim)
            Normalized input.

        Returns
        -------
        gamma : Tensor, shape (B, output_dim)
            Predicted mean (μ).
        nu : Tensor, shape (B, output_dim)
            Evidence for mean (> 0).
        alpha : Tensor, shape (B, output_dim)
            Evidence for variance (> 1).
        beta : Tensor, shape (B, output_dim)
            Scale parameter (> 0).
        """
        h = self.backbone(x)
        out = self.head(h)

        # Split into 4 NIG parameters (same as DenseNormalGamma)
        gamma, log_nu, log_alpha, log_beta = out.chunk(4, dim=-1)

        # Activation constraints (matches original EDL exactly):
        #   gamma:  unconstrained (linear)
        #   nu:     softplus → strictly > 0
        #   alpha:  softplus + 1 → strictly > 1
        #   beta:   softplus → strictly > 0
        nu = F.softplus(log_nu) + 1e-6       # strictly > 0
        alpha = F.softplus(log_alpha) + 1.0 + 1e-6  # strictly > 1
        beta = F.softplus(log_beta) + 1e-6   # strictly > 0

        return gamma, nu, alpha, beta


def evidential_loss(gamma, nu, alpha, beta, targets, lambda_reg=0.01):
    """Evidential regression loss L_DER (Eq. 8 in Amini et al., 2020).

    L = L_NLL + λ · L_REG

    where:
        L_NLL = NIG negative log-likelihood
        L_REG = |y - γ| · (2ν + α)   (evidence regularizer)

    This matches the original implementation in:
        evidential_deep_learning/losses/continuous.py → EvidentialRegression()

    Parameters
    ----------
    gamma : Tensor, (B, D)  — predicted mean
    nu : Tensor, (B, D)     — evidence for mean (> 0)
    alpha : Tensor, (B, D)  — evidence for variance (> 1)
    beta : Tensor, (B, D)   — scale (> 0)
    targets : Tensor, (B, D)
    lambda_reg : float
        Regularization coefficient (called 'coeff' in original code).

    Returns
    -------
    loss : scalar Tensor
    """
    # --- NIG Negative Log-Likelihood (NIG_NLL in original) ---
    # Ω = 2β(1 + ν)
    omega = 2.0 * beta * (1.0 + nu)

    # Numerical stability: clamp arguments to log to avoid -inf/NaN
    _eps = 1e-8
    # L_NLL = ½·log(π/ν) - α·log(Ω) + (α+½)·log(ν·(y-γ)² + Ω) + log Γ(α) - log Γ(α+½)
    nll = (
        0.5 * torch.log(torch.clamp(np.pi / nu, min=_eps))
        - alpha * torch.log(torch.clamp(omega, min=_eps))
        + (alpha + 0.5) * torch.log(torch.clamp(nu * (targets - gamma) ** 2 + omega, min=_eps))
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    # --- Evidence Regularizer (NIG_Reg in original) ---
    # L_REG = |y - γ| · (2ν + α)
    reg = torch.abs(targets - gamma) * (2.0 * nu + alpha)

    # Total loss
    loss = (nll + lambda_reg * reg).mean()

    return loss


def epistemic_score(nu, alpha, beta):
    """Epistemic (model) uncertainty: β / (ν · (α - 1)).

    Summed over output dimensions to give a single scalar per sample.
    
    NOT bounded to [0, 1]. This is a variance in NIG parameter space.
    Values > 1.0 are normal and indicate high model uncertainty.
    For visualization, use log scale or percentile-based normalization.
    """
    # Var_epistemic = β / (ν · (α - 1))
    # Clamp (α - 1) to avoid division by zero
    var_epi = beta / (nu * torch.clamp(alpha - 1.0, min=1e-6))
    return var_epi.sum(dim=-1)


def aleatoric_score(nu, alpha, beta):
    """Aleatoric (data noise) uncertainty: β / (α - 1).

    Summed over output dimensions to give a single scalar per sample.
    This represents irreducible noise in the data.
    """
    return (beta / (alpha - 1.0).clamp(min=1e-6)).sum(dim=-1)