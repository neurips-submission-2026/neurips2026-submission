"""Continual Learning regularization methods for evidential regression.

Implements:
  - EWCRegularizer    : Elastic Weight Consolidation (Kirkpatrick et al., 2017)
  - OnlineEWC         : Online variant with exponential moving average Fisher
  - SynapticIntelligence : SI (Zenke et al., 2017) — path-integral importance

These are used as baselines in the NeurIPS comparison against Adaptive EDL.

Usage pattern (EWC):
    ewc = EWCRegularizer(model, lambda_ewc=500.0)
    # ... train on task 1 ...
    ewc.update_task(dataloader_task1)   # snapshot Fisher + optimal params
    # ... train on task 2 ...
    loss = task2_loss + ewc.penalty()   # add EWC term
"""
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


# =========================================================================
# Elastic Weight Consolidation (Kirkpatrick et al., 2017)
# =========================================================================

class EWCRegularizer:
    """Diagonal-Fisher EWC for evidential regression continual learning.

    After each task, computes the diagonal of the Fisher Information Matrix
    (approximated as the expected squared gradient of the log-likelihood)
    and stores the current parameter values as the task optimum.

    The penalty term is:
        P_EWC(θ) = λ_EWC/2 · Σ_i F_i · (θ_i - θ*_i)²

    where F_i = E[( ∂ log p(y|x,θ) / ∂θ_i )²] is the Fisher diagonal.

    Parameters
    ----------
    model : nn.Module  — the evidential network being trained
    lambda_ewc : float — regularization strength (default 500.0)
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 500.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        # Accumulated Fisher and parameter snapshots across all tasks
        self._fisher: dict[str, torch.Tensor] = {}
        self._theta_star: dict[str, torch.Tensor] = {}
        self._n_tasks = 0

    def compute_fisher(self, dataloader, loss_fn, n_samples: int = 500,
                       device: str = "cpu"):
        """Estimate diagonal Fisher via Monte Carlo gradient squaring.

        Parameters
        ----------
        dataloader : iterable of (inputs, targets) — normalized batches
        loss_fn    : callable(gamma, nu, alpha, beta, targets) → scalar loss
        n_samples  : max samples to use for estimation
        device     : torch device string

        Returns
        -------
        fisher : dict[str, Tensor]  — diagonal Fisher per parameter
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters() if p.requires_grad}

        total = 0
        for inp, tgt in dataloader:
            if total >= n_samples:
                break
            inp = inp.to(device)
            tgt = tgt.to(device)

            self.model.zero_grad()
            gamma, nu, alpha, beta = self.model(inp)
            loss = loss_fn(gamma, nu, alpha, beta, tgt)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 * inp.shape[0]
            total += inp.shape[0]

        if total > 0:
            for n in fisher:
                fisher[n] /= total
        return fisher

    def update_task(self, dataloader, loss_fn, n_samples: int = 500,
                    device: str = "cpu"):
        """Call once after training on each task.

        Accumulates Fisher across tasks and stores current params as θ*.
        """
        new_fisher = self.compute_fisher(dataloader, loss_fn,
                                         n_samples=n_samples, device=device)
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            f = new_fisher.get(n, torch.zeros_like(p))
            if n in self._fisher:
                self._fisher[n] = self._fisher[n] + f
            else:
                self._fisher[n] = f.clone()
            self._theta_star[n] = p.data.clone()

        self._n_tasks += 1

    def penalty(self) -> torch.Tensor:
        """EWC quadratic penalty term to add to the task loss."""
        if self._n_tasks == 0:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(iter(self._fisher.values())).device)
        for n, p in self.model.named_parameters():
            if n in self._fisher:
                diff = p - self._theta_star[n]
                loss = loss + (self._fisher[n] * diff * diff).sum()
        return self.lambda_ewc / 2.0 * loss

    @property
    def n_tasks(self) -> int:
        return self._n_tasks


# =========================================================================
# Online EWC (online Fisher via exponential moving average)
# =========================================================================

class OnlineEWC(EWCRegularizer):
    """Online EWC with EMA Fisher update — no discrete task boundaries needed.

    The Fisher is updated after every gradient step using an EMA:
        F_t ← γ · F_{t-1} + (1-γ) · g_t²

    This is better suited to the continual learning setting where task
    boundaries are not known in advance.

    Parameters
    ----------
    model : nn.Module
    lambda_ewc : float
    ema_gamma : float  — EMA decay factor (default 0.99)
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 500.0,
                 ema_gamma: float = 0.99):
        super().__init__(model, lambda_ewc)
        self.ema_gamma = ema_gamma
        # Initialize Fisher with zeros; θ* with current params
        self._init_from_model()

    def _init_from_model(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._fisher[n] = torch.zeros_like(p)
                self._theta_star[n] = p.data.clone()
        self._n_tasks = 1  # treat as always initialized

    def step_update(self, loss: torch.Tensor):
        """Update Fisher EMA after a backward pass.

        Call AFTER loss.backward() but BEFORE optimizer.step().
        Also refreshes θ* to current params.
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                g2 = p.grad.detach() ** 2
                self._fisher[n] = self.ema_gamma * self._fisher[n] + (1 - self.ema_gamma) * g2
                self._theta_star[n] = p.data.clone()


# =========================================================================
# Synaptic Intelligence (Zenke et al., 2017)
# =========================================================================

class SynapticIntelligence:
    """SI: path-integral importance measure for continual learning.

    Tracks Ω_i = Σ_t [ -g_t · Δθ_t ] / (Δθ_T² + ε)  (importance per param)
    where g_t is the gradient at step t and Δθ_t is the parameter change.

    The penalty is:
        P_SI = λ_si · Σ_i Ω_i · (θ_i - θ*_i)²

    Parameters
    ----------
    model : nn.Module
    lambda_si : float  — regularization strength
    eps : float        — denominator damping
    """

    def __init__(self, model: nn.Module, lambda_si: float = 1.0,
                 eps: float = 1e-3):
        self.model = model
        self.lambda_si = lambda_si
        self.eps = eps

        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._omega: dict = {n: torch.zeros_like(p) for n, p in params.items()}
        self._theta_star: dict = {n: p.data.clone() for n, p in params.items()}
        self._theta_prev: dict = {n: p.data.clone() for n, p in params.items()}
        # Accumulated path integral W (numerator before normalization)
        self._W: dict = {n: torch.zeros_like(p) for n, p in params.items()}
        self._initialized = False

    def before_step(self):
        """Snapshot current params before the gradient step."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._theta_prev[n] = p.data.clone()

    def after_step(self):
        """Update W using gradient × parameter delta. Call AFTER optimizer.step()."""
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                delta = p.data - self._theta_prev[n]
                # W accumulates path integral: -g·Δθ (positive when gradient & step aligned)
                self._W[n] = self._W[n] + (-p.grad.detach() * delta).clamp(min=0)

    def consolidate(self):
        """Call at end of task to compute Ω and reset path integral."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                delta_T = p.data - self._theta_star[n]
                self._omega[n] += self._W[n] / (delta_T ** 2 + self.eps)
                self._omega[n] = torch.clamp(self._omega[n], min=0.0)
                # Reset
                self._W[n].zero_()
                self._theta_star[n] = p.data.clone()
        self._initialized = True

    def penalty(self) -> torch.Tensor:
        """SI quadratic penalty."""
        if not self._initialized:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._omega:
                diff = p - self._theta_star[n]
                loss = loss + (self._omega[n] * diff * diff).sum()
        return self.lambda_si * loss
