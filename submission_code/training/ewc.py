"""EWC (Elastic Weight Consolidation) online adapter for inverse dynamics.

Implements the Kirkpatrick et al. (2017) regularisation-based continual
learner adapted to the residual feedforward setting used throughout this
project.

Key design choices (matched to the rest of the benchmark for fairness):

* The base task loss is MSE on the predictive mean ``gamma`` of the
  evidential head, identical to :class:`UniformCLAdapter` (ER baseline).
* Replay is uniform over the same ring buffer so the only methodological
  difference vs ER is the EWC penalty.
* The Fisher diagonal is the empirical squared-gradient of the NIG NLL
  computed from a fixed, held-out subset of the offline training
  distribution. This reuses the NIG model the network was trained on.
* The penalty is ``(lambda_ewc / 2) * sum_i F_i (theta_i - theta_0_i)^2``
  where ``theta_0`` is the offline checkpoint.
* Same ``lr``, ``batch_size``, ``update_every``, ``min_buffer_size``, and
  ``buffer_capacity`` as ER, so any difference in tracking error comes
  from the regulariser, not from infrastructure.
"""
from __future__ import annotations

import os
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as torchF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.train_online import UniformCLAdapter
from models.evidential_net import evidential_loss


def compute_fisher_diagonal(model: torch.nn.Module,
                            inputs: np.ndarray,
                            targets: np.ndarray,
                            num_samples: int = 4096,
                            batch_size: int = 64,
                            device: str = "cpu",
                            normalize: bool = True) -> dict[str, torch.Tensor]:
    """Empirical Fisher diagonal under the NIG NLL loss.

    ``inputs`` and ``targets`` are expected to be the *raw* (un-normalised)
    feature and label arrays from the offline dataset; if ``normalize`` is
    True they are standardised against ``model.input_mean / std`` and
    ``model.target_mean / std`` (matching how the online adapters feed
    data to the network).

    Returns a dict mapping parameter name to a non-negative Fisher value
    of the same shape, averaged over the samples actually used.
    """
    model.eval()
    if normalize:
        inp_m = model.input_mean.cpu().numpy()
        inp_s = model.input_std.cpu().numpy()
        tgt_m = model.target_mean.cpu().numpy()
        tgt_s = model.target_std.cpu().numpy()
        inputs = (inputs - inp_m) / inp_s
        targets = (targets - tgt_m) / tgt_s

    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters() if p.requires_grad}

    n_used = 0
    rng = np.random.default_rng(0)
    n_total = len(inputs)
    if n_total == 0:
        return fisher
    idx = rng.permutation(n_total)
    while n_used < num_samples:
        take = min(batch_size, num_samples - n_used)
        sel = idx[n_used:n_used + take]
        if len(sel) == 0:
            sel = rng.choice(n_total, take, replace=True)
        x = torch.tensor(inputs[sel], dtype=torch.float32, device=device)
        y = torch.tensor(targets[sel], dtype=torch.float32, device=device)

        model.zero_grad()
        gamma, nu, alpha, beta = model(x)
        # NIG NLL — use evidential_loss with no regularisation term to
        # get the pure log-likelihood gradient. Sum so each sample
        # contributes its own grad squared (averaged below by n_used).
        loss = evidential_loss(gamma, nu, alpha, beta, y, lambda_reg=0.0)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2 * len(sel)
        n_used += len(sel)

    for n in fisher:
        fisher[n] = fisher[n] / max(n_used, 1)
    return fisher


def _try_load_offline_dataset() -> tuple[np.ndarray, np.ndarray] | None:
    """Best-effort load of the cached offline dataset for Fisher
    estimation. Returns (inputs, targets) or None if unavailable."""
    candidates = [
        os.path.join(ROOT, "results", "step2_dataset.npz"),
        os.path.join(ROOT, "data", "step2_dataset.npz"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            d = np.load(p)
            if "inputs" in d and "targets" in d:
                return d["inputs"], d["targets"]
    return None


def _synthetic_dataset_from_model(model: torch.nn.Module,
                                  n: int = 4096,
                                  device: str = "cpu",
                                  seed: int = 0
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Fall-back: sample inputs around the training distribution from the
    cached normalisation statistics, then label them with the offline
    network's predictive mean. Used when no cached offline dataset is on
    disk; gives a valid Fisher in the same regime the network was
    trained, just less informative than the true dataset.
    """
    rng = np.random.default_rng(seed)
    inp_m = model.input_mean.cpu().numpy()
    inp_s = model.input_std.cpu().numpy()
    tgt_m = model.target_mean.cpu().numpy()
    tgt_s = model.target_std.cpu().numpy()
    # Standard-normal samples in normalised space, mapped back to raw.
    z = rng.normal(0.0, 1.0, size=(n, len(inp_m))).astype(np.float32)
    inputs_raw = z * inp_s + inp_m
    with torch.no_grad():
        x = torch.tensor((inputs_raw - inp_m) / inp_s,
                         dtype=torch.float32, device=device)
        gamma, _, _, _ = model(x)
        targets_norm = gamma.cpu().numpy()
    targets_raw = targets_norm * tgt_s + tgt_m
    return inputs_raw, targets_raw


class EWCAdapter(UniformCLAdapter):
    """Uniform-replay adapter with an EWC quadratic penalty.

    All construction-time arguments are forwarded to
    :class:`UniformCLAdapter`; the EWC-specific extras are
    ``lambda_ewc``, ``fisher`` (precomputed dict, optional), and
    ``num_fisher_samples`` (used when ``fisher`` is None).
    """

    def __init__(self, model, env, dt,
                 lambda_ewc: float = 1.0,
                 fisher: dict[str, torch.Tensor] | None = None,
                 num_fisher_samples: int = 4096,
                 device: str = "cpu",
                 **kwargs):
        super().__init__(model, env, dt, device=device, **kwargs)
        self.lambda_ewc = float(lambda_ewc)

        # Build Fisher if not supplied. Use the original (pre-deepcopy)
        # ``model`` snapshot for both the Fisher computation and the
        # anchor; the underlying ``self.model`` is a deep copy that the
        # adapter owns and trains.
        if fisher is None:
            data = _try_load_offline_dataset()
            inp_dim = int(self.model.input_mean.numel())
            tgt_dim = int(self.model.target_mean.numel())
            if (data is not None
                    and data[0].shape[-1] == inp_dim
                    and data[1].shape[-1] == tgt_dim):
                inputs, targets = data
            else:
                # Either no cached dataset on disk OR the cached dataset
                # was collected for a different platform/feature mode.
                # Fall back to a synthetic dataset sampled around the
                # offline normalisation stats so the Fisher is still
                # computed in the right input regime.
                inputs, targets = _synthetic_dataset_from_model(
                    self.model, n=num_fisher_samples, device=device)
            fisher = compute_fisher_diagonal(self.model, inputs, targets,
                                             num_samples=num_fisher_samples,
                                             device=device)
        self.fisher = {n: f.to(device) for n, f in fisher.items()}
        # Anchor parameters: snapshot of the current (offline) weights.
        self.theta_0 = {n: p.detach().clone()
                        for n, p in self.model.named_parameters()
                        if p.requires_grad}

    @property
    def info_str(self):
        return (f"EWC: buf={len(self.buffer)}/{self.buffer.capacity} "
                f"updates={self.update_count} λ_ewc={self.lambda_ewc:.3g}")

    # ----- gradient step --------------------------------------------------
    def _do_update(self):
        """One gradient step on MSE(gamma, target) plus EWC penalty."""
        warmup_steps = 50
        if self.update_count < warmup_steps:
            warmup_frac = 0.1 + 0.9 * (self.update_count / warmup_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.target_lr * warmup_frac

        inputs, targets = self.buffer.sample(self.batch_size)
        inp_t = torch.FloatTensor(inputs).to(self.device)
        tgt_t = torch.FloatTensor(targets).to(self.device)

        self.model.train()
        gamma, nu, alpha, beta = self.model(inp_t)
        task_loss = torchF.mse_loss(gamma, tgt_t)
        ewc_term = torch.zeros((), device=inp_t.device)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                diff = p - self.theta_0[n]
                ewc_term = ewc_term + (self.fisher[n] * diff * diff).sum()
        loss = task_loss + 0.5 * self.lambda_ewc * ewc_term

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()
        self.update_count += 1
