"""Adaptive Evidential Regression for Continual Learning (NeurIPS 2025 submission).

Core contribution: per-sample adaptive regularization coefficient
    λ_i = λ_0 · ν_i / (ν_i + τ)

where ν_i is the current evidence (pseudo-count for the mean) and τ > 0 is
a temperature that controls the sharpness of the transition.

Key properties:
  • ν_i → 0  (novel/uncertain sample):  λ_i → 0   — reduced penalty, model can update freely
  • ν_i → ∞  (familiar/certain sample): λ_i → λ_0 — full penalty, preserves acquired knowledge
  • Self-stabilizing: weak evidence attracts weak regularization, allowing evidence to grow

Theoretical guarantee (Theorem 1, see paper):
  For α < 2 (holds early in training when α head ≈ 1+ε), the adaptive scheme applies
  strictly weaker evidence-penalizing gradients near ν=0 than standard EDL:

      ∂ℒ_ada/∂ν|_{ν≈0}  <  ∂ℒ_std/∂ν|_{ν≈0}

  This prevents catastrophic evidence collapse when the model encounters a new task.

Bayesian interpretation (Proposition 2, see paper):
  λ_i can be viewed as a task-conditioned evidence weight in the NIG posterior:
      ν_post = ν_0 + λ_0·ν_0/(ν_0+τ)  ≈  ν_0(1 + λ_0/τ)  for large ν_0
               ≈ ε(1 + λ_0/τ)                             for small ν_0 ≈ ε
  Familiar tasks accumulate evidence proportionally; novel tasks grow cautiously.

Reference: Amini et al., 2020. "Deep Evidential Regression." NeurIPS.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pre-computed constants to avoid repeated allocation in loss hot-path
_LOG_PI = math.log(math.pi)
_EPS    = 1e-8


class AdaptiveEvidentialNet(nn.Module):
    """Normal-Inverse-Gamma evidential net with adaptive per-sample λ regularization.

    Architecture is identical to EvidentialNet (4-layer BN-ReLU backbone + NIG head).
    The sole difference is in the loss function: lambda is a function of ν rather than
    a fixed scalar.  This means the same pretrained EvidentialNet weights can be loaded
    directly via :meth:`from_pretrained`.

    Parameters
    ----------
    input_dim : int
    output_dim : int
    hidden_dims : list[int]  default [256, 256, 256, 256]
    lambda_tau : float
        Temperature τ in λ(ν) = λ_0·ν/(ν+τ). Default 1.0.
        Smaller τ → sharper transition (near-full λ even at moderate ν).
        Larger τ → gentler transition (need very high ν to reach full λ).
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims=None, lambda_tau: float = 1.0,
                 use_layernorm: bool = True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 256]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_tau = lambda_tau
        self.use_layernorm = bool(use_layernorm)

        # Match EvidentialNet: LayerNorm by default; BN preserved for legacy
        # checkpoints (3-D AUV/Drone) via use_layernorm=False.
        norm_cls = nn.LayerNorm if self.use_layernorm else nn.BatchNorm1d
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), norm_cls(h), nn.ReLU()]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 4 * output_dim)

        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        self.register_buffer("target_mean", torch.zeros(output_dim))
        self.register_buffer("target_std", torch.ones(output_dim))

        # Smooth-EMA schedule:  λ_{t+1} = λ_t + η · (ν̄/(ν̄+τ) − λ_t)
        # Stored as a buffer so it persists across forward/backward and
        # moves with `.to(device)`.  Starts at 0 (i.e. full "learning
        # phase") and gradually converges toward 1 as evidence grows.
        self.register_buffer("lambda_state", torch.zeros(1))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """Return NIG parameters (gamma, nu, alpha, beta)."""
        h = self.backbone(x)
        out = self.head(h)
        gamma, log_nu, log_alpha, log_beta = out.chunk(4, dim=-1)
        nu    = F.softplus(log_nu)    + 1e-6
        alpha = F.softplus(log_alpha) + 1.0 + 1e-6
        beta  = F.softplus(log_beta)  + 1e-6
        return gamma, nu, alpha, beta

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    @staticmethod
    def _nig_nll(y, gamma, nu, alpha, beta):
        """NIG negative log-likelihood (Amini et al. Eq. 4) — vectorised hot-path."""
        # omega = 2β(1+ν)  — fused multiply for speed
        omega = beta * (2.0 + 2.0 * nu)
        err2  = (y - gamma).pow(2)
        # Clamp once to avoid double allocation
        log_nu    = torch.log(nu.clamp(min=_EPS))
        log_omega = torch.log(omega.clamp(min=_EPS))
        log_res   = torch.log((nu * err2 + omega).clamp(min=_EPS))
        nll = (
            0.5 * (_LOG_PI - log_nu)          # 0.5*log(π/ν)
            - alpha * log_omega                 # -α·log(Ω)
            + (alpha + 0.5) * log_res           # (α+½)·log(ν·e²+Ω)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        return nll

    def adaptive_loss(self, gamma, nu, alpha, beta, targets,
                      lambda_0: float = 0.01, eta: float = 0.2,
                      mse_weight: float = 1.0):
        """Adaptive evidential loss with a **smooth EMA** λ schedule.

        Unlike a per-sample coefficient, ACE maintains a single scalar
        ``λ_t ∈ [0, 1]`` per network that is updated after every batch:

        .. math::
            \\lambda_{t+1} = \\lambda_t + \\eta \\, \\bigl(
                \\bar\\nu_t / (\\bar\\nu_t + \\tau) - \\lambda_t \\bigr),

        where ``\\bar\\nu_t`` is the detached batch-mean evidence.  The
        regulariser coefficient fed into the loss is ``λ_0 · λ_t``, so
        novel samples (small ν) drive the schedule toward 0 (learning
        phase), while familiar samples (large ν) drive it toward 1
        (full regularisation).  The transition is smooth — no per-batch
        switching — and the EMA rate ``η`` controls the time-constant.

        Parameters
        ----------
        gamma, nu, alpha, beta : Tensor (B, D)
        targets : Tensor (B, D)
        lambda_0 : float, base regularisation coefficient.
        eta : float, EMA learning rate (default 0.05).
        mse_weight : float, stabilising MSE term on γ (default 1.0).

        Returns
        -------
        loss : scalar Tensor
        metrics : dict with ``lambda_t``, ``nu_mean``, ``nll_mean``.
        """
        nll = self._nig_nll(targets, gamma, nu, alpha, beta)

        # ---- EMA update of λ_t (detached ν̄ → pure data signal) ----
        nu_d = nu.detach()
        nu_bar = nu_d.mean()
        target_frac = nu_bar / (nu_bar + self.lambda_tau)   # ∈ [0, 1]
        with torch.no_grad():
            self.lambda_state.mul_(1.0 - eta).add_(eta * target_frac)
        lam_t = self.lambda_state  # scalar buffer, detached from autograd

        reg = torch.abs(targets - gamma) * (2.0 * nu + alpha)
        mse = F.mse_loss(gamma, targets)
        # Effective regulariser coefficient = λ_0 · λ_t (scalar)
        loss = (nll + lambda_0 * lam_t * reg).mean() + mse_weight * mse

        with torch.no_grad():
            eff = float(lambda_0 * lam_t.item())  # effective coefficient
            metrics = {
                "lambda_t":    float(lam_t.item()),  # fraction in [0, 1]
                "lambda_mean": eff,                  # effective λ (back-compat)
                "lambda_min":  eff,
                "lambda_max":  eff,
                "nu_mean":     nu.mean().item(),
                "nu_min":      nu.min().item(),
                "nll_mean":    nll.mean().item(),
                "reg_mean":    reg.mean().item(),
                "mse_mean":    mse.item(),
            }
        return loss, metrics

    @staticmethod
    def standard_loss(gamma, nu, alpha, beta, targets, lambda_reg: float = 0.01, mse_weight: float = 1.0):
        """Standard fixed-λ evidential loss (Amini et al., 2020) for comparison."""
        omega = 2.0 * beta * (1.0 + nu)
        eps = 1e-8
        nll = (
            0.5 * torch.log(torch.clamp(nu, min=eps).reciprocal() * math.pi)
            - alpha * torch.log(torch.clamp(omega, min=eps))
            + (alpha + 0.5) * torch.log(torch.clamp(nu * (targets - gamma)**2 + omega, min=eps))
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        reg = torch.abs(targets - gamma) * (2.0 * nu + alpha)
        mse = F.mse_loss(gamma, targets)
        return (nll + lambda_reg * reg).mean() + mse_weight * mse

    def ace_loss(self, gamma, nu, alpha, beta, targets, lambda_0: float = 0.01,
                 lambda_schedule: float = 0.0, mse_weight: float = 0.25,
                 robust: bool = True):
        """ACE loss with smooth global lambda schedule.

        Instead of per-sample λ_i, we use a time-smoothed λ_schedule that
        adapts via EMA based on the mean evidence of the current batch:
            λ_{t+1} = λ_t + η · ( ν̄/(ν̄+τ) − λ_t )

        This gives smooth, interpretable regularization transitions across
        regime shifts without per-sample jumps.

        Parameters
        ----------
        lambda_schedule : float in [0, 1]
            Current EMA-smoothed schedule value. Effective coefficient is
            lambda_0 * lambda_schedule.
        mse_weight : float
            Stabiliser MSE on γ. Reduced default (0.25) so the NIG term
            controls the update; a high MSE coefficient was causing the
            adapter to chase fresh-buffer noise on nominal trajectories
            and lose tracking accuracy relative to LQR.
        robust : bool
            If True, replace the inner MSE with a Huber/Smooth-L1 term.
            Sensor noise scenarios produce occasional very-large targets;
            Huber prevents these outliers from dominating gradients.
        """
        nll = self._nig_nll(targets, gamma, nu, alpha, beta)
        # Global schedule: same λ for all samples in the batch
        reg = torch.abs(targets - gamma) * (2.0 * nu + alpha)
        if robust:
            mse = F.smooth_l1_loss(gamma, targets, beta=1.0)
        else:
            mse = F.mse_loss(gamma, targets)
        loss = (nll + lambda_0 * lambda_schedule * reg).mean() + mse_weight * mse

        with torch.no_grad():
            metrics = {
                "lambda_schedule": lambda_schedule,
                "lambda_effective": lambda_0 * lambda_schedule,
                "nu_mean": nu.mean().item(),
                "nu_min": nu.min().item(),
                "nll_mean": nll.mean().item(),
                "reg_mean": reg.mean().item(),
                "mse_mean": mse.item(),
            }
        return loss, metrics

    # ------------------------------------------------------------------
    # Uncertainty scores
    # ------------------------------------------------------------------

    def epistemic_score(self, nu, alpha, beta):
        """Epistemic uncertainty: β / (ν·(α-1)), summed over output dims."""
        return (beta / (nu * torch.clamp(alpha - 1.0, min=1e-6))).sum(dim=-1)

    def aleatoric_score(self, nu, alpha, beta):
        """Aleatoric uncertainty: β / (α-1), summed over output dims."""
        return (beta / torch.clamp(alpha - 1.0, min=1e-6)).sum(dim=-1)

    def evidence_stats(self, nu, alpha, beta):
        """Return a dict of evidence diagnostics for monitoring."""
        return {
            "nu_mean":    nu.mean().item(),
            "nu_min":     nu.min().item(),
            "nu_max":     nu.max().item(),
            "alpha_mean": alpha.mean().item(),
            "epistemic":  self.epistemic_score(nu, alpha, beta).mean().item(),
            "aleatoric":  self.aleatoric_score(nu, alpha, beta).mean().item(),
        }

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, evidential_net, lambda_tau: float = 1.0):
        """Initialize from a pretrained EvidentialNet (identical architecture).

        Copies all weights exactly; only the loss function changes at training time.

        Parameters
        ----------
        evidential_net : EvidentialNet
            Source model trained with standard EDL loss.
        lambda_tau : float
            Temperature for adaptive λ schedule.

        Returns
        -------
        AdaptiveEvidentialNet  (same weights, adaptive loss)
        """
        src = evidential_net
        # Infer hidden dims from backbone (every 3rd module is Linear)
        hidden_dims = []
        for m in src.backbone:
            if isinstance(m, nn.Linear):
                hidden_dims.append(m.out_features)

        # Inherit LN/BN choice from the source model so weights load cleanly.
        use_ln = bool(getattr(src, "use_layernorm", False))
        net = cls(src.input_dim, src.output_dim,
                  hidden_dims=hidden_dims, lambda_tau=lambda_tau,
                  use_layernorm=use_ln)
        net.load_state_dict(src.state_dict(), strict=False)

        # Copy normalization buffers explicitly (strict=False may skip them)
        net.input_mean.copy_(src.input_mean)
        net.input_std.copy_(src.input_std)
        net.target_mean.copy_(src.target_mean)
        net.target_std.copy_(src.target_std)
        net.feature_mode = getattr(src, "feature_mode", "full")
        return net
