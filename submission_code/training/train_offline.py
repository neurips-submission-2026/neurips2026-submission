"""Offline training: Phase 1 (MSE MLP) + Phase 2 (Evidential distillation)."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.evidential_net import EvidentialNet, evidential_loss


# ─── Residual MLP ─────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class InverseModelMLP(nn.Module):
    """Residual MLP: (state_t, delta) → action_t"""
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_blocks=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.output_proj(self.blocks(self.input_proj(x)))


# ─── Training ─────────────────────────────────────────────────────────

def train_two_phase(inputs, targets, device="cpu",
                    hidden_dim=256, n_blocks=4,
                    evid_hidden_dims=(256, 256, 256, 256),
                    phase1_epochs=300, phase2_epochs=150,
                    batch_size=1024, patience=60, fast=False,
                    phase2_lambda_reg=0.005):
    """Two-phase training.

    Phase 1: MSE on residual MLP (gets the mapping right)
    Phase 2: Evidential loss distilled from MLP (adds uncertainty)

    Parameters
    ----------
    fast : bool
        If True, use fewer epochs and smaller architecture for quick testing.

    Returns: (mlp, evidential_model, train_losses, val_losses)
    """
    if fast:
        phase1_epochs = min(phase1_epochs, 60)
        phase2_epochs = min(phase2_epochs, 40)
        patience = min(patience, 20)
        hidden_dim = min(hidden_dim, 128)
        n_blocks = min(n_blocks, 2)
        evid_hidden_dims = tuple(min(d, 128) for d in evid_hidden_dims[:2])
    # Normalize
    inp_mean, inp_std = inputs.mean(0), inputs.std(0) + 1e-8
    tgt_mean, tgt_std = targets.mean(0), targets.std(0) + 1e-8
    inputs_n = (inputs - inp_mean) / inp_std
    targets_n = (targets - tgt_mean) / tgt_std

    n = len(inputs_n)
    n_val = max(int(0.1 * n), min(512, n // 5))
    n_train = n - n_val

    # Ensure batch_size doesn't exceed training samples
    batch_size = min(batch_size, n_train)

    x_train = torch.tensor(inputs_n[:n_train], dtype=torch.float32)
    y_train = torch.tensor(targets_n[:n_train], dtype=torch.float32)
    x_val = torch.tensor(inputs_n[n_train:], dtype=torch.float32)
    y_val = torch.tensor(targets_n[n_train:], dtype=torch.float32)

    train_dl = DataLoader(TensorDataset(x_train, y_train),
                          batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=min(2048, n_val))

    input_dim = inputs.shape[1]
    output_dim = targets.shape[1]

    # ── Phase 1: MSE ──
    print(f"\n  Phase 1: MLP (hidden={hidden_dim}, blocks={n_blocks})")
    mlp = InverseModelMLP(input_dim, output_dim, hidden_dim, n_blocks).to(device)
    print(f"    Params: {sum(p.numel() for p in mlp.parameters()):,}")

    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, steps_per_epoch=len(train_dl), epochs=phase1_epochs)

    best_val, best_state, pat = float("inf"), None, 0
    train_losses, val_losses = [], []

    for epoch in range(phase1_epochs):
        mlp.train()
        el = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.mse_loss(mlp(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 5.0)
            opt.step(); sched.step()
            el += loss.item() * len(xb)
        train_losses.append(el / n_train)

        mlp.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vl += nn.functional.mse_loss(mlp(xb), yb).item() * len(xb)
        val_losses.append(vl / n_val)

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            pat = 0
        else:
            pat += 1

        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:4d}  train={train_losses[-1]:.6f}  "
                  f"val={val_losses[-1]:.6f}")
        if pat >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    mlp.load_state_dict(best_state); mlp.eval()
    print(f"  Phase 1 done. Best val MSE: {best_val:.6f}")

    # ── Phase 2: Evidential ──
    print(f"\n  Phase 2: Evidential (distilled from MLP)")
    evid = EvidentialNet(input_dim, output_dim, hidden_dims=list(evid_hidden_dims)).to(device)

    # Soft labels from MLP
    with torch.no_grad():
        soft_train = mlp(x_train.to(device)).cpu()
        soft_val = mlp(x_val.to(device)).cpu()

    ev_dl = DataLoader(TensorDataset(x_train, soft_train),
                       batch_size=batch_size, shuffle=True, drop_last=True)
    ev_val_dl = DataLoader(TensorDataset(x_val, soft_val), batch_size=2048)

    ev_opt = torch.optim.AdamW(evid.parameters(), lr=5e-4, weight_decay=1e-5)
    ev_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ev_opt, T_max=phase2_epochs)

    best_ev, best_ev_state = float("inf"), None
    ev_train_l, ev_val_l = [], []

    for epoch in range(phase2_epochs):
        evid.train()
        el = 0.0
        for xb, yb in ev_dl:
            xb, yb = xb.to(device), yb.to(device)
            mu, nu, alpha, beta = evid(xb)
            loss = evidential_loss(mu, nu, alpha, beta, yb, lambda_reg=phase2_lambda_reg)
            ev_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(evid.parameters(), 5.0)
            ev_opt.step()
            el += loss.item() * len(xb)
        ev_train_l.append(el / n_train)
        ev_sched.step()

        evid.eval()
        vl, mu_mse = 0.0, 0.0
        with torch.no_grad():
            for xb, yb in ev_val_dl:
                xb, yb = xb.to(device), yb.to(device)
                mu, nu, alpha, beta = evid(xb)
                vl += evidential_loss(mu, nu, alpha, beta, yb, lambda_reg=phase2_lambda_reg).item() * len(xb)
                mu_mse += nn.functional.mse_loss(mu, yb).item() * len(xb)
        ev_val_l.append(vl / n_val)

        if ev_val_l[-1] < best_ev:
            best_ev = ev_val_l[-1]
            best_ev_state = {k: v.cpu().clone() for k, v in evid.state_dict().items()}

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:4d}  ev_loss={ev_val_l[-1]:.6f}  mu_mse={mu_mse/n_val:.6f}")

    evid.load_state_dict(best_ev_state); evid.eval()
    print(f"  Phase 2 done. Best: {best_ev:.6f}")

    # Attach normalization
    norm = {
        "input_mean": torch.tensor(inp_mean, dtype=torch.float32),
        "input_std": torch.tensor(inp_std, dtype=torch.float32),
        "target_mean": torch.tensor(tgt_mean, dtype=torch.float32),
        "target_std": torch.tensor(tgt_std, dtype=torch.float32),
    }
    for m in [mlp, evid]:
        for k, v in norm.items():
            setattr(m, k, v)

    all_train = train_losses + ev_train_l
    all_val = val_losses + ev_val_l
    return mlp, evid, all_train, all_val
