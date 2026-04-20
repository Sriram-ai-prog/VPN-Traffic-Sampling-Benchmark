# dl_samplers.py
"""
Deep-learning-based sampling techniques.
Updated: Added missing KS/CVR reporting to WGAN loop.
"""

import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler

from .config import DEVICE, SEED, WGAN_EPOCHS
from .data_loader import enforce_logical_consistency
from .metrics_utils import compute_distribution_fidelity, compute_constraint_violation_rate
from .models import evaluate_models


# =============================================================================
# 1. FEATURE JITTERING
# =============================================================================
class FeatureJittering:
    def __init__(self, noise_scale=0.05):
        self.noise_scale = noise_scale

    def fit_resample(self, X, y):
        rng = np.random.default_rng(SEED)
        X_np, y_np = X.values, np.asarray(y)
        counts = Counter(y_np)
        target = counts.most_common(1)[0][1]
        X_out, y_out = [X_np], [y_np]

        for cls, cnt in counts.items():
            if cnt >= target: continue
            needed = target - cnt
            idx = np.where(y_np == cls)[0]
            base = X_np[idx]
            sel = rng.choice(len(base), size=needed, replace=True)
            noise = rng.normal(0.0, self.noise_scale, base[sel].shape)
            X_out.append(base[sel] + noise)
            y_out.append(np.full(needed, cls))

        return np.vstack(X_out), np.concatenate(y_out)


def run_feature_jittering(X_train, y_train, X_test, y_test, encoder):
    print("\n--- Feature Jittering ---")
    aug = FeatureJittering(0.05)
    X_res, y_res = aug.fit_resample(X_train, y_train)
    X_res, y_res = enforce_logical_consistency(X_res, y_res, X_train.columns.tolist())

    # Metrics reporting (This was already here)
    fid = compute_distribution_fidelity(X_train, pd.DataFrame(X_res, columns=X_train.columns))
    cvr = compute_constraint_violation_rate(pd.DataFrame(X_res, columns=X_train.columns))
    print(f"[Jittering] KS={fid['mean_ks']:.4f} | CVR={cvr:.4f}")

    return evaluate_models(X_res, y_res, X_test, y_test, encoder, "FeatureJittering")


# =============================================================================
# 2. WGAN-GP (Fixed)
# =============================================================================

class Generator(nn.Module):
    def __init__(self, noise_dim, feat_dim, class_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + class_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, feat_dim)
        )

    def forward(self, z, y): return self.net(torch.cat([z, y], 1))


class Discriminator(nn.Module):
    def __init__(self, feat_dim, class_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + class_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
            # Note: No Sigmoid here (Correct for WGAN)
        )

    def forward(self, x, y): return self.net(torch.cat([x, y], 1))


def _gradient_penalty(D, real, fake, labels):
    alpha = torch.rand(real.size(0), 1, device=DEVICE)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    out = D(interp, labels)
    grad = autograd.grad(out, interp, torch.ones_like(out), True, True, True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def run_wgan(X_train, y_train, X_test, y_test, encoder, epochs=WGAN_EPOCHS, repeats=3):
    print("\n--- Conditional WGAN-GP ---")
    unique_classes = np.unique(y_train)
    class_weights = {c: 1.0 / cnt for c, cnt in Counter(y_train).items()}
    sample_weights = [class_weights[y] for y in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(y_train), replacement=True)

    for run in range(repeats):
        print(f"[WGAN] Run {run + 1}/{repeats}")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_train)

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
        y_oh = torch.nn.functional.one_hot(y_t, len(unique_classes)).float()

        loader = DataLoader(TensorDataset(X_t, y_oh), batch_size=64, sampler=sampler)
        G = Generator(32, X_train.shape[1], len(unique_classes)).to(DEVICE)
        D = Discriminator(X_train.shape[1], len(unique_classes)).to(DEVICE)

        opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
        opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9))

        # Training Loop
        for _ in range(epochs):
            for rx, rlab in loader:
                bs = rx.size(0)
                z = torch.randn(bs, 32, device=DEVICE)
                fake = G(z, rlab)
                d_loss = D(fake.detach(), rlab).mean() - D(rx, rlab).mean() + 10 * _gradient_penalty(D, rx, fake, rlab)
                opt_D.zero_grad();
                d_loss.backward();
                opt_D.step()

                z = torch.randn(bs, 32, device=DEVICE)
                g_loss = -D(G(z, rlab), rlab).mean()
                opt_G.zero_grad();
                g_loss.backward();
                opt_G.step()

        # Generation Loop
        counts = Counter(y_train)
        target = max(counts.values())
        X_syn, y_syn = [X_scaled], [y_train]

        with torch.no_grad():
            for cls, cnt in counts.items():
                if cnt >= target: continue
                n = target - cnt
                z = torch.randn(n, 32, device=DEVICE)
                l = torch.full((n,), cls, device=DEVICE)
                l_oh = torch.nn.functional.one_hot(l, len(unique_classes)).float()
                X_syn.append(G(z, l_oh).cpu().numpy())
                y_syn.append(np.full(n, cls))

        X_final = scaler.inverse_transform(np.vstack(X_syn))
        y_final = np.concatenate(y_syn)

        # Enforce consistency
        X_final, y_final = enforce_logical_consistency(X_final, y_final, X_train.columns.tolist())

        # --- FIX START: Calculate and Print Metrics ---
        X_final_df = pd.DataFrame(X_final, columns=X_train.columns)

        fid = compute_distribution_fidelity(X_train, X_final_df)
        cvr = compute_constraint_violation_rate(X_final_df)

        print(f"[WGAN-GP] KS={fid['mean_ks']:.4f} | CVR={cvr:.4f}")
        # --- FIX END ---

        return evaluate_models(X_final, y_final, X_test, y_test, encoder, "WGAN-GP")