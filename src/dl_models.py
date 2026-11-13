"""LSTM forecaster for the t+1 PM2.5 target (Stage 10).

A small LSTM reads the window from src.windows and predicts PM2.5 at t + 1. Like
the XGBoost model it can learn the change pm(t+1) - pm(t) instead of the level
("delta"), which is easier for such a persistent series. The learning target is
standardised internally for stable training and inverted on predict.

Regularisation: dropout, weight decay, gradient clipping and early stopping on a
held-out tail of the training rows. Seeds are fixed, though MPS/GPU runs can still
vary slightly (checked separately, as for XGBoost). All passes through the network
are mini-batched so memory stays small on the GPU.
"""

import numpy as np
import torch
from torch import nn

SEED = 0


def pick_device():
    """Prefer the Apple GPU (MPS), then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _LSTMNet(nn.Module):
    def __init__(self, n_channels, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]                 # hidden state at the issuance hour t
        return self.head(self.drop(last)).squeeze(-1)


class LSTMForecaster:
    """Plain fit/predict wrapper around a single-output LSTM.

    fit(X, y, pm_t) and predict(X, pm_t) take the 3D window tensor X
    (samples, lookback, channels), the raw t+1 target y, and the raw pm(t) used by
    the "delta" formulation (ignored for "level").
    """

    def __init__(self, formulation="delta", hidden=32, layers=1, dropout=0.1,
                 lr=1e-3, weight_decay=1e-4, epochs=40, batch_size=256,
                 patience=5, val_frac=0.15, seed=SEED):
        assert formulation in ("delta", "level")
        self.formulation = formulation
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        self.seed = seed

    def _base(self, X, pm_t):
        """The value added back after prediction: pm(t) for delta, 0 for level."""
        if self.formulation == "delta":
            return np.asarray(pm_t, dtype="float32")
        return np.zeros(len(X), dtype="float32")

    def _forward(self, X, batch=512):
        """Network output z over X, in mini-batches (keeps GPU memory small)."""
        self.net_.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), batch):
                xb = torch.tensor(X[s:s + batch], device=self.device_)
                out.append(self.net_(xb).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype="float32")

    def fit(self, X, y, pm_t=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device_ = pick_device()

        X = X.astype("float32")
        raw = (np.asarray(y, dtype="float32") - self._base(X, pm_t))   # delta or level
        self.t_mean_ = float(raw.mean())
        self.t_std_ = float(raw.std()) or 1.0
        z = ((raw - self.t_mean_) / self.t_std_).astype("float32")

        # early-stopping holdout: the most recent val_frac of the training rows
        cut = int(len(X) * (1 - self.val_frac))
        Xtr, ztr, Xva, zva = X[:cut], z[:cut], X[cut:], z[cut:]

        self.net_ = _LSTMNet(X.shape[2], self.hidden, self.layers, self.dropout).to(self.device_)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_loss, best_state, bad = float("inf"), None, 0
        order = np.arange(len(Xtr))
        for _ in range(self.epochs):
            self.net_.train()
            np.random.shuffle(order)
            for s in range(0, len(order), self.batch_size):
                b = order[s:s + self.batch_size]
                xb = torch.tensor(Xtr[b], device=self.device_)
                zb = torch.tensor(ztr[b], device=self.device_)
                opt.zero_grad()
                loss = loss_fn(self.net_(xb), zb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()

            val_loss = float(np.mean((self._forward(Xva) - zva) ** 2))
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net_.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        self.net_.load_state_dict(best_state)
        self.n_train_ = int(len(X))
        if self.device_.type == "mps":
            torch.mps.empty_cache()
        return self

    def predict(self, X, pm_t=None):
        X = X.astype("float32")
        z = self._forward(X)
        return z * self.t_std_ + self.t_mean_ + self._base(X, pm_t)
